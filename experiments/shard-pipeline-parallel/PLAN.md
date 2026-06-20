# Execution Plan — Shard-Style Pipeline Parallel Inference

**Branch:** `experiments/shard-pipeline-parallel`  
**Created:** 2026-06-20  
**Reviewed by:** GLM-5.2 (big model pass)

---

## Critical Finding: Phase 1 Is Simpler Than AGENTS.md Estimates

The AGENTS.md estimates 500-800 lines and 2-3 weeks for async RPC. After reading the actual code, the implementation is significantly smaller. Here's why:

**GRAPH_COMPUTE is already fire-and-forget.** The client sends the graph data but never reads a response. The server's `rpc_serve_client` handler for `RPC_CMD_GRAPH_COMPUTE` processes the graph and sends nothing back:

```cpp
// Server side — no response sent after graph_compute:
case RPC_CMD_GRAPH_COMPUTE: {
    std::vector<uint8_t> input;
    recv_msg(sock, input);
    server.graph_compute(input);
    break;  // no send_msg — fire and forget
}
```

```cpp
// Client side — send_rpc_cmd without response:
bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
// returns immediately after send, no recv
```

**Synchronization currently happens implicitly.** The server processes commands sequentially on a single TCP connection. When the client sends `GET_TENSOR` after `GRAPH_COMPUTE`, the server processes `GRAPH_COMPUTE` first, then `GET_TENSOR`. The `GET_TENSOR` response implicitly confirms the computation is done.

**The `synchronize` function is a no-op:**
```cpp
static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}
```

**This means async RPC only needs:**
1. One new command: `RPC_CMD_SYNC` (server responds with empty message — acts as a barrier)
2. Event functions that use SYNC to wait for remote computation
3. `synchronize()` that sends SYNC and waits
4. Flip `caps.async` and `caps.events` to `true`

**Revised estimate: ~200-300 lines, not 500-800.** The connection pool mentioned in AGENTS.md is unnecessary because the server is single-threaded per connection and commands are processed in order. The fire-and-forget GRAPH_COMPUTE already gives us non-blocking dispatch.

---

## Architecture Refinement: Blocking vs Non-Blocking Events

**CUDA events** are non-blocking on the host. `event_record` enqueues an event on a stream. `event_wait` enqueues a stream-level dependency. The host thread never blocks.

**RPC events** can't easily be non-blocking because we're over TCP. The simplest implementation uses **blocking events**: `event_wait` sends `RPC_CMD_SYNC` to the event's endpoint and blocks until the response arrives.

**Impact on pipeline parallelism:** With blocking events, there's no host-side overlap between splits. The total time is `compute_A + copy_AB + compute_B + copy_BC + compute_C` (sequential). With non-blocking events, it would be `max(compute_A, copy_AB) + max(compute_B, copy_BC) + compute_C` (overlapped).

**Decision:** Start with blocking events (simple, correct, unblocks the code path). Measure performance. If blocking is the bottleneck, implement non-blocking async tensor copy with deferred sync in a follow-up.

Even with blocking events, there IS a benefit over the current fully-synchronous model: the scheduler can dispatch `GRAPH_COMPUTE` (fire-and-forget) and move to the next split's setup before blocking on the sync. The server-side computation on different endpoints also overlaps naturally.

---

## Revised Phase Structure

The AGENTS.md has 4 phases with direct-return as a separate tier. I'm restructuring:

- **Direct-return merged into Phase 2** — it's integral to the coordinator, not separable
- **Phase 0 added** — baseline verification before changing anything
- **Phase 2 split into smaller tasks** — the coordinator is the hardest part, each task must be atomic
- **WAN compression demoted** — it's a 10% optimization, not a phase

---

## Task Breakdown

Each task is designed for a lesser model to execute in 1-3 sessions. Tasks include exact files, line numbers, what to change, how to verify, and dependencies.

---

### Phase 0: Baseline Verification

#### T0.1 — Verify RPC works on current branch

**Goal:** Confirm the existing RPC backend works before we modify anything.

**Steps:**
1. Build the RPC server binary: `build_turbo4.bat` should produce `build/bin/rpc-server.exe` (or `llama-rpc-server.exe`). Check `build/bin/` for the binary.
2. Start an RPC server on port 8082: `rpc-server.exe --host 127.0.0.1 --port 8082`
3. Start a llama-server with RPC backend: `llama-server.exe --model <small-model> --rpc 127.0.0.1:8082 -ngl 999 --split-mode layer`
4. Send a test prompt via curl and verify correct output.
5. Check server logs for "pipeline parallelism" — it should say **disabled** because `caps.async = false`.

**Files to read:**
- `tools/rpc/rpc-server.cpp` — CLI args for the server
- `examples/main/main.cpp` or `examples/server/server.cpp` — how `--rpc` flag is handled

**Verify:**
- RPC server starts and accepts connections
- llama-server connects and loads model layers across local + RPC device
- Inference produces correct output
- Log shows pipeline parallelism is disabled

**Dependencies:** None.  
**Effort:** 1 session.  
**Deliverable:** Working RPC baseline + notes on exact commands used.

---

### Phase 1: Async RPC (Unlocks Pipeline Parallelism)

#### T1.1 — Add RPC_CMD_SYNC command (server-side)

**Goal:** Add a new RPC command that the server responds to with an empty message. This acts as a synchronization barrier — since the server processes commands sequentially, the response confirms all prior commands (including GRAPH_COMPUTE) are complete.

**Files to modify:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Changes:**

1. Add to `enum rpc_cmd` (line ~74, before `RPC_CMD_COUNT`):
```cpp
RPC_CMD_SYNC,
```

2. Add handler in `rpc_serve_client` switch statement (after `RPC_CMD_GET_DEVICE_MEMORY`):
```cpp
case RPC_CMD_SYNC: {
    // no request data to read
    // send empty response (just size = 0)
    if (!send_msg(sock, nullptr, 0)) {
        return;
    }
    break;
}
```

3. Update `RPC_CMD_COUNT` assertion if needed.

**Verify:**
- Build succeeds
- Server starts without errors
- Manual test: connect with a raw TCP client, send SYNC command byte, receive response

**Dependencies:** None.  
**Effort:** 1 session.  
**Deliverable:** SYNC command works server-side.

---

#### T1.2 — Implement RPC event functions

**Goal:** Implement `event_new`, `event_free`, `event_record`, `event_wait`, `event_synchronize` for the RPC backend.

**Files to modify:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Changes:**

1. Define RPC event context (add near top of file, after existing structs):
```cpp
struct rpc_event_context {
    std::string endpoint;     // endpoint of the backend that recorded this event
    bool       has_pending;   // true after event_record, false after event_wait/synchronize
};
```

2. Implement device-level event functions:
```cpp
static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    rpc_backend_rpc_device_context * dev_ctx = 
        (rpc_backend_rpc_device_context *)dev->context;
    rpc_event_context * ev_ctx = new rpc_event_context;
    ev_ctx->endpoint = dev_ctx->endpoint;
    ev_ctx->has_pending = false;
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev_ctx,
    };
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    delete (rpc_event_context *)event->context;
    delete event;
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    rpc_event_context * ev_ctx = (rpc_event_context *)event->context;
    if (!ev_ctx->has_pending) return;
    auto sock = get_socket(ev_ctx->endpoint);
    if (!sock) return;
    // send SYNC command, wait for response
    bool status = send_rpc_cmd(sock, RPC_CMD_SYNC, nullptr, 0, nullptr, 0);
    RPC_STATUS_ASSERT(status);
    ev_ctx->has_pending = false;
}
```

3. Implement backend-level event functions:
```cpp
static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    rpc_event_context * ev_ctx = (rpc_event_context *)event->context;
    // graph_compute was already sent fire-and-forget
    // mark that we need to sync before reading results
    ev_ctx->has_pending = true;
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // delegate to event_synchronize — for RPC, waiting means blocking
    ggml_backend_rpc_device_event_synchronize(event->device, event);
}
```

4. Wire into device interface (line ~1851):
```cpp
/* .event_new            = */ ggml_backend_rpc_device_event_new,
/* .event_free           = */ ggml_backend_rpc_device_event_free,
/* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
```

5. Wire into backend interface (line ~737):
```cpp
/* .event_record            = */ ggml_backend_rpc_event_record,
/* .event_wait              = */ ggml_backend_rpc_event_wait,
```

**Verify:**
- Build succeeds
- No crashes when events are created/freed
- Event record + wait cycle works (send graph, record event, wait event = blocks until server done)

**Dependencies:** T1.1 (SYNC command must exist).  
**Effort:** 1-2 sessions.  
**Deliverable:** Event functions implemented and wired in.

---

#### T1.3 — Implement synchronize() and enable caps

**Goal:** Make `synchronize()` actually sync, and flip the caps flags to enable pipeline parallelism.

**Files to modify:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp`

**Changes:**

1. Implement `synchronize` (replace no-op at line ~652):
```cpp
static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = 
        (ggml_backend_rpc_context *)backend->context;
    auto sock = get_socket(rpc_ctx->endpoint);
    if (!sock) return;
    // send SYNC, wait for response — blocks until all prior commands processed
    bool status = send_rpc_cmd(sock, RPC_CMD_SYNC, nullptr, 0, nullptr, 0);
    RPC_STATUS_ASSERT(status);
}
```

2. Enable caps (line ~1798):
```cpp
props->caps = {
    /* .async                 = */ true,
    /* .host_buffer           = */ false,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ true,
};
```

**Verify:**
- Build succeeds
- Start llama-server with 2+ devices (1 local GPU + 1 RPC)
- Server log now says **"pipeline parallelism enabled"**
- Inference produces correct output (no garbage, no crashes)
- Compare output with baseline T0.1 — must be byte-identical for greedy decoding

**Dependencies:** T1.1, T1.2.  
**Effort:** 1 session.  
**Deliverable:** Pipeline parallelism enabled and working with correct output.

---

#### T1.4 — Test pipeline parallel with 2 RPC servers

**Goal:** Verify pipeline parallelism works with multiple RPC backends, not just one.

**Steps:**
1. Start 2 RPC servers on different ports (8082, 8083)
2. Start llama-server with both: `--rpc 127.0.0.1:8082,127.0.0.1:8083 --split-mode layer -ngl 999`
3. Run inference, verify correct output
4. Benchmark: compare tok/s with pipeline parallel vs without (disable by setting caps back to false temporarily)
5. Test with longer generations (200+ tokens) to verify stability

**Verify:**
- Both RPC servers accept connections
- Layers distributed across all 3 devices (local + 2 RPC)
- Correct output for greedy decoding
- No deadlocks or hangs
- Performance with pipeline parallel >= performance without (at least not worse)

**Dependencies:** T1.3.  
**Effort:** 1-2 sessions.  
**Deliverable:** Working multi-RPC pipeline parallel inference + benchmark numbers.

---

### Phase 2: Coordinator + Direct-Return + n-gram Draft

This is the hard part. The coordinator is new code that implements Shard-style draft/verify loop. Direct-return routing is integral — the tail node sends results directly to the coordinator, not through the relay chain.

#### T2.1 — Direct-return connection infrastructure

**Goal:** Add the ability for an RPC server (tail node) to push results directly to the coordinator, bypassing the relay chain.

**Current state:** RPC servers only accept inbound connections and respond to commands. They never initiate outbound connections. The client (coordinator) pulls results via `GET_TENSOR`.

**Target state:** The tail node can be configured with a "return endpoint" (coordinator's address). After computing a graph, it pushes the output tensor to the return endpoint.

**Files to modify:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — add return-endpoint config and push logic
- `tools/rpc/rpc-server.cpp` — add `--return-endpoint` CLI arg
- `ggml/include/ggml-rpc.h` — expose return-endpoint config if needed

**Changes:**

1. Add `--return-endpoint <host:port>` to rpc-server CLI. Store in a global or pass to `rpc_serve_client`.

2. Add a new command `RPC_CMD_PUSH_TENSOR` — a server can send this to another server (or coordinator) to push a tensor:
```
Request:  | cmd (1B) | tensor_id (8B) | data_size (8B) | data (data_size bytes) |
Response: | status (1B) |
```

3. After `graph_compute` on the tail node, if a return endpoint is configured, send the output tensor via `RPC_CMD_PUSH_TENSOR` to the return endpoint.

4. The coordinator (llama-server) needs a listener for push commands. This could be a separate thread that accepts connections and processes `RPC_CMD_PUSH_TENSOR` commands, storing the received tensors in a queue.

**Design note:** This is the most significant protocol change. The RPC backend currently has a strict client-server model. Direct-return requires server-to-server (or server-to-coordinator) communication. The simplest approach:
- The coordinator opens a listener socket (like an RPC server)
- The tail node connects to the coordinator's listener
- After graph_compute, the tail pushes the output tensor
- The coordinator reads from the push queue when it needs the result

**Verify:**
- Tail node connects to coordinator's return listener
- After graph_compute, tail pushes output tensor
- Coordinator receives the tensor correctly (byte-identical to GET_TENSOR result)
- No deadlock when both forward and return connections are active

**Dependencies:** T1.4 (need working multi-RPC first).  
**Effort:** 2-3 sessions.  
**Deliverable:** Direct-return TCP channel works, tail pushes results to coordinator.

---

#### T2.2 — Coordinator skeleton: draft/verify loop with n-gram

**Goal:** Build the coordinator that manages the Shard-style draft/verify loop using n-gram drafting (zero-cost, no model needed).

**Files to create:**
- `src/llama-coordinator.h` — coordinator interface
- `src/llama-coordinator.cpp` — coordinator implementation

**Files to modify:**
- `src/llama-context.cpp` — integrate coordinator into decode loop
- `common/common.h` — add coordinator parameters

**Coordinator interface:**
```cpp
struct llama_coordinator_params {
    int    n_draft_k = 2;           // tokens to draft per round
    int    n_pipeline_depth = 2;    // chunks in flight (start conservative)
    bool   use_direct_return = false; // use direct-return routing
    // n-gram params
    int    ngram_n_min = 3;
    int    ngram_n_max = 5;
};

class llama_coordinator {
public:
    llama_coordinator(llama_context * ctx, const llama_coordinator_params & params);
    
    // Main loop: draft K tokens, send to ring, verify, accept
    // Returns accepted tokens
    std::vector<llama_token> decode_step();
    
    // Prefill: send entire prompt through the ring
    void prefill(const std::vector<llama_token> & tokens);
    
private:
    llama_context * ctx;
    llama_coordinator_params params;
    // ... n-gram cache, pipeline state, etc.
};
```

**Draft/verify loop (simplified, no pipelining yet):**
1. Draft K tokens using n-gram-k4v or ngram-mod (call existing speculative decode functions)
2. Send `[cur, d₁, ..., dₖ]` through the pipeline (GRAPH_COMPUTE on each stage in order)
3. Wait for tail to return verified logits (via direct-return or GET_TENSOR)
4. Greedy accept: find longest prefix match, accept n draft + 1 correction
5. Crop KV caches on all nodes (lazy — piggyback on next verify)
6. Repeat

**Key integration point:** The coordinator replaces the normal decode loop in `llama_context`. When coordinator mode is enabled, `llama_decode` calls the coordinator's `decode_step` instead of the normal single-token decode.

**Verify:**
- Coordinator drafts K tokens using n-gram
- Tokens sent through pipeline correctly
- Greedy acceptance produces correct output (byte-identical to plain greedy decode)
- Output is coherent for 100+ token generations

**Dependencies:** T2.1 (direct-return for result retrieval).  
**Effort:** 3-4 sessions.  
**Deliverable:** Basic draft/verify loop works end-to-end with n-gram drafting.

---

#### T2.3 — KV cache crop protocol

**Goal:** Implement lazy KV cache crop on divergence. When a draft token is rejected, all nodes need to crop their KV caches back to the accepted prefix.

**Approach:** Crop is piggybacked on the next verify message. Each verify message includes a `start_pos` field. When a node receives `start_pos < current_cache_size`, it crops its cache to `start_pos` before processing.

**Files to modify:**
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — add `start_pos` to GRAPH_COMPUTE request, implement crop on server side
- `src/llama-coordinator.cpp` — send `start_pos` with each verify

**Changes:**

1. Extend GRAPH_COMPUTE request format:
```
Old: | device (4B) | n_nodes (4B) | nodes (...) | n_tensors (4B) | tensors (...) |
New: | device (4B) | start_pos (8B) | n_nodes (4B) | nodes (...) | n_tensors (4B) | tensors (...) |
```

2. Server-side: when `start_pos < current_seq_len`, call `llama_kv_cache_seq_rm` or equivalent to crop the cache before computing the graph.

3. Coordinator: on divergence, set `start_pos` to the corrected position for the next verify.

**Verify:**
- After divergence, KV caches are correctly cropped on all nodes
- Subsequent generations are correct (no garbage from stale cache)
- Test: intentionally cause divergence (bad draft), verify recovery
- 500+ token generation with multiple divergences produces coherent output

**Dependencies:** T2.2.  
**Effort:** 2 sessions.  
**Deliverable:** KV cache crop works correctly on divergence.

---

#### T2.4 — Async pipelining (multiple verify chunks in flight)

**Goal:** Allow the coordinator to fire multiple verify chunks without waiting, overlapping WAN latency with computation.

**This is where the real performance gains come from** — Shard saw 5.6× improvement from pipelining.

**Files to modify:**
- `src/llama-coordinator.cpp` — add pipeline state management

**Changes:**

1. Add pipeline state:
```cpp
struct pipeline_chunk {
    int                pos;       // starting position
    std::vector<llama_token> drafts;  // K draft tokens
    bool               active;    // in flight
};

std::vector<pipeline_chunk> inflight;  // chunks currently in the pipeline
int max_depth = 2;  // conservative start
```

2. Implement the three-phase loop (from Shard's algorithm):
```
PHASE 1: Fill pipeline — while inflight.size() < max_depth, draft K tokens, send to ring
PHASE 2: Wait for one result — blocking recv from tail
PHASE 3: Accept/correct — greedy accept, crop on divergence, clear inflight on divergence
```

3. On divergence: all inflight chunks are stale. Clear them. Crop draft cache. Restart from corrected position.

4. Overlap-by-1: consecutive chunks share one token position for continuity.

**Verify:**
- Multiple chunks in flight simultaneously (verify with debug logging)
- Pipeline fill/drain cycle works correctly
- Divergence correctly invalidates all inflight chunks
- Output is byte-identical to non-pipelined version (pipelining is an optimization, not a semantic change)
- Performance: measure tok/s at depth=1 (baseline), depth=2, depth=4 — should see improvement

**Dependencies:** T2.2, T2.3.  
**Effort:** 2-3 sessions.  
**Deliverable:** Async pipelining works, performance scales with depth.

---

#### T2.5 — End-to-end test with n-gram draft

**Goal:** Comprehensive testing of the full pipeline: n-gram draft → pipeline verify → greedy accept → KV cache crop → async pipelining.

**Tests:**
1. **Correctness:** Output byte-identical to plain greedy decode for 5 different prompts
2. **Stability:** 1000+ token generation with 0% garbage
3. **Divergence recovery:** Intentionally bad drafts → verify recovery is correct
4. **Performance:** tok/s at various pipeline depths (1, 2, 4)
5. **Multi-node:** Test with 2 and 3 RPC servers
6. **KV cache:** Verify cache sizes on all nodes after divergence + crop

**Dependencies:** T2.4.  
**Effort:** 1-2 sessions.  
**Deliverable:** All tests pass, performance numbers recorded.

---

#### T2.6 — Performance tuning

**Goal:** Find optimal pipeline depth and K (tokens per verify) for the hardware setup.

**Experiments:**
1. K = 1, 2, 3, 4 at depth = 1, 2, 4, 6
2. Measure: tok/s, acceptance rate, wasted compute ratio
3. Adaptive K: `K = max(1, min(K_max, round(ema_accepted) + 1))`
4. Adaptive depth: based on measured RTT, `depth = max(2, rtt_ms / draft_time_ms)`

**Dependencies:** T2.5.  
**Effort:** 1-2 sessions.  
**Deliverable:** Optimal parameters identified, adaptive tuning implemented.

---

### Phase 3: MTP Integration

#### T3.1 — Extract MTP predictions from tail node

**Goal:** The tail node's forward pass produces MTP predictions (multiple future token logits). Extract these and send them to the coordinator via direct-return.

**Files to read first:**
- `src/llama-context.cpp` — search for `LLAMA_CONTEXT_TYPE_MTP`, understand how MTP predictions are produced
- `src/llama-model.cpp:2007-2100` — MTP graph building
- `include/llama.h:208` — `LLAMA_CONTEXT_TYPE_MTP`

**Changes:**
1. On the tail node, after graph_compute, extract the MTP prediction tensors
2. Include them in the direct-return push message (alongside the verified hidden states)
3. The coordinator receives both verified logits AND MTP predictions

**Key question to resolve:** Does MTP produce token IDs directly, or logits that need argmax? This determines how much data to send over the wire.

**Verify:**
- MTP predictions correctly extracted from tail node
- Predictions match what a single-GPU MTP run would produce
- Direct-return message includes both verified output and MTP predictions

**Dependencies:** T2.1 (direct-return), T2.2 (coordinator).  
**Effort:** 2 sessions.  
**Deliverable:** MTP predictions flow from tail to coordinator.

---

#### T3.2 — Coordinator uses MTP predictions as draft

**Goal:** Instead of n-gram drafting, the coordinator uses the MTP predictions from the tail node as the draft for the next verify round.

**Changes:**
1. Coordinator: when MTP predictions are available, use them as draft tokens instead of n-gram
2. Fallback: if MTP predictions are not available (e.g., first token), use n-gram as fallback
3. The draft/verify loop becomes: receive MTP predictions from previous round → use as draft → verify → receive new MTP predictions → repeat

**Verify:**
- MTP-driven drafting produces correct output
- Acceptance rate with MTP >= acceptance rate with n-gram
- Fallback to n-gram works for first token and after divergence

**Dependencies:** T3.1.  
**Effort:** 1-2 sessions.  
**Deliverable:** MTP-driven pipeline parallel inference works.

---

#### T3.3 — Benchmark MTP vs n-gram over WAN

**Goal:** Compare MTP and n-gram drafting for pipeline parallel inference.

**Metrics:**
- tok/s (decode speed)
- Acceptance rate (fraction of draft tokens accepted)
- Wasted compute ratio (inflight chunks discarded on divergence)
- Latency per token

**Dependencies:** T3.2.  
**Effort:** 1 session.  
**Deliverable:** Benchmark comparison table, recommendation for which drafting method to use.

---

### Phase 4: WAN Optimization

#### T4.1 — Tensor compression

**Goal:** Compress tensor data sent over the network to reduce bandwidth usage on WAN links.

**Files to modify:**
- `ggml/src/ggml-rpc/transport.cpp` — add zstd or lz4 compression for large payloads
- `ggml/src/ggml-rpc/ggml-rpc.cpp` — compress SET_TENSOR and GRAPH_COMPUTE data

**Approach:**
- Only compress payloads > 10KB (small payloads have overhead)
- Use zstd (good ratio, fast) or lz4 (fast, lower ratio)
- Compress on send, decompress on recv
- Negotiate compression support in HELLO handshake

**Verify:**
- Compression reduces bandwidth by >= 50% for typical tensor sizes
- No correctness issues (compressed + decompressed data is byte-identical)
- Latency doesn't increase (compression time < network time saved)

**Dependencies:** None (can be done independently of Phase 2/3).  
**Effort:** 1-2 sessions.  
**Deliverable:** Tensor compression works, bandwidth measured.

---

#### T4.2 — Topology solver

**Goal:** Port Shard's Held-Karp TSP solver to compute optimal stage ordering for minimum WAN latency.

**Approach:**
- Can be a standalone tool (Python or C++) that takes RTT measurements and outputs optimal ordering
- Or integrate into the coordinator startup
- Shard's Python implementation in `shard/topology.py` can be used as-is for setup

**Files to create:**
- `tools/topology/topology.cpp` (optional — or just use Shard's Python)

**Verify:**
- Given a 4x4 RTT matrix, solver outputs the correct minimum-latency ordering
- Compare with Shard's Python solver for same input — must match

**Dependencies:** None.  
**Effort:** 1 session.  
**Deliverable:** Topology solver works, optimal ordering computed.

---

#### T4.3 — WAN benchmark + adaptive pipeline depth

**Goal:** Test over real WAN (or simulated WAN latency) and tune parameters.

**Steps:**
1. Use `tc` (traffic control) on Linux or Clumsy on Windows to add artificial latency (20-100ms)
2. Benchmark at various latency levels
3. Implement adaptive pipeline depth: `depth = max(2, measured_rtt_ms / draft_time_ms)`
4. Test with real WAN if possible (2 machines on different networks)

**Verify:**
- Performance degrades gracefully with increasing latency
- Adaptive depth improves performance vs fixed depth at high latency
- No instability at high latency (200+ ms)

**Dependencies:** T2.6, T4.1, T4.2.  
**Effort:** 2 sessions.  
**Deliverable:** WAN-ready configuration with adaptive tuning.

---

## Dependency Graph

```
T0.1 (baseline)
  └─► T1.1 (SYNC cmd)
        └─► T1.2 (events)
              └─► T1.3 (synchronize + caps)
                    └─► T1.4 (multi-RPC test)
                          └─► T2.1 (direct-return)
                                └─► T2.2 (coordinator)
                                      └─► T2.3 (KV crop)
                                            └─► T2.4 (pipelining)
                                                  └─► T2.5 (e2e test)
                                                        └─► T2.6 (tuning)
                                                              └─► T4.3 (WAN bench)

T3.1 (MTP extract) — depends on T2.1, T2.2
  └─► T3.2 (MTP draft)
        └─► T3.3 (MTP vs ngram)

T4.1 (compression) — independent
T4.2 (topology) — independent
```

**Critical path:** T0.1 → T1.1 → T1.2 → T1.3 → T1.4 → T2.1 → T2.2 → T2.3 → T2.4 → T2.5 → T2.6 → T4.3

**Parallelizable:** T4.1 and T4.2 can be done at any time. T3.x can start after T2.2.

---

## Revised Effort Estimate

| Phase | Tasks | Sessions | Calendar (1 person) |
|-------|-------|----------|---------------------|
| Phase 0 | 1 | 1 | 1 day |
| Phase 1 | 4 | 4-6 | 1 week |
| Phase 2 | 6 | 11-16 | 2-3 weeks |
| Phase 3 | 3 | 4-5 | 1 week |
| Phase 4 | 3 | 4-5 | 1 week |
| **Total** | **17** | **24-33** | **5-6 weeks** |

This is tighter than the AGENTS.md's 7-11 weeks because:
1. Phase 1 is ~200 lines, not 500-800 (fire-and-forget discovery)
2. Direct-return merged into Phase 2, not separate
3. WAN compression and topology demoted to minor tasks
4. Each task is well-scoped for subagent execution

---

## What NOT to Do

1. **Don't implement a connection pool.** The single-connection-per-endpoint model works because the server processes commands sequentially. GRAPH_COMPUTE is fire-and-forget. A connection pool adds complexity with no benefit for the current architecture.

2. **Don't implement non-blocking events in Phase 1.** Blocking events (send SYNC, wait for response) are correct and simple. Non-blocking events require async tensor copy and deferred synchronization, which is a separate optimization.

3. **Don't try to support DFlash in the initial implementation.** DFlash (block-diffusion) has a fundamentally different architecture (it's a diffusion model, not a draft/verify model). Out of scope.

4. **Don't modify the scheduler.** The scheduler in `ggml-backend.cpp` already does the right thing — split at backend boundaries, use events for synchronization. We just need to enable the code path by implementing the RPC backend's event interface.

5. **Don't change the wire protocol for existing commands.** Only ADD new commands (SYNC, PUSH_TENSOR). Changing existing command formats breaks compatibility.

6. **Don't skip T0.1 (baseline).** If the existing RPC doesn't work on this branch, everything else is wasted. Verify first.

---

## Build & Test Infrastructure

**Build:** Use `build_turbo4.bat` (the established build script). Do NOT use ad-hoc cmake/ninja commands.

**Test model:** Use a small model (e.g., 1-3B params) for Phase 0-1 testing. Switch to the 27B Ornstein model for Phase 2+ testing.

**RPC server:** `build/bin/rpc-server.exe` (or `llama-rpc-server.exe`). Check `build/bin/` after build.

**Test commands:**
```bash
# Terminal 1: RPC server
rpc-server.exe --host 127.0.0.1 --port 8082

# Terminal 2: llama-server with RPC
llama-server.exe --model <model.gguf> --rpc 127.0.0.1:8082 -ngl 999 --split-mode layer -c 4096

# Terminal 3: test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10}'
```

**For multi-RPC testing:**
```bash
# Terminal 1: RPC server A
rpc-server.exe --host 127.0.0.1 --port 8082
# Terminal 2: RPC server B  
rpc-server.exe --host 127.0.0.1 --port 8083
# Terminal 3: llama-server with both
llama-server.exe --model <model.gguf> --rpc 127.0.0.1:8082,127.0.0.1:8083 -ngl 999 --split-mode layer -c 4096
```

---

## Risk Register (Updated)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Blocking events limit pipeline overlap | High | Medium | Accept for Phase 1. Measure. Non-blocking is a follow-up optimization. |
| KV cache crop has edge cases | Medium | High | T2.3 has dedicated testing. Start with eager crop (not lazy) if lazy is buggy. |
| Direct-return requires protocol changes | Medium | Medium | T2.1 is well-scoped. Fallback to GET_TENSOR if direct-return is too complex. |
| MTP predictions format unclear | Medium | Low | T3.1 starts by reading the MTP code. Can fall back to n-gram if MTP is too coupled. |
| Multi-RPC testing on single machine | Low | Low | Use different ports. For real WAN testing, use T4.3 with artificial latency. |
| Build issues on experiment branch | Low | High | Use `build_turbo4.bat`. Check `build/bin/` for binaries before assuming build failed. |

---

## Success Metrics

| Milestone | Metric | Target |
|-----------|--------|--------|
| Phase 1 complete | Pipeline parallel enabled | Log says "pipeline parallelism enabled" |
| Phase 1 complete | Correctness | Byte-identical to single-GPU greedy decode |
| Phase 1 complete | Performance | >= 50% of single-GPU tok/s (LAN, 2 nodes) |
| Phase 2 complete | Correctness | 1000+ tokens, 0% garbage |
| Phase 2 complete | Performance | >= 30% of single-GPU tok/s (WAN, n-gram draft) |
| Phase 2 complete | Pipelining | depth=4 >= 2× depth=1 tok/s |
| Phase 3 complete | MTP acceptance | >= n-gram acceptance rate |
| Phase 4 complete | WAN performance | >= 60% of single-GPU tok/s over real WAN |

---

**Document version:** 1.0  
**Last updated:** 2026-06-20  
**Reviewer:** GLM-5.2
