# Shard-Style Pipeline Parallel Inference — Experiment Branch

**Branch:** `experiments/shard-pipeline-parallel`  
**Base:** buun-llama-cpp master (commit TBD)  
**Goal:** Adapt buun-llama-cpp's RPC backend to support Shard-style pipeline-parallel inference over WAN with speculative decoding.

---

## Executive Summary

**Shard** (https://github.com/leyten/shard) serves 744B-parameter models across 6-7 scattered GPUs over WAN at ~30 tok/s using:
- Pipeline-parallel layer distribution (contiguous blocks per GPU)
- Speculative decoding over WAN (draft K tokens locally, verify across all nodes in one traversal)
- Async pipelining (multiple verify chunks in flight simultaneously)
- Direct-return routing (tail → coordinator in 1 hop, not relay-back)

**buun-llama-cpp** already has:
- RPC backend with tensor serialization, graph compute, up to 16 servers
- Layer split mode (`LLAMA_SPLIT_MODE_LAYER`) distributes layers across devices
- RDMA support (optional)
- All speculative decoding types (MTP, ngram-k4v, ngram-mod, DFlash, etc.)

**Critical blocker:**
- RPC backend reports `caps.async = false` and `caps.events = false`
- This **completely disables pipeline parallelism** — every operation is synchronous
- The scheduler's pipeline mode requires async + events to overlap operations

**Verdict:** Feasible but requires 7-11 weeks of focused work. The RPC transport exists. Missing pieces: async support, coordinator layer, direct-return routing.

---

## Shard Architecture Deep-Dive

### Core Innovation: Speculative Decoding Over WAN

**Problem:** Over WAN, round-trip latency (50-80ms per hop) dominates. Without speculation, each token = full WAN round-trip → ~1-2 tok/s.

**Solution:** Draft K tokens locally (no WAN cost), then verify all K+1 tokens in a single pipeline traversal. Amortizes latency across multiple tokens.

**The Round:**
1. **Draft** (local, coordinator): Small model (GLM-4-9B or n-gram) proposes K candidate tokens greedily: `[d₁, d₂, ..., dₖ]`
2. **Ship** to stage 0: Send `[cur, d₁, d₂, ..., dₖ]` (K+1 tokens) into the pipeline
3. **Verify** (one full traversal): All nodes process K+1 tokens in sequence, producing `[r₁, r₂, ..., rₖ₊₁]`
4. **Accept** (greedy, lossless): Find longest prefix where `dⱼ == rⱼ`. Accept `n` draft tokens + 1 correction token `r[n+1]`. Output is byte-identical to plain greedy decode.
5. **Crop** KV caches back to accepted length (lazy — piggybacked on next verify)

**Performance impact:**
- Plain KV decode: 1.87 tok/s
- + spec-decode (K=2): 1.99 tok/s
- + direct-return: 2.94 tok/s (48% improvement)
- + async pipelining (depth=6): **16.6 tok/s** (5.6× improvement)
- + CUDA-graphed draft: **~30 tok/s** (draft was 94% of loop after WAN hidden)

### Direct-Return Routing

**Problem:** Naive relay-back: tail → stage N-2 → ... → stage 0 → coordinator = 6 hops back.

**Solution:** Tail opens separate TCP connection directly to coordinator, bypassing relay chain.

**Architecture:**
```
coordinator ──fwd──► stage0 ──► stage1 ──► ... ──► stageN-1 (tail)
     ▲                                                    │
     └────────────── ret (direct return) ─────────────────┘
```

**Implementation:**
- Coordinator opens forward connection to stage 0
- Coordinator also opens return listener on separate port
- Tail accepts TWO connections: one forward (from predecessor), one return (to coordinator)
- Tail sends hidden states (not tokens) back to coordinator
- Coordinator applies norm + lm_head locally to compute argmax

**Why hidden states, not tokens?** Coordinator holds embedding table and lm_head. Needs raw hidden states to compute argmax locally. Also allows batching norm+lm_head efficiently.

**Performance:** 1.99 → 2.94 tok/s (48% improvement from eliminating return relay).

### Async Pipelining

**Problem:** Without pipelining: draft → send → wait → accept → repeat. Loop runs at latency (one full round-trip per K tokens).

**Solution:** Fire multiple verify chunks without waiting. Loop runs at throughput (multiple chunks in flight).

**Algorithm:**
```python
while not done:
    # PHASE 1: Fill pipeline up to `depth` inflight chunks
    while len(inflight) < depth and not done:
        ds = draft_k(K)  # K draft tokens
        send_chunk(pos, [cur] + ds)
        inflight.append((pos, ds))
        stream.extend(ds)
        cur = ds[-1]
        pos += K
    
    # PHASE 2: Wait for one verification result
    r = recv_logits()  # blocks until tail returns
    
    # PHASE 3: Verify and accept
    start, ds = inflight.pop(0)
    n = greedy_accept(ds, r)
    
    if n == K:
        # Full accept: all K tokens correct
        out.extend(ds)
        pos += K
        cur = ds[-1]
    else:
        # Divergence: accept prefix + correction
        out.extend(ds[:n] + [r[n]])
        cur = r[n]
        pos += n + 1
        
        # Invalidate all inflight chunks (they're stale)
        inflight.clear()
        dcache.crop(pos)  # Rollback draft cache
```

**Key details:**
- Consecutive chunks overlap by 1 position (last token of chunk j = first token of chunk j+1)
- Pipeline depth ≈ ring latency / draft time (typical: 2-6)
- Draft for chunk j+1 runs **while** chunk j traverses WAN (async overlap)
- On divergence: all inflight chunks discarded, draft cache cropped

**Performance:** 2.94 → 16.6 tok/s (5.6× improvement).

### Ring Topology and Stage Assignment

**Topology Solver:**
- Given N GPUs with measured pairwise RTTs (asymmetric — internet peering ≠ geographic)
- Find minimum-latency Hamiltonian loop through all nodes with coordinator as depot
- ≤16 nodes: Held-Karp (exact TSP DP, O(k²·2^k))
- >16 nodes: Nearest-Neighbor + 2-opt heuristic

**Stage Assignment:**
- Model layers divided into N contiguous blocks, one per GPU
- Each block sized to fit GPU's VRAM
- Example: GLM-5.2 744B (78 layers) → 13 layers per node × 6 GPUs

**Key insight:** Geography ≠ latency. Optimal path exploits fast peering pipes (e.g., CA↔VA at 9ms despite geographic distance). Measured RTT matrix essential.

**Demo results** (10 US cities):
- Arbitrary join order: 123.2 ms
- Geographic guess: 112.5 ms
- **Optimal loop: 56.2 ms** (2.19× better)
- Best 6 of 10: 44.2 ms (21% further reduction)

### KV Cache Management Per-Node

**Per-Node Cache:**
- Each node maintains its own KV cache for its layer block
- Layer indices re-indexed 0-based within each node's block
- Prefill: entire prompt sent at once, each node fills its cache
- Decode: only new token's hidden state sent, each node appends to cache

**Crop (rollback on divergence):**
- Lazy crop — piggybacked on next verify message (saves one WAN hop)
- Each stage crops its cache when it receives the next verify with `start_pos`
- Coordinator crops draft cache immediately on divergence

**CUDA-Graph Compatible Cache:**
- StaticCache: pre-allocated, fixed MAXLEN
- Rollback trick: overwrite at `start` position, rewind write position
- No actual crop needed — just reset write pointer
- Monkeypatches for CUDA graph safety (global `_WRITE_POS` tensor)

---

## buun-llama-cpp RPC Backend — Current State

### What Exists

**Protocol & Transport** (`ggml/src/ggml-rpc/ggml-rpc.cpp`, `transport.cpp`):
- TCP socket transport with optional RDMA (RoCE) support
- Connection capability negotiation via HELLO handshake
- Protocol versioning: Major v4, Minor v0, Patch v0
- Up to 16 RPC servers (`GGML_RPC_MAX_SERVERS = 16`)

**RPC Commands** (17 total):
| Command | Purpose |
|---------|---------|
| `RPC_CMD_ALLOC_BUFFER` | Allocate remote GPU buffer |
| `RPC_CMD_GET_ALLOC_SIZE` | Query allocation size |
| `RPC_CMD_GET_ALIGNMENT` | Get device alignment |
| `RPC_CMD_GET_MAX_SIZE` | Max allocation size |
| `RPC_CMD_BUFFER_GET_BASE` | Get buffer base pointer |
| `RPC_CMD_FREE_BUFFER` | Free remote buffer |
| `RPC_CMD_BUFFER_CLEAR` | Zero-fill buffer |
| `RPC_CMD_SET_TENSOR` | Upload tensor data (with optional hash-based skip) |
| `RPC_CMD_SET_TENSOR_HASH` | Hash-based dedup for unchanged tensors |
| `RPC_CMD_GET_TENSOR` | Download tensor data |
| `RPC_CMD_COPY_TENSOR` | Cross-device tensor copy on server |
| `RPC_CMD_GRAPH_COMPUTE` | Serialize & execute full compute graph |
| `RPC_CMD_GRAPH_RECOMPUTE` | Re-execute last graph (fast path) |
| `RPC_CMD_INIT_TENSOR` | Initialize tensor on remote |
| `RPC_CMD_DEVICE_COUNT` | Query server's GPU count |
| `RPC_CMD_HELLO` | Protocol handshake |
| `RPC_CMD_GET_DEVICE_MEMORY` | Query free/total memory |

**Graph Serialization** (`serialize_graph`):
- Format: `| device(4B) | n_nodes(4B) | node_ptrs(n_nodes×8B) | n_tensors(4B) | tensors(n_tensors×sizeof(rpc_tensor)) |`
- Recursively walks all tensor sources and `view_src`
- Server reconstructs graph via `create_node()` with tensor pointer remapping

**Server-Side** (`rpc_server::graph_compute`):
- Deserializes graph into `ggml_context` with `no_alloc=true`
- Executes via `ggml_backend_graph_compute(backends[device], graph)` on server's local CUDA backend
- Stores graph for `GRAPH_RECOMPUTE` fast-path (same UID → skip re-serialization)

### Split Modes

**Three split modes** (from `include/llama.h`):
| Mode | Value | Behavior |
|------|-------|----------|
| `LLAMA_SPLIT_MODE_NONE` | 0 | Single GPU, all layers on one device |
| `LLAMA_SPLIT_MODE_LAYER` | 1 | Layers distributed round-robin across GPUs by free memory ratio |
| `LLAMA_SPLIT_MODE_ROW` | 2 | Tensor parallelism — individual weight rows split across GPUs |

**Layer Assignment** (`llama-model.cpp:1168-1252`, `load_tensors`):
1. Builds `buft_list` per device
2. Computes split points from free memory ratios: `splits[i] = cumulative_free / total_free`
3. For each layer `il`: `layer_gpu = upper_bound(splits, (il - i_gpu_start) / act_gpu_layers)`
4. Stores in `pimpl->dev_layer[il]` = `{dev, buft_list}`

**Pipeline Parallelism** (`llama-context.cpp:346-379`):
- Enabled when: `n_devices > 1 && n_gpu_layers > n_layer && split_mode == LAYER && offload_kqv && !tensor_overrides`
- **Requires all non-CPU backends to support `caps.async` and `caps.events`**
- Passed to scheduler: `ggml_backend_sched_new(..., pipeline_parallel=true)`

**Scheduler Split Execution** (`ggml-backend.cpp:1550-1722`):
- Graph split into subgraphs at backend boundaries
- Each split: copy input tensors → `ggml_backend_graph_compute_async` → record event
- Cross-split synchronization via `event_wait` / `event_synchronize`

### Critical Blocker: No Async Support

```cpp
// ggml-rpc.cpp:1798-1803
props->caps = {
    /* .async       = */ false,
    /* .host_buffer = */ false,
    /* .buffer_from_host_ptr = */ false,
    /* .events      = */ false,
};

// Backend interface:
/* .event_record = */ NULL,
/* .event_wait   = */ NULL,
/* .set_tensor_async = */ NULL,
/* .cpy_tensor_async = */ NULL,
```

**Impact:** Pipeline parallelism is **DISABLED** when any backend is RPC. Every graph compute is synchronous — coordinator blocks waiting for each RPC server.

---

## Compatibility with Existing Speculative Decoding

### ngram-k4v and ngram-mod: Trivially Compatible ✓

These are local, model-free drafting methods — suffix matching from generated context, no model forward pass needed.

**How it works with Shard-style:**
- Coordinator drafts locally using n-gram (just table lookups, zero-cost)
- Sends draft tokens to ring for verification
- This is exactly how Shard works with its external draft model, except draft is zero-cost

**No changes needed to n-gram code.** The coordinator layer (Phase 2) just needs to call the existing n-gram drafting functions.

### MTP (Multi-Token Prediction): Compatible but Needs Work ⚠️

MTP is built into the model — final layers have prediction heads that output multiple future tokens in one forward pass.

**How it works with Shard-style:**
- MTP heads live on the **last pipeline stage** (tail node)
- Tail node sends MTP predictions back to coordinator along with verified tokens
- Coordinator uses MTP predictions as draft for next verify round
- Different from Shard (separate draft model on coordinator), but flow is similar: draft → verify → accept → repeat

**Key difference:** Shard's coordinator drafts locally (separate model). With MTP, "draft" happens on tail node during verification, then fed back as next draft. Tighter coupling but still works.

**Changes needed:**
- Tail node must send MTP predictions to coordinator (modify direct-return protocol)
- Coordinator must use MTP predictions as next draft (modify speculative decode loop)
- Not a blocker, just additional work (Phase 3)

### DFlash: Not Assessed

DFlash (block-diffusion speculative decoding) is more complex. Not assessed for Shard-style distribution. Likely requires significant architectural changes. Out of scope for initial implementation.

---

## Gap Analysis: What's Missing

### Tier 1: Async RPC (Critical Path)

**What's needed:**
- Implement `event_record`/`event_wait` over the wire
- Set `caps.async = true`, `caps.events = true`
- Implement `set_tensor_async`, `get_tensor_async`, `cpy_tensor_async`

**Files to modify:**
| File | Function/Location | Change |
|------|-------------------|--------|
| `ggml/src/ggml-rpc/ggml-rpc.cpp:1798-1803` | `ggml_backend_rpc_device_get_props` | Set `.async = true`, `.events = true` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:726-730` | Backend interface | Implement `set_tensor_async`, `get_tensor_async`, `cpy_tensor_async` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:737-738` | Backend interface | Implement `event_record`, `event_wait` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:56-75` | `enum rpc_cmd` | Add `RPC_CMD_EVENT_RECORD`, `RPC_CMD_EVENT_WAIT`, `RPC_CMD_SET_TENSOR_ASYNC` |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:1323-1391` | `rpc_server::graph_compute` | Make async — return immediately, provide completion notification |
| `ggml/src/ggml-rpc/transport.cpp` | Socket layer | Add multiplexing or multiple connections per server |

**Effort:** 2-3 weeks, ~500-800 lines

**Deliverable:** Multi-GPU inference over RPC with async overlap (no speculative decoding yet). This alone unlocks the existing scheduler pipeline mode.

### Tier 2: Coordinator Layer (New Code)

**What's needed:**
- Draft/verify loop management
- Token batching
- Result collection
- KV cache management with crop/rollback

**Files to create/modify:**
| File | Change |
|------|--------|
| New: `src/llama-coordinator.cpp` | Shard-style coordinator: draft model, token batching, verify dispatch, result collection |
| `src/llama-context.cpp:3839-3866` (`graph_compute`) | Add coordinator mode that dispatches sub-graphs to RPC backends without waiting |
| `src/llama-context.cpp:346-379` (pipeline_parallel detection) | Allow pipeline_parallel even with RPC backends once async is implemented |
| `ggml/src/ggml-backend.cpp:1550-1722` (split execution) | Modify to support overlapping split execution across RPC backends |

**Effort:** 3-4 weeks, ~1000-1500 lines

**Deliverable:** Shard-style pipeline with n-gram drafting over WAN.

### Tier 3: Direct-Return Routing

**What's needed:**
- Tail node pushes results to coordinator instead of relay-back
- Server must initiate outbound connections (currently only accepts inbound)

**Files to modify:**
| File | Change |
|------|--------|
| `ggml/src/ggml-rpc/ggml-rpc.cpp:699-721` (`ggml_backend_rpc_graph_compute`) | Add "output routing" — server pushes result to coordinator |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:1387` (server graph_compute) | After compute, optionally push output tensor to specified endpoint |
| `ggml/src/ggml-rpc/transport.cpp` | Server needs ability to initiate outbound connections |
| `tools/rpc/rpc-server.cpp` | Server needs coordinator endpoint configuration for direct-return |

**Effort:** 1-2 weeks, ~300-500 lines

**Deliverable:** Tail → coordinator in 1 hop (48% latency reduction).

### Tier 4: MTP Integration

**What's needed:**
- Tail node sends MTP predictions to coordinator
- Coordinator uses MTP predictions as next draft

**Files to modify:**
| File | Change |
|------|--------|
| `src/llama-context.cpp` (decode loop) | Modify speculative decode path to use MTP predictions from tail |
| `src/llama-model.cpp:2148-2166` (`build_graph`) | Support building separate draft and verify graphs for MTP |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` (direct-return protocol) | Extend to include MTP predictions |

**Effort:** 1-2 weeks, ~200-400 lines

**Deliverable:** MTP-driven pipeline parallel inference.

### Tier 5: WAN Optimization

**What's needed:**
- Compression (zstd/lz4) for tensor data
- Connection pooling
- Keepalive tuning for high-latency links

**Files to modify:**
| File | Change |
|------|--------|
| `ggml/src/ggml-rpc/transport.cpp` | Add compression, connection pooling |
| `ggml/src/ggml-rpc/ggml-rpc.cpp:79-80` | Tune `HASH_THRESHOLD` for high-latency links |
| `ggml/src/ggml-rpc/transport.cpp:44-93` | RDMA path — WAN-friendly tuning |
| `ggml/include/ggml-rpc.h:17` | Raise `GGML_RPC_MAX_SERVERS` beyond 16 |

**Effort:** 1-2 weeks, ~200-400 lines

**Deliverable:** Production-ready distributed inference over WAN.

### Tier 6: Topology Solver

**What's needed:**
- Port Shard's Held-Karp TSP solver for optimal stage ordering
- Measure pairwise RTTs between nodes
- Compute minimum-latency Hamiltonian loop

**Files to create:**
| File | Change |
|------|--------|
| New: `tools/topology/topology.cpp` | Held-Karp TSP solver, RTT measurement, optimal ordering |

**Effort:** 1 week, ~150-300 lines (or use Shard's Python code as-is for setup)

**Deliverable:** Optimal stage ordering for minimum latency.

---

## Implementation Phases

### Phase 1: Async RPC (Unlocks Pipeline Parallelism)

**Goal:** Implement async support in RPC backend to enable existing scheduler pipeline mode.

**Tasks:**
1. Add `RPC_CMD_EVENT_RECORD`, `RPC_CMD_EVENT_WAIT`, `RPC_CMD_SET_TENSOR_ASYNC` commands
2. Implement `event_record`/`event_wait` in backend interface
3. Implement `set_tensor_async`/`get_tensor_async`/`cpy_tensor_async`
4. Set `caps.async = true`, `caps.events = true` in device properties
5. Make `graph_compute` async — return immediately, provide completion notification
6. Add multiplexing or multiple connections per server in transport layer
7. Test with existing `LLAMA_SPLIT_MODE_LAYER` to verify basic pipeline works

**Deliverable:** Multi-GPU inference over RPC with async overlap (no speculative decoding yet).

**Effort:** 2-3 weeks

**Risk:** Medium — requires deep understanding of ggml backend interface. Debugging will be painful.

### Phase 2: Coordinator + n-gram Draft (Shard-Style with Local Draft)

**Goal:** Build coordinator layer that manages draft/verify loop with n-gram drafting.

**Tasks:**
1. Create `src/llama-coordinator.cpp` with draft/verify loop
2. Integrate existing n-gram drafting (ngram-k4v or ngram-mod)
3. Implement direct-return routing (tail → coordinator in 1 hop)
4. Implement KV cache management with lazy crop
5. Implement async pipelining (multiple verify chunks in flight)
6. Tune pipeline depth and K (tokens per verify)

**Deliverable:** Shard-style pipeline with n-gram drafting over WAN.

**Effort:** 3-4 weeks

**Risk:** Medium — new code, but Shard provides reference implementation.

### Phase 3: MTP Integration

**Goal:** Use MTP predictions from tail node as draft for next verify round.

**Tasks:**
1. Modify tail node to send MTP predictions to coordinator
2. Modify coordinator to use MTP predictions as next draft
3. Tune pipeline depth and K for MTP acceptance rates
4. Benchmark MTP vs n-gram drafting over WAN

**Deliverable:** MTP-driven pipeline parallel inference.

**Effort:** 1-2 weeks

**Risk:** Low — builds on Phase 2, MTP already works locally.

### Phase 4: WAN Optimization + Topology

**Goal:** Optimize for real WAN conditions, compute optimal stage ordering.

**Tasks:**
1. Add compression (zstd/lz4) for tensor data
2. Add connection pooling
3. Tune keepalive for high-latency links
4. Port topology solver for optimal stage ordering
5. Benchmark over real WAN (not just LAN)

**Deliverable:** Production-ready distributed inference.

**Effort:** 1-2 weeks

**Risk:** Low — incremental improvements.

---

## Key Risks

### 1. Async RPC Complexity

The ggml backend interface is complex. Getting async + events right across the RPC boundary is non-trivial. Debugging will be painful.

**Mitigation:** Start with simple async operations (e.g., async tensor copy), verify correctness, then build up to full async graph compute.

### 2. KV Cache Synchronization

Each node maintains its own KV cache. On divergence (draft token rejected), all nodes need to crop their caches. Lazy crop (piggyback on next verify) helps, but edge cases exist.

**Mitigation:** Implement comprehensive KV cache tests. Verify crop behavior on divergence. Test with long generations to ensure no drift.

### 3. WAN Latency Variance

Shard's topology solver minimizes average latency, but WAN jitter can still cause pipeline stalls. Need adaptive pipeline depth.

**Mitigation:** Implement adaptive pipeline depth based on measured RTT variance. Start conservative (depth=2), increase if stable.

### 4. MTP Acceptance Rate Over Distributed Setup

MTP is tuned for single-GPU inference. Over WAN with higher latency, acceptance rates may differ. Need benchmarking.

**Mitigation:** Benchmark MTP acceptance rates over WAN in Phase 3. Compare with n-gram drafting. Use whichever has higher acceptance.

---

## Success Criteria

### Phase 1 Success
- [ ] Async RPC operations work correctly (tensor copy, graph compute)
- [ ] `LLAMA_SPLIT_MODE_LAYER` with RPC backends uses pipeline parallelism
- [ ] Performance scales with number of RPC servers (at least on LAN)
- [ ] No deadlocks or race conditions in async operations

### Phase 2 Success
- [ ] Coordinator drafts K tokens using n-gram
- [ ] Verify round processes K+1 tokens across all nodes
- [ ] Greedy acceptance works correctly (byte-identical to plain decode)
- [ ] Async pipelining overlaps multiple verify chunks
- [ ] Direct-return routing works (tail → coordinator in 1 hop)
- [ ] KV cache crop works correctly on divergence

### Phase 3 Success
- [ ] MTP predictions sent from tail to coordinator
- [ ] Coordinator uses MTP predictions as next draft
- [ ] MTP acceptance rate ≥ n-gram acceptance rate over WAN
- [ ] End-to-end inference works with MTP-driven drafting

### Phase 4 Success
- [ ] Compression reduces WAN bandwidth by ≥50%
- [ ] Topology solver computes optimal stage ordering
- [ ] Performance over real WAN ≥ 80% of LAN performance
- [ ] Stable for 1000+ token generations with 0% garbage

---

## Technical Constraints

### Hardware
- Target: 2-8 GPUs across WAN (RTX 3090/4090 or similar)
- Each GPU: ≥24GB VRAM
- WAN latency: 20-100ms per hop (typical)
- WAN bandwidth: ≥100 Mbps per GPU

### Software
- OS: Windows (primary), Linux (secondary)
- CUDA: 12.8+ (for buun-llama-cpp)
- Compiler: MSVC (Windows), GCC (Linux)
- Build system: CMake + Ninja

### Model Constraints
- Model must fit across all GPUs (total VRAM ≥ model size)
- Each GPU must hold at least 1 layer (minimum block size)
- KV cache must fit in remaining VRAM after model layers
- MTP models require special GGUF format with MTP heads

### Performance Targets
- Phase 1: ≥50% of single-GPU performance (LAN, no speculative)
- Phase 2: ≥30% of single-GPU performance (WAN, n-gram speculative)
- Phase 3: ≥40% of single-GPU performance (WAN, MTP speculative)
- Phase 4: ≥60% of single-GPU performance (WAN, optimized)

---

## References

### Shard Repository
- Main: https://github.com/leyten/shard
- Architecture: https://github.com/leyten/shard/blob/master/README.md
- Speculative decoding: https://github.com/leyten/shard/blob/master/phase0/specdec.py
- Pipeline: https://github.com/leyten/shard/blob/master/phase0/specpipe.py
- Topology: https://github.com/leyten/shard/blob/master/shard/topology.py

### buun-llama-cpp RPC
- RPC backend: `ggml/src/ggml-rpc/ggml-rpc.cpp`
- RPC server: `tools/rpc/rpc-server.cpp`
- Transport: `ggml/src/ggml-rpc/transport.cpp`
- Split modes: `src/llama-model.cpp:1168-1252`
- Pipeline parallel: `src/llama-context.cpp:346-379`
- Scheduler: `ggml/src/ggml-backend.cpp:1550-1722`

### Speculative Decoding
- MTP: `src/llama-context.cpp` (search for `LLAMA_CONTEXT_TYPE_MTP`)
- n-gram: `common/common.h:159-174` (enum `common_speculative_type`)
- DFlash: `src/llama-context.cpp:1097-1237` (dflash eval callback)

---

## Next Steps

1. **Review this document** — ensure all technical details are correct
2. **Prioritize phases** — decide which phases to tackle first
3. **Assign work** — break phases into tasks, assign to lesser models
4. **Start Phase 1** — async RPC is the critical path
5. **Benchmark early** — verify async RPC works on LAN before WAN

---

**Document version:** 1.0  
**Last updated:** 2026-06-20  
**Author:** Hermes Agent (with research from Mercury and researcher subagents)
