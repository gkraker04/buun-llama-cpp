# CUDA Error 3093 Investigation — DFlash + mmproj Illegal Memory Access

**Date**: 2026-05-12
**Branch**: `fix/cuda-error-3093` off `master`
**Model**: Qwen3.6-27B (qwen35 architecture) + CLIP vision encoder (mmproj-BF16.gguf)
**Drafter**: DFlash draft-3.6-q8_0.gguf (5 target layers [1,16,31,46,61], n_target_features=25600)

## Error Details

```
E CUDA error: an illegal memory access was encountered
E   current device: 0, in function ggml_backend_cuda_synchronize
E   at /home/zg/buun-llama-cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3093
E   cudaStreamSynchronize(cuda_ctx->stream())
```

Line 3093 is `cudaStreamSynchronize` — this just **surfaces** the error. The illegal memory access happened in a kernel launched earlier on that stream; synchronize catches it here.

## Crash Logs Analyzed

| Log | Timestamp | Crash Point | Phase |
|-----|-----------|-------------|-------|
| 20-52-03.clean.log | 7.25.597 | After prompt processing done (42,991 tokens), first gen token spec cycle | Verify batch (12 tok) |
| 21-02-33.log | 7.16.362 | After prompt processing done (18,166 tokens), first gen token spec cycle | Verify batch (12 tok) |
| 21-15-51.log | 4.23.059 | During decoding (n_tokens=23,484, ctx=182,95) | Draft decode (9 tok) |

## Key Observations

### 1. Non-consecutive Token Positions
Persistent `W find_slot: non-consecutive token position X after Y for sequence 0 with N new tokens` throughout all runs. This is **expected** behavior during DFlash speculative decoding — the drafter predicts tokens in bursts (8 at a time), creating gaps in the recurrent state cell positions. The warning fires in `llama-memory-recurrent.cpp:797`.

### 2. Crash Timing
Crashes happen in two patterns:
- **Pattern A** (Logs 1 & 2): Immediately after prompt processing completes, during the very first generation token spec cycle. The verify ubatch of ~12 tokens triggers the crash.
- **Pattern B** (Log 3): During mid-decoding, during a draft decode batch of 9 tokens.

### 3. System Configuration
- GPU: RTX 3090 (24 GB VRAM), CUDA ARCH 860
- Context reduced from 262144 to 131072 due to memory constraints
- n_gpu_layers = -2 (partial offload — last 2 layers on CPU)
- GPU cross-ring: 5 layers × 512 slots × 5120 embd (~512 MB ring + staging)
- CLIP vision encoder on CUDA0
- Flash attention enabled, kv_unified = true

### 4. Previously Applied Fixes (now merged to master)
Three commits were in `fix/issue-38-dflash-mmproj` and have been merged:

1. **#44 — DFlash tape_replay crash with partial GPU offload**: Added host-memory fallback in `tape_replay()` when recurrent state lives on CPU
2. **#38 — Allow DFlash spec decoding with multimodal**: Removed 3 guards that disabled speculation when mmproj loaded
3. **Pre-evict cache entries**: Prevent OOM by evicting old prompt cache before allocation

## Suspected Root Causes

### Hypothesis 1: GPU Ring Buffer Index Mismatch During Spec Cycles

The cross-ring interleave kernel (`cross-ring-interleave.cu`) reads from per-layer ring buffers using circular indexing. With speculative decoding and mmproj, the `ring_write_pos` and `ring_filled` counters may get out of sync because:

- Vision tokens in the context have different hidden state dimensions (mmproj projects CLIP output to model n_embd)
- The eval_callback captures layer hiddens for ALL tokens (text + vision), but ring buffer writes assume uniform token semantics
- During spec verify batches with non-consecutive positions, the ring write position may advance past where data was actually written

### Hypothesis 2: Recurrent State Cell Overlap with Speculative Tokens

The `non-consecutive token position` warnings show gaps of 5 tokens between expected and actual positions (e.g., pos 14937 after 14932, with 8 new draft tokens). The recurrent memory's `find_slot()` allocates cells for these spec tokens but the hidden state capture may not populate them correctly when vision tokens are present.

The eval_callback (`dflash_eval_callback` in `llama-context.cpp:1036`) captures hidden states per layer per slot. With mmproj, the callback fires for vision encoder layers too, which have different tensor shapes than text layers.

### Hypothesis 3: Cross-Attention Tensor Size Mismatch

In `dflash_draft.cpp`, `set_input()` copies cross-attention data into the `target_hidden` tensor of shape `[n_target_features, ctx_len]`. With mmproj present, `cross.n_embd` is set to `n_layers * n_embd_layer` (25600), but vision tokens may have a different `n_embd` flow through the eval callback.

The GPU D2D path (`fn_set_tensor_d2d`) copies from the ring staging buffer using `cross.n_embd` for sizing, but if vision tokens produce hidden states with different dimensions, the copy could read past buffer boundaries.

### Hypothesis 4: Partial Offload + Vision Tokens Interaction

With `-ngl -2` (partial offload), recurrent layers -1 and -2 live on CPU. The tape_replay fix (#44) handles this for replay, but the hidden state capture path (`append_target_hiddens` → `ring_write`) doesn't check whether the layer's hidden state buffer is host memory before writing to the GPU ring. Writing H2D data from a host-buffered tensor could cause an illegal device access when the drafter kernel tries to read it.

## Updates — May 16, 2026

### Production Test with Fix Applied (Commit 2: GPU ring clear)

**Build**: `b9322-318676356` (GPU ring clear fix + diagnostics)
**Server config**: `buun-server_qwen3.6.sh` — `--parallel 2`, `turbo3_tcq`/`turbo2_tcq`, DFlash + mmproj, `--cache-ram 7680`

**Result**: Server started and ran successfully. Slot 1 processed a 15,095-token prompt with DFlash, generated ~530 tokens output — **completed successfully** (4m 52s). This is the longest we've seen it run without crashing.

**Then**: User opened the WebUI and sent a short chat message (17 tokens) → **CRASH INSTANTLY**.

```
CUDA synchronize error on device 0: an illegal memory access was encountered
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize at ...ggml-cuda.cu:3109
```

**Crash sequence:**
```
4.55.028.566 | prompt processing done, n_tokens=17, batch.n_tokens=5
4.55.159.843 | created context checkpoint (pos_min=12, pos_max=12, n_tokens=13, size=150.896 MiB)
4.55.295.087 | verify ubatch: 5 tok, 135.2ms (27.03ms/tok)
4.55.306.412 | sched_reserve: reserving ...
4.55.332.731 | CUDA0 compute buffer size = 123.75 MiB
4.55.332.767 | graph nodes = 210, graph splits = 4
4.55.332.769 | reserve took 26.32 ms, sched copies = 1
4.55.659.373 | CUDA synchronize error ← 326ms AFTER sched_reserve completes
```

### Root Cause Analysis

**Key differences from our test:**
- Our test used `--parallel 1` → ✅ works
- Production uses `--parallel 2` → ❌ crashes on second request after long first request

**What's already in our build:**
- SD-075 multi-slot DFlash tape fix (skip tree-mode on multi-seq, pass seq_id through tape_replay) ✅ merged
- SD-078 batched DFlash draft (B2.0-B2.6: wider drafter graph, per-seq cross-data routing) ✅ merged
- Our GPU ring clear (cudaMemset on slot reset) ✅ deployed

**Despite all these fixes, `--parallel 2` still crashes.**

### New Hypothesis: Shared Drafter Context State Pollution

With `--parallel 2`, each slot gets its own `common_speculative` instance (and its own DFlash ring), but ALL slots share the same **drafter context (`ctx_dft`)**. The shared drafter has:
- A single KV cache sized for both sequences (`n_seq_max=2`)
- A single compute buffer allocation
- Shared GPU ring interleave

When slot 1 processes 15K tokens:
1. Fills its sequence (seq_id=1) in the drafter's KV cache
2. Writes hidden states to its own CPU ring and GPU ring
3. Allocates compute buffers in the shared scheduler

When slot 1's request completes → `slot.release()` calls `reset()` but does NOT destroy or reset `slot.spec`. The DFlash state persists.

When slot 0 starts a new 17-token request:
1. `flush_prefill()` clears the GPU ring ✅
2. 17 hidden states are written to the new ring
3. The target model processes the prompt (17 tokens) OK
4. The first DFlash verify batch (5 tokens, target verification of draft) — SUCCEEDS ✅
5. **BUT**: The drafter's KV cache still has slot 1's data at various positions
6. The next decode step launches — `sched_reserve` re-computes buffer sizes
7. The NEXT speculative cycle runs with cross-attention → GPU kernel accesses stale drafter KV cache → illegal memory access

**Why sched_reserve is called mid-generation:**
After prefill (batch=17) and the first verify batch, the `cross.n_enc` bucket changes from the prefill value to a generation value. `set_cross_data_gpu` detects the bucket mismatch and sets `sched_need_reserve=true`. The scheduler is rebuilt with new compute buffer sizes. On the FIRST compute after the rebuild, the illegal access surfaces at synchronize.

### To Do / Fix Plan

1. ~~Add CUDA error context** in `ggml-backend_cuda_synchronize` — log device and error string before aborting~~ ✅ Done (commit 1)
2. ~~Add `n_embd` consistency guard in `ring_write()` — skip GPU write path when captured hidden state dim != expected `n_embd`~~ ✅ Done (commit 1)
3. ~~Add CUDA error checks in `dflash_cross_ring_gpu_interleave` — check both launch and sync errors, with full context logging~~ ✅ Done (commit 1)
4. ~~Add bounds validation in `dflash_cross_ring_gpu_write` — verify `n_embd` matches ring allocation before writing~~ ✅ Done (commit 1)
5. ~~Add ring buffer diagnostic logging — log ring state (`write_pos`, `filled`, `committed_len`) on every write and cross-data build~~ ✅ Done (commit 1)
6. ~~Test with `-ngl 99` (full offload)** — eliminate partial offload as variable~~ ✅ No change
7. ~~**Test with `--spec-type dflash`** — determine if copyspec triggers crash~~ ✅ Copyspec still in chain
8. ~~**Test with mmproj unloaded** — isolate mmproj interaction~~ ✅ No crash without mmproj
9. ✅ **GPU ring clear on slot reset** — applied, helps single-slot but not `--parallel 2`
10. **⬜ Investigate shared drafter KV cache stale entries** — Most promising lead. The drafter's KV cache isn't cleared when a slot finishes. Need to check if `llama_kv_cache_seq_rm` (or equivalent) is called on `ctx_dft` during slot release.
11. **⬜ Test with `CUDA_LAUNCH_BLOCKING=1`** — Pinpoints the exact kernel causing the illegal access (slow but diagnostic)
12. **⬜ Test with `--parallel 1`** — Already confirmed working. Official workaround.

## Test Results (May 16, 2026)

All three root-cause isolation tests completed successfully. Each config:
- Loaded Qwen3.6-27B model with DFlash drafter, turbo2_tcq KV cache
- Initialized GPU cross ring (5 layers x 512 slots x 5120 embd)
- Completed inference without CUDA errors
- Verified diagnostics firing correctly

### Key Findings

1. **`-ngl 99` vs `auto`**: No difference in observed behavior.
2. **`--spec-type dflash`**: Does NOT disable copyspec — still initialized as fallback.
3. **mmproj**: Works without mmproj, crash triggers with mmproj loaded.
4. **`--parallel 1`**: Works perfectly — no crash, full DFlash + mmproj inference.
5. **`--parallel 2`**: ❌ Crashes on second request after long first request, even with GPU ring clear fix.
6. **Diagnostics verified working**: Both `LOG_DBG` messages confirmed in verbose server output.

## Cross-Configuration Findings

### Crash reproduces across KV quant types
| Log | KV Type | Parallel | Context | Crash Phase |
|-----|---------|----------|---------|-------------|
| 20-52-03 | turbo4 | 1 slot | 131072 | First gen token verify (12 tok) |
| 21-02-33 | turbo4 | 1 slot | 96512 | First gen token verify (12 tok) |
| 21-15-51 | turbo2_tcq | 2 slots (--parallel 2) | 175872 | Mid-decode draft batch (9 tok) |

**Critical**: Crash is NOT limited to TCQ. Reproduces with both turbo4 and turbo2_tcq, ruling out our previous narrow hypothesis that TCQ + parallel was the sole trigger. The crash path shared across configs points to DFlash + mmproj interaction itself.

### Backward Position Pattern Before Crash (Logs 1 & 2)
```
find_slot: position 14937 after 14932 with 8 new tokens     ← DFlash draft (forward gap)
verify ubatch: 12 tok                                         ← first gen token verify OK
done request: POST /v1/chat/completions 200
spec cycle: accept=11.9ms                                     ← 11 tokens accepted
find_slot: position 14931 after 14933 with 1 new tokens      ← BACKWARDS! copyspec suffix lookup
CUDA error: illegal memory access ~250ms later               ← crash at synchronize
```

The backward position (14931 after 14933) comes from copyspec's secondary impl doing suffix-based speculation, looking up a historical position behind the current cell. The crash fires during the next `cudaStreamSynchronize` — likely in `tape_replay_sync()` or the next decode graph execution where the GDN kernel accesses recurrent state that was modified by tape replay on a different stream.

### Tape Replay + Partial Offload Interaction
With `-ngl -2` (partial offload, last 2 layers on CPU):
- Recurrent S state: 576 MiB across 64 layers, 4 cells
- R state (DeltaNet): 22.5 MiB
- Fix #44 added host-memory fallback for `tape_replay()` GPU path
- But `append_target_hiddens` → `ring_write` → GPU ring upload uses `cudaMemcpyAsync(..., cudaMemcpyHostToDevice)` regardless of offload config
- The GDN kernel for partially-offloaded layers may access state tensors that live on different devices

### Hypothesis Refinement
**Primary suspect**: During spec cycles with mmproj loaded, the verify batch decode triggers `tape_replay` which launches an async GDN graph. The next synchronize (either in `tape_replay_sync()` or in the subsequent `sched_reserve()` when cross-attention bucket changes) catches a CUDA illegal memory access from:
1. GDN kernel accessing a recurrent state tensor that was partially modified by tape replay conv rebuild
2. Concat K/V tensors in drafter attention with mismatched dimensions during transition phases
3. GPU ring buffer staging data not fully synchronized before target model reads it

## Server Arguments (from log 1)

```
llama-server -m Qwen3.6-27B-Q4_K_M.gguf --mmproj mmproj-BF16.gguf
  --draft-model dflash-draft-3.6-q8_0.gguf -ngl -2 -c 262144
  --port 8080 --threads 6 --threads-batch 6
  --draft-max 16 -b 256 -ub 64
```

## To Do / Fix Plan

1. **Reproduce with DFlash-only** (disable copyspec) — isolate whether copyspec backward-position pattern triggers crash
2. **Reproduce with mmproj unloaded** — confirm crash is mmproj-specific
3. **Add CUDA error context logging** in `ggml-cuda.cu:3093` — log last kernel or add cudaGetError before synchronize to catch which operation faults
4. **Verify tape_replay stream isolation** — ensure GDN replay graph uses separate stream/device from target decode graph, especially with partial offload
5. **Test with `-ngl 99` (full offload)** — eliminate partial-offload as variable
