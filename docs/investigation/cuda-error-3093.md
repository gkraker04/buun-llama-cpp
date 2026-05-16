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

## To Do / Fix Plan

1. ~~Add CUDA error context** in `ggml-backend_cuda_synchronize` — log device and error string before aborting~~ ✅ Done (commit 1)
2. ~~Add `n_embd` consistency guard in `ring_write()` — skip GPU write path when captured hidden state dim != expected `n_embd`~~ ✅ Done (commit 1)
3. ~~Add CUDA error checks in `dflash_cross_ring_gpu_interleave` — check both launch and sync errors, with full context logging~~ ✅ Done (commit 1)
4. ~~Add bounds validation in `dflash_cross_ring_gpu_write` — verify `n_embd` matches ring allocation before writing~~ ✅ Done (commit 1)
5. ~~Add ring buffer diagnostic logging — log ring state (`write_pos`, `filled`, `committed_len`) on every write and cross-data build~~ ✅ Done (commit 1)
6. ~~Test with `-ngl 99` (full offload)** — eliminate partial offload as variable~~ ✅ All 3 test configs pass: server starts, DFlash + copyspec speculators initialize, inference returns correct output. No CUDA errors in any config. Diagnostics confirmed working: `dflash ring:` and `dflash cross:` messages firing correctly.
7. ~~**Test with `--no-copyspec`** — disable copyspec secondary to test if backward-position pattern triggers crash~~ ✅ `--spec-type dflash` was used, but copyspec is STILL initialized and called (4 times in test). The `--spec-type` flag registers primary speculator but doesn't disable copyspec fallback chain. Copyspec generated 0 draft tokens — it's part of the built-in chain.
8. ~~**Test with mmproj unloaded** — confirm crash is mmproj-specific~~ ✅ Server starts and infers without mmproj. No CUDA errors.

## Test Results (May 16, 2026)

All three root-cause isolation tests completed successfully. Each config:
- Loaded Qwen3.6-27B model with DFlash drafter, turbo2_tcq KV cache
- Initialized GPU cross ring (5 layers x 512 slots x 5120 embd)
- Completed inference without CUDA errors
- Verified diagnostics firing correctly

### Key Findings

1. **`-ngl 99` vs `auto`**: No difference in observed behavior. Both configs load all 65/65 layers to GPU (auto already uses all VRAM).
2. **`--spec-type dflash`**: Does NOT disable copyspec. Copyspec is still initialized and called as a fallback (4 calls in ~200ms). This is hardcoded into the speculative chain — copyspec is part of the secondary/backup mechanism.
3. **mmproj**: Server starts and runs without mmproj. No immediate errors.
4. **Diagnostics verified working**: Both `LOG_DBG` messages confirmed in verbose server output:
   ```
   D dflash ring: wrote 4 tok (+4 ntok) pos=11->15 filled=15 committed=11
   D dflash cross: seq=0 GPU ring write_pos=15 filled=15 n_layers=5 n_embd=5120 window=512
   D dflash ring: wrote 2 tok (+2 ntok) pos=15->17 filled=17 committed=15
   ```

### Next Steps
- The short-request tests don't reproduce the crash (requires multi-turn 23k+ token context)
- To reproduce, need sustained usage with `--parallel 2` and multiple long-prompt requests
- The `--spec-type dflash` approach doesn't disable copyspec — to truly test without copyspec, the fallback chain in `speculative.cpp` needs code modification
- No `n_embd` mismatches detected — the ring buffers are consistent (n_embd=5120 throughout)

## Fixes Applied (May 2026)

These fixes were implemented on `fix/cuda-error-3093` branch:

### Commit 1: cuda-error-3093 diagnostic guards and bounds checks

**Files changed**: 3 files, +48/-4 lines

1. **`ggml/src/ggml-cuda/ggml-cuda.cu`** — `ggml_backend_cuda_synchronize`:
   - Now captures `cudaStreamSynchronize` return value explicitly, logs device ID and error string before forwarding to `CUDA_CHECK_GEN` abort
   - Helps identify which device faulted (relevant for multi-GPU setups)

2. **`ggml/src/ggml-cuda/cross-ring-interleave.cu`** — GPU ring operations:
   - `dflash_cross_ring_gpu_write`: Added `n_embd` bounds guard — if caller passes mismatched `n_embd` vs ring allocation, skips the write with a diagnostic instead of silently corrupting adjacent ring data
   - `dflash_cross_ring_gpu_interleave`: Added explicit `cudaGetLastError` after kernel launch and `cudaStreamSynchronize` error check, both with full context (read_start, cross_len, ring_size, layers, embd) logged to stderr

3. **`common/speculative.cpp`** — `ring_write` and `build_cross_data`:
   - `ring_write`: Added `embd != n_embd` guard that warns and skips GPU upload when dimensions mismatch
   - `ring_write`: Added diagnostic logging of ring state (write_pos before/after, filled, committed_len) at DEBUG level
   - `build_cross_data` GPU path: Added diagnostic logging of seq_id, write_pos, filled, n_layers, n_embd, ctx_window

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
