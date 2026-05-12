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

## Next Steps / Fix Plan

1. **Add CUDA error context**: Before `cudaStreamSynchronize` in ggml-cuda.cu:3093, capture the last launched kernel identity via logging or cudaGetError before/after key DFlash operations
2. **Check ring buffer bounds**: Add assertions to verify `ring_write_pos`, `ring_filled`, and GPU staging buffer dimensions are consistent during spec cycles with mmproj
3. **Verify hidden state dims for vision tokens**: Ensure eval_callback captures with correct `n_embd` regardless of token type (text vs vision)
4. **Test partial offload path**: Verify ring_write correctly handles host-buffered layer hiddens for partially-offloaded recurrent layers when mmproj is loaded

## Server Arguments (from log 1)

```
llama-server -m Qwen3.6-27B-Q4_K_M.gguf --mmproj mmproj-BF16.gguf
  --draft-model dflash-draft-3.6-q8_0.gguf -ngl -2 -c 262144
  --port 8080 --threads 6 --threads-batch 6
  --draft-max 16 -b 256 -ub 64
```
