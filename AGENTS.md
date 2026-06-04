# TURBO4 STATUS — 2026-06-04

**Branch:** `experiments/turbo4-quantize` (rebased on buun/master)
**Build:** `build_turbo4.bat` — CUDA 13.2, Ninja, MSVC vcvars64, sm_86, GGML_CUDA_FA_ALL_QUANTS=ON
**Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf
**Status:** ✅ Build succeeds, server loads and serves correct tokens
**Measured speed:** 1.09 tok/s (128 tokens, q8_1 vec_dot path, v6-i3 model)
**Server:** Port 8081, `experiments/turbo4-weight-quant\test_start_server.bat`

══════════════════════════════════════════════
CURRENT STATE
══════════════════════════════════════════════

**Working paths (decode, ne11=1):**
1. ✅ **cuBLAS fallback** — full dequant → fp16 → cublasGemmEx. Correct but slow (~0.46 tok/s).
2. ✅ **q8_1 vec_dot** (`k_turbo4_mul_mat_vec_q8` in turbo-matmul.cu) — pre-rotate activations via WHT, quantize to q8_1, dot with centroid LUT × norm × q8_1 activation × q8_1 scale. **1.09 tok/s.** Correct output. Bottleneck: byte-scalar weight reads (32 threads, 2048 iterations/thread).
3. ❌ **MMVQ vec_dot** (`vec_dot_turbo4_0_q8_1` in vecdot-turbo4.cuh) — ~24-36 tok/s but **garbage output**. Root cause: WHT pre-rotation at mmvq.cu:1191 may have incorrect sign handling, or the vec_dot dp4a math doesn't account for the WHT-domain centroid structure.

**Prefill (ne11>1):** Falls back to cuBLAS (unchanged).

══════════════════════════════════════════════
FIXES APPLIED
══════════════════════════════════════════════

**Non-LIFO pool allocator fix (ggml-cuda.cu:575-588):**
- `ggml_cuda_pool_vmm::free()` had a `GGML_ASSERT` requiring all frees in strict LIFO order
- Turbo matmul path triggers non-LIFO frees (allocations occur in graph capture/eval and are freed in different order)
- Replaced assert with non-fatal warning + pool reset (`pool_used = 0`)
- Server now survives inference without crashing

══════════════════════════════════════════════
UPSTREAM (buun/master) — as of last rebase
══════════════════════════════════════════════
0461612ba cuda: stage TCQ KV codebook in shared memory for fused-MMA decode
9e7c78a2d cuda: fix asymmetric q8_0-K + turbo-V flash-attention (garbage output)
d71ef816f cuda: add bf16-K/V flash-attention support for asymmetric turbo KV

All in `fattn.cu` / `fattn-mma-f16.cuh` — flash-attention for KV cache.
None touch the weight quantization path.

══════════════════════════════════════════════
LOCAL PATCHES (unstaged, experiments/turbo4-quantize)
══════════════════════════════════════════════
- `ggml-cuda.cu`: Non-LIFO pool free → warning instead of assert + pool reset
- `turbo-matmul.cu`: `k_turbo4_mul_mat_vec_q8` — q8_1 vec_dot kernel
- `turbo-matmul.cu`: `k_turbo4_mul_mat_vec` — original byte-parallel vec_dot
- `turbo-matmul.cu`: `k_turbo4_mul_mat_vec_simple` — simple threaded vec_dot
- `turbo-matmul.cu`: `k_turbo4_mul_mat_vec_wht` — WHT dequant + inverse WHT
- `turbo-matmul.cu`: Updated dispatch — vector kernel for ne11=1, cuBLAS else
- `ggml-cuda.cu`: Turbo dispatch restored (routes to updated function)
- `turbo-wht.cu/cuh`: inverse WHT wrapper
- `build_turbo4.bat`: +DGGML_CUDA_ALL_QUANTS=ON
- `test_start_server.bat`: defaults to f16 cache, v6-i3 model, flash-attn on
- `tools/quantize/quantize.cpp`: Added TURBO4_0 and aliases to QUANT_OPTIONS
- `vecdot-turbo4.cuh`: MMVQ vec_dot (broken — garbage output)
- `mmvq.cu`: MMVQ pipeline with WHT pre-rotation

══════════════════════════════════════════════
MMVQ VEC_DOT STATUS
══════════════════════════════════════════════

`vec_dot_turbo4_0_q8_1` in `vecdot-turbo4.cuh` is wired into the MMVQ pipeline.
Speed: ~24-36 tok/s (measured from earlier MMVQ runs).
Output: Garbage.
Symptoms: The WHT pre-rotation at mmvq.cu:1191-1205 applies `k_turbo_wht` with
direction=0 to activations. The vec_dot then dots rotated activations against
turbo4 centroid weights. Hypothesis: either (a) the sign convention in the WHT
forward/centroid codebook is wrong, or (b) the dp4a math in the vec_dot needs
to account for the centroid structure differently, or (c) the q8_1
quantization of the rotated activations loses precision needed for correct
results.

══════════════════════════════════════════════
NEXT UP
══════════════════════════════════════════════
1. Debug MMVQ vec_dot — get 24+ tok/s with correct output
   a. Enable MMVQ dispatch by removing turbo bypass in ggml-cuda.cu lines 2626-2629
   b. Compare pre-rotation kernels (q8_1 path vs MMVQ path)
   c. Verify centroids/vec_dot math matches between working q8_1 and broken MMVQ
2. If MMVQ fixed: benchmark, push branch, publish to localmaxxing.com

══════════════════════════════════════════════
MEMORY
══════════════════════════════════════════════
- build_turbo4.bat is the canonical build script
- test_start_server.bat in experiments/turbo4-weight-quant/ is the server script
- v6-i3 model works; v4-i3 is zeroed-out (NTFS truncation from G: space exhaustion)
- Server at port 8081, api-key dummythicc
- Flash-attn is ON (upstream fixed asymmetric turbo KV in buun/master)
- Non-LIFO pool fix in ggml-cuda.cu:575 — necessary for turbo4 matmul to run without aborting
- Speed baseline: 1.09 tok/s decode, q8_1 vec_dot path, 128 tokens, 27B model on RTX 3090 (250W)
