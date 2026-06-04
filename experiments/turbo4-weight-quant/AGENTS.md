# TURBO4 Q8_1 VEC_DOT STATUS — 2026-06-04

## Current State
The turbo4 weight-quant custom matmul path (`ggml_cuda_mul_mat_turbo` → dispatch in `turbo-matmul.cu`) uses a **q8_1 pre-quantized vec_dot** for single-token decode:

1. **Pre-rotate** activations via `k_turbo_wht` direction=0 (s1 pre-multiply → FWHT → s2·inv_sqrt_128 post-multiply)
2. **Quantize** rotated activations to q8_1 blocks via `quantize_row_q8_1_cuda`
3. **Vec_dot** with `k_turbo4_mul_mat_vec_q8` — centroid LUT × norm × q8_1 activation × q8_1 scale

**Result**: Correct output at **1.09 tok/s** (bottleneck: byte-level weight reads, 32 threads, 2048 iterations/thread)

## Known Issues
- **Speed**: 1 tok/s is the byte-scalar bottleneck. True speed requires vectorized weight loads (uint32_t) or MMVQ integration.
- **Non-LIFO pool free**: ggml-cuda.cu:583 assert replaced with non-fatal warning + pool reset. Pathological allocations from turbo matmul graph captures don't free in LIFO order.
- **MMVQ vec_dot** (`vec_dot_turbo4_0_q8_1` in vecdot-turbo4.cuh): Produces ~24-36 tok/s but **garbage output**. Root cause suspected: the MMVQ WHT pre-rotation at mmvq.cu:1191 might have incorrect sign handling or the vec_dot dp4a math doesn't account for the WHT-domain centroid structure correctly.
- **cuBLAS path**: Works correctly for multi-token batches (uses full dequant to fp16 + cublasGemmEx).

## Next Steps
1. Fix MMVQ vec_dot correctness — get 24+ tok/s with correct output
2. Enable MMVQ dispatch by removing turbo bypass in `ggml-cuda.cu` lines 2626-2629

## File Locations
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\turbo-matmul.cu` — q8_1 vec_dot kernel + dispatch
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\vecdot-turbo4.cuh` — MMVQ vec_dot (broken)
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\mmvq.cu` — MMVQ pipeline with WHT pre-rotation
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\ggml-cuda.cu` — dispatch bypass (lines 2626-2629)

## Script Usage
| Action | Command | Description |
|--------|---------|-------------|
| Build | `build_turbo4.bat` | Full build (CUDA 13.2, Ninja, sm_86) |
| Start server | `experiments\turbo4-weight-quant\test_start_server.bat` | v6 model, port 8081 |
| Stop server | `taskkill /PID <pid>` | Graceful stop (no /F for GPU processes) |
