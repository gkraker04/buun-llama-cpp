# TURBO4 Q8_1 VEC_DOT STATUS — 2026-06-03

## Current State
The turbo4 weight-quant custom matmul path (`ggml_cuda_mul_mat_turbo` → dispatch in `turbo-matmul.cu`) now uses a **q8_1 pre-quantized vec_dot** for single-token decode:

1. **Pre-rotate** activations via `k_turbo_wht` direction=0 (s1 pre-multiply → FWHT → s2·inv_sqrt_128 post-multiply)
2. **Quantize** rotated activations to q8_1 blocks via `quantize_row_q8_1_cuda`
3. **Vec_dot** with `k_turbo4_mul_mat_vec_q8` — centroid LUT × norm × q8_1 activation × q8_1 scale

**Result**: Correct output at ~1 tok/s (bottleneck: byte-level weight reads, 32 threads, 2048 iterations/thread)

## Known Issues
- **Speed**: 1 tok/s is the byte-scalar bottleneck. True speed requires vectorized weight loads (uint32_t) or MMVQ integration.
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

## Script Usage — MANDATORY
**ALWAYS use the .bat scripts. Never run ad-hoc commands.**

| Action | Command | Description |
|--------|---------|-------------|
| Build | `build_llama targets=llama-server` | Full build with defaults |
| Quick CUDA rebuild | `build_llama targets=ggml-cuda` | Rebuild just the CUDA lib |
| Start server | `start_server` | v6 model, port 8081 |
| Start alt port | `start_server port=8082 mtp_p=0.5` | With different params |
| Stop server | `stop_server` | Kill + GPU cleanup |
| Stop specific port | `stop_server port=8082` | | 

### Named Parameter Convention
All scripts accept `key=value` parameters. Defaults are documented at the top of each script under `DEFAULTS (LOCKED)`.

### Why Scripts
- Solves zombie process/NTFS lock issues (cleanup before launch)
- Solves wrong model path issues (resolved from named versions)
- Solves zombie GPU memory issues (port-based PID tracking)
- Solves forgotten args issues (known-good defaults always used)
