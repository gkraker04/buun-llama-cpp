# Turbo4 Speed Optimization Plan — 2026-06-03

## Current Baseline
- **q8_1 pre-quantized vec_dot**: ~1.0 tok/s, correct output
- **MMVQ path** (bypass disabled): ~24-36 tok/s, **garbage output**
- **Target**: 15-18 tok/s (match Q4_K_M)

## Verification: q8_1 vec_dot produces garbage
The q8_1 vec_dot (with correct quantize stride `s01=ncols_x`) still produces garbage output at ~1 tok/s.
Root cause unknown — centroids, norm, and activation math match the working simple kernel.
Possible causes:
1. `quantize_row_q8_1_cuda` produces wrong results for WHT-rotated activation values
2. The vec_dot kernel has an alignment or indexing bug
3. `block_q8_1.ds` access with `__low2float` is wrong

**Next**: Add diagnostic — use `cudaMemcpy` to copy q8_1 quantized output to host and verify.

## High-Impact Target: MMVQ vec_dot
**MMVQ produces 36 tok/s but garbage output. Fix this first.**

### Known
- WHT pre-rotation at mmvq.cu:1191 calls `ggml_cuda_turbo_wht_forward` — same as prerotate_activations
- `vec_dot_turbo4_0_q8_1` in vecdot-turbo4.cuh uses dp4a intrinsics
- The dp4a uses LUT for centroids and 8-bit activation values

### Investigation Plan
1. **Read vecdot-turbo4.cuh** — understand the dp4a math and check for bugs
2. **Trace the math**: compare with correct dequant formula
3. **Test with minimal vec_dot** — return simple scalar dot (no dp4a) to isolate dp4a vs math issue
4. **If dp4a is wrong**: rewrite as scalar dot (still 30+ tok/s vs current 1 tok/s)
5. **If math is wrong**: correct the formula and test

### Fallback: Speed up q8_1 vec_dot
If MMVQ proves intractable, optimize the working q8_1 kernel:
1. **Vectorized uint32_t loads**: 4 bytes at once instead of byte-level. Coalesced 4-byte reads → 4x fewer memory transactions
2. **Register blocking**: process 4 elements per thread per iteration
3. **Warp-level q8_1 quantization**: Use shared memory to quantize activations in one warp-wide pass

## Script Usage (MANDATORY)
| Action | Command |
|--------|---------|
| Build | `build_llama targets=llama-server` (or `targets=ggml-cuda` for quick rebuild) |
| Start | `start_server model=v6-i3 port=8081` |
| Stop  | `stop_server` |
| Quick GPU memory clear | `stop_server` (kills zombies + port holders) |

## Key Files
- `turbo-matmul.cu` — q8_1 vec_dot kernel + dispatch (WORKING)
- `vecdot-turbo4.cuh` — MMVQ vec_dot (BROKEN)
- `mmvq.cu` — MMVQ pipeline + WHT pre-rotation
- `ggml-cuda.cu` — dispatch bypass at lines 2626-2629
