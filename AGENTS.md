# TURBO4 STATUS — 2026-05-31 (UPDATED)

**Branch:** `experiments/turbo4-quantize` (HEAD: `ca4c3d20d`)
**Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v5-i3.gguf
**Server:** port 8081

## ROOT CAUSE NARROWED

**cuBLAS fallback (dequant→fp16→GEMM) produces CORRECT output.**
The custom matmul kernel and MMVQ vec_dot paths produce gibberish.

This confirms:
1. ✅ GGUF weight data is correct (no corruption)
2. ✅ `turbo4_dequant_block_to_half` is correct (used by cuBLAS path)
3. ✅ WHT activation rotation (`k_turbo_wht`) is correct
4. ❌ Bug is in the custom matmul kernel (`k_turbo4_mul_mat_vec` / `k_turbo4_mul_mat_multi` in `turbo-matmul.cu`)

Also confirmed: both `experiments/turbo4-quantize` (custom matmul) and `master` (MMVQ) produce gibberish → the bug is common to both paths, NOT specific to either kernel implementation.

## What's Different Between cuBLAS and Custom Matmul

Both paths:
- Pre-rotate activations with the same `ggml_cuda_turbo_wht_forward` kernel
- Read `block_turbo4_0` from the same weight data
- Look up `d_turbo_centroids_4bit[idx] * norm` identically

Differences:
- **cuBLAS:** Dequant ALL weights to fp16 array first, then one cuBLAS GEMM call. Uses `turbo4_dequant_block_to_half` which loops over ALL 128 elements per block.
- **Custom matmul:** Reads ONE nibble at a time, does F32 MACC inline. 32 threads × 4 elements per block × warp reduction.

## Possible Root Causes (untested)

1. **Thread indexing bug in custom matmul kernel** — `elem = lane + je * 32` gives elements 0..127 in interleaved order. The WHT outputs in sequential order (thread 0 → element 0, thread 1 → element 1, etc.). The kernel reads `vy[block_base + elem]` in interleaved order (0, 32, 64, 96, 1, 33, 65, 97, ...). Since each thread sums its 4 elements, then the warp reduction adds them all, the interleaving shouldn't matter for a sum. But verify.

2. **Shared memory or warp shuffle bug** — The `__shfl_xor_sync` reduction at the end. If the shuffle doesn't correctly reduce across all 32 lanes, only a subset of elements contribute to the result.

3. **`__half2float` vs `__float2half` casts** — `blk->norm` is read as `__half2float`. But the norm was stored as `GGML_FP32_TO_FP16` on the CPU side (which rounds to fp16). The GPU reads it as fp16 and converts to F32. Could there be a subtle difference?

4. **MMVQ vec_dot uses int8 centroid approximation** — `kvalues_turbo4[idx] * TURBO4_SCALE_FACTOR` instead of the exact `d_turbo_centroids_4bit[idx]`. Small int8 quantization errors (~0.3% per centroid) could compound across layers.

## Next Steps

1. **Rebuild and use cuBLAS permanently** — Accept the speed hit (~3 tok/s from earlier testing vs ~12+ tok/s with custom matmul). This is the simplest path to a working model.

2. **Or fix the custom matmul kernel** — The kernel is in `ggml/src/ggml-cuda/turbo-matmul.cu`. Focus on the thread indexing (lane-based element assignment vs WHT output order) and the warp shuffle reduction. The fact that BOTH custom matmul and MMVQ produce gibberish suggests the common element is wrong (norm handling? centroid lookup? dp4a path?).

3. **Or try routing turbo4 through the MMQ path** — The MMQ batched matmul kernel (for ncols_dst > 1) doesn't use vec_dot — it uses a different pattern. If MMQ works, the bug is specific to MMVQ's vec_dot path.

## Validation Test Results (from validate_turbo4_weights.py)

| Check | Result |
|-------|--------|
| GGUF magic | ✓ |
| Tensor count | 866 (505 turbo4, 361 F16/F32) |
| First tensor | token_embd.weight [5120, 248320] |
| Row 0 L2 norm | 0.1075 |
| Row 1 L2 norm | 0.1139 |
| NaNs/Infs/Zeros | None |
| Centroid distribution | Reasonable (skew toward centroid 0 and 8) |
| Dequantized values | Mean -0.000021, Std 0.0015, Range [-0.009, 0.013] |
| **Verdict** | GGUF data is sound — bug is in GPU pipeline |
