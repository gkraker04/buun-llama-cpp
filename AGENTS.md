# TURBO4 STATUS — 2026-05-31 (UPDATED)

**Branch:** `experiments/turbo4-quantize` (HEAD: `5f6f2c561`)
**Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v5-i3.gguf
**Server:** port 8081

## ROOT CAUSE ISOLATED

**Simple single-thread kernel produces CORRECT output at ~6 tok/s.**
**cuBLAS fallback (dequant→fp16→GEMM) also produces CORRECT output.**
Original parallel kernel (`k_turbo4_mul_mat_vec`) and MMVQ vec_dot produce gibberish.

This confirms:
1. ✅ GGUF weight data is correct (no corruption)
2. ✅ `turbo4_dequant_block_to_half` is correct (used by cuBLAS path)
3. ✅ WHT activation rotation (`k_turbo_wht`) is correct
4. ✅ Basic dequant+dot math is correct (`d_turbo_centroids_4bit[idx] * norm * activation[i]`)
5. ❌ Bug is in the **parallel kernel's thread indexing/reduction** (warp shuffle or element assignment)

## What Works Now

- **Single-token decode (ncols_dst == 1):** Uses `k_turbo4_mul_mat_vec_simple` — 1 thread per row, correct output at ~6 tok/s
- **Multi-token/large batch:** Falls back to cuBLAS path (verified correct)
- **Dispatch code:** Fixed syntax errors in `ggml_cuda_mul_mat_turbo` (had `break`/`case` without switch)

## What's Broken (Not Used Currently)

- `k_turbo4_mul_mat_vec` — parallel kernel with 32 threads per row, warp shuffle reduction
- `k_turbo4_mul_mat_multi<N>` — multi-token parallel kernels (same bug pattern)
- MMVQ vec_dot path (`vec_dot_turbo4_0_q8_1`) — produces gibberish

Suspect: thread indexing `elem = lane + je * 32` gives interleaved access to activations, but WHT outputs in sequential order. The sum reduction should be order-independent, so the warp shuffle or element assignment is likely wrong.
