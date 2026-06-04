# TURBO4 STATUS — 2026-06-04 (16:00)

**Branch:** `experiments/turbo4-quantize` (commit `252d11e14`)
**Build:** `build_turbo4.bat` — CUDA 13.2, Ninja, MSVC vcvars64, sm_86
**Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf
**Server:** Port 8082, `test_start_mmvq.bat` (no MTP, f16 cache, --no-warmup)

══════════════════════════════════════════════
CURRENT STATE — MMVQ WORKING ✓
══════════════════════════════════════════════

**MMVQ vec_dot (decode, ne11=1):**
- ✅ **Correct output** — " Paris." for "The capital of France is"
- ✅ **6.51 tok/s** (128-token benchmark, 8-token prefill, 19.7s decode)
- ✅ **No CUDA graph hangs**, no deadlocks
- ✅ Uses `__shfl_xor_sync` with `__activemask()` for warp-level cooperation

**Approach:**
- Register-only iWHT on centroids (s2→butterfly→s1×inv_sqrt)
- Thread pair (2 threads per turbo4 block): each does 64-element butterfly passes 1-6, exchange for pass 7 via warp shuffle
- Float dot product with unrotated q8_1 activations
- No shared memory, no dp4a, no int8 approximation

**Prefill (ne11>1):** Falls back to cuBLAS (3.7–4.2 tok/s)

══════════════════════════════════════════════
HISTORY
══════════════════════════════════════════════

**Stage 1 — The bug (WHT pre-rotation) ✗**
- Applied WHT forward to activations, then dp4a with unrotated int8 centroids
- ~31 tok/s but **completely garbage output** (HTTP 500)
- Root cause: WHT forward and inverse are NOT the same (different s1/s2 sign application)
- WHT_fwd: s1→butterfly→s2×inv_sqrt. WHT_inv: s2→butterfly→s1×inv_sqrt.
- Sign pattern mismatch makes M_ij ≠ M_ji, so Parseval doesn't apply in the simple form

**Stage 2 — Naive float iWHT ✓ (slow)**
- Each thread independently computed ALL 128 iWHT values, then dotted with its 64-element half
- Correct but 1.24 tok/s — each thread did 7-pass 128-element butterfly + 64-element dot

**Stage 3 — Shuffle-optimized ✓ (current)**
- Thread pair: 64-element butterfly each, exchange for pass 7 via __shfl_xor_sync
- 6.51 tok/s — correct, ~5× faster than naive float
- __activemask() ensures deadlock-free across varying iteration counts

══════════════════════════════════════════════
PERFORMANCE
══════════════════════════════════════════════

**Decode (MMVQ vec_dot, ne11=1):**
| Variant | Speed | Correct? | Notes |
|---------|-------|----------|-------|
| cuBLAS fallback | ~0.46 tok/s | ✅ | Full dequant→fp16→cublasGemmEx |
| q8_1 vec_dot | ~1.09 tok/s | ✅ | Centroid LUT × norm × q8_1 |
| WHT pre-rot (broken) | ~31 tok/s | ❌ | Wrong sign convention |
| Naive float iWHT | 1.24 tok/s | ✅ | Stage 2: full 128-element per thread |
| Shuffle iWHT | **6.51 tok/s** | ✅ | Stage 3: current best |

**Prefill (cuBLAS fallback, ne11>1):** 3.7–4.2 tok/s

**Q4_K_M baseline (reference):** ~15-18 tok/s decode

**Comparison with broken 31 tok/s path:**
- Old path: 1 instruction per 4 elements (dp4a), 0 WHT ops
- New path: ~1152 float ops per thread per turbo4 block (WHT + dot)
- The dp4a path is ~100× more compute-efficient, but was mathematically wrong

══════════════════════════════════════════════
ROOT CAUSE ANALYSIS
══════════════════════════════════════════════

**Why WHT pre-rotation produced garbage:**
The WHT forward and inverse transforms differ in their sign application order:
- Forward: s1 → butterfly → s2 × inv_sqrt
- Inverse: s2 → butterfly → s1 × inv_sqrt

The matrix M_ij = s2[i] × H_ij × s1[j] is NOT symmetric because s1[i] ≠ s2[i]
for many elements. Therefore Σ WHT_inv(C) · A ≠ Σ C · WHT_fwd(A).
The sign mismatch causes element-level corruption, explaining the "garbage output" quality.

**Key lesson:** Always match the CPU dequant transform exactly. The CPU applies
inverse WHT to centroids (s2→butterfly→s1). The vec_dot must do the same.

══════════════════════════════════════════════
OUTLOOK
══════════════════════════════════════════════

**Current speed:** 6.51 tok/s decode. Usable but ~2.5× slower than Q4_K_M.

**The bottleneck:** ~1152 float ops per vec_dot call (WHT butterfly + dot product)
vs ~4 int ops for Q4_K_M's dp4a path. The iWHT mixes ALL 128 elements, preventing
simple dp4a lookup. A precomputed lookup table would need 16^128 entries.

**Potential speedups from here:**
1. Dequant turbo4→fp16 once at layer load time, cache in temp buffer, use fp16 MMVQ
2. Fuse iWHT into the weight-quantization step (store weights pre-iWHT'd in GGUF)
   — Would need modified quantizer and new model file. ~31 tok/s reachable.
3. Write a custom CUDA kernel that does iWHT + dp4a in one pass
4. Accept 6.5 tok/s as the fast-correct baseline and move on to other work

══════════════════════════════════════════════
MODIFIED FILES
══════════════════════════════════════════════

- `ggml/src/ggml-cuda/vecdot-turbo4.cuh` — Shuffle-optimized iWHT vec_dot
- `ggml/src/ggml-cuda/mmvq.cu` — WHT pre-rotation removed, clean dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu` — Non-LIFO pool fix
- `experiments/turbo4-weight-quant/test_start_mmvq.bat` — Added --no-warmup
- `experiments/turbo4-weight-quant/bench_mmvq.py` — Benchmark script

══════════════════════════════════════════════
MEMORY (your personal notes)
══════════════════════════════════════════════
- MMVQ vec_dot working correctly at 6.51 tok/s with shuffle-optimized iWHT approach
- WHT pre-rotation was fundamentally wrong (s1≠s2 sign pattern breaks Parseval)
- Always match CPU dequant: iWHT on centroids (s2→butterfly→s1), not WHT on activations
- __shfl_xor_sync with __activemask() for deadlock-free warp cooperation
- --no-warmup needed for testing (float iWHT is slow enough that warmup takes >60s)
- test_start_mmvq.bat at port 8082, api-key dummythicc, flash-attn on, f16 cache
- build_turbo4.bat for builds, _just_ninja.bat for quick iter
- Current branch: experiments/turbo4-quantize on buun/master
- Speed comparison: 31 tok/s dp4a (broken) vs 6.5 tok/s float iWHT (correct)
- ~1152 float ops per turbo4 block per vec_dot call. Bottleneck is compute, not bandwidth.
- Potential 2×: fuse iWHT into quantizer (pre-iWHT centroids in GGUF). Would need new model.
- Server WITHOUT --no-warmup takes ~2min to warm up at 1-6 tok/s
