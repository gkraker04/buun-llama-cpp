# TURBO4 STATUS — 2026-06-18 (02:15)

**Branch:** `experiments/turbo4-dp4a-fusion` (commit `d61726155` + dp4a attempt reverted)
**Build:** `build_turbo4.bat` — CUDA 13.2, Ninja, MSVC vcvars64, sm_86
**Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v10-i6.gguf
**Server:** Port 8082, `experiments/turbo4-weight-quant/test_start_server.bat` (no MTP, f16 cache)

══════════════════════════════════════════════
CURRENT STATE — WHT PRE-ROTATION WORKING ✓
══════════════════════════════════════════════

**MMVQ vec_dot (decode, ne11=1):**
- ✅ **Correct output** — " Paris." for "The capital of France is"
- ✅ **21-22 tok/s** generation (64-128 token benchmark)
- ✅ **24-32 tok/s** prompt processing (short prompts)
- ✅ **No CUDA graph hangs**, no deadlocks

**Approach:**
- Forward WHT pre-rotation on activations (once per row, before q8_1 quantization)
- Simplified vec_dot: codebook lookup + float dot product (no iWHT butterfly)
- Uses `ggml_cuda_turbo_wht_forward` from `turbo-wht.cu`
- Parseval identity: Σ iWHT(C)·A = Σ C·WHT(A)

**Prefill (ne11>1):** Falls back to cuBLAS (untested with new approach)

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

**Stage 3 — Shuffle-optimized iWHT ✓ (previous best)**
- Thread pair: 64-element butterfly each, exchange for pass 7 via __shfl_xor_sync
- 6.51 tok/s — correct, ~5× faster than naive float
- __activemask() ensures deadlock-free across varying iteration counts

**Stage 4 — WHT pre-rotation ✓ (current)**
- Pre-rotate activations with forward WHT before q8_1 quantization
- Simplified vec_dot: no iWHT butterfly, just codebook lookup + dot
- 21-22 tok/s — ~35% faster than Stage 3, ~17× faster than Stage 2

**Stage 5 — dp4a vec_dot attempt ✗ (reverted)**
- Converted float centroids to int8, used `ggml_cuda_dp4a()` for 4× throughput
- 24 tok/s but **garbage output** — subtle bug in int8 arithmetic or scale handling
- Reverted to Stage 4 (rotated vec_dot) for stability

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
| Shuffle iWHT | 6.51 tok/s | ✅ | Stage 3: previous best |
| **WHT pre-rotation** | **21-22 tok/s** | ✅ | Stage 4: current best |
| dp4a vec_dot | 24 tok/s | ❌ | Stage 5: int8 bug, reverted |

**Prefill (cuBLAS fallback, ne11>1):** 3.7–4.2 tok/s (untested with new approach)

**Q4_K_M baseline (reference):** ~15-18 tok/s decode

**Comparison with broken 31 tok/s path:**
- Old path: 1 instruction per 4 elements (dp4a), 0 WHT ops
- New path: ~1152 float ops per thread per turbo4 block (WHT + dot)
- The dp4a path is ~100× more compute-efficient, but was mathematically wrong

══════════════════════════════════════════════
ROOT CAUSE ANALYSIS
══════════════════════════════════════════════

**Why WHT pre-rotation produced garbage (Stage 1):**
The WHT forward and inverse transforms differ in their sign application order:
- Forward: s1 → butterfly → s2 × inv_sqrt
- Inverse: s2 → butterfly → s1 × inv_sqrt

The matrix M_ij = s2[i] × H_ij × s1[j] is NOT symmetric because s1[i] ≠ s2[i]
for many elements. Therefore Σ WHT_inv(C) · A ≠ Σ C · WHT_fwd(A).
The sign mismatch causes element-level corruption, explaining the "garbage output" quality.

**Key lesson:** Always match the CPU dequant transform exactly. The CPU applies
inverse WHT to centroids (s2→butterfly→s1). The vec_dot must do the same.

**Why Stage 4 works:** Forward WHT on activations (direction=0: s1→butterfly→s2×inv_sqrt)
gives a_rot such that Σ iWHT(C)·A = Σ C·a_rot (Parseval identity).
The vec_dot then just loads centroids and dots with pre-rotated q8_1 activations.

**Why dp4a failed (Stage 5):**
The dp4a approach quantizes float centroids to int8 and uses `ggml_cuda_dp4a()` for
4× throughput. However, the implementation has a subtle bug in scale handling or
int8 packing that produces garbage output. The theoretical speedup is real (24 tok/s),
but correctness is non-negotiable. Debugging int8 arithmetic is time-consuming.

══════════════════════════════════════════════
OUTLOOK
══════════════════════════════════════════════

**Current speed:** 21-22 tok/s decode. ~30% short of buun's dp4a target (29-30 tok/s).

**Remaining gap analysis:**
- The vec_dot itself is now O(1) per element (just lookup + multiply-add)
- Bottleneck likely: q8_1 quantization overhead, memory bandwidth, or register pressure
- CUDA profiling needed to pinpoint exact bottleneck

**Potential speedups from here:**
1. Fuse WHT pre-rotation into q8_1 quantization kernel (avoid separate pass) — ~10-15% gain
2. Debug dp4a vec_dot implementation — potential 24→29 tok/s if bug fixed
3. Try different memory layout (interleave activations for better coalescing)
4. Profile with nvprof/nsight to find exact bottleneck
5. Accept 22 tok/s as viable baseline and move on to other work

**Target:** 29-30 tok/s (buun's dp4a baseline for turbo4 KV cache)

══════════════════════════════════════════════
MODIFIED FILES
══════════════════════════════════════════════

- `ggml/src/ggml-cuda/vecdot-turbo4.cuh` — Added vec_dot_turbo4_0_q8_1_rotated (simplified)
- `ggml/src/ggml-cuda/mmvq.cu` — WHT pre-rotation dispatch, calls ggml_cuda_turbo_wht_forward
- `ggml/src/ggml-cuda/turbo-wht.cu` — Forward/inverse WHT kernels (direction=0/1)
- `ggml/src/ggml-cuda/turbo-wht.cuh` — Header declarations
- `experiments/turbo4-weight-quant/test_start_server.bat` — Test server on port 8082
- `bench_turbo4.py` — Benchmark script (requests-based)

══════════════════════════════════════════════
MEMORY (your personal notes)
══════════════════════════════════════════════
- MMVQ vec_dot working correctly at 21-22 tok/s with WHT pre-rotation approach
- Stage 4: forward WHT on activations (direction=0) + simplified vec_dot = 35% speedup
- Always match CPU dequant: iWHT on centroids (s2→butterfly→s1), not WHT on activations
- __shfl_xor_sync with __activemask() for deadlock-free warp cooperation (Stage 3)
- test_start_server.bat at port 8082, api-key dummythicc, flash-attn on, f16 cache
- build_turbo4.bat for builds, _just_ninja.bat for quick iter
- Current branch: experiments/turbo4-dp4a-fusion on buun/master
- Speed comparison: 31 tok/s dp4a (broken) vs 22 tok/s WHT pre-rot (correct)
- ~1152 float ops per turbo4 block per vec_dot call. Bottleneck is compute, not bandwidth.
- Potential 2×: fuse iWHT into quantizer (pre-iWHT centroids in GGUF). Would need new model.
- Server WITHOUT --no-warmup takes ~2min to warm up at 1-6 tok/s (Stage 3)
- Stage 4 server warms up faster (~45s for model load)
- dp4a vec_dot (Stage 5) reached 24 tok/s but produced garbage output — reverted
- int8 centroid quantization: t4c_i8[] with scale 1/525.8, but scale handling has bug
