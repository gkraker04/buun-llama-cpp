# Competitive Analysis: Ours vs TheTom vs Duster (2026-03-31)

## Repos Tested

- **Ours**: `/root/llama-tcq-clean` (master, TCQ codebooks compiled-in)
- **TheTom**: `github.com/TheTom/llama-cpp-turboquant` feature/turboquant-kv-cache @ 8ad0f00
- **Duster**: `github.com/dusterbloom/llama-cpp-turboquant-cuda` feature/turboquant-kv-cache @ a7a6d10
- **Server**: RTX 3090 24GB, Qwen3.5-27B Q6_K, wikitext-2-raw (MD5 7c0137fc)

## Full Benchmark Results

### 3-bit PPL (turbo3 K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| **ours (TCQ)** | **6.507** | **6.883** | **7.005** | 7.053 |
| TheTom turbo3 | 6.548 | 6.934 | 7.089 | 7.114 |
| Duster turbo3 | 6.562 | 6.917 | 7.088 | 7.115 |
| Duster TBQ3 | 6.565 | 6.921 | 7.056 | **7.034** |

### 2-bit PPL (turbo2 K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| ours (TCQ) | 6.742 | 7.266 | 7.294 | 7.484 |
| TheTom turbo2 | **6.739** | 7.386 | 7.478 | 7.652 |
| Duster turbo2 | BROKEN | BROKEN | BROKEN | BROKEN |
| **Duster TBQ2** | 6.798 | **7.233** | **7.186** | **7.332** |

Note: our 2-bit with best codebook (not env var loaded here): 6.708 @2K, 7.222 @64K

### 4-bit PPL (turbo4 K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| ours turbo4 | 6.498 | 6.865 | 6.942 | 6.940 |
| TheTom turbo4 | 6.552 | 6.972 | 7.056 | 7.058 |
| Duster turbo4 | 6.498 | 6.865 | 6.942 | 6.940 |
| **Duster TBQ4** | **6.492** | **6.856** | **6.920** | **6.909** |

### Speed (tok/s) — turbo3

| Impl | pp=512 | pp=8K | pp=32K | decode |
|------|--------|-------|--------|--------|
| ours (TCQ) | 892 | 878 | 796 | 28.7 |
| **TheTom** | **1137** | **1109** | **989** | **30.8** |
| Duster turbo3 | 1131 | 1102 | 986 | 30.1 |

### Speed (tok/s) — turbo4 (all identical)

| Impl | pp=512 | pp=8K | pp=32K | decode |
|------|--------|-------|--------|--------|
| ours | 1135 | 1100 | 978 | 30.0 |
| TheTom | 1134 | 1107 | 986 | 30.7 |
| Duster | 1135 | 1103 | 973 | 30.1 |

## Speed Gap Root Cause Analysis

The speed gap is **TCQ-specific** — turbo4 (no TCQ) matches everyone perfectly.

### Encode: Viterbi is inherently slower (~partially fixable)

- Our TCQ: 512 threads/block, 128 sequential barrier-synced Viterbi iterations, 34.5KB shared mem, `__launch_bounds__(512,1)` limits to 1 block/SM
- Duster TBQ: 128 threads/block, fully parallel scalar quantize (one binary search per element), ~660B shared mem
- **Inherent to TCQ**: trellis is sequential along time axis. Can't parallelize the 128 steps.
- **Fixable**: thread-0 serial sections (FWHT rotation, backtracking, packing) could be parallelized. Maybe ~5% gain.

### Decode: Full-context dequant to f16 (~7% fixable)

- Our approach: dequant ALL KV tokens to f16 temp buffer, then run f16 flash attention
- Duster TBQ: native `vec_dot_fattn_vec_KQ_tbq3_0` reads compressed data directly in FA inner loop
- **NOT inherent to TCQ**: a native `vec_dot_fattn_vec_KQ_turbo3_tcq` could read 9-bit states and do codebook lookup inline
- Our TCQ dequant is actually SIMPLER than TBQ's (just codebook lookup, no inverse Hadamard), so our native vec_dot would be easier to implement

### Prefill: No chunked cuBLAS GEMM (20-27% fixable)

- Our approach: bulk dequant entire KV cache to f16, then MMA flash attention
- Duster: chunked cuBLAS GEMM prefill — dequant 4096 tokens at a time, use `cublasGemmStridedBatchedEx` for Q@K^T and P@V, custom online softmax kernels between
- **NOT inherent to TCQ**: cuBLAS only needs dequanted f16 chunks, doesn't care how dequant works
- Our TCQ dequant per chunk would be FASTER than TBQ's (no inverse Hadamard needed)
- Duster's implementation: 3-kernel pipeline (init, softmax-update, finalize) + cuBLAS GEMMs
- Enables 350K+ context on single RTX 3090

## Duster's TBQ: What It Is

Duster independently implemented TBQ as SRHT + Lloyd-Max quantization:
- File: `tbq-quant.cu` (566 lines, clean standalone implementation)
- Same rotation concept as turbo: L2 norm → normalize → Rademacher sign flips (fixed seed=42) → 128-point Hadamard → quantize
- Uses textbook-optimal Lloyd-Max centroids for N(0,1) distribution (no training needed)
- 3 types: TBQ2 (4 centroids), TBQ3 (8 centroids), TBQ4 (16 centroids)
- Hardcoded boundaries and centroids as `__constant__` arrays

## Quality Findings

### Where TCQ wins (short-medium context)
- 3-bit: TCQ beats all at 2K-32K by 0.04-0.08 PPL
- Trellis coding gain provides ~0.04 PPL advantage at short context

### Where TBQ wins (long context)
- 3-bit @64K: TBQ3 7.034 vs TCQ 7.053 (margin: 0.019)
- 2-bit @8K+: TBQ2 consistently better (7.233 vs 7.266 @8K, 7.186 vs 7.294 @32K)
- 4-bit: TBQ4 marginally better everywhere (~0.01-0.03)

### Why TBQ scales better at long context — BOMBSHELL FINDING

**TBQ is accidentally doing 1-bit quantization!**

TBQ uses N(0,1) Lloyd-Max centroids on post-FWHT data that is N(0, 1/sqrt(128)) = N(0, 0.088).
The data std dev (0.088) is ~2.8x smaller than the innermost centroid distance (0.245).
**Every single value maps to one of two inner bins** — just encoding the sign.
100% of values land in bins 3 or 4 (out of 0-7). Effective entropy: 0.997 bits out of 3.0 bits.

**TBQ's "quality advantage" comes from an accidental temperature scaling bug:**
- TBQ stores raw L2 norm without correction. Reconstructed values are 2.77x too large (3-bit).
- This acts as attention temperature reduction to T=0.36 (sharpens attention).
- At long context, sharper attention helps (less noise from irrelevant KV entries).
- The 64K ordering (TBQ4=6.909, TBQ3=7.034, TBQ2=7.332) is monotonically ordered by
  temperature severity — exactly as predicted by this theory.

**Our norm correction is technically correct but suboptimal for attention:**
- We preserve exact L2 norm (raw_norm / recon_norm correction).
- This gives 976x better per-element MSE than TBQ.
- But at long context, the temperature sharpening from TBQ's scale error helps more than MSE.

**Second factor: TCQ error correlation:**
- Our trellis (right-shift, k=3, L=9) shares 6/9 state bits between consecutive positions.
- This constrains consecutive centroid choices → correlated quantization errors.
- Correlated errors average out more slowly in Q@K dot products.
- Scalar Lloyd-Max has zero autocorrelation; TCQ likely has autocorrelation ~0.15-0.30 at lag 1.

### Actionable Experiments from Quality Analysis (ranked by expected impact)

1. **Temperature scaling (EASY, HIGH IMPACT)**: Multiply corrected_norm by alpha (try 1.5, 2.0, 2.5)
   in encode kernel. Combines our 976x better MSE with TBQ's temperature benefit.

2. **Optimal temperature grid search**: PPL at 2K/8K/32K/64K across alpha values.

3. **Remove norm correction for K only**: Raw norm for K (temperature on attention logits),
   keep correction for V (accurate weighted sum). K temperature helps attention routing,
   V accuracy helps output quality.

4. **TCQ error decorrelation (HARD)**: Apply fixed permutation (e.g., bit-reversal) to element
   indices after FWHT before trellis encoding. Decorrelates errors across d_k dimension
   without changing MSE. Requires matching inverse permutation in decode.

## Infrastructure Comparison

| Feature | Ours | Duster | Gap? |
|---------|------|--------|------|
| Persistent dequant buffers | YES | YES | No |
| Graph reuse + pipeline parallelism | YES | YES | No |
| Layer-adaptive KV cache | YES | YES | No |
| Asymmetric K/V support | YES | YES | No |
| Padding non-128 heads | In progress | Done | Minor |
| Chunked cuBLAS GEMM prefill | **NO** | YES | **Major** |
| Native vec_dot for decode | **NO** | YES (TBQ only) | **Major** |

## Action Items (Priority Order)

All tracked in `experiments.md` with full details.

### P0: Temperature scaling (experiments #69, #70, #76)
- #69: Multiply corrected_norm by alpha (1.5-2.5) — one-line change, potentially beats TBQ everywhere
- #70: Asymmetric K/V — raw norm for K (temperature), corrected for V (accuracy)
- #76: Full alpha grid search across bit-rates and context lengths

### P1: Native TCQ vec_dot for decode (#71)
- Implement `vec_dot_fattn_vec_KQ_turbo3_tcq` in fattn-common.cuh
- Read 9-bit state from bitstream → `codebook[state] * norm` → dot product
- Expected: ~7% decode speedup, reduced VRAM at long context

### P2: Chunked cuBLAS GEMM prefill (#72, #84)
- Port Duster's 3-kernel pipeline. 4096-token chunks + cuBLAS GEMMs.
- Expected: 20-27% prefill speedup, enables 350K+ context
- NOT TCQ-specific — applies to all our quant types

### P3: Encode speedup (#73)
- Parallelize thread-0 FWHT + bitpacking. Expected ~5%.

### P4: Quality investigations (#74, #75, #77, #78, #82, #83)
- #74: TCQ error decorrelation via element permutation
- #75: Lloyd-Max boundaries as TCQ initial state prior
- #77: Investigate turbo4 vs TBQ4 quality gap (Lloyd-Max centroids?)
- #78: Measure TCQ error autocorrelation empirically
- #82: Replicate TBQ's exact 1-bit behavior to validate temperature theory
- #83: Adaptive per-layer temperature scaling

### P5: Integration & validation (#79, #80, #81)
- #79: TBQ-style fast encode as speed-critical fallback
- #80: Finish head_dim padding verification
- #81: Sparse V dequant with TCQ path (+22.8% decode at 32K)

## Server State After Analysis

- `/root/llama-thetom` — TheTom's repo, built, feature/turboquant-kv-cache branch
- `/root/llama-duster` — Duster's repo, built, feature/turboquant-kv-cache branch
- `/root/llama-tcq-clean` — our repo, built
- Benchmark results in `/tmp/bench_comparison.txt`, `/tmp/bench_turbo4.txt`
- Quality agent transcript in agent output file (check task output)
