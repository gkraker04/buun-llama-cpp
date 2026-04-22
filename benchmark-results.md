# TurboQuant Benchmark Results

Hardware: RTX 3090 24GB, Qwen3.5 27B Q6_K (20.56 GiB)
Date: 2026-03-26
Build: feature/turboquant-kv-cache + FA_ALL_QUANTS=ON

## PPL (2K ctx, 8 chunks)

| Config | PPL | vs q8_0 | Notes |
|--------|-----|---------|-------|
| q8_0 baseline | 5.8375 | — | |
| turbo3 uniform | 5.8323 | -0.09% | with norm correction |
| turbo4 uniform | 5.8186 | -0.32% | with norm correction |
| LA-2 turbo3 | 5.8140 | -0.40% | TURBO_LAYER_ADAPTIVE=2, last 8/40 layers q8_0 |
| LA-2 turbo4 | 5.8077 | -0.51% | TURBO_LAYER_ADAPTIVE=2, last 8/40 layers q8_0 |
| **LA-1 turbo3** | **5.7958** | **-0.71%** | TURBO_LAYER_ADAPTIVE=1, first4+last4 q8_0. BEST PPL |
| LA-1 turbo4 | 5.8989 | +1.05% | WORSE than uniform turbo4! Early layers need turbo4's QJL |
| LA-3 turbo3 (last4) | 5.8091 | -0.49% | only 4 layers q8_0 = ~4.2x compression |
| LA-4 turbo3 (first4) | 5.8211 | -0.28% | only 4 layers q8_0 |
| LA-5 turbo3 (2+2) | 5.8091 | -0.49% | same as mode 3! Only 4 layers q8_0 = ~4.2x |
| turbo4-K + q8_0-V | 5.8451 | +0.13% | asymmetric |
| q8_0-K + turbo3-V | 5.8451 | +0.13% | asymmetric |
| turbo4-K + turbo3-V | 5.8653 | +0.48% | asymmetric, worst mixed combo |
| turbo3-K + turbo4-V | 5.8212 | -0.28% | asymmetric, values matter more! |

## Decode Speed tg64 (tok/s)

| Config | 4K | 16K | 32K | ratio @32K |
|--------|-----|------|------|-----------|
| q8_0 baseline | 31.02 | 30.77 | 30.69 | 1.000 |
| turbo3 uniform | 29.93 | 29.65 | 29.83 | 0.972 |
| turbo4 uniform | 29.43 | 29.41 | 29.47 | 0.960 |
| LA-2 turbo3 | 30.14 | 29.94 | 29.98 | 0.977 |
| LA-2 turbo4 | 29.69 | 29.68 | 29.69 | 0.967 |
| LA-1 turbo3 | 30.12 | — | 29.98 | 0.977 |
| turbo4-K + q8_0-V | 30.21 | 30.14 | 30.15 | 0.982 |
| q8_0-K + turbo3-V | 30.40 | 30.34 | 30.32 | 0.988 |
| turbo4-K + turbo3-V | 29.70 | 29.57 | 29.62 | 0.965 |
| turbo3-K + turbo4-V | 29.70 | 29.57 | 29.63 | 0.965 |

## Prefill Speed pp4096 (tok/s)

| Config | tok/s | ratio |
|--------|-------|-------|
| q8_0 | 1134.64 | 1.000 |
| turbo3 | 631.09 | 0.556 |
| turbo4 | 586.71 | 0.517 |

## Extreme Context (65K) — Decode Speed tg64

| Config | 65K tok/s | VRAM | vs 32K speed |
|--------|----------|------|-------------|
| LA-1 turbo3 | 29.98 | ~22.3 GiB | identical to 32K (29.98) |
| LA-5 turbo3 (2+2) | 29.90 | ~22.3 GiB | -0.3% |
| turbo4 uniform | 29.51 | ~22.2 GiB | +0.1% |

Note: q8_0 would need ~28+ GiB at 65K — would OOM on 24GB RTX 3090.

## Layer-Adaptive Mode Comparison (all turbo3, PPL 2K/8chunks)

| Mode | Which layers q8_0 | #layers q8_0 | PPL | vs q8_0 | Compression |
|------|-------------------|-------------|-----|---------|------------|
| 1 (4+4) | first 4 + last 4 | 8 | 5.7958 | -0.71% | ~3.5x |
| 3 (last4) | last 4 | 4 | 5.8091 | -0.49% | ~4.2x |
| 5 (2+2) | first 2 + last 2 | 4 | 5.8091 | -0.49% | ~4.2x |
| 2 (last8) | last 8 | 8 | 5.8140 | -0.40% | ~3.5x |
| 4 (first4) | first 4 | 4 | 5.8211 | -0.28% | ~4.2x |
| 0 (uniform) | none | 0 | 5.8323 | -0.09% | 4.9x |

## FWHT Inline FA Rotation (experiment/fwht-inline-fa)

Q pre-rotation moved from graph-level ggml_turbo_wht op to inline in FA kernels:
- Vec kernel (decode): shared memory FWHT rotation, zero extra kernel launches
- Prefill MMA: separate forward rotation kernel with persistent buffer
- V un-rotation remains at graph level (CUDA graph compatible)

| Metric | Before (graph-level) | After (inline Q) | Delta |
|--------|---------------------|-------------------|-------|
| Decode tg64 | 30.25 | 30.14 | -0.4% |
| Prefill pp512 | 1149.90 | 1146.21 | -0.3% |
| PPL (10-chunk) | 19.7152 | 19.7152 | identical |
| CUDA graphs | working | working | no change |

Key finding: `cudaMallocAsync` for Q rotation buffer caused NaN on graph replay.
Fixed by using a persistent `cudaMalloc` buffer that grows as needed.

## Observations

1. Norm correction makes turbo3 and turbo4 BEAT q8_0 in PPL
2. Layer-adaptive mode 2 improves PPL further (best: LA-2 turbo4 at 5.8077)
3. Layer-adaptive also improves decode speed (q8_0 layers dequant faster)
4. No context scaling regression — turbo/q8 ratio improves at longer contexts
5. Asymmetric turbo+q8 PPL is slightly worse than pure q8_0 (norm correction mismatch?)
6. q8_0-K + turbo3-V is fastest asymmetric config (98.8% of q8_0)
7. Prefill is the big gap (0.5x) — vec kernel only, no tensor core support yet
8. Mixed turbo-turbo: turbo3-K + turbo4-V (5.8212) beats turbo4-K + turbo3-V (5.8653) — contradicts "More Keys Less Values" paper. Values need more precision on this model.
9. Both mixed turbo-turbo combos decode at same speed (~29.65 tok/s) — no speed difference from K/V asymmetry
10. 65K context fits on 24GB RTX 3090 with all turbo configs (~22.2-22.3 GiB). q8_0 would OOM.
11. Decode speed at 65K is virtually identical to 32K — zero degradation even at 2x the context
12. Mode 3 (last4) = Mode 5 (first2+last2) in PPL. Last 2 layers dominate quality impact.
13. Mode 5 (first2+last2) is the max-compression sweet spot: only 4 layers q8_0, ~4.2x compression, -0.49% PPL
14. Asymmetric layer-adaptive (V-only or K-only promotion) does NOT help — norm correction mismatch between turbo+q8_0 within a layer hurts. Both K+V must be promoted together.

## FWHT Rotation Results (CORRECTED)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 WITH rotation + norm correction (committed HEAD) | 5.8323 | -0.09% (BETTER) |
| turbo3 WITHOUT rotation + WITH norm correction | 6.2357 | +6.8% worse |
| turbo3 WITHOUT rotation + WITHOUT norm correction | 6.5249 | +11.8% worse |
| q8_0 baseline | 5.8375 | — |

**CORRECTION**: Previous session incorrectly concluded rotation HURTS PPL. The 6.51 measurement was from a BROKEN double-rotation state (inline FA rotation applied on top of graph-level rotation). The committed code (721880c00+) with SET_ROWS rotation + graph-level TURBO_WHT gives correct PPL = 5.83.

**Conclusion**: FWHT rotation is ESSENTIAL for quality. It provides a 0.39 PPL improvement (6.24 → 5.83). Norm correction provides an additional 0.29 improvement (6.52 → 6.24). Together they make turbo3 beat q8_0.

## Prefill Dequant+MMA Optimization (experiment/prefill-dequant-attend)

| Config | pp4096 old (vec) | pp4096 new (dequant+MMA) | vs q8_0 (1134.64) | Speedup |
|--------|-----------------|-------------------------|-------------------|---------|
| turbo3 | 631.09 | 1121.33 | 98.8% | 1.78x |
| turbo4 | 586.71 | 1121.30 | 98.8% | 1.91x |

**PPL with optimization**: turbo3 = 5.8501 (vs 5.8323 baseline, within error bars)

**Decode speed**: 30.10 tok/s (unchanged from baseline ~30 tok/s)

**How it works**: During prefill (Q->ne[1] > 1), bulk-dequantize turbo K/V to fp16 temp buffers, then use the fast MMA (tensor core) kernel. During decode (Q->ne[1] == 1), use the existing vec kernel with inline dequant. The fp16 temp buffer is allocated with cudaMallocAsync and freed after attention completes.

**Memory overhead**: ~16 MB per KV head group (seq_len × head_dim × 2 bytes × 2 for K+V). Temporary, freed after each attention call.

**Mixed K/V prefill speed** (pp4096):

| Config | tok/s | vs q8_0 |
|--------|-------|---------|
| q8_0 K + turbo3 V | 1131.15 | 99.7% |
| turbo3 K + q8_0 V | 1129.26 | 99.5% |
| turbo3 K + turbo3 V | 1121.33 | 98.8% |
| turbo4 K + turbo4 V | 1121.30 | 98.8% |

## Extreme Context with Prefill Optimization (turbo3)

| Context | pp tok/s | tg64 tok/s | Notes |
|---------|----------|-----------|-------|
| 4K | 1123 | 30.03 | baseline |
| 32K | 980 | 29.83 | |
| 65K | 847 | 29.79 | q8_0 would OOM (~28+ GiB) |
| 98K | 748 | 29.86 | |
| 112K | 707 | 29.82 | |
| 128K | 669 | 29.85 | full model context window! |

**Key finding**: 27B model running at 128K context on a 24GB RTX 3090 with turbo3 KV cache. Impossible with q8_0. Decode speed is constant across all context lengths (~30 tok/s). Prefill scales sub-linearly with context length.

## Asymmetric Layer-Adaptive (turbo3, PPL 2K/8chunks)

| Mode | Strategy | PPL | vs q8_0 | Decode 4K | Notes |
|------|----------|-----|---------|-----------|-------|
| 6 | V-only q8_0 last 8 | 5.8390 | +0.03% | 30.16 | worse than uniform! |
| 7 | K-only q8_0 last 8 | 5.8390 | +0.03% | — | identical to mode 6 |
| 8 | V-only q8_0 first2+last2 | 5.8330 | -0.08% | — | ~= uniform |
| 2 (ref) | both K+V q8_0 last 8 | 5.8140 | -0.40% | 30.14 | much better |
| 0 (ref) | uniform turbo3 | 5.8323 | -0.09% | 29.93 | baseline |

**Key finding**: Asymmetric layer-adaptive does NOT help. Promoting only K or only V gives identical PPL (5.8390), both worse than uniform turbo3. The norm correction mismatch between turbo and q8_0 within the same layer hurts quality. Both K and V must be promoted together (mode 2) for the quality improvement to work.

## turbo4 Prefill Dequant+MMA Investigation

**Bug found**: turbo4 dequant_f16 kernel didn't handle ne0 > QK_TURBO4 (256 vs 128 for Qwen3.5-27B). Elements j >= 128 read from wrong block. Fix: add `blk_idx = j / QK_TURBO4` block indexing (matching turbo3 pattern).

| Config | PPL (prefill via MMA) | PPL (prefill via vec) | Difference |
|--------|----------------------|----------------------|------------|
| turbo3 | 5.8501 | 5.8323 | +0.3% (acceptable) |
| turbo4 (fixed) | 5.8966 | 5.8186 | +1.3% (too much) |

**Root cause of turbo4 regression**: turbo4's QJL correction adds ~0.001 magnitude adjustments per element. This is at the limit of fp16 precision (10-bit mantissa). The fp16 round-trip (dequant → fp16 buffer → MMA read) rounds away the QJL signal. turbo3 is unaffected because its 8 centroids are coarse enough (~0.3 spacing) for fp16.

**Decision**: Prefill dequant+MMA enabled for turbo3 only. turbo4 continues to use vec kernel for prefill (preserves PPL 5.8186 at cost of 588 tok/s prefill speed vs 1121 tok/s).

## Layer-Adaptive + Prefill MMA (turbo3, comprehensive)

| Config | PPL | vs q8_0 | pp4096 tok/s | pp/q8 | tg64 tok/s | tg/q8 | Compression |
|--------|-----|---------|-------------|-------|-----------|-------|-------------|
| **LA-1 turbo3** | **5.7690** | **-1.17%** | **1128** | **99.6%** | **30.25** | **97.5%** | ~3.5x |
| LA-5 turbo3 | 5.8246 | -0.22% | 1119 | 98.8% | 30.03 | 96.8% | ~4.2x |
| turbo3 uniform | 5.8501 | +0.22% | 1125 | 99.3% | 30.04 | 96.8% | 4.9x |
| q8_0 baseline | 5.8375 | — | 1133 | 100% | 31.04 | 100% | 1.0x |

**Recommended config: LA-1 turbo3** (TURBO_LAYER_ADAPTIVE=1)
- 1.17% BETTER PPL than q8_0
- 99.6% prefill speed, 97.5% decode speed
- 3.5x KV cache compression
- Enables 128K context on 24GB GPU where q8_0 OOMs at ~65K

## Experiment: Drop QJL from turbo4 (branch: experiment/drop-qjl)

| Config | PPL | vs q8_0 | pp4096 tok/s | tg64 tok/s | Compression |
|--------|-----|---------|-------------|-----------|-------------|
| turbo4 WITH QJL (baseline) | 5.8186 | -0.32% | 588 (vec) | 29.43 | 4.25 bits |
| turbo4 NO QJL | 5.8501 | +0.22% | 1124 (MMA!) | 29.40 | 4.25 bits* |
| turbo3 (reference) | 5.8323 | -0.09% | 1125 (MMA) | 29.93 | 3.5 bits |

*Block layout unchanged (rnorm+signs still present but zeroed). True format redesign would be 3.125 bits.

**Key finding**: QJL is worth +0.3 PPL points for turbo4. Without QJL, turbo4 is slightly WORSE than turbo3 in quality, speed, and compression. QJL + norm correction is the reason turbo4 beats q8_0. Dropping QJL does fix the fp16 prefill issue (MMA works = 1124 tok/s), but turbo3 already gets the same prefill speed.

**Conclusion**: Keep QJL. turbo4's value is the QJL+norm-correction combo. TheTom's "QJL unnecessary" finding may not apply when norm correction is present.

## Long-Context PPL Comparison (turbo3 vs q8_0)

| Context | Chunks | q8_0 PPL | turbo3 uniform PPL | turbo3 LA-1 PPL | LA-1 vs q8_0 |
|---------|--------|----------|-------------------|----------------|-------------|
| 2K | 8 | 5.8375 | 5.8323 (-0.09%) | 5.7690 (-1.17%) | **turbo3 wins** |
| 4K | 4 | 6.2677 | 6.3252 (+0.92%) | 6.3198 (+0.83%) | q8_0 wins |
| 8K | 4 | 7.4241 | 7.3783 (-0.62%) | 7.3952 (-0.39%) | **turbo3 wins** |

**Key finding**: Quality comparison is noisy across context lengths. Error bars ±0.16-0.18 are larger than the measured differences (0.03-0.09 PPL). turbo3 generally competitive with q8_0 at all context lengths. The PPL increase from 2K→8K is a data effect (wikitext text becomes harder to predict), not a quantization degradation — both turbo3 and q8_0 show the same pattern.

## Sign+Magnitude Encoding (branch: experiment/sign-magnitude-encoding)

turbo3 decode speed: 30.05 tok/s (4K) / 29.91 tok/s (32K). Identical to baseline. q8_0: 31.03 tok/s. The 3% gap is memory-bound, not ALU-bound. Encoding change has no effect.

## 128K Context Test

| Config | pp131072 tok/s | tg64 tok/s | Fits on 24GB? |
|--------|---------------|-----------|---------------|
| turbo3 uniform | 671.42 | 29.89 | YES |
| LA-5 turbo3 (first2+last2) | 673.95 | 30.01 | YES |
| LA-1 turbo3 (first4+last4) | — | — | **NO (OOM)** |

**Key finding**: LA-1 (8 q8_0 layers) OOMs at 128K on 24GB RTX 3090. LA-5 (4 q8_0 layers) and uniform turbo3 both work. LA-5 is the recommended config for 128K: PPL 5.8091 (-0.49% vs q8_0), 674 tok/s prefill, 30.01 tok/s decode, fits on 24GB.

**Context length recommendations**:
- Up to 65K: Use LA-1 turbo3 (best PPL: -1.17%)
- 65K-128K: Use LA-5 turbo3 (best balance: -0.49% PPL, 4.2x compression)
- 128K+: Use turbo3 uniform (maximum compression: 4.9x)

## Experiment: Vec Q Pre-rotation (experiment/vec-q-prerotate)

Moves FWHT Q rotation from inside the vec kernel (22 syncthreads for D=256) to a separate
kernel launch before vec kernel dispatch. Reduces register pressure and eliminates barrier stalls.

### MoE Model: Qwen3.5-35B-A3B-Q4_K_S (19.24 GiB)

| Context | KV Type | Before | After | Change |
|---------|---------|--------|-------|--------|
| p=0 | turbo3 | 118.10 | 125.77 | **+6.5%** |
| p=0 | q8_0 | — | 138.01 | (turbo3=91.1% of q8_0) |
| p=8K | turbo3 | — | 114.44 | — |

### Dense Model: Qwen3.5-27B-Q6_K (20.56 GiB)

| Context | KV Type | Before | After | Change |
|---------|---------|--------|-------|--------|
| p=0 | turbo3 | 30.05 | 30.13 | no regression |
| p=8K | turbo3 | — | 30.03 | — |
| p=32K | turbo3 | — | 29.94 | — |
| p=42K | turbo3 | — | 29.95 | — |
| p=32K | q8_0 | — | 30.99 | (turbo3=96.6%) |
| p=42K | q8_0 | — | 30.99 | (turbo3=96.6%) |

### Community Report (different GPU, Qwen3.5-27B-Q8_0, 42K context)

| KV Type | Prefill | Decode |
|---------|---------|--------|
| q8_0 | 2492.44 | 23.96 |
| turbo3 | 2445.92 | 16.30 (68% of q8_0) |

Note: regression is worse on the tester's GPU (likely lower bandwidth) where FFN is slower
and attention is a larger fraction of total compute. On our RTX 3090, turbo3 holds at 96.6% of q8_0.

PPL verification: 19.7152 (identical to baseline — FWHT is linear, so in-kernel vs pre-kernel produces same result).

## Experiment: Decode Dequant to FP16 (experiment/turbo-decode-fp16)

Dequant turbo3 K/V to fp16 before vec kernel for decode, so the inner loop uses
simple fp16 dot product instead of turbo3 bit-extract+LUT. Gated by GGML_TURBO_DECODE_FP16=1.

### MoE Model: Qwen3.5-35B-A3B-Q4_K_S

| Context | No flag | With flag | q8_0 | vs q8_0 |
|---------|---------|-----------|------|---------|
| p=0 | 125.77 | 127.98 | 138.01 | 92.7% |
| p=8K | 114.44 | **127.37** | — | — |
| p=32K | — | **126.92** | — | — |

### Dense Model: Qwen3.5-27B-Q6_K

| Context | No flag | With flag | q8_0 | vs q8_0 |
|---------|---------|-----------|------|---------|
| p=0 | 30.13 | 30.29 | 31.0 | 97.7% |
| p=32K | 29.94 | 30.05 | 30.99 | 97.0% |
| p=42K | 29.95 | 30.06 | 30.99 | 97.0% |

PPL verification: 19.7152 (bit-exact match — turbo3 centroids are lossless in fp16)

**Key finding**: FP16 dequant eliminates context scaling degradation on MoE models
(127 t/s flat from p=0 to p=32K) and has ZERO cost on dense models at all context lengths.
Could be made the default for turbo3 decode.

## Experiment: Attention-Sink Token Protection (#23)

Store first N KV positions at fp16 precision (pre-quantization), overwrite dequanted fp16
buffer before flash attention. Sink tokens receive disproportionate attention weights.

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 baseline (no sink) | 5.8501 ± 0.165 | — |
| N=4 sink tokens | 5.8246 ± 0.164 | -0.026 |
| N=8 sink tokens | 5.8506 ± 0.165 | +0.001 |
| N=16 sink tokens | 5.8894 ± 0.167 | +0.039 |

**Conclusion**: No significant improvement. All results within error bars (±0.165).
turbo3 + FWHT + norm correction already achieves high enough quality that protecting
sink positions provides no measurable benefit. The attention-sink phenomenon amplifies
quantization error at those positions, but turbo3's error is too small for the effect
to matter.

## Experiment: NSNQuant Per-Token DC Removal (#22)

Subtract per-element mean and renormalize before FWHT rotation. Simplified version of
NSNQuant (which uses cross-token channel means, incompatible with per-token SET_ROWS).

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 baseline | 5.8501 ± 0.165 | — |
| turbo3 + DC removal | 5.8827 ± 0.166 | +0.033 (noise) |
| turbo4 baseline | 5.8186 (ref) | — |
| turbo4 + DC removal | 17.4134 ± 0.618 | **+11.6 (catastrophic)** |

**Conclusion**: Per-token DC removal provides no benefit for turbo3 (values already
near-zero-mean after normalization + FWHT) and is catastrophic for turbo4 (QJL residual
computed relative to DC-removed signal but decoded without DC restoration).

## Experiment: MSE-Optimal Norm Correction (bonus)

Replace L2-preserving norm correction (β = ||x||/||q||) with MSE-optimal scaling
(α = ||x|| · dot(x,q) / ||q||²). Theoretically halves per-element MSE.

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 L2-preserving (baseline) | 5.8501 ± 0.165 | — |
| turbo3 MSE-optimal | 5.9083 ± 0.167 | +0.058 (worse) |

**Conclusion**: MSE-optimal scaling reduces the norm by cos(θ), effectively lowering
attention temperature (making softmax more uniform). L2-preserving is better for
attention because it maintains the intended dot-product magnitudes.

## Session 2026-03-27 Night — Experiment Run Summary

Working through Ready experiments, then Needs Research. Progress:
- **#39 GSR Walsh ordering**: DONE — NEUTRAL (random signs negate benefit)
- **#40 HadaNorm mean-centering**: DROPPED (duplicate of #22, incompatible with per-token quant)
- **#49/#17 parallel_blocks tuning**: DONE — NO EFFECT (FFN dominates decode, not attention)
- **#45 Gemma-3 SWA V bug**: DONE — **FIXED**. Added V un-rotation to iSWA build_attn overload. Gemma-3 turbo3 K+V PPL 5.8867 (+3.3% vs q8_0 5.6995). Was 45T before fix.
- **#31 Turbo in speculative decoding**: DONE — NO BENEFIT (draft KV is tiny, spec decode slower than normal decode for this pair)
- **#16b turbo4 prefill fix**: NOT YET STARTED
- **#42 KVLinC asymmetric K/V rotation**: DONE — NEGATIVE (rotation helps both K and V)
- **turbo4 head_dim=128 bug**: FOUND ROOT CAUSE — see below

Server state: iSWA fix deployed at `/root/llama-cuda-turbo/` on `root@dorei`. Fully rebuilt.

## BUG FOUND: turbo4 K broken on head_dim=128

turbo4-K produces PPL 33K on Qwen3-14B (head_dim=128). turbo4-V works fine (PPL 6.62).

**Root cause**: In `fattn.cu` line 702, the Q pre-rotation check is:
```c
const bool turbo_kv = K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0;
```
This only checks for TURBO3_0, NOT TURBO4_0! So when using turbo4 K:
1. Q pre-rotation is SKIPPED (turbo_kv is false)
2. Vec kernel gets UNROTATED Q dotted with ROTATED K → garbage dot products → PPL 33K

This same `turbo_kv` variable gates THREE things in the else branch (lines 702-773):
1. Line 713: `do_decode_dequant` — fp16 decode dequant (turbo4 doesn't need this anyway)
2. Line 759: Q pre-rotation — **THIS IS THE BUG**. turbo4 K is stored rotated but Q never gets rotated
3. The fp16 dequant kernels at lines 722/738 only handle TURBO3_0 blocks (correct, turbo4 uses vec kernel)

**Fix**: Change line 759 to check for turbo4 K type too:
```c
const bool turbo_k_any = (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0);
if (turbo_k_any && Q->ne[0] % 128 == 0) {
```
Only need to check K type (not V) since Q pre-rotation is for the Q·K dot product.

**Mystery**: turbo4-K + q8_0-V gave PPL 5.8451 on Qwen3.5-27B (head_dim=256) in experiment #10. Need to verify this wasn't a fluke or different code state. The bug should affect ALL turbo4-K regardless of head_dim.

**Test data** (Qwen3-14B Q5_K_M, head_dim=128, 2K/8chunks):
- q8_0 baseline: PPL 6.5020
- turbo3 K+V: PPL 6.7458 (+3.7%)
- turbo4 K+V: PPL 32643 (BROKEN)
- turbo4 K + q8_0 V: PPL 33890 (BROKEN — confirms K is the issue)
- q8_0 K + turbo4 V: PPL 6.6232 (+1.9% — V works great, better than turbo3!)
- GGML_TURBO_DECODE_FP16=1 turbo4 K: still broken (same PPL 33890 — rules out vec_dot inline bug)
Local branch: `feature/turboquant-kv-cache` with iSWA fix + experiments.md + benchmark-results.md updates.

## Gemma-3-27B-it turbo3 Results (post-#45 iSWA fix)

| Config | PPL (2K/8chunks) | vs q8_0 | Notes |
|--------|------------------|---------|-------|
| q8_0 baseline | 5.6995 ± 0.174 | — | |
| turbo3 K+V | 5.8867 ± 0.170 | +3.3% | **was 45T PPL before fix** |
| turbo3-K + q8_0-V | 5.9633 ± 0.174 | +4.6% | K-only |

**Finding**: With iSWA V un-rotation fix, Gemma-3 turbo3 matches the head_dim=128 degradation pattern (+3-4% PPL). Same as MN-Violet-Lotus (+2.6%) and Qwen3-14B (+3.8%). K-only is slightly worse than K+V, consistent with "values matter more" finding.

## Experiment: KVLinC No-K-Rotation (#42)

| Config | PPL (2K/8chunks) | vs q8_0 |
|--------|------------------|---------|
| turbo3 baseline (both rotated) | 5.8323 | -0.09% |
| turbo3 K unrotated, V rotated | 6.1647 | +5.6% |
| turbo3 neither rotated (prior) | 6.2357 | +6.8% |
| q8_0 | 5.8375 | — |

**Conclusion**: KVLinC's "rotation hurts keys" finding does NOT apply to turbo3 Lloyd-Max codebook. Rotation helps both K and V. K rotation alone contributes ~0.07 PPL, V rotation ~0.33 PPL, together ~0.40 PPL improvement.

Experiment branches created: `experiment/gsr-walsh-ordering`, `experiment/parallel-blocks-tuning`, `experiment/kvlinc-no-k-rotation`.

## Experiment: parallel_blocks Tuning (#49/#17)

Forced different parallel_blocks values via GGML_PARALLEL_BLOCKS env var.
turbo3 tg64 at 32K context:

| parallel_blocks | tok/s |
|-----------------|-------|
| default (auto) | 29.95 ± 0.07 |
| 1 | 29.97 ± 0.07 |
| 2 | 29.95 ± 0.10 |
| 4 | 29.95 ± 0.06 |
| 8 | 29.96 ± 0.05 |
| 16 | 29.96 ± 0.07 |
| 32 | 29.93 ± 0.08 |
| q8_0 baseline | 30.81 ± 0.07 |

**Conclusion**: NO EFFECT. All values within noise. Attention is <5% of decode
time — FFN dominates. 2.8% turbo3→q8_0 gap is structural dequant overhead.

## Experiment: GSR Walsh Ordering (#39)

Reorder FWHT output by sequency (sign-change count) to group similar-frequency
components into turbo3 quantization blocks.

| Config | PPL (2K/8chunks) | vs q8_0 |
|--------|------------------|---------|
| turbo3 baseline (natural Hadamard) | 5.8323 | -0.09% |
| turbo3 + Walsh ordering | 5.8248 | -0.22% |
| q8_0 reference | 5.8375 | — |

**Conclusion**: NEUTRAL (-0.13%, within ±0.164 error bars). Random sign arrays
in PolarQuant already decorrelate all 128 FWHT output elements, making them
identically distributed. Sequency reordering cannot improve intra-block variance
when frequency structure is already destroyed by random signs. GSR paper's gains
(PPL 20.29→11.59) were with non-randomized Hadamard.

## Multi-Model Validation (turbo3 uniform, 2K context)

**BUG FIX**: KV cache context OOM for turbo types — the ggml context allocation didn't
account for turbo rotation matrix tensors (2 extra objects). Fixed by adding `n_turbo_extra`
to the context size calculation in `llama-kv-cache.cpp`.

| Model | Architecture | head_dim | q8_0 PPL | turbo3 PPL | Delta |
|-------|-------------|----------|----------|-----------|-------|
| Qwen3.5-27B Q6_K | qwen3 | 256 | 5.8375 | 5.8501 | +0.2% |
| Qwen3.5-35B-A3B Q4_K_S (MoE) | qwen3 | 256 | 6.4155 | 6.4334 | +0.3% |
| MN-Violet-Lotus-12B Q4_K_M | llama | 128 | 5.6051 | 5.7494 | +2.6% |
| Qwen3-14B Q5_K_M | qwen3 | 128 | 8.3084 | 8.6230 | +3.8% |
| Gemma-3-27B-it Q4_K_M (K only) | gemma3 | 128 | 5.6995 | 7.4946 | +31% |
| Gemma-3-27B-it Q4_K_M (V only) | gemma3 | 128 | 5.6995 | **45T** | **broken** |

**Key findings**:
1. turbo3 works excellently on Qwen3.5 models (head_dim=256): <0.3% PPL increase
2. Quality degrades moderately on head_dim=128 models: +2.6% to +3.8% PPL
3. **Gemma-3 V is completely broken** — turbo3 V produces catastrophic PPL on Gemma-3's
   SWA/global hybrid attention architecture. turbo3 K works (degraded). Root cause:
   likely related to the interleaved sliding-window attention cache architecture.
4. The KV cache context OOM was a latent bug affecting all non-Qwen3.5 models with turbo types

## Experiment #50: turbo4 Q Pre-rotation Fix (APPLIED)

**Fix**: In `fattn.cu`, changed Q pre-rotation guard from `turbo_kv` (only checks TURBO3_0) to
`turbo_k_any = (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0)`.

### Qwen3.5-27B Q6_K (head_dim=256, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 K+V | 5.8375 | — |
| q8_0-K + turbo4-V | 5.8372 | -0.01% |
| turbo4-K + q8_0-V | 5.8451 | +0.13% |
| **turbo4 K+V** | **5.8186** | **-0.32%** |
| turbo3 uniform | 5.8501 | +0.22% |

**turbo4 K+V BEATS q8_0 on head_dim=256!** Also beats turbo3.

### Qwen3-14B Q5_K_M (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 K+V | 6.5020 | — |
| q8_0-K + turbo4-V | 6.6232 | +1.9% |
| turbo3 K+V | 6.7458 | +3.7% |
| turbo4-K + q8_0-V | 6.8322 | +5.1% |
| turbo4 K+V | 6.9118 | +6.3% |

On head_dim=128, turbo4-V (+1.9%) beats turbo3 K+V (+3.7%), but turbo4-K (+5.1%) is the weak link.

### Summary
- Fix resolves turbo4 K from garbage (33K PPL) to functional on both head_dim sizes
- head_dim=256: turbo4 is the BEST quantization option (-0.32% vs q8_0)
- head_dim=128: turbo4-V excellent, turbo4-K degrades more than turbo3
- Experiment #10's turbo4-K result (5.8451) confirmed reproducible after fix

## Sparse V Dequant (credit: TheTom, sparse-v-dequant)

Skip V dequantization for KV positions where `exp(score - max) < 1e-6`.
At long context, 90%+ of attention weights are negligible — eliminating those
dequant operations removes work without quality loss.

### Quality (turbo3, Qwen3.5-27B, 2K/8chunks)
PPL = 5.8501 — **bit-identical** to baseline. Zero quality impact.

### Dense Model: Qwen3.5-27B Q6_K — Decode tg64

| Context | turbo3 (fp16 dq) | turbo3 (native) | turbo4 |
|---------|------------------|-----------------|--------|
| 4K | 30.25 | — | 29.68 |
| 32K | 30.03 | 30.02 | — |

No meaningful improvement on dense model (attention is <5% of decode compute).

### MoE Model: Qwen3.5-35B-A3B Q4_K_S — Decode tg64

**KEY FINDING: sparse V eliminates context scaling regression on native dequant!**

| Context | fp16 dq (sparse V) | native (sparse V) | native (no sparse V, old) |
|---------|--------------------|--------------------|---------------------------|
| p=0 | — | — | 125.77 |
| 4K | 127.53 | 127.13 | — |
| 8K | — | **126.89** | **114.44** (-9% regression!) |
| 32K | 126.70 | **126.21** | (worse) |

Native dequant with sparse V matches fp16 dequant speed at all contexts.
The fp16 dequant path was originally created to fix this regression — sparse V
achieves the same fix with zero extra memory bandwidth (no temp fp16 buffer).

### Experiment #16b: turbo4 Prefill MMA Enabled

Enabled fp16 dequant + MMA for turbo4 prefill. QJL correction loses ~1% PPL in fp16,
but 1.9x prefill speedup is worth it (only prompt tokens affected).

| Metric | turbo4 before | turbo4 after | turbo3 (ref) | q8_0 (ref) |
|--------|---------------|--------------|--------------|------------|
| pp4096 tok/s | 588 | **1113** | 1125 | 1133 |
| PPL (all-prefill) | 5.8186 | 5.8966 | 5.8501 | 5.8375 |
| Decode tg64 | 29.43 | 29.66 | 30.25 | 31.04 |

turbo3 PPL verified unchanged (5.8501).

### CUDA 13.2 Compatibility
Built and tested with CUDA 13.2.0 (nvcc V13.2.51). No segfault, identical results.
Turbo3 PPL: 5.8501 (identical). Turbo4 PPL: 5.8186 (identical).
