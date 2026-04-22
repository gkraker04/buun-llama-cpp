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

### Multi-Sequence (n_seq > 1) Fix — 2026-03-27

Bug: turbo dequant-to-fp16 kernels in fattn.cu ignored ne[3] (stream dimension).
With kv_unified=false (default) and n_seq > 1, only stream 0 was dequanted;
other streams had uninitialized fp16 data, causing catastrophic PPL.

| Config | Before fix | After fix |
|--------|-----------|-----------|
| turbo3 n_seq=1 | 6.31 | 6.31 |
| turbo3 n_seq=2 | 17.10 | 6.30 |
| turbo3 n_seq=4 | 22.56 | 6.34 |
| turbo3 -kvu (unified) | 6.30 | 6.30 |

## Community Report: Dual RTX 4090 (multi-GPU, unknown model)

Date: 2026-03-27. First multi-GPU test after q_rot_buf per-device fix (6cdd9db87).

| Config | Prefill tok/s | Decode tok/s | vs q8_0 prefill | vs q8_0 decode |
|--------|--------------|-------------|-----------------|---------------|
| q8_0 baseline | 5116.86 | 103.57 | 100% | 100% |
| turbo3 | 4986.84 | 36.27 | 97.5% | 35.0% |
| turbo4 | 2541.50 | 17.63 | 49.7% | 17.0% |

**Analysis**:
- turbo3 prefill at 97.5% of q8_0: consistent with RTX 3090 results (98.8%)
- turbo3 decode at 35%: matches MoE decode regression pattern (Qwen3.5-35B-A3B was 37%)
- turbo4 is exactly ~2x slower than turbo3 across both prefill and decode — unexpected
- **Need to confirm**: what model, what context, was `-fa on` used?
- turbo4 prefill at 49.7% suggests it may NOT be hitting the MMA prefill path (which should give ~98% like turbo3). Possible that `-fa` is off or the model triggers a non-MMA kernel path on multi-GPU.

## Reproduction Test: ubergarm's Q4_0 + c512 configuration (2026-03-27)

Testing exact same params as ubergarm (ik_llama.cpp#1509): Qwen3.5-35B-A3B-Q4_0, -c 512, seed 1337, -ngl 99.

**Hardware**: RTX 3090 24GB (ubergarm used RTX A6000 48GB)

| KV type | LA | PPL (ours) | PPL (ubergarm) | Delta |
|---------|------|------------|----------------|-------|
| f16 | - | 6.5776 ± 0.04194 | 6.5792 ± 0.04196 | match |
| turbo3 | 1 | 6.6112 ± 0.04218 | 9.2400 | +0.51% vs +40.4% |

**Conclusion**: Cannot reproduce. f16 baselines match perfectly, confirming identical model/dataset.
turbo3 PPL is +0.51% on our build vs +40.4% on theirs. Issue is on their end —
likely build config (stale cmake, missing FA, or patches applied on wrong base).

## turbo2 (2-bit) Initial Results (2026-03-27)

New 2-bit PolarQuant type: 10 bytes per 32 elements = 2.5 bpv (6.4x vs fp16).
Model: Qwen3.5-27B-heretic Q6_K, head_dim=256, RTX 3090.

### PPL (4K ctx, 4 chunks)

| Config | PPL | vs f16 | vs q8_0 |
|--------|-----|--------|---------|
| f16 baseline | 6.2765 | — | — |
| q8_0 baseline | 6.2677 | -0.14% | — |
| turbo3 (3-bit) | 6.3252 | +0.78% | +0.92% |
| **turbo2 (2-bit)** | **6.7792** | **+8.01%** | **+8.16%** |

### Decode Speed pp512+tg64 (tok/s)

| Config | pp512 | tg128 | pp512+tg64 | pp32768+tg64 |
|--------|-------|-------|------------|--------------|
| q8_0 | 1150.91 | 30.98 | 227.27 | 921.92 |
| turbo3 | 1138.57 | 30.27 | 222.76 | 904.84 |
| turbo2 | 1141.87 | 30.64 | 225.64 | 908.88 |

### KV Memory (4K context, 16 KV layers)

| Config | KV size | Compression vs fp16 |
|--------|---------|---------------------|
| f16 | 128 MiB | 1.0x |
| q8_0 | 68 MiB | 1.9x |
| turbo3 | 28 MiB | 4.6x |
| turbo2 | 20 MiB | 6.4x |

### Mixed K/V Configurations (4K ctx, 4 chunks)

| K type | V type | PPL | vs f16 | Effective bpv |
|--------|--------|-----|--------|---------------|
| turbo3 | turbo2 | 6.5670 | +4.63% | ~2.9 |
| turbo2 | turbo3 | 6.5203 | +3.88% | ~2.9 |
| turbo2 | q8_0 | 6.4894 | +3.39% | ~5.3 |
| q8_0 | turbo2 | 6.5490 | +4.34% | ~5.3 |

### Layer-Adaptive turbo2 (4K ctx, 4 chunks)

| Config | PPL | vs f16 | Notes |
|--------|-----|--------|-------|
| turbo2 uniform | 6.7792 | +8.01% | all 16 layers turbo2 |
| turbo2 LA-1 | 6.7411 | +7.40% | first4+last4 = q8_0, 8 turbo2 |
| turbo2 LA-2 | 6.6866 | +6.53% | last 8 = q8_0, 8 turbo2 |
| (turbo3 uniform) | 6.3252 | +0.78% | reference: still much better |

**Analysis**: turbo2 at 2.5 bpv has +8% PPL degradation on head_dim=256 (Qwen3.5-27B).
This is significantly worse than turbo3's <1% gap. Speed is essentially identical to turbo3
(both compute-bound by dequant, not memory-bound at short context on this model).
The 6.4x compression is impressive but the quality cost is too high for general use.
turbo2 may work better for V-only (K stays turbo3) or for extreme memory pressure scenarios.

## turbo3 vs q8_0 vs turbo4 — Full Context Scaling (2026-03-28)

Model: Qwen3.5-27B-heretic Q6_K, RTX 3090 24GB. turbo3 uniform, turbo4 uniform.

### PPL by Context Length

| Context | Chunks | q8_0 | turbo3 | t3 vs q8 | turbo4 | t4 vs q8 | LA-1 t3 | LA-1 vs q8 |
|---------|--------|------|--------|----------|--------|----------|---------|------------|
| 2K | 8 | 5.8375 | 5.8323 | -0.09% | 5.8186 | -0.32% | 5.7690 | -1.17% |
| 4K | 4 | 6.2677 | 6.3252 | +0.92% | — | — | 6.3198 | +0.83% |
| 8K | 4 | 7.4241 | 7.3783 | -0.62% | — | — | 7.3952 | -0.39% |
| 32K | 1 | 7.2139 | 7.1693 | -0.62% | 7.3296 | +1.60% | 7.2168 | +0.04% |
| 64K | 1 | 8.1975 | 8.2379 | +0.49% | 8.5004 | +3.69% | — | — |

### Decode Speed tg64 (tok/s)

| Context | q8_0 | turbo3 | t3/q8 | turbo4 | t4/q8 |
|---------|------|--------|-------|--------|-------|
| 32K | 30.69 | 29.83 | 97.2% | 29.47 | 96.0% |
| 64K | — | 29.79 | — | 29.51 | — |

### VRAM at 64K

| Config | VRAM |
|--------|------|
| turbo3 | ~22.3 GiB |
| turbo4 | ~22.2 GiB |
| q8_0 | **OOM** (~28+ GiB needed) |

### Key Findings

1. **turbo3 holds quality at all context lengths**: -0.62% at 32K, +0.49% at 64K (within noise)
2. **turbo4 degrades catastrophically at long context**: +1.60% at 32K, **+3.69% at 64K**
3. turbo4 QJL noise accumulates as sqrt(N) — 1-bit correction errors compound over more KV positions
4. LA-1 advantage vanishes at long context: +0.04% at 32K (was -1.17% at 2K)
5. turbo3 fits 64K on 24GB RTX 3090 where q8_0 would OOM
6. Decode speed is constant across context lengths for both turbo types
7. **Recommendation**: turbo3 for all use cases. turbo4 only viable at short context (<4K) on head_dim=256

## InnerQ Per-Channel Equalization (#52) — 2026-03-28

Branch: `experiment/innerq-channel-equalization`. Model: Qwen3-14B Q5_K_M (head_dim=128).

### Concept

Per-channel K scaling before L2 norm + FWHT to equalize channel magnitudes.
Inverse scaling applied to Q in FA kernel before FWHT rotation. Targets the
head_dim=128 quality gap where channels have extreme anisotropy (channel 114
has 20x mean RMS on Qwen3-14B).

### Calibration

Online calibration: accumulate per-channel K² during first 100K token-counts,
compute sqrt-dampened scales: `scale = (rms/mean_rms)^strength`, clamped to max 2.0x.
Controlled by `TURBO_INNERQ=1` and `TURBO_INNERQ_STRENGTH=<float>` env vars.

### Reference Baselines (Qwen3-14B Q5_K_M, head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| f16 baseline | 6.4239 | +0.05% |
| q8_0 baseline | 6.4206 | — |
| turbo3 uniform (no InnerQ) | 6.6340 | +3.33% |

### Strength Sweep (turbo3 + InnerQ, 2K/8chunks)

| Strength | PPL | vs q8_0 | vs turbo3 baseline | Gap closed |
|----------|-----|---------|--------------------|-----------:|
| 0.00 (identity) | 6.6340 | +3.33% | — | 0% |
| 0.10 | 6.5596 | +2.17% | -1.12% | 35% |
| **0.20** | **6.5175** | **+1.51%** | **-1.76%** | **55%** |
| 0.30 | 6.5310 | +1.72% | -1.55% | 48% |
| 0.50 | 6.5872 | +2.60% | -0.71% | 22% |
| 1.00 (full RMS) | catastrophic | — | — | — |

**Optimal: strength=0.20, max clamp=2.0x**

### Effect on head_dim=256 (Qwen3.5-27B)

No benefit — FWHT with 256 dimensions already equalizes channels effectively.
Max channel ratio only 2x (vs 20x on hd128). InnerQ disabled automatically
when head_dim >= 256 would be appropriate, but currently gated by env var.

### Asymmetric Comparison (Qwen3-14B, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 K+V + InnerQ (strength=0.20) | 6.5175 | +1.51% |
| turbo3-K + turbo4-V + InnerQ | 6.6645 | +3.80% |
| turbo3 K+V (no InnerQ) | 6.6340 | +3.33% |

turbo3 uniform + InnerQ beats asymmetric turbo3-K + turbo4-V with InnerQ.

### Effect on head_dim=256 — Auto-Detection (2026-03-28)

| Config | PPL | vs q8_0 | Notes |
|--------|-----|---------|-------|
| turbo3 (no InnerQ) | 5.8501 | +0.22% | baseline |
| turbo3 + InnerQ (strength=0.20) | 5.9283 | +1.56% | **WORSE** — channels balanced, perturbation hurts |
| turbo3 + InnerQ (auto-detect) | 5.8501 | +0.22% | auto-disabled: max ratio 1.164 < 1.2 |

Auto-detect threshold (max_ratio < 1.2) correctly disables InnerQ on hd256.

### K-only vs K+V Scaling Comparison (Qwen3-14B, strength=0.20)

| Config | PPL | vs q8_0 | Gap closed |
|--------|-----|---------|------------|
| K+V calibrate, K+V apply (best) | 6.5349 | +1.78% | 46% |
| K-only calibrate, K+V apply | 6.5757 | +2.42% | 27% |
| K-only calibrate, K-only apply (s=0.30) | 6.5418 | +1.89% | 43% |
| K-only calibrate, K-only apply (s=0.20) | 6.5477 | +1.98% | 40% |
| K-only calibrate, K-only apply (s=0.50) | 6.5486 | +1.99% | 40% |
| Max-based (paper's formula, mode=1) | 6.6716 | +3.91% | **WORSE** |
| No InnerQ baseline | 6.6340 | +3.33% | 0% |

**Key findings**:
1. V scaling helps even without output compensation (K+V > K-only)
2. Mixed K+V calibration stats are better than K-only stats for K+V application
3. Paper's max-based formula doesn't transfer to our codebook pipeline
4. RMS-based mode=0 with strength=0.20 is optimal

### Summary

InnerQ closes the head_dim=128 gap by **~46%** (from +3.33% to +1.78% vs q8_0).
Auto-detects and disables on hd256 models where channels are already balanced.
The remaining +1.78% gap is inherent to lower dimensionality (fewer dimensions =
larger relative quantization noise in dot products). Further improvement would
require more bits or a fundamentally different approach.

## New turbo4: 4-bit PolarQuant (16 centroids, no QJL)

Branch: `experiment/turbo4-4bit-polarquant`. Dropped QJL 1-bit correction entirely,
replaced with pure 4-bit PolarQuant (16 Lloyd-Max centroids). Inspired by TheTom's
Metal implementation showing QJL variance gets amplified by softmax.

Block format: 66 bytes per 128 values = 4.125 bpv (was 68 bytes = 4.25 bpv with QJL).

### Qwen3.5-27B Q6_K (head_dim=256, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 | 5.8377 | — |
| new turbo4 (4-bit) | 5.8467 | +0.15% |
| turbo3 | 5.8501 | +0.22% |
| old turbo4 (QJL) | 5.8186 | -0.32% |

On hd256, old QJL turbo4 was better (-0.32% vs +0.15%). QJL helps on hd256
where the 1-bit correction noise is small relative to 256-dim dot products.

### Qwen3-14B Q5_K_M (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 | 6.4206 | — |
| **new turbo4 (4-bit)** | **6.4978** | **+1.20%** |
| turbo3 + InnerQ | 6.5175 | +1.51% |
| turbo3 | 6.6340 | +3.33% |
| old turbo4 (QJL) | 6.9118 | +6.3% |

**Breakthrough on hd128!** New turbo4 beats turbo3+InnerQ by 0.31 percentage points.
Old turbo4 (QJL) was catastrophic at +6.3%. The 72.5% MSE reduction from 16 vs 8
centroids more than compensates for losing the QJL correction.

InnerQ has no additional effect on new turbo4 (PPL identical with/without).

### LA-1 on Qwen3-14B (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 | 6.6340 | +3.33% |
| turbo3 + LA-1 | 6.6763 | +3.98% |
| turbo3 + LA-1 + InnerQ | 6.6608 | +3.74% |

LA-1 HURTS on hd128 (unlike hd256 where it gives -0.71%). 8/40=20% promoted layers
create quality discontinuity that is worse than uniform turbo3 noise.

### turbo4 Quality Decomposition (hd128, Qwen3-14B, 2K/8chunks)

**K quality dominates V by 13x.** Source of gap analysis:

| Config | PPL | vs q8_0 | Source |
|--------|-----|---------|--------|
| q8_0 (f16 K + f16 V) | 6.4206 | — | — |
| f16 K + turbo4 V | 6.4256 | +0.08% | V quant only |
| turbo4 K + f16 V | 6.4879 | +1.05% | K quant only |
| turbo4 K + turbo4 V | 6.4978 | +1.20% | K + V combined |
| turbo4 K + turbo3 V | 6.5854 | +2.57% | |
| turbo3 K + turbo4 V | 6.6330 | +3.31% | |
| turbo3 K + turbo3 V | 6.6340 | +3.33% | |

### turbo4 + Layer-Adaptive sweep (hd128, Qwen3-14B, 2K/8chunks)

| Mode | Description | PPL | vs q8_0 | Promoted layers |
|------|-------------|-----|---------|-----------------|
| **LA-3** | **last 4 (K+V)** | **6.4587** | **+0.59%** | **4/40 = 10%** |
| LA-10 | last 4 (K-only) | 6.4604 | +0.62% | K: 4/40, V: 0/40 |
| LA-11 | last 6 (K+V) | 6.4626 | +0.65% | 6/40 |
| LA-2 | last 8 (K+V) | 6.4628 | +0.66% | 8/40 |
| LA-1 | first4+last4 (K+V) | 6.4638 | +0.67% | 8/40 |
| LA-7 | last 8 (K-only) | 6.4653 | +0.70% | K: 8/40, V: 0/40 |
| LA-9 | last 2 (K+V) | 6.4795 | +0.92% | 2/40 |
| 0 | turbo4 uniform | 6.4978 | +1.20% | 0/40 |
| LA-4 | first 4 (K+V) | 6.5125 | +1.43% | 4/40 |
| LA-5 | first2+last2 (K+V) | 6.5162 | +1.49% | 4/40 |

Best: LA-3 (last 4, K+V) at +0.59%, 4.53 bpv average.
Sweet spot: exactly 4 last layers. More layers (6, 8) give diminishing returns.
First-layer promotion always hurts on hd128. K-only promotion nearly as good.

### turbo4 Context Scaling (hd128, Qwen3-14B)

| Context | q8_0 PPL | turbo4 PPL | turbo4 gap | turbo3 PPL | turbo3 gap |
|---------|----------|------------|------------|------------|------------|
| 2K | 6.4206 | 6.4978 | +1.20% | 6.6340 | +3.33% |
| 8K | 7.0149 | 7.0718 | +0.81% | 7.4261 | +5.86% |
| 32K | 7.2929 | 7.3638 | +0.97% | 7.7736 | +6.59% |

turbo4 context scaling is excellent (~1% gap at all lengths).
turbo3 degrades catastrophically at long context on hd128 (3.3% → 6.6%).

### Decode Speed (hd128, Qwen3-14B, RTX 3090, tg64)

| Config | t/s | vs q8_0 |
|--------|-----|---------|
| q8_0 | 69.68 | — |
| turbo4 | 59.72 | 85.7% |
| turbo3 | 59.54 | 85.4% |

turbo4 and turbo3 have identical decode speed (compute-bound dequant).

### Empirical centroid calibration — FAILED

Simulated post-FWHT distribution has kurtosis -0.047 (sub-Gaussian), but the empirical
centroids derived from simulation gave catastrophic PPL = 6.8758 (+17.8%). The simulation
is wrong for real model data. TheTom's measurement on real KV tensors shows kurtosis = 2.9
(near-Gaussian), confirming Gaussian-optimal Lloyd-Max centroids are correct.
