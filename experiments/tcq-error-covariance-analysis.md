# TCQ Error Covariance Analysis (2026-03-30 14:41)

## Hypothesis Tested

**Theory**: The attention dot product error is `q · e`, and for correlated errors:
```
Var_j(q · e_j) = q^T · Σ_e · q
```
Averaged over model queries: `E[Var] = tr(Σ_q · Σ_e)`. If Σ_q ≠ I (Q is anisotropic), then the eigenvector alignment between Σ_e and Σ_q determines attention quality — not MSE alone.

**Predictions**:
1. Σ_q is NOT proportional to identity
2. Old codebook Σ_e is more diagonal than trained codebooks
3. tr(Σ_q · Σ_e) correlates with PPL better than MSE alone

## Method

- Extracted Q vectors from Qwen3.5-27B inference (TURBO_DUMP_Q=1, 512 tokens, 4 heads/layer, 40 layers)
- Applied FWHT in Python to get rotated Q (same domain as quantized K)
- Used existing post-FWHT K data (/tmp/turbo_postrot.bin, 10K vectors)
- Ran Viterbi with each codebook to get error vectors
- Computed 128×128 covariance matrices Σ_e and Σ_q
- Computed tr(Σ_q · Σ_e) and eigenvector alignment

## Results

### Prediction #1: Q anisotropy — MASSIVELY CONFIRMED (14:02)

Aggregate Σ_q (all layers, 4 heads, 163K samples):

| Metric | Value |
|--------|-------|
| Eigenvalue max/min | 937:1 |
| Effective rank | 4.9 / 128 |
| Top eigenvalue | 322.44 |
| 2nd eigenvalue | 5.55 |
| Bottom eigenvalue | 0.34 |

**Q lives in a ~5-dimensional subspace of the 128-dim rotated space.** The first principal component captures ~96% of variance.

Per-layer Σ_q varies wildly:

| Layer | Head 0 EigMax/Min | Effective Rank |
|-------|-------------------|----------------|
| 0 | 417 | 26.7 |
| 10 | 579 | 17.7 |
| 19 | 9,932 | 1.9 |
| 20 | 9.5 | 114.2 |
| 30 | 1,128 | 12.6 |
| 39 | 152 | 59.2 |

Layer 19: Q is nearly 1-dimensional (eff rank 1.5-1.9). Layer 20: Q is nearly isotropic (eff rank 114).

### Prediction #2: Old Σ_e is more diagonal — CONFIRMED BUT MONOTONIC (13:45)

| Codebook | Off/Diag Frob | lag-2 AC | EigMax/Min | Eff Rank | MSE | PPL |
|----------|---------------|----------|------------|----------|-----|-----|
| Coset 0-iter | 0.301 | -0.157 | 24.93 | 120.2 | 0.000349 | 5.9194 |
| CUDA 3-iter | 0.198 | -0.098 | 15.22 | 123.9 | 0.000267 | 5.8450 |
| CUDA 10-iter | 0.130 | -0.040 | 10.78 | 125.6 | 0.000211 | 5.9386 |
| CUDA 30-iter | 0.118 | -0.021 | 7.85 | 126.0 | 0.000184 | 5.8733 |
| Old numpy 100-iter | 0.114 | -0.008 | 6.84 | 126.1 | 0.000174 | 5.8236 |
| Scalar LM (no trellis) | 0.174 | -0.001 | 3.20 | 124.6 | 0.000345 | — |

**Σ_e diagonal-ness improves MONOTONICALLY with GLA iterations.** More training → less off-diagonal structure. The coset init (0 iter) has the WORST diagonal ratio (0.301), not the best.

Key finding: **strong negative lag-2 autocorrelation** in coset-initialized codebooks (-0.157). The trellis has 6 bits of state memory = 2 steps of K=3 bits. GLA training gradually eliminates this correlation.

**But PPL is NOT monotonic** — 10-iter has better Σ_e metrics than 3-iter on every measure, yet worse PPL. Off-diagonal structure alone does not explain PPL oscillation.

### Prediction #3: tr(Σ_q · Σ_e) correlates with PPL — REFUTED (14:02)

| Codebook | MSE | tr(Σ_q·Σ_e) | Ratio | PPL |
|----------|-----|-------------|-------|-----|
| Coset 0-iter | 0.000349 | 0.142 | 0.976 | 5.9194 |
| CUDA 3-iter | 0.000267 | 0.110 | 0.989 | 5.8450 |
| CUDA 10-iter | 0.000211 | 0.086 | 0.981 | 5.9386 |
| CUDA 30-iter | 0.000184 | 0.075 | 0.983 | 5.8733 |
| Old numpy 100-iter | 0.000174 | 0.073 | 1.005 | 5.8236 |

Correlations with PPL:
- MSE: Pearson +0.41, Spearman +0.50
- tr(Σ_q·Σ_e): Pearson +0.39, Spearman +0.50
- **tr(Σ_q·Σ_e) does NOT predict PPL better than MSE.**

Eigenvector alignment analysis — error variance along Q's sensitive vs insensitive directions:

| Codebook | top5 eigvec err | avg err | top/avg |
|----------|-----------------|---------|---------|
| Coset | 0.000345 | 0.000349 | 0.988 |
| 3-iter | 0.000267 | 0.000267 | 1.002 |
| 10-iter | 0.000211 | 0.000210 | 1.002 |
| 30-iter | 0.000183 | 0.000184 | 0.997 |
| Old numpy | 0.000175 | 0.000174 | 1.007 |

**All ratios within 1% of 1.0.** TCQ errors are isotropic in Q's eigenvector basis. The massive Q anisotropy (937:1) is invisible to the error structure.

## Key Insight: Untapped Potential

Q has effective rank ~5. This means 123/128 error dimensions don't affect attention. Current TCQ distributes errors isotropically — only ~4% of error energy matters.

**If errors could be concentrated in Q's 123 insensitive directions**, attention error would drop ~25x at the same MSE. This is the theoretical ceiling for Q-weighted TCQ training.

## What DOESN'T Explain PPL Oscillation

- Off-diagonal Σ_e structure (improves monotonically, PPL doesn't)
- Eigenvector alignment between Σ_e and Σ_q (all codebooks are equally isotropic)
- tr(Σ_q · Σ_e) (proportional to MSE for all codebooks)
- Lag-k autocorrelation (decreases monotonically)
- Eigenvalue spread of Σ_e (decreases monotonically)

## Remaining Hypotheses (to test)

1. **Quantization bias E[e] ≠ 0** — np.cov strips out the mean. Systematic bias in errors could shift attention weights directionally. Different codebooks may have different bias profiles.
2. **Higher-order moments / tail behavior** — softmax is exp(), exponentially sensitive to outlier errors. Error kurtosis/tails may differ between codebooks in ways covariance misses.
3. **Per-layer K data** — our Σ_e uses aggregate K from all layers. Layer-specific K distributions could interact differently with layer-specific Q structure.
4. **Measurement noise** — PPL differences of 0.01-0.15 on 8 chunks of 2048 tokens may be within statistical uncertainty.

## Files

- `/tmp/q_vectors_raw.bin` — raw Q vectors (512 tokens, 4 heads, 40 layers, 80MB)
- `/tmp/turbo_postrot.bin` — post-FWHT K data (1M vectors, 512MB)
- `/tmp/old_codebook_3bit.bin` — old numpy codebook (512 floats)
- `scripts/analyze_error_covariance.py` — Σ_e analysis script
- `scripts/analyze_q_covariance.py` — combined Σ_q + Σ_e + tr(Σ_q·Σ_e) analysis
- Q extraction: `TURBO_DUMP_Q=1` env var in fattn.cu (currently on server build)
- `scripts/analyze_bias_and_tails.py` — bias, kurtosis, tails, attention simulation

---

## Deeper Analysis: Bias, Tails, and Attention Simulation (2026-03-30 15:15)

### Quantization bias E[e]

| Codebook | Bias% of RMS err | Mean kurtosis | Max kurtosis | Mag corr | PPL |
|----------|-------------------|---------------|--------------|----------|-----|
| Coset 0-iter | 5.08% | +9.97 | +46.47 | +0.537 | 5.9194 |
| CUDA 3-iter | 3.48% | +5.18 | +45.75 | +0.394 | 5.8450 |
| CUDA 5-iter | 2.71% | +2.49 | +21.72 | +0.284 | 5.8576 |
| CUDA 10-iter | 2.18% | +1.09 | +13.49 | +0.217 | 5.9386 |
| CUDA 20-iter | 2.09% | +0.74 | +8.82 | +0.193 | 5.9712 |
| CUDA 30-iter | 1.96% | +0.56 | +6.62 | +0.176 | 5.8733 |
| Old numpy 100 | 2.09% | +0.36 | +3.77 | +0.156 | 5.8236 |

Key observations:
- **All metrics improve monotonically** with GLA iterations: bias decreases, kurtosis decreases, mag correlation decreases
- **Coset init has extreme kurtosis** — max channel kurtosis +46 (vs +3.8 for old numpy). The trellis creates channels with very heavy error tails
- **Error-input magnitude correlation** is 0.537 for coset (large K → large error) vs 0.156 for old numpy. GLA training reduces this
- **Position-dependent MSE**: first 16 positions have 7-14% higher MSE than last 16 (trellis warmup)

### Simulated attention error (with real Q vectors)

| Codebook | Attn L1 | Attn KL | Dot err var | PPL |
|----------|---------|---------|-------------|-----|
| Coset 0-iter | 0.0309 | 0.00132 | 0.00270 | 5.9194 |
| CUDA 3-iter | 0.0283 | 0.00110 | 0.00220 | 5.8450 |
| CUDA 5-iter | 0.0267 | 0.00098 | 0.00195 | 5.8576 |
| CUDA 10-iter | 0.0252 | 0.00088 | 0.00176 | 5.9386 |
| CUDA 20-iter | 0.0240 | 0.00079 | 0.00155 | 5.9712 |
| CUDA 30-iter | 0.0237 | 0.00077 | 0.00154 | 5.8733 |
| Old numpy 100 | 0.0233 | 0.00074 | 0.00148 | 5.8236 |

**Attention L1, KL, and dot product variance all improve monotonically.** Even with real Q vectors, the simulated attention error says 10-iter should be better than 3-iter. But PPL disagrees.

### Correlation with PPL — NO metric explains the oscillation

| Metric | Pearson | Spearman |
|--------|---------|----------|
| MSE | +0.10 | +0.14 |
| RMS bias | +0.07 | +0.04 |
| Bias fraction | +0.03 | -0.18 |
| Mean kurtosis | +0.05 | +0.14 |
| Mag correlation | +0.05 | +0.14 |
| P99.9 error | +0.03 | +0.14 |
| Attention L1 | +0.03 | +0.14 |
| Attention KL | +0.05 | +0.14 |
| Dot error variance | +0.05 | +0.14 |
| **Dot error mean** | **+0.75** | **+0.79** |

The only metric with meaningful correlation is **dot error mean** — but the absolute values are ~0.0002, deep in the noise floor for 200×500 = 100K dot products (SE ≈ 0.004). Almost certainly spurious.

### Conclusion: The oscillation is NOT in the aggregate statistics

Every aggregate error metric we tested improves monotonically with GLA iterations:
- Second-order: covariance diagonal-ness, eigenvalue spread, lag correlations
- Bias: mean error, per-channel bias
- Higher-order: kurtosis, skewness, tail percentiles
- Input-dependent: magnitude correlation, position-dependent MSE
- Attention-specific: simulated L1, KL, weighted error

PPL oscillates while ALL of these are monotonic. Pearson correlations with PPL are 0.03-0.10 for everything except the spurious dot error mean.

### Remaining hypotheses

The PPL oscillation must come from something NOT captured by aggregate statistics on random/mixed-layer data:

1. **Test-text sensitivity** — the oscillation may be specific to wikitext-2. Testing on a different dataset (or more chunks) could show a different pattern. The PPL differences (0.01-0.13) may be within the effective uncertainty for this specific test text.

2. **Per-layer codebook interaction** — our K data mixes all 40 layers. If layer 19 (eff rank 1.9, nearly 1-D Q) is the sensitive one, its signal is diluted 40x in the aggregate. Per-layer K extraction + per-layer analysis is needed.

3. **Token-level effects** — attention sinks, BOS tokens, and rare tokens may have K values that interact specifically with certain codebook configurations. Not visible in aggregate statistics.

### Actionable finding: Q anisotropy → weighted TCQ training

Regardless of the oscillation mystery, the Q anisotropy finding (eff rank 5/128) has clear implications:

- 123/128 error dimensions don't affect attention → ~96% of quantization error is wasted
- Current codebooks distribute errors isotropically (top/avg = 1.0 ± 1%)
- Theoretical ceiling: 25x attention error reduction at same MSE via Q-weighted training
- This is the TCQ analogue of GPTQ's Hessian-weighted quantization
- Implementation: use per-channel weights w_i ∝ Σ_q eigenvalues in the GLA objective

---

## PPL Robustness Test: The "Oscillation" Was Noise (2026-03-30 15:45)

Tested 5 codebooks across 3 datasets with many more chunks to tighten confidence intervals.

### Full results

| Codebook | test 64ch | valid 64ch | train 32ch | 8ch (old) |
|----------|-----------|-----------|-----------|-----------|
| Old numpy 100-iter | 6.507 ±0.065 | 6.909 ±0.071 | 6.956 ±0.099 | 5.824 |
| CUDA 3-iter | 6.502 ±0.065 | 6.910 ±0.071 | 6.959 ±0.098 | 5.845 |
| CUDA 10-iter | 6.595 ±0.066 | 7.026 ±0.073 | 7.063 ±0.101 | 5.939 |
| CUDA 20-iter | 6.568 ±0.066 | 7.008 ±0.072 | 7.050 ±0.101 | 5.971 |
| CUDA 30-iter | 6.560 ±0.066 | 7.022 ±0.073 | 7.057 ±0.101 | 5.873 |

Delta from old numpy (positive = worse):

| Codebook | Δ test | Δ valid | Δ train | Δ 8ch (old) |
|----------|--------|---------|---------|-------------|
| 3-iter | **-0.005** | **+0.001** | **+0.003** | +0.021 |
| 10-iter | +0.088 | +0.117 | +0.107 | +0.115 |
| 20-iter | +0.061 | +0.099 | +0.094 | +0.148 |
| 30-iter | +0.054 | +0.113 | +0.101 | +0.050 |

### Key findings

**1. The "lucky codebook" was NOT lucky.** CUDA 3-iter (seed 42, 100K samples) matches old numpy (unknown seed, 4K samples) within ±0.005 across all 3 datasets. The old codebook is not a special outlier — it's just a normal codebook at an iteration count that works.

**2. The 10-20 iter crash is REAL.** +0.09-0.12 PPL degradation persists across test, valid, and train sets. Not text-specific.

**3. The "chaotic oscillation" was NOISE from small sample size.** At 8 chunks:
- 30-iter appeared to "recover" to PPL 5.873 (Δ=+0.050)
- At 64 chunks on valid/train, 30-iter is Δ=+0.10-0.11 — NO recovery, same as 10-iter

The 8-chunk test oversampled easy text at the start of wikitext-2, amplifying random variation between codebooks.

**4. The real pattern is simple and roughly monotonic:**
- 3 iterations: sufficient. Matches best known codebook.
- 10+ iterations: genuine degradation of +0.06-0.12 PPL
- No recovery at higher iterations — 30-iter is as bad as 10-iter on valid/train

### Revised narrative

The MSE-PPL divergence is REAL but SIMPLER than we thought:
- There is no "chaotic oscillation" — that was measurement noise
- There is no "lucky seed" — any 3-iter codebook works
- The mechanism: 3 GLA iterations provides coding gain without moving too far from coset init. 10+ iterations over-optimizes for MSE in ways that hurt attention quality.
- The specific mechanism (WHY 10+ iters hurts despite better MSE, better Σ_e structure, lower kurtosis, etc.) remains unknown — no aggregate error statistic correlates with the degradation.

### Implications

- **Use 3 GLA iterations for TCQ codebook training.** No seed search needed.
- **The old numpy codebook can be replaced** by any fresh 3-iter CUDA training (much faster).
- **The MSE-PPL divergence paper story changes**: not chaotic oscillation, but a clean "early stopping" story — 3 iterations is the sweet spot where trellis coding gain is captured without over-fitting the codebook to the MSE objective.

---

## Signal-Dependent Quantization Noise Test — REFUTED (2026-03-30 ~16:30)

### Theory

As GLA iterations increase, the TCQ quantizer evolves from memoryless (coset structure, all states in a coset produce the same output) to predictive (state-dependent reconstruction). This makes quantization errors correlated with the input signal.

In the transformer, Q and K come from the same residual stream (Q = W_Q·x, K = W_K·x). If K error is signal-dependent (corr(k, e) > 0), then through Q-K correlation, the attention logit error q·e becomes correlated with the logit value q·k. This is equivalent to a temperature perturbation in softmax — systematically changing attention sharpness — which compounds across 40 layers and degrades PPL.

Key prediction: corr(k, e) and corr(q·k, q·e) should INCREASE with GLA iterations.

### Results — OPPOSITE of prediction

| Codebook | MSE | corr(k,e) | corr(q·k, q·e) | ε (temp) | ΔH (entropy) |
|----------|-----|-----------|-----------------|----------|--------------|
| Scalar LM (no trellis) | .000345 | +0.358 | +0.356 | +0.068 | -0.0033 |
| Coset 0-iter | .000350 | +0.373 | +0.372 | +0.071 | -0.0034 |
| CUDA 3-iter | .000267 | +0.252 | +0.250 | +0.042 | -0.0021 |
| CUDA 5-iter | .000234 | +0.197 | +0.194 | +0.030 | -0.0016 |
| CUDA 10-iter | .000211 | +0.165 | +0.158 | +0.022 | -0.0013 |
| CUDA 20-iter | .000191 | +0.154 | +0.147 | +0.020 | -0.0012 |
| CUDA 30-iter | .000184 | +0.147 | +0.143 | +0.020 | -0.0012 |
| Old numpy | .000174 | +0.142 | +0.137 | +0.019 | -0.0011 |

Cross-position trellis memory effect (corr(k[i-1], e[i])):

| Codebook | lag-1 | lag-2 |
|----------|-------|-------|
| Scalar LM | -0.004 | -0.000 |
| Coset 0-iter | -0.010 | -0.068 |
| CUDA 3-iter | -0.007 | -0.052 |
| CUDA 10-iter | -0.006 | -0.029 |
| CUDA 30-iter | -0.005 | -0.019 |
| Old numpy | -0.003 | -0.011 |

PPL correlation: all metrics Pearson < 0.2 with PPL.

### Analysis

1. **Signal-error correlation DECREASES monotonically** with training — opposite of prediction. From +0.252 (3-iter) to +0.147 (30-iter). More GLA iterations make errors LESS signal-dependent.

2. **This is purely a MSE effect.** Coarser quantization = larger error relative to signal = higher correlation mechanically. The trellis memory adds negligible extra correlation beyond what MSE explains.

3. **Temperature perturbation DECREASES** — from ε=+0.042 (3-iter) to ε=+0.020 (30-iter). The attention is being LESS perturbed with more training.

4. **Entropy change DECREASES** — all codebooks cause slight attention sharpening (ΔH < 0), but this weakens with training. No systematic broadening/sharpening difference between good and bad codebooks.

5. **Coset ≈ scalar** (0.373 vs 0.358): the memoryless trellis adds minimal signal-error correlation beyond what scalar quantization creates. Trellis memory effects (lag-2 cross-correlation) shrink with training, not grow.

### What this means

The signal-dependent noise theory fails because it predicted the wrong direction. The quantizer becomes MORE memoryless-like (less signal-dependent) with training, not less. The improvement in MSE mechanically reduces all correlation metrics.

### Cumulative refutation list (as of this test)

Every single-layer error property tested improves monotonically with GLA iterations:

| Theory | Metric | Direction with training |
|--------|--------|------------------------|
| Error covariance structure | Off-diagonal ratio | Improves (decreases) |
| Q-weighted error | tr(Σ_q · Σ_e) | Proportional to MSE |
| Eigenvector alignment | top/avg ratio | Always ≈ 1.0 (isotropic) |
| Bias | E[e], RMS bias | Improves (decreases) |
| Higher-order moments | Kurtosis, skewness | Improves (decreases) |
| Tail behavior | p99, p99.9 error | Improves (decreases) |
| Magnitude correlation | corr(|k|, |e|) | Improves (decreases) |
| Position-dependent MSE | First/last ratio | Improves (→ 1.0) |
| Simulated attention | KL div, L1, weighted | All improve |
| Signal-error correlation | corr(k, e) | Improves (decreases) |
| Attention logit-error | corr(q·k, q·e) | Improves (decreases) |
| Temperature perturbation | ε | Improves (decreases) |
| Entropy change | ΔH | Improves (→ 0) |

**None of these predict PPL.** The mechanism behind 10+ iteration degradation is invisible to all aggregate single-layer error analysis.

### Files

- `scripts/analyze_signal_error_correlation.py` — signal-error correlation analysis

---

## Rare State Fragility Test — REFUTED (2026-03-30 ~17:00)

### Theory

With more GLA iterations, the trellis state frequency distribution becomes skewed. Rare states (trained on few samples) have poorly-determined centroids. These rarely fire, so aggregate statistics miss them, but when they DO fire on critical tokens, the reconstruction is poor in a way that matters for PPL.

### Results — No rare states exist, distribution is nearly uniform

| Codebook | Entropy (max 9.0) | Gini | Dead | <100 | <500 | R/C MSE | max state MSE |
|----------|-------------------|------|------|------|------|---------|---------------|
| Coset 0-iter | 8.850 (98.3%) | 0.255 | 0 | 0 | 4 | 4.48x | 0.00226 |
| CUDA 3-iter | 8.852 (98.4%) | 0.251 | 0 | 0 | 13 | 3.48x | 0.00208 |
| CUDA 10-iter | 8.851 (98.3%) | 0.246 | 0 | 0 | 15 | 3.40x | 0.00124 |
| CUDA 30-iter | 8.848 (98.3%) | 0.250 | 0 | 0 | 15 | 3.46x | 0.00110 |
| Old numpy | 8.881 (98.7%) | 0.221 | 0 | 0 | 10 | 2.99x | 0.00088 |

Transition structure (all codebooks identical):
- Transitions used: ~62/4096 (1.5%) — trellis constrains paths identically
- Conditional entropy: ~2.92/3.0 bits — near-maximal choice freedom
- Unique states per block: ~110 (out of 512) — all codebooks similar

Per-state MSE distribution evolves with training:

| Codebook | state MSE CV | max/mean | skewness | corr(freq, MSE) |
|----------|-------------|----------|----------|-----------------|
| Coset 0 | 1.15 | 5.3x | +1.6 | -0.33 |
| 3-iter | 1.00 | 6.3x | +2.6 | -0.44 |
| 10-iter | 0.67 | 4.8x | +3.1 | -0.61 |
| 30-iter | 0.59 | 5.0x | +3.5 | -0.65 |
| Old numpy | 0.53 | 4.4x | +3.8 | -0.67 |

Per-state MSE CV **decreases** (becomes more uniform). The corr(freq, state_MSE) becomes more negative (rare states have relatively higher MSE), but the absolute MSE for ALL states — including the rarest — decreases with training.

### PPL correlation

| Metric | Pearson with PPL |
|--------|-----------------|
| State entropy | -0.63 |
| Gini | +0.53 |
| Rare/Common MSE | +0.47 |
| MSE | +0.10 |

State entropy shows the strongest correlation (-0.63), but the trend is flat across CUDA iterations (8.852 → 8.848). The correlation is driven by the old numpy outlier (8.881).

### Analysis

1. **No dead states, no truly rare states.** All 512 states are active for every codebook. Minimum frequency is 117 (in 1.28M assignments). With 2,500 avg assignments per state, even the rarest states have >100 samples.

2. **State distribution barely changes with training.** Entropy, Gini, CV all flat within the CUDA sweep. The trellis structure dominates — the same ~62 transitions are used regardless of codebook.

3. **The trained codebook doesn't create specialized paths.** Conditional entropy is 2.92/3.0 bits for ALL codebooks — the Viterbi encoder makes near-uniform choices at every step. There's no "path specialization" happening.

4. **Rare/Common MSE ratio is constant** (~3.4x) across iterations. Rare states always have ~3.4x higher MSE than common states, but both decrease proportionally with training.

---

## Status: All Single-Layer Hypotheses Exhausted (2026-03-30 ~17:00)

### Complete list of refuted hypotheses

Every single-layer error property we can measure improves monotonically with GLA iterations. None predict PPL.

| # | Hypothesis | Metric tested | Result |
|---|-----------|---------------|--------|
| 1 | Error covariance structure | Off-diag ratio, lag-k AC | Improves monotonically |
| 2 | Q-weighted error | tr(Σ_q · Σ_e) | ∝ MSE, no additional info |
| 3 | Eigenvector alignment | Error in Q's top/bottom eigvecs | Always isotropic (ratio ≈ 1.0) |
| 4 | Quantization bias | E[e], per-channel bias | Improves monotonically |
| 5 | Higher-order moments | Kurtosis, skewness | Improves monotonically |
| 6 | Tail behavior | p99, p99.9 | Improves monotonically |
| 7 | Input magnitude correlation | corr(|k|, |e|) | Improves monotonically |
| 8 | Position-dependent error | First/last MSE ratio | Improves monotonically |
| 9 | Simulated attention error | KL div, L1, weighted | All improve |
| 10 | Signal-dependent noise | corr(k, e), corr(q·k, q·e) | DECREASES (opposite of theory) |
| 11 | Temperature perturbation | ε, ΔH | DECREASES with training |
| 12 | Rare state fragility | State entropy, freq dist | Nearly uniform, unchanged |
| 13 | Path specialization | Transition entropy, paths used | Identical across codebooks |

### What remains

The PPL degradation at 10+ GLA iterations is real (+0.06-0.12 across 3 datasets), but invisible to ANY aggregate, single-layer error analysis. This leaves:

1. **Multi-layer error propagation** — per-layer attention error is ~0.025%, undetectable in single-layer analysis, but compounds coherently across 40 layers. Testing requires per-layer codebook ablation or intermediate activation dumping.

2. **Something we haven't conceived of** — a property of the codebook-model interaction that doesn't manifest in error statistics at all.

### Actionable conclusions regardless of mechanism

- **Use 3 GLA iterations.** Works as well as any seed, matches the "lucky" old codebook.
- **Q anisotropy is exploitable.** Effective rank 5/128 means 96% of error energy is wasted. Q-weighted TCQ training could yield ~25x attention error reduction. This doesn't require understanding the MSE-PPL divergence.
- **The MSE-PPL divergence itself is paper-worthy.** 13 distinct error metrics all improve monotonically while PPL degrades. This is a clean, novel finding for the quantization literature.

### Files

- `scripts/analyze_signal_error_correlation.py` — signal-error + temperature test
- `scripts/analyze_rare_states.py` — state frequency + rare state analysis

---

## Multi-Layer Analysis: Per-Layer Codebook Ablation (2026-03-30 ~18:00)

### Infrastructure

Added dual-codebook support controlled by `TURBO_TCQ_SPLIT` env var:
- Codebook A = old numpy (good, PPL baseline)
- Codebook B = CUDA 10-iter (bad, +0.088 at 64ch)
- Modes: `0` (all A), `all` (all B), `kv` (K=A, V=B), `vk` (K=B, V=A), `<N>` (layers >=N use B)

Code changes: `d_turbo3_tcq_codebook_B[512]` in turbo-quant-cuda.cuh and fattn-common.cuh, per-layer selection via `sscanf(dst->name, "cache_k_l%d")`, env var routing in set-rows.cu and fattn.cu.

### Phase 1: K/V Isolation

| Config | PPL (32ch) | Δ PPL | Contribution |
|--------|-----------|-------|-------------|
| All A (baseline) | 6.574 | — | — |
| All B | 6.657 | +0.083 | 100% |
| K=A, V=B | 6.627 | +0.053 | 64% |
| K=B, V=A | 6.620 | +0.047 | 57% |

**K and V contribute roughly equally** (~55%/45%). Sum of individual effects (+0.100) exceeds total (+0.083) — slight cancellation when both are degraded together.

This was unexpected: we predicted K would dominate given Q anisotropy (eff rank 5/128). V errors affect attention output linearly and apparently matter just as much.

### Phase 2: Per-Layer Binary Search

| Config | B on layers | PPL (32ch) | Δ PPL |
|--------|------------|-----------|-------|
| SPLIT=0 | none | 6.574 | baseline |
| SPLIT=10 | 10-39 | 6.634 | +0.060 |
| SPLIT=20 | 20-39 | 6.619 | +0.045 |
| SPLIT=30 | 30-39 | 6.599 | +0.025 |
| SPLIT=all | 0-39 | 6.657 | +0.083 |

Per-group contribution (by subtraction):

| Layer group | Δ PPL | % of total |
|-------------|-------|-----------|
| Layers 0-9 | +0.023 | 28% |
| Layers 10-19 | +0.015 | 18% |
| Layers 20-29 | +0.020 | 24% |
| Layers 30-39 | +0.025 | 30% |
| **Sum** | **+0.083** | **100%** |

### Key Finding: Uniformly Distributed, Perfectly Additive

1. **No concentrated sensitivity.** Every 10-layer group contributes 18-30% of the degradation. No single layer or small group dominates.

2. **Perfectly additive.** The sum of group contributions (0.023+0.015+0.020+0.025 = 0.083) matches the total exactly. Effects are linear, not compounding.

3. **Per-layer contribution: ~+0.002 PPL.** Each of the 40 layers contributes roughly +0.002 PPL — far below the detection threshold for individual layers (±0.065 at 64 chunks).

4. **This is Scenario B** from our analysis plan: the mechanism is a universal property of the codebook that creates slightly worse reconstruction at every layer, compounding linearly across 40 layers.

### What This Rules Out

- **Per-layer sensitivity** (Scenario A): Ruled out. Q anisotropy (layer 19 eff rank 1.9) does not create concentrated sensitivity.
- **Nonlinear compounding**: Ruled out. Effects sum linearly — no error amplification across layers.
- **K-dominant theory**: Ruled out. V contributes equally.

### Implications for Mechanism

The degradation is universal and additive. This means:
- Each layer independently contributes ~+0.002 PPL from the codebook swap
- This per-layer contribution is too small to detect by any error metric at single-layer resolution
- The effect is real but below measurement noise at any single observation point
- It sums to a detectable +0.083 only because 40 independent layers accumulate it

**The mechanism must be a subtle per-block property of the codebook** that:
1. Affects every layer equally (universal, not layer-dependent)
2. Is invisible to aggregate statistics (MSE, covariance, kurtosis all improve)
3. Contributes +0.002 PPL per layer (below detection in single-layer error metrics)
4. Is present in both K and V quantization (both contribute ~equally)

### Next Steps

The question changes from "which layers are sensitive?" to "what subtle codebook property creates +0.002 PPL per layer while having better aggregate statistics?"

Candidates:
1. **Reconstruction norm interaction**: the norm correction `saved_norm / recon_norm` interacts differently with different codebooks. Small norm errors compound additively.
2. **Block-boundary effects**: the trellis has warm-up/cool-down at block boundaries. 128-element blocks with 40 layers × 4 heads × 2 blocks/head = 320 boundaries per token.
3. **Encoding path sensitivity**: the Viterbi encoder makes different path choices with different codebooks. Even with lower MSE, the paths might be slightly worse for specific activation patterns that repeat across all layers.

### Files

- `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` — added `d_turbo3_tcq_codebook_B[512]`, shared memory codebook in SET_ROWS
- `ggml/src/ggml-cuda/fattn-common.cuh` — added `d_turbo3_tcq_codebook_B_fattn[512]`
- `ggml/src/ggml-cuda/set-rows.cu` — TURBO_TCQ_SPLIT env var parsing, layer index extraction
- `ggml/src/ggml-cuda/fattn.cu` — dequant kernel `use_alt` parameter, split logic for both prefill and decode

---

## Norm Correction Analysis — REFUTED (2026-03-30 ~19:00)

### Theory

The SET_ROWS kernel normalizes input to unit norm, runs Viterbi, then computes `corrected_norm = saved_norm / recon_norm` where `recon_norm = ||reconstruction||`. Different codebooks produce different `recon_norm` values → different norm corrections → systematic magnitude errors that compound across layers.

### Results

| Codebook | recon_norm bias | |norm-1| mean | norm_std | unit_MSE | cos_dist |
|----------|----------------|-------------|----------|----------|----------|
| old numpy | -0.010 | 0.015 | 0.016 | 0.000173 | 0.0110 |
| cuda_3 | -0.030 | 0.032 | 0.023 | 0.000265 | 0.0168 |
| cuda_10 | -0.014 | 0.018 | 0.017 | 0.000209 | 0.0133 |
| cuda_30 | -0.011 | 0.015 | 0.015 | 0.000183 | 0.0117 |

1. **Output norm ratio = exactly 1.0** for all codebooks. The norm correction perfectly restores the input norm.
2. **fp16 precision loss < 0.001%** of the corrected norm. Not significant.
3. **All norm metrics improve monotonically** with GLA iterations: bias shrinks, variance shrinks.
4. **Cosine distance also improves monotonically** (0.017 → 0.011). Direction quality gets better.
5. **10-iter vs old systematic difference**: recon_norm is -0.004 lower for 10-iter, statistically significant (|mean|/SE = 7.9), but 10-iter is actually CLOSER to 1 than 3-iter. Directionality of this difference doesn't explain PPL.

### Conclusion

Norm correction is **not the mechanism**. The norm correction works perfectly regardless of codebook. This is hypothesis #15 to show monotonic improvement while PPL doesn't.

---

## Status: Complete Refutation of All Testable Hypotheses (2026-03-30 ~19:00)

### Updated refutation list (15 hypotheses + multi-layer ablation)

| # | Hypothesis | Result |
|---|-----------|--------|
| 1-13 | (See earlier table) | All improve monotonically |
| 14 | Per-layer sensitivity | **Uniform across 40 layers, additive** |
| 15 | K dominates V | **Both contribute equally (~55/45)** |
| 16 | Norm correction error | **Norm is exact; all metrics improve** |

### What we've definitively established

1. **The PPL degradation is real**: +0.083 on test, confirmed across 3 datasets
2. **It's universal**: affects all 40 layers equally, both K and V
3. **It's additive**: per-layer contributions sum linearly (no compounding)
4. **Every measurable error property improves monotonically** with GLA iterations
5. **The mechanism is invisible to ANY single-layer error analysis** — including covariance, weighted covariance, bias, kurtosis, tails, autocorrelation, signal-error correlation, temperature perturbation, state frequency, norm correction, and simulated attention error with real Q vectors

### What this means

The 10-iter codebook is **measurably better than the 3-iter codebook on every error metric we can compute**, yet produces **+0.088 worse PPL**. Meanwhile the 3-iter codebook (with 53% HIGHER MSE than old numpy) matches old numpy's PPL within ±0.005.

The MSE→PPL relationship for TCQ codebooks is:
- 3-iter: MSE 0.000267, PPL 6.502 (BEST PPL despite HIGHEST MSE!)
- 10-iter: MSE 0.000211, PPL 6.595 (worst PPL)
- 30-iter: MSE 0.000184, PPL 6.560
- old numpy: MSE 0.000174, PPL 6.507

This is analogous to **overfitting in machine learning**: more GLA iterations optimize the MSE objective but degrade generalization (PPL). The 3-iter codebook benefits from "early stopping" — just enough training for trellis coding gain without over-specializing.

### Paper narrative

This is the most important finding for the paper:

**TCQ codebook optimization exhibits an early-stopping phenomenon where 3 GLA iterations provide optimal perplexity despite 53% higher MSE than fully-trained codebooks. The MSE-PPL divergence is uniformly distributed across all 40 transformer layers, affects K and V equally, is perfectly additive, and is invisible to 16 distinct error metrics. This represents a fundamental disconnect between element-wise MSE and attention quality that no known error analysis technique can bridge at single-layer resolution. The practical recommendation: use 3 GLA iterations for TCQ codebook training.**

### Files

- `scripts/analyze_norm_correction.py` — norm correction analysis

---

## CRITICAL CORRECTION: Codebook A Was Mislabeled (2026-03-30 ~20:00)

### Discovery

While adding runtime codebook loading infrastructure (`TURBO_TCQ_CB_B` env var, `cudaMemcpyToSymbol`), we discovered that the codebook compiled into the A slot (`d_turbo3_tcq_codebook[512]`) in the uncommitted working tree was **NOT** the old numpy codebook. The source code comment reads:

> "CUDA GLA fine-tuned: 50 iters, 100K samples from coset init"

This is the **50-iter fine-tuned codebook** (trained starting FROM old numpy via `--init`), with 52.8% MSE reduction. The real old numpy codebook (50.1% MSE reduction) exists in:
- The committed source (`git show HEAD:...turbo-quant-cuda.cuh`)
- The binary file `/tmp/old_codebook_3bit.bin`

The two codebooks are completely different: max value difference = 0.035, all 512 entries differ.

### Impact on previous results

**Every PPL measurement labeled "old numpy" in this document was actually the 50-iter fine-tuned codebook.** The previous conclusions were based on comparing 50-iter-fine-tuned (PPL ~6.50) vs 3-iter-seed-42 (PPL ~6.50) and concluding they matched. In reality, both were being compared against a codebook that happens to give similar PPL to 3-iter — not the actual old numpy.

### Corrected measurements

All measurements below use the same build with runtime codebook loading through the B pathway (`TURBO_TCQ_CB_B=<file> TURBO_TCQ_SPLIT=all`). Validated: loading old numpy via B pathway gives identical PPL to the clean committed build (6.575 vs 6.575). No pathway bias.

| Codebook | Training details | MSE | PPL (64ch test) | Δ vs old numpy |
|----------|-----------------|-----|-----------------|----------------|
| **Old numpy** | ~100 iter, 4K samples, numpy trainer | 0.000174 | **6.575** | baseline |
| 50-iter fine-tuned | 50 iter FROM old numpy (`--init`), 100K | ~0.000168 | 6.599 | +0.024 |
| 3-iter seed 42 | coset→3 iter, 100K samples | 0.000267 | 6.609 | +0.034 |
| 3-iter seed 123 | coset→3 iter, 100K samples | 0.000272 | 6.616 | +0.041 |
| 3-iter seed 999 | coset→3 iter, 100K samples | 0.000272 | 6.618 | +0.043 |
| 3-iter seed 7 | coset→3 iter, 100K samples | 0.000271 | 6.619 | +0.044 |
| 50-iter coset | coset→50 iter, 100K, seed 42 | 0.000179 | 6.620 | +0.045 |
| 10-iter seed 42 | coset→10 iter, 100K samples | 0.000211 | 6.630 | +0.055 |

Note: absolute PPL values shifted ~+0.07 vs earlier measurements due to intervening code changes (turbo4 inverse-FWHT, etc.). Relative ordering is what matters.

### What this changes

**1. "3-iter matches old numpy" — WRONG.** 3-iter seed 42 is +0.034 worse than old numpy. All other seeds are +0.041-0.044 worse. No 3-iter codebook from any seed matches old numpy.

**2. "Seed 42 is special at 3 iterations" — WRONG.** Seed 42 at 3-iter (6.609) is only marginally better than seeds 7/123/999 (6.616-6.619). The spread across seeds (0.010) is much smaller than the gap to old numpy (0.034-0.044). Seed 42 is not an outlier.

**3. "The old codebook is NOT lucky" — WRONG.** Old numpy IS a lucky codebook. No CUDA-trained codebook from coset initialization matches it at any iteration count. The closest approach is 50-iter fine-tuned FROM old numpy (+0.024), which partially preserves old numpy's properties but still degrades.

**4. MSE-PPL divergence is REAL but smaller than reported.** The anomaly was reported as +0.088 (old numpy vs 10-iter). The actual gap is +0.055 (old numpy 6.575 vs 10-iter 6.630). There IS rough MSE-PPL correlation: lower MSE codebooks generally have better PPL. But old numpy violates this — it has higher MSE than 50-iter-fine-tuned yet better PPL.

**5. The multi-layer ablation results remain valid.** Those experiments compared codebook A vs B within the SAME build. The relative finding (uniform distribution, additive, K≈V) is not affected by which codebook was in A — just the interpretation of "which codebook is better."

### Revised understanding

The PPL ordering is:

```
old numpy (6.575) > 50-iter-ft (6.599) > 3-iter (6.609) > other seeds (6.616-6.620) > 10-iter (6.630)
    MSE: 0.000174        ~0.000168        0.000267         0.000271-272        0.000211
```

Key observations:
- Old numpy is the best despite NOT having the lowest MSE
- Fine-tuning FROM old numpy partially preserves its quality (+0.024 degradation after 50 more GLA iterations)
- All coset-init codebooks cluster at +0.034 to +0.055 regardless of iteration count
- The 3-iter vs 10-iter gap within coset-init codebooks is only +0.021 (6.609 vs 6.630), not +0.088

### What makes old numpy special?

The old numpy codebook was trained with:
- The numpy-based Viterbi trainer (not CUDA)
- 4K samples per iteration (not 100K)
- ~100 iterations
- Unknown seed (original training is not reproducible)
- Previous tests with 4 numpy seeds (7, 42, 123, 999) at 100 iterations all gave PPL 5.88-5.93 — none matched old numpy

The old numpy codebook occupies a PPL-favorable local minimum that:
1. Cannot be reached from coset initialization with the CUDA trainer at any iteration count
2. Cannot be replicated with the numpy trainer at different seeds
3. Is partially preserved when used as initialization for further GLA training
4. Is gradually eroded by additional GLA iterations (50-iter-ft degrades +0.024 from old numpy)

This points to the codebook's PPL quality being determined by its position in the optimization landscape, not by aggregate error statistics. The old numpy codebook landed in a rare basin that happens to produce attention-friendly quantization errors.

### Remaining questions

1. **What structural property of old numpy makes it special?** Direct comparison of codebook values between old numpy and other codebooks (spacing, group monotonicity, dynamic range, etc.) might reveal this.
2. **Can we find the basin intentionally?** Multi-seed search with PPL-in-the-loop evaluation could find other codebooks in the same basin.
3. **Does the encode/decode mismatch test reveal anything?** Encode with old numpy states but decode with another codebook's values (and vice versa) would isolate whether the advantage is in state selection vs reconstruction values.

### Infrastructure added

- `TURBO_TCQ_CB_B=<path>`: Load codebook B from a binary file (512 floats) at runtime via `cudaMemcpyToSymbol`. Eliminates recompilation for each codebook test.
- `TURBO_TCQ_DECODE_SPLIT=<mode>`: Independent control of fattn decode codebook selection. Falls back to `TURBO_TCQ_SPLIT` when not set. Enables encode/decode mismatch experiments.
- `--seed <N>` flag added to `scripts/tcq_train_cuda.cu` for reproducible codebook training.

### Files

- Binary codebooks on server: `/tmp/old_codebook_3bit.bin`, `/tmp/tcq_3bit_{3,10,50}iter_s{7,42,123,999}.bin`, `/tmp/cb_50iter_finetuned.bin`
- `scripts/tcq_train_cuda.cu` — updated with `--seed` flag
- `ggml/src/ggml-cuda/set-rows.cu` — runtime codebook loading
- `ggml/src/ggml-cuda/fattn.cu` — runtime loading + `TURBO_TCQ_DECODE_SPLIT`

---

## Test 3: Encode/Decode Codebook Mismatch (2026-03-30)

**Question**: Is old numpy's PPL advantage from its state selection (encode) or its reconstruction values (decode)?

### Setup

- Codebook A (compiled-in): Old numpy (verified -0.24244059f first value, matching committed HEAD)
- Codebook B (runtime-loaded via TURBO_TCQ_CB_B): CUDA 10-iter seed 42
- `TURBO_TCQ_SPLIT` controls encode path (SET_ROWS Viterbi)
- `TURBO_TCQ_DECODE_SPLIT` controls decode path (fattn dequant)
- Both codebook A arrays updated (turbo-quant-cuda.cuh AND fattn-common.cuh)
- 64 chunks, wikitext-2 test, c=512

### Results

| Config | Encode CB | Decode CB | PPL | Δ vs A/A |
|--------|-----------|-----------|-----|----------|
| A/A | Old numpy | Old numpy | **6.575** | baseline |
| B/B | 10-iter | 10-iter | **6.630** | +0.055 |
| A/B | Old numpy | 10-iter | **6.753** | +0.178 |
| B/A | 10-iter | Old numpy | **6.783** | +0.208 |

### Analysis

1. **Both mismatches are catastrophically worse than either pure case.** Encoding with one codebook and decoding with another produces states optimized for wrong reconstruction values. The +0.12 to +0.21 penalty confirms the Viterbi state path and reconstruction values are tightly coupled.

2. **Old numpy's advantage is holistic, not decomposable.** If the advantage were purely in reconstruction values, A/B (old numpy states + 10-iter decode) would be close to A/A. If purely in state selection, B/A would be close to A/A. Neither is — both are far worse.

3. **B/A is slightly worse than A/B** (+0.208 vs +0.178). This means old numpy's state paths are slightly more robust to codebook perturbation than 10-iter's. Old numpy selects states that are more "generally good" — their quality degrades less gracefully when decoded with wrong values.

4. **The mismatch asymmetry is small** (0.03 PPL). Neither encode nor decode dominates — the codebook functions as an indivisible unit where state selection and reconstruction values must be co-optimized.

### Implications

- Codebook quality cannot be improved by independently optimizing state selection vs reconstruction values
- The MSE-PPL divergence is not explained by a "better states, worse values" decomposition
- Old numpy genuinely found a better joint solution in the (states × values) space despite achieving it with fewer GLA iterations
- This is consistent with GLA iterations getting trapped in local optima that are MSE-optimal but PPL-suboptimal

---

## Q-Weighted TCQ Training — Feasibility Assessment (2026-03-31)

### Concept

Standard TCQ minimizes MSE (treats all K dimensions equally). Since Q is highly anisotropic (effective rank 5/128 aggregate), ~96% of K quantization error lands in directions Q doesn't use for attention. A Q-weighted Viterbi path metric would concentrate codebook precision where it matters:

```
standard: path_cost += ||k - k̂||²
weighted: path_cost += (k - k̂)ᵀ Σ_Q (k - k̂)
```

This is the GPTQ analogy — GPTQ uses per-layer Hessians to weight which weight dimensions matter. Here, Σ_Q weights which K dimensions matter for attention.

### Theoretical upper bound: ~25x attention error reduction at same MSE

If Q only uses 5/128 directions and error is currently isotropic, then only ~4% of error is in Q's subspace. Redirecting quantization budget → up to 1/0.04 = 25x reduction in attention-weighted error.

### Why 25x is unrealistic in practice

1. **Per-layer Q anisotropy varies wildly.** Layer 19 has effective rank 1.9 (eigenvalue ratio 9932:1), layer 20 has rank 114. The "rank 5/128 aggregate" masks this. Near-full-rank layers get almost no benefit.

2. **Single shared codebook.** TCQ codebook is the same for all 40 layers. Can't simultaneously optimize for layer 19's rank-2 Q subspace and layer 20's rank-114 subspace. Optimal for one hurts the other.

3. **Trellis constrains error redistribution.** Unlike scalar quantization (independently optimize each coordinate), TCQ processes dimensions sequentially through state transitions. Trellis structure limits how freely error can be pushed out of specific directions.

4. **Error isotropicity result (this document).** We measured top/avg eigenvalue ratio of error covariance at ~1.0 for ALL codebooks. Better MSE codebooks don't put more error in Q's sensitive direction — the MSE-PPL divergence mechanism isn't about error directionality.

5. **Codebook trained offline, Q varies at inference.** Q statistics differ per layer, per head, per token, per input. A single codebook can't be simultaneously optimal for all Q subspaces across all inputs.

### Realistic estimate: 2-5x attention-weighted error reduction

Accounting for per-layer variation and trellis constraints, a shared Q-weighted codebook would likely achieve 2-5x improvement in attention-weighted error. Whether this translates to meaningful PPL improvement depends on:
- Whether the MSE-PPL divergence is actually an error-direction problem (our isotropicity data suggests NOT)
- Whether the gain at anisotropic layers (19, etc.) survives averaging with near-isotropic layers (20, etc.)

### Path forward if pursued

Would require **per-layer codebooks** (or layer-group codebooks) to be effective:
- 40 codebooks × 512 states × 4 bytes = 80KB (3-bit), fits in CUDA constant memory
- Training pipeline change: extract per-layer Q covariance, train per-layer codebooks with weighted Viterbi
- Architectural change: layer index passed to encode/decode kernels to select codebook (Phase 2 of the multi-layer plan already adds this infrastructure)

### Status: DEFERRED

Promising research direction but blocked by two findings:
1. Error isotropicity suggests the MSE-PPL gap isn't about error direction
2. Per-layer codebooks are a prerequisite for meaningful gains, which is a significant architectural change

Revisit after the multi-layer ablation (plan Phase 2) identifies whether specific layers drive the PPL regression.
