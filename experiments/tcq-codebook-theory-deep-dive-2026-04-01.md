# TCQ Codebook Theory Deep Dive — 2026-04-01

## Motivation

We observe that more GLA iterations produce codebooks with lower MSE but worse PPL.
Our "lucky numpy" codebook (few iterations, old seed) beats all CUDA-trained variants.
Product-aware training (Ordentlich-Polyanskiy inspired) produced byte-identical codebooks
to standard MSE training. We need to understand WHY mathematically.

Three parallel research agents investigated: (1) TCQ codebook design theory, (2) quantization
error propagation through attention, (3) TCQ trellis path specifics.

---

## Core Finding: MSE is Provably the Wrong Objective

Three independent theoretical results converge:

### 1. Ordentlich & Polyanskiy (2024) — Matrix Multiply Quantization

For matrix multiplication A^T * B (which attention IS — Q^T * K), the rate-distortion
function is fundamentally different from entrywise MSE R(D).

- MSE-optimal quantizers are **3.4x worse** than task-optimal ones at realistic dimensions
- Phase transition at R* = 0.906 bits: below this, optimal strategy zeros out coordinates
- The optimal quantizer density for matmul: `(f(x)·E[Y²|X=x])^{1/3}`
- Standard MSE-optimal density: `f(x)^{1/3}`
- These are DIFFERENT functions with different shapes

### 2. Zamir-Feder (1996) — Lattice Quantization Noise

Lattice (regular/structured) quantizers produce WHITE noise — uniform, uncorrelated,
signal-independent. As dimension increases, quantization noise approaches white Gaussian.

Trained VQ quantizers produce signal-dependent, correlated noise. After softmax (which
exponentially amplifies variance), signal-dependent noise is catastrophically worse than
white noise at the same MSE.

QuIP# empirically confirmed: E8 lattice OUTPERFORMS k-means trained codebooks despite
k-means having lower MSE on training data.

### 3. Kasner & Marcellin (UTCQ, 1999) — No Training Needed

Universal TCQ with uniform thresholds (NO training at all) achieves "performance comparable
with fully optimized ECTCQ." The coding gain comes from the trellis structure and coset
set-partitioning, NOT from codeword position optimization.

---

## Analysis of Our Two Training Methods

### Method 1: Standard GLA

GLA alternates Viterbi encoding with centroid updates. Each iteration provably decreases MSE.

**Why more iterations hurt:**
1. MSE is the wrong objective — lower MSE ≠ better attention quality
2. Over-training breaks coset regularity (interleaved structure)
3. Irregular cosets → signal-dependent error variance → exponentially amplified by softmax
4. UTCQ proves the trellis structure provides coding gain, not codeword positions
5. The "lucky numpy" codebook preserves regularity due to few iterations + good initialization

**GLA failure modes (TCQ-specific):**
- Coupled optimization through trellis: moving one codeword cascades through all Viterbi paths
- Coset structure degrades: codewords from different cosets "cross over"
- Path concentration: most Viterbi paths converge to subset of states
- Reduced effective free distance: coding gain drops as set partitioning breaks down
- Exponentially more local minima than unconstrained VQ

### Method 2: Product-Aware Training

Attempted to optimize for attention product distortion (K·Q interaction) rather than raw MSE.
Implemented as a gentle regularizer on GLA centroids.

**Why it failed:**
- GLA centroid = global MSE minimizer for each partition cell
- Weak regularizer barely moves centroids → byte-identical to standard MSE
- The Ang et al. (2026) result shows the correct objective has a completely different SHAPE
- For correlated Q,K with |ρ| > 1/√3: optimal density is BIMODAL (peaks at tails)
- A regularizer cannot turn a unimodal quantizer into a bimodal one

**The idea was correct, but the implementation was wrong.** Need a fundamentally different
training objective, not a regularizer on MSE.

---

## K vs V Error Propagation (Asymmetric)

From AsymKV (2410.13212), Theorem 1:

**V errors propagate LINEARLY:**
- O = A * V, so V error E_v → A * E_v (simple weighted average)
- Per-token V quantization confines error within each token

**K errors propagate NONLINEARLY (exponentially amplified):**
- K error enters inside softmax exponential: exp(E_q / sqrt(h))
- Two stages: (1) query multiplies K error, (2) exponential amplifies it
- K quantization is ~5x more damaging than V at same bit width

**Softmax Lipschitz bounds (2025):**
- Softmax is exactly 1/2-Lipschitz (tight bound, previously assumed 1)
- Local Lipschitz depends on attention distribution:
  - Near-uniform OR near-categorical: stable (small constant)
  - Intermediate entropy: maximum sensitivity (1/2)
  - Attention heads with moderate concentration are MOST vulnerable

---

## Rate-Distortion Fundamentals

For i.i.d. Gaussian (our post-FWHT data):
- Shannon bound: D(R) = σ² · 2^(-2R)
- Optimal scalar quantizer: 1.53 dB worse (fundamental, unclosable gap)
- 8-state TCQ: ~0.5 dB from R(D)
- 256-state TCQ: ~0.2 dB from R(D)

**Gain decomposition (Lookabaugh & Gray):**
- Granular gain (cell shape/sphericity)
- Shaping gain (boundary matches source)
- Space-filling gain (no wasted space)
- TCQ primarily captures granular gain through trellis structure

**TurboQuant proof:** After rotation, each coordinate follows known Beta distribution.
Lloyd-Max quantizer analytically computable. Within 2.72x of Shannon bound. No training needed.

---

## Context-Length Dependence Explanation

Why codebook quality is context-dependent:
- At longer contexts, softmax distributions become more diffuse (higher entropy)
- Softmax Lipschitz constant is LARGEST at intermediate entropy
- K quantization errors compound across more tokens
- Regular codebooks: consistent error properties across all tokens → graceful degradation
- Irregular codebooks: worst-case errors get more chances to cause damage at longer contexts

---

## Theoretically-Grounded Paths Forward

### Path A: Analytical codebook (no training)
- Post-FWHT data is known Gaussian → Lloyd-Max is computable
- Within 2.72x of Shannon bound (TurboQuant proof)
- UTCQ proves untrained TCQ matches trained TCQ
- Coset regularity preserved by construction
- Simple, defensible, publishable
- **CAVEAT**: UTCQ explicitly caveats "most encoding rates" — breaks down at 2-bit where
  codebook is too coarse for structure alone. 3-iter numpy at 2-bit is terrible everywhere.

### Path B: Train for the right objective
- Replace MSE centroid with attention-output-error-minimizing centroid
- Use Ang et al. density: `(f(x)·E[Y²|X=x])^{1/3}` instead of `f(x)^{1/3}`
- Hard constraint: maintain monotonic coset interleaving
- Ambitious, needs careful implementation

### Path C: More trellis states
- 8 states → 0.5 dB from R(D); 256 states → 0.2 dB
- Diminishing returns, but the gain is structural (free)
- Trade-off: more states = larger codebook = more memory

### Path D: Stepped codebooks (context-dependent selection)
- Use conservative codebook at short contexts, aggressively-trained at long contexts
- Already have runtime loading mechanism (TURBO_TCQ_CB env var)
- Crossover point ~8K at 3-bit (empirically determined)
- Could expose as flag: `--kv-codebook short|long|auto`

---

## WHY Trained Codebooks Win at Long Context — Literature Findings

Research agent found 7 mechanisms. The three most important:

### 1. CLT Averaging of Quantization Error (PRIMARY MECHANISM)

Softmax attention computes weighted averages of V vectors. At longer contexts, more tokens
contribute to the average. By central limit theorem, quantization error variance is
suppressed by ~1/√n_eff (effective number of attended tokens).

Two components of codebook error:
- **Bias** (systematic error / noise floor) — determined by MSE. NOT suppressed by averaging.
- **Variance** (random error / spikes) — determined by regularity. IS suppressed by averaging.

**Short context**: n_eff small → variance dominates → regularity wins
**Long context**: n_eff large → CLT suppresses variance → bias/MSE floor dominates → trained wins

This cleanly explains the crossover: at short contexts, error variance from irregular codebooks
causes attention corruption. At long contexts, the averaging dilutes variance, and the lower
MSE of trained codebooks directly translates to lower output error.

### 2. Finite-Blocklength Dispersion (Kostina-Verdu 2012)

The gap between actual quantizer performance and R(D) bound scales as:

    gap ∝ V(R) / √n

where V(R) is the rate-dispersion function and n is block length.

A trained codebook's residual gap is dispersion-dominated (shrinks with n).
An untrained/mismatched codebook has a CONSTANT mismatch gap + dispersion:

    gap_untrained = D_mismatch + V(R) / √n

At short n: dispersion term large → trained vs untrained gap small
At large n: dispersion term small → trained codebook's advantage GROWS because its
gap shrinks while the untrained gap converges to D_mismatch > 0.

The trained codebook's advantage literally grows with sequence length.

### 3. Overload Distortion at Low Rate (explains 2-bit)

At 2-bit with only 4 codewords per coset, overload distortion (clipping of tail samples)
dominates. Training is essential to place the sparse levels correctly. Structure alone
cannot compensate — there simply aren't enough levels for the coset interleaving to help.

This is why:
- 3-iter numpy at 2-bit is terrible (levels are random, massive overload)
- UTCQ's "no training needed" result doesn't apply at R=1
- The compiled-in 2-bit codebook has higher kurtosis error → 64K cliff

### Other mechanisms found

4. **Viterbi path mixing** — trained codebooks have better-matched stationary distributions.
   At long sequences, the stationary distribution dominates over initial conditions.
5. **Attention dilution** (Scalable-Softmax 2501.19399) — at long context, attention weights
   flatten, amplifying noise floor relative to spikes.
6. **Codebook mismatch theory** (Gray & Linder 2003) — mismatched codebook penalty is
   persistent and constant with block length. Trained codebook gap vanishes.
7. **Error kurtosis** — two codebooks with same MSE can have very different tail behavior.
   Softmax sensitivity to outliers increases with context length. The compiled-in 2-bit
   anomaly at 64K (7.484 vs 7.222 for 100-iter) is likely a kurtosis/tail effect.

### Reconciled picture

The "regularity > MSE" thesis from UTCQ and Zamir-Feder is correct for:
- Higher rates (3+ bits with many codewords per coset)
- Short block lengths (where variance dominates)
- Entropy-constrained settings (where entropy coder compensates for irregularity)

It breaks down for:
- Low rates (2-bit, where level placement is critical)
- Long block lengths (where CLT averaging suppresses variance advantage)
- Fixed-rate settings (where mismatch penalties are persistent)

**Both MSE and regularity matter. Which dominates depends on rate and context length.**
The ideal codebook minimizes MSE WITHOUT sacrificing regularity. The ideal training procedure
is: enough GLA iterations to match the distribution well, constrained to preserve coset
structure. Or: stepped codebooks that use different codebooks at different context lengths.

---

## Testable Predictions

1. **Coset regularity correlates with KLD at short contexts**: Measure monotonic group fraction.
   Predict: more regular codebook wins at 2K-8K.
2. **MSE correlates with KLD at long contexts**: Predict: lower MSE wins at 32K+.
3. **Error kurtosis explains compiled-in 2-bit anomaly**: Measure kurtosis of quantization
   error for compiled-in vs 100-iter numpy at 2-bit. Predict: compiled-in has higher kurtosis.
4. **Stepped codebook beats any single codebook**: Use compiled-in for ≤8K, cb_50iter for >8K.
   Predict: better than either alone across all context lengths.
5. **Analytical Lloyd-Max codebook works at 3-bit but not 2-bit**: Compute optimal Gaussian
   quantizer levels analytically. Predict: competitive at 3-bit, bad at 2-bit.

### Key references for context-dependent codebook quality
- Kostina & Verdu (2012): Finite-blocklength lossy compression, arXiv 1102.3944
- Ingber & Kochman (2011): Dispersion of lossy source coding, arXiv 1102.2598
- Gray & Linder (2003): Mismatch in high-rate ECVQ
- Kasner & Marcellin (1999): UTCQ — "most encoding rates" caveat
- Scalable-Softmax (2501.19399): Attention dilution at long context

---

## CRITICAL: Theory vs Data Mismatch — Context-Dependent Codebook Quality

**Our data CONTRADICTS the simple "regularity > MSE" thesis.**

### What the data actually shows

**3-bit PPL**: Crossover at ~8K. More-trained (cb_50iter) is WORSE at short contexts
(2K-8K) but BETTER at long contexts (32K+). The aggressively-trained codebook wins where
it matters most (long context = production use case).

**2-bit PPL**: Training helps at ALL contexts up to a point. 3-iter numpy (least trained)
is terrible everywhere. 100-iter beats CUDA 200-iter at 64K but loses at 2K.

**Pattern**: Conservative/analytical codebooks → better at short context (2K-8K).
Aggressively-trained codebooks → better at long context (32K+).

### Why the "regularity wins" thesis is incomplete

The UTCQ result ("no training needed") may only apply at:
- Higher rates (3+ bits per coset member) where each coset covers the space adequately
- Or in the asymptotic (infinite block length) regime

At low rates (2-bit) and short sequences, untrained codebooks are just bad — they don't
match the distribution well enough. The regularity advantage doesn't compensate for having
codewords in the wrong places.

### Proposed mechanism: MSE floor vs error variance

Two competing effects determine codebook quality at a given context length:

1. **Mean error level (MSE)** = the noise floor. Lower MSE = less average distortion.
2. **Error variance/regularity** = noise spikes. Irregular codebooks have signal-dependent
   error that creates occasional large errors.

**At short contexts (2K-8K)**:
- Attention is concentrated on few tokens (low entropy softmax)
- Individual error spikes can corrupt the entire attention pattern
- ERROR VARIANCE dominates → regularity wins
- Softmax Lipschitz constant is small at near-categorical attention → less amplification
  of mean error, but spikes still cause pattern corruption

**At long contexts (32K+)**:
- Attention is diffuse across many tokens (higher entropy softmax)
- Individual spikes are diluted by averaging over many tokens
- MEAN ERROR LEVEL (MSE) dominates → lower MSE wins
- The trained codebook's lower noise floor outweighs its occasional spikes
- Softmax Lipschitz is at intermediate values → amplifies mean error more

This explains the crossover: short context cares about worst-case (regularity),
long context cares about average-case (MSE).

### Stepped codebooks — practical path forward

Since we have runtime codebook loading (`TURBO_TCQ_CB` env var), we could implement
context-dependent codebook selection:

- **Short context (≤8K)**: Use conservative codebook (compiled-in or analytical)
- **Long context (>8K)**: Use aggressively-trained codebook (cb_50iter or similar)

This could be implemented as:
1. Detect context length at KV cache init time
2. Load appropriate codebook binary
3. Switch point determined empirically per bit-width (crossover is ~8K at 3-bit)

Even simpler: since most users know their target context length at launch time,
expose it as a flag: `--kv-codebook short|long|auto`

### What this means for training

The goal is NOT "train less" or "train more" — it's:
1. For short-context use: train moderately (preserve structure, match distribution)
2. For long-context use: train aggressively (minimize MSE, structure matters less)
3. OR: find a training objective that's good at all contexts (the holy grail)

The product-aware training idea was pointing in the right direction (task-specific
objective) but implemented wrong (regularizer too weak). A properly weighted objective
that accounts for context-length-dependent sensitivity could produce codebooks that
work well everywhere.

---

## Key References

### Foundational
- Shannon (1959): Rate-distortion theory
- Gray & Neuhoff (1998): "Quantization" — 59-page definitive survey
- Zamir & Feder (1996): Lattice quantization noise is white
- Gersho (1979): Asymptotically optimal block quantization

### TCQ
- Marcellin & Fischer (1990): TCQ of memoryless and Gauss-Markov sources
- Fischer & Wang (1992): Entropy-constrained TCQ
- Kasner & Marcellin (1999): Universal TCQ
- Sabin & Gray (1986): GLA convergence proof
- Cappellari (arXiv:0704.1411): Max-Hamming-distance codes for TCQ
- Kieffer & Liao (arXiv:1010.1286): Exact Hamming distortion of Viterbi-encoded TCQ

### Task-Specific Quantization
- Ordentlich & Polyanskiy (2410.13780): Optimal quantization for matrix multiplication
- Ordentlich & Polyanskiy (2601.17187): High-rate comparison
- Ang, Kim & Pilanci (2603.19559): Optimal scalar quantizer for matmul, phase transition

### Error Propagation
- AsymKV (2410.13212): K vs V error asymmetry
- QEP (2504.09629): Error accumulation across layers
- Softmax 1/2-Lipschitz (2510.23012): Tight perturbation bound
- Local Lipschitz for attention (2507.07814): Distribution-dependent bounds

### Modern Applications
- QTIP (NeurIPS 2024): TCQ for LLM weight quantization
- QuIP# (2402.04396): Hadamard + E8 lattice
- TurboQuant (ICLR 2026): Online VQ with near-optimal distortion rate
- LLVQ (2603.11021): Leech lattice for LLM compression
- GPTQ-as-Babai (2507.18553): Geometric view of quantization error

Full memory files saved at:
- memory/reference_tcq_codebook_theory_deep_dive.md
- memory/reference_tcq_codebook_training_deep_dive.md
- memory/reference_quantization_error_propagation_theory.md
