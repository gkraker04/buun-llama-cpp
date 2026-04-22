# TCQ Codebook Optimization: Research Notes (2026-03-30, updated later same day)

## The MSE-PPL Divergence Problem

Training TCQ codebooks to deeper MSE optima **hurts perplexity**. BUT the relationship is non-monotonic — PPL oscillates chaotically as GLA iterations increase.

### Full GLA Iteration Sweep (3-bit)

| GLA Iters | Samples/iter | MSE Reduction | PPL (2K, 8ch) | Delta vs old |
|-----------|-------------|---------------|---------------|--------------|
| 0 (coset init) | — | 0.1% | 5.9194 | +1.64% |
| 1 | 100K | -0.1% | 5.9194 | +1.64% |
| 3 | 100K | 24.3% | 5.8450 | +0.37% |
| 5 | 100K | 33.8% | 5.8576 | +0.58% |
| 10 | 100K | 40.4% | 5.9386 | +1.97% |
| 20 | 100K | 46.0% | 5.9712 | +2.53% |
| 30 | 100K | 48.2% | 5.8733 | +0.85% |
| **50** | **100K** | **52.8%** | **5.8313** | **+0.13%** |
| 100 | 100K | 54.1% | 5.8889 | +1.12% |
| 200 | 100K | 54.7% | 5.9094 | +1.47% |
| 100 (small batch) | 4K | 53.2% | 5.9600 | +2.34% |
| From scratch (real) | 500×30 | 54.9% | 5.8741 | +0.87% |
| From scratch (synth) | 500×30 | 55.5% | 5.8885 | +1.11% |
| **Old numpy** | **4K** | **50.1%** | **5.8236** | **baseline** |

2-bit:
| Old (numpy) | 4K | 33.1% | 6.0158 | baseline |
| Fine-tuned (100K) | 50 | 34.1% | 5.9958 | -0.33% |

### Key insight: PPL is chaotic, not monotonic

Previous hypothesis "sharp cliff between 50 and 100 iterations" was WRONG — we only measured at 50 and 100. The full sweep reveals PPL oscillates wildly:
- 0-1 iters: bad (5.92) — coset init has no trellis benefit
- 3 iters: good (5.845) — first useful codebook
- 5 iters: slightly worse (5.858)
- 10-20 iters: CRASH to 5.94-5.97 — worse than 0 iters!
- 30 iters: recovery (5.873)
- 50 iters: best CUDA result (5.831)
- 100+ iters: degrades again

### Disproven hypotheses

**Small-batch regularization**: Hypothesis was that old numpy codebook (4K samples/iter) benefits from noisy gradients like SGD. DISPROVEN — CUDA trainer with 4K samples gives PPL 5.96 (terrible).

**CUDA trainer is broken**: Hypothesis was that something in the CUDA implementation (float32 precision, cuRAND distribution, Viterbi bug) produces systematically worse codebooks. DISPROVEN — wrote an equivalent numpy trainer, tested 4 seeds (7, 42, 123, 999) at 100 iterations each. All produce PPL 5.88-5.93, comparable to or worse than CUDA results. The implementation doesn't matter.

**More iterations always hurts (monotonic degradation)**: DISPROVEN by the full sweep. PPL oscillates chaotically: 3 iters is good (5.845), 10-20 iters crashes (5.94-5.97), 50 iters recovers (5.831). Not monotonic.

### Confirmed: The old codebook is a lucky local minimum

Tested 4 numpy seeds and multiple CUDA configurations — none replicate the old codebook's PPL:

| Impl | Seed | Iters | Samples/iter | MSE Red. | PPL | vs old |
|------|------|-------|-------------|----------|------|--------|
| **Old numpy** | **?** | **100** | **4K** | **50.1%** | **5.8236** | **—** |
| Numpy | 42 | 100 | 4K | 52.0% | 5.8979 | +1.28% |
| Numpy | 42 | 200 | 4K | 54.3% | 5.8914 | +1.16% |
| Numpy | 7 | 100 | 4K | 51.5% | 5.8801 | +0.97% |
| Numpy | 123 | 100 | 4K | 52.4% | 5.9300 | +1.83% |
| Numpy | 999 | 100 | 4K | 52.5% | 5.8853 | +1.06% |
| CUDA | 42 | 50 | 100K | 52.8% | 5.8313 | +0.13% |
| CUDA | 42 | 100 | 4K | 53.2% | 5.9600 | +2.34% |

The old codebook at PPL 5.8236 is ~0.06 PPL better than the best new seed (5.8801). All new seeds cluster in the 5.88-5.93 range. The old codebook is a significant outlier — either a 1-in-100+ lucky draw or trained with a slightly different algorithm we can't reproduce.

**Patching verified**: Loaded old numpy codebook from binary file, patched using the same regex method as all other tests → PPL 5.8236 (exact match). All measurements are valid.

### What we ruled out (tested and disproven)

The deeply-optimized codebook is **better on every metric we can measure on random data**:

| Metric | OLD | NEW | Winner |
|--------|-----|-----|--------|
| Element-wise MSE | 0.000174 | 0.000158 | NEW |
| Max absolute error | 0.0396 | 0.0367 | NEW |
| Error autocorrelation | -0.016 | -0.005 | NEW |
| Error kurtosis | 0.22 | 0.18 | NEW |
| Error skewness | 0.037 | -0.003 | NEW |
| Cosine similarity | 0.9888 | 0.9898 | NEW |
| Norm preservation | 0.9884 | 0.9900 | NEW |
| Dot product error | 0.0102 | 0.0098 | NEW |
| Dot product MSE | 0.000174 | 0.000151 | NEW |
| Attention KL div | 0.000016 | 0.000014 | NEW |
| Attention L1 | 0.00378 | 0.00365 | NEW |
| 95th pctile MSE | 0.000228 | 0.000200 | NEW |
| 99th pctile MSE | 0.000266 | 0.000229 | NEW |

Var(q·e) / (||q||² × MSE) ≈ 1.0 for both codebooks — errors are effectively uncorrelated.

**Every test on random synthetic data says NEW is better. Yet PPL is worse.**

### Structural differences between codebooks

- Old: 64/64 groups monotonically increasing (clean coset structure)
- 50-iter fine-tune: 62/64 monotonic
- Fully trained: 48/64 monotonic

### Other confirmed facts

- No synthetic-to-real gap: post-FWHT KV data is near-perfect Gaussian (σ=1/√128, kurtosis=-0.087)
- CUDA trainer matches numpy MSE quality at equivalent iterations, 280x faster
- The PPL measurements are deterministic (same result on re-runs)
- PPL is NOT monotonic with GLA iterations — it oscillates chaotically (see sweep above)

## Leading Hypothesis: Model-Specific Interaction

The tests above use **random Q and K vectors**. But in the actual model, Q and K are computed by learned weight matrices. After FWHT rotation, K is approximately Gaussian — but "approximately" hides per-layer, per-head structure that the model relies on.

**The old codebook's coset structure produces a specific error fingerprint that happens to be benign for this model's learned representations. The new codebook produces a different error fingerprint that happens to interfere with directions the model is sensitive to.**

This is undetectable by random-vector analysis because it depends on the specific learned weight matrices of Qwen3.5-27B. The FWHT rotation decorrelates the DATA but doesn't decorrelate the model's SENSITIVITY to errors.

This is analogous to why GPTQ uses the Hessian (H = X^T X) to weight quantization error — uniform MSE is the wrong objective because the loss landscape is anisotropic.

## Ideas to Test (Next Session)

### 1. Layer Sensitivity Scan (~80 min, no code changes)

For each of the 40 layers, test PPL with that layer's KV cache at degraded quality (e.g., turbo3 instead of turbo3_tcq, or a deliberately bad codebook for that layer only).

This identifies which layers are most sensitive to codebook choice. If 5 layers drive 90% of the regression, we focus optimization there.

**Implementation**: Modify SET_ROWS to accept a layer index and use layer-specific codebook. Or simpler: test with mixed -ctk/-ctv settings if the infrastructure supports per-layer type selection (it probably doesn't — may need code changes).

**Simpler version**: Use TURBO_EXTRACT to dump per-layer data, then for each layer compute MSE with old vs new codebook. If some layers show larger MSE differences, those might be the sensitive ones.

### 2. Per-Layer Codebook Implementation

Engineering is straightforward:

```cuda
// Global memory: per-layer codebooks
__device__ float d_layer_codebooks[40][512]; // 80KB — doesn't fit __constant__, use global

// In SET_ROWS kernel (512 threads = 1 entry each):
__shared__ float s_codebook[512];
s_codebook[threadIdx.x] = d_layer_codebooks[layer_id][threadIdx.x];
__syncthreads();
// Use s_codebook[state] everywhere instead of d_turbo3_tcq_codebook[state]

// In VEC attention kernel:
// Same approach — load layer's codebook into shared memory
```

Memory: 80KB for 40 layers × 512 entries. Exceeds 64KB __constant__ limit → use global memory + shared memory cache. The SET_ROWS kernel already has 512 threads, so loading is 1 entry/thread = zero overhead.

### 3. Per-Layer GLA Stopping Point

The clever optimization: instead of trying to find PPL-optimal codebooks (infeasible), **train each layer's codebook with a different number of GLA iterations**.

We know: 50 iters is globally safe, 100 is globally bad. But different layers might have different cliffs. Binary search each layer's cliff:
- Start all layers at 50 iters
- Try pushing one layer to 100 iters, measure PPL
- If PPL holds, push to 150; if not, try 75
- Repeat for each layer

Cost: ~40 layers × 4 binary search steps × 2 min/eval = ~5 hours. Feasible overnight.

### 4. Weighted MSE Training (GPTQ Analogy)

Instead of uniform MSE: minimize `Σ w_i × (x_i - x̂_i)²` where `w_i` reflects per-channel importance.

To estimate importance:
1. Run one inference pass on wikitext
2. For each attention head at each layer, compute Fisher information: how much does perturbing K[dim_i] change the attention output?
3. After FWHT, these importances transform: w_rotated = H^T × w_original × H (where H is the Hadamard matrix)
4. Use w_rotated as per-channel weights in the GLA training

If some rotated channels are 10x more important, the GLA would allocate more codebook precision to those channels, potentially finding codebooks that sacrifice MSE on unimportant channels to protect important ones.

**This is the most principled approach** — it's the TCQ analogue of GPTQ. But it requires infrastructure for per-channel importance estimation.

### 5. K vs V Isolation Test

Currently can't be done without code changes because SET_ROWS encodes both K and V with the same codebook. Need to:
1. Add a flag to SET_ROWS that selects codebook based on whether the destination is K or V
2. Test: old codebook for K encoding/decoding + new codebook for V encoding/decoding, and vice versa
3. This tells us whether the regression comes from K quantization, V quantization, or both

## Files and State (updated 2026-03-30 afternoon)

### Modified files (uncommitted on master):
- `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` — old numpy 3-bit codebook (the good one) + 50-iter 2-bit
- `ggml/src/ggml-cuda/fattn-common.cuh` — same codebooks (fattn copy)
- `scripts/tcq_train_cuda.cu` — added `--init` flag, fixed 0-iter support
- `benchmark-results.md` — full sweep data + numpy comparison

### Server state (root@dorei):
- Server build has OLD numpy codebook (PPL 5.8236) compiled
- `/tmp/old_codebook_3bit.bin`, `/tmp/old_codebook_2bit.bin` — original codebooks as binary files
- `/tmp/tcq_train_sweep` — CUDA trainer binary (supports --init, 0-iter)
- `/tmp/tcq_train_numpy.py` — numpy TCQ trainer (equivalent to CUDA)
- `/tmp/tcq_sweep_3bit_*.txt` — CUDA sweep outputs (0-30 iters)
- `/tmp/tcq_numpy_100.txt`, `/tmp/tcq_numpy_200.txt` — numpy seed=42 outputs
- `/tmp/tcq_numpy_seed{7,123,999}.txt` — numpy multi-seed outputs
- `/tmp/tcq_sweep_3bit_4k_100.txt` — CUDA small-batch output
- `/tmp/turbo_postrot.bin` — 512MB extracted post-FWHT KV data (1M vectors)

### Recommended next steps:
1. **Multi-seed CUDA search** — train 100+ seeds (seconds each), test PPL for top candidates. Cheapest way to find a codebook as good as or better than the old numpy.
2. **Investigate old codebook structure** — what structural property makes it special? Compare value distributions, spacing regularity, group statistics.
3. **Per-layer codebooks** — still viable regardless of seed issue
4. **Weighted MSE (GPTQ analogy)** — most principled, hardest

## For the Paper

The MSE-PPL divergence is itself a paper-worthy finding:
- Novel observation: TCQ codebooks optimized for element-wise MSE produce worse attention quality
- The relationship is NON-MONOTONIC: PPL oscillates chaotically as GLA iterations increase
- Codebook quality for attention is seed-dependent — different local minima have vastly different PPL (~0.1 PPL spread across seeds at similar MSE)
- Connects to the broader quantization literature on loss-aware vs MSE-aware optimization
- The coset structure acts as implicit regularization — parallels to how structured initialization helps in deep learning
- Practical recommendation: keep the codebook that works, don't optimize further without PPL-in-the-loop training
- Gap to Shannon bound (1.82 dB at 3-bit with old codebook) is competitive with QTIP (0.84 dB) despite much simpler trellis
- The gap between our MSE and the best-PPL codebook (50.1% vs achievable 55%) represents the cost of not having loss-aware training
