# turbo1_tcq baked-in codebooks (E7, k=1/L=8 trellis, 1.25 bpw)

Trained on the 14×4090 box (sm_89). Joint median KLD **0.050923** @ ctx8192/24ch on
Qwen3.6-27B-Q6_K vs f16-KV base, NO centering tap (uncentered post-FWHT corpus).

Ladder: anchor (untrained turbo2-seed) 0.0691 → old cq-way trained ref 0.0569 → **0.0509** (this).

- `best_K_s5_iter135.bin` — K encode/decode codebook (product_mono, seed 5, Lloyd iter 135)
- `best_V_s1_iter490.bin` — V encode/decode codebook (product_mono, seed 1, Lloyd iter 490)

Baked as `__constant__` literals into:
- encode: `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` (`d_turbo1_tcq_codebook` / `_v`)
- decode: `ggml/src/ggml-cuda/fattn.cu` (`d_turbo1_tcq_cb_fattn` / `_v_fattn`)
Encode/decode arrays are bit-identical; faithful to these .bin to 7.45e-9 (float32 %.8f ULP).
`TURBO1_TCQ_CB_K` / `TURBO1_TCQ_CB_V` env still override at runtime for future sweeps.

## Sweep method (deterministic KLD; teacher-forced, no sampling)
1. Harvested 312.5k unit-norm post-FWHT K/V records (TURBO_EXTRACT, no tap) → 156k K / 156k V.
2. k=1 CUDA trainer (product_mono: --mode qweights --constrain-monotonicity), unit-norm data AS-IS.
3. Seed-select: 5 K-seeds × 5 V-seeds @ iter200 → **seeds null** (all ~0.05217; converged monotone
   codebooks are near-identical encoders). Picked K-seed5 / V-seed1 (marginal).
4. Iter sweep (the real lever): coarse 25→550 then fine, **flat + non-monotone** (lottery).
   Best K=iter135 (robust across V), best V=iter490.
5. Joint grid confirmed K135×V490 = 0.050923 (K590 was a lucky V500-only pairing — rejected).
