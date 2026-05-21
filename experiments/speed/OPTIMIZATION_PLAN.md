# Speed Optimization Plan — buun-llama-cpp (Native Windows)

**Last updated:** 2026-05-20

## Target Configuration

- **Model:** Ornstein3.6-27B-MTP-NSC-ACE-SABER Q4_K_M (multimodal, MoE, MTP heads)
- **Hardware:** RTX 3090 (24GB VRAM, sm_86), i7-8700K, 32GB DDR4
- **Priority:** Fastest decode tok/s at full context (262K)
- **Environment:** Native Windows (PowerShell + MSVC + CUDA 12.8 + Ninja)
- **Build target:** `86-real` (SASS, no PTX JIT)

---

## Current Best Results (May 2026 — 86-real binary)

### Winning Config: `batch=256, ubatch=128` -> **32.88 tok/s decode**

Full 18-config batch/ubatch sweep results (`sweep_results.csv`):

| batch | ubatch | decode tok/s | prompt tok/s | VRAM (MB) |
|-------|--------|-------------|-------------|-----------|
| 256   | **128** | **32.88** | 98.9 | 22,519 |
| 512   | 64     | 32.34       | 107.1 | 22,408 |
| 256   | 64     | 31.94       | 102.7 | 22,431 |
| 1024  | 512    | 32.21       | 101.8 | 23,111 |
| 2048  | 2048   | 30.47       | 95.5  | 23,985 |

**Key server flags:**
```
--batch-size 256 --ubatch-size 128 --flash-attn on
--cache-type-k turbo3_tcq --cache-type-v turbo2_tcq
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75
--ctx-size 262144 --fit off --parallel 1
```

### Established Facts

| Claim | Status | Evidence |
|-------|--------|----------|
| `86-real` gives ~5% over plain `86` | Verified | +5% on 18-config sweep |
| `p_min=0.75` is the floor | Confirmed | <0.75 produces gibberish draft tokens |
| `n_max=2` beats `n_max=3+` on RTX 3090 Q4_K_M | Confirmed | 3rd chain verify cost outweighs marginal acceptance (pre-in-graph MTP) |
| `ubatch=128` is the sweet spot | Confirmed | Wins or ties at every batch size tested |
| `turbo3_tcq`/`turbo2_tcq` KV frees VRAM | Confirmed | Freed enough VRAM for stable 262K context |
| In-graph MTP (from Buun, May 2026) | In master | `bf22e115e` - 1.77x bench reported; graph-level, no new flags needed |
| MoE MTP support | In master | `1c4788192` - critical for Ornstein (MoE model) |

---

## Available Levers (Untested / Partially Tested)

### Build-Level CMake Flags — **AUDITED: no wins here**

A full audit of every flag in `build_server.ps1` found **nothing useful**:

| Flag | Verdict |
|------|---------|
| `GGML_CUDA_FORCE_MMQ=ON` | **No-op on RTX 3090.** Ampere has Turing MMA -> batch size already 128. Only helps on Volta. |
| `GGML_CUDA_GRAPHS=ON` | **Already default.** `GGML_CUDA_GRAPHS_DEFAULT=ON` in CMakeLists. Redundant. |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE=128` | **Matches default + single GPU only.** Irrelevant. |
| `GGML_SCHED_MAX_COPIES=4` | **Matches default** (4 in ggml/CMakeLists.txt:188). Redundant. |
| `GGML_CUDA_MMV_Y=2` | **Does not exist** in this codebase. Fictional flag. |
| `GGML_CUDA_FORCE_DMMV` | **Does not exist** in this codebase. Fictional flag. |

**Conclusion:** Our current `test_build_server.bat` is already optimal for RTX 3090. No CMake flag changes needed.


### MTP Depth with In-Graph Improvements

The `n_max=2` finding was from **before** Buun's in-graph MTP changes. With graph-level optimizations, `n_max=3` may now be viable.

### Buun's Unmerged Experiment Branches

#### PFlash (SD-089) -- Highest Potential
Branch: `buun/experiment/SD-089-pflash`

FlashPrefill kernels + prompt compression for speculative decoding. Compresses prompt KV before passing to the drafter during prefill, reducing cross-attention work.

- **Files:** flashprefill.cu/cuh, pflash-graph.cpp, pflash-loader, pflash-score
- **Status:** Working, wired into server-context.cpp. Needs merge and testing.
- **Risk:** Moderate.

#### SD-086 Fused Chunked GDN -- Low Hanging Fruit
Branch: `buun/experiment/SD-086-fused-chunked-gdn`

Single-file change to `gated_delta_net.cu`: uses `exp2` SFU fast path for GDN gate decay. Ornstein uses these recurrent layers.

- **Files changed:** 1 (`ggml/src/ggml-cuda/gated_delta_net.cu`)
- **Risk:** Very low.
- **Effort:** Cherry-pick `e0ffe82bc`

#### SD-083 Decode SWA
Branch: `buun/experiment/SD-083-decode-swa`

Sliding window attention for decode + two-phase tape deferral. Reduces KV work during decode.

- **Risk:** Higher -- architectural change.
- **Prerequisite:** SD-084v2 (already in master).

### Upstream Sync

Buun has `rebase/upstream-sync-20260518`. We don't have an `upstream` remote configured. Worth checking for upstream perf fixes.

---

## Next Experiments (Priority Order)

### [ ] ~~1. Rebuild with extra CMake flags~~ — **AUDITED: all fictional/no-op. Skip.**

### [ ] 1. Test MTP depth=3 with in-graph optimizations
Change `--spec-draft-n-max 2` -> `--spec-draft-n-max 3` against current binary.

**Target:** >33 tok/s without quality regression

### [ ] 2. Cherry-pick SD-086 (fused GDN)
Single commit, single file. Rebuild and bench.

**Target:** Measurable improvement on MoE-heavy decode

### [ ] 3. Merge and test PFlash (SD-089)
Merge experiment branch, resolve conflicts, build, bench prefill-heavy workload.

**Target:** Faster prefill + TTFT without decode regression

### [ ] 4. Set up upstream remote + rebase
Add ggml-org/llama.cpp as `upstream`, check for relevant perf improvements.

**Target:** No regression; identify 1+ upstream improvement

---

## Experimental Record

| Date | Experiment | Config | Result | Verdict |
|------|-----------|--------|--------|---------|
| May 19 | Batch/ubatch sweep 18-config | 86-real, MTP p_min=0.75, turbo3/2 KV | Winner: 256/128 -> 32.88 tok/s | Production config |
| May 18 | Build arch test | 86 vs 86-real | 86-real wins ~5% | Adopted |
| May 17 | MTP n_max sweep | n_max=2,3,4 | n_max=2 wins (pre-in-graph MTP) | Revisit |

---

## Result Submissions

- **32.88 tok/s (batch=256, ubatch=128)** -- Not yet submitted to localmaxxing.com
- **22,519 MB VRAM** @ 262K context with turbo3/2 KV

---

*This plan replaces the previous version based on 86-real + batch/ubatch sweep results and Buun's in-graph MTP merge.*
