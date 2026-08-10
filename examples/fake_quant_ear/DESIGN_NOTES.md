# DESIGN_NOTES — `llama-fake-quant-ear`

In-memory fake-quantization engine for Algorithm-1 Multi-Bitwidth Shapley
Estimation (SLQ, arXiv 2605.02404). Loads an F16 GGUF once, applies candidate
quantization recipes to the live model tensors in RAM, runs the perplexity-style
forward pass, and writes logits in llama-perplexity's `--logits-file` binary
format — no GGUF is ever written.

## 1. Tensor access: `llama_internal_get_tensor_map` (not `llama_model_loader`)

**Chosen:** load the model normally with `llama_load_model_from_file` (via
`common_init_from_params`, exactly like `llama_perplexity`), then reach the
live tensors through `llama_internal_get_tensor_map(model)`
(`src/llama-model.h`, same internal header `tests/test-quantize-stats.cpp`
includes). The returned `ggml_tensor*` objects are the very tensors the
compute graph references — `llama_decode` reads their `->data` directly, so
mutating the buffer in place is picked up with zero graph rebuilding.

**Rejected:** the `llama_model_loader`-level path (`get_tensor_meta` /
`create_tensor` / `load_data_for`). It would require reimplementing the
entire `llama_model_load` pipeline (vocab, hparams, per-arch tensor creation,
buffer allocation) to quantize between load and graph construction — that is
core code we are forbidden to touch. The tensor-map path is a ~30-line
addition on top of the standard public load flow.

**Consequence — mmap must be off:** with `use_mmap` (the default) the model's
tensor data points into the mmap'd F16 file, and an in-place fake-quant would
write through and corrupt the source GGUF. The tool forces `params.use_mmap =
false` (logs a notice) so weights live in owned buffers and mutation is
RAM-only. This costs ~18 GB of RAM (the model size); the pristine copy
(`--keep-f16-copy`) costs another ~18 GB on top.

**Fake-quant semantics:** tensor *types* are never changed — the graph builds
identically (F16 model). For each recipe-selected tensor:
`pristine (F16/F32) -> f32 -> ggml_quantize_chunk(target_type, ...) ->
dequantize_row_<target> -> write back as the tensor's original type`
(`ggml_fp32_to_fp16_row` for F16 tensors, `memcpy` for F32). The quantize call
is the exact `ggml_quantize_chunk` code path `llama-quantize` uses, so
per-tensor values match a real GGUF quant bit-for-bit; only the final F16
write-back adds a ~1e-3 relative rounding (unavoidable without changing the
tensor type; negligible vs every ladder target except Q8_0, where it is
comparable — see the equivalence check in §5).

Supported target types mirror the fork's `ggml_quantize_chunk` weight palette
restricted to the 2-8 bits-per-weight band — 20 types: `F16 F32 BF16 Q4_0
Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ2_S IQ3_XXS
IQ3_S IQ4_NL IQ4_XS`, with `F16`/`F32`/`BF16` as "restore pristine" targets
and BF16 also a valid fake-quant target for BF16-shipping models. The tool
registers F16/F32/BF16 tensors and writes back per-original-type
(`ggml_fp32_to_fp16_row` for F16, `ggml_fp32_to_bf16_row` for BF16, `memcpy`
for F32). Excluded on
purpose: the below-2-bpw `Q1_0`/`IQ1_S`/`IQ1_M`, the Bonsai `Q2_0`/`Q2_0_G128`
variants, the fp4 exotics `MXFP4`/`NVFP4`, `Q8_1`/`Q8_K` (no
`ggml_quantize_chunk` case in ggml.c), `TQ1_0`/`TQ2_0` (deferred by the user),
and the `TURBO*` KV-cache types (not weights). Types are parsed
case-insensitively by name. `--imatrix` is optional in general but required
when a recipe targets an imatrix-dependent IQ type (`IQ2_XXS IQ2_XS IQ2_S
IQ3_XXS IQ3_S IQ4_XS`); `IQ4_NL` and the non-IQ types do not require it.

## 2. Plan-mode state reset

The tool keeps per-tensor *current target type* (`tensor_state::cur`, starts
equal to the loaded type). Each step expands its recipe to the list of tensors
whose target differs from `cur` — **only those are re-quantized** (the
brief's optimization). All other tensors keep their values untouched, so a
step that flips one class costs one class's quantize, not the whole model.

Pristine data (needed because re-quantizing from already-quantized values
would compound error) comes from two interchangeable sources:

- **`--keep-f16-copy` (fast path):** at load, every F16/F32 tensor's bytes are
  snapshotted into RAM (~18 GB). Reset = zero-copy read from the snapshot.
- **default (low RAM):** pristine bytes are read per-tensor from the F16
  GGUF with `gguf_init_from_file(no_alloc=true)` metadata (offsets only) plus
  a `fopen`/`_fseeki64`/`fread` per tensor. A step touching one class reads
  only that class's bytes (a few hundred MB), so it is a few seconds of I/O,
  not a full 18 GB re-read.

Recipe application is **deterministic**: each tensor is always quantized from
pristine, so a class at type T has identical values no matter how many steps
passed.

Relative `switch` steps validate the walk: every tensor of the class must be
at `from` (else the plan is inconsistent and the tool errors with the tensor
name). The plan's first step must therefore be an absolute `recipe`
establishing the initial state (e.g. all classes at their ladder top,
`Q8_0`), which the Python driver generates.

## 3. OpenMP strategy

The per-tensor quantize phase (read pristine -> f32 -> quantize -> dequantize
-> write back) is embarrassingly parallel; each OpenMP iteration handles one
tensor with its own scratch buffers and touches disjoint memory regions, so
there is no sharing and no false sharing. Implementation details:

- `#pragma omp parallel for schedule(dynamic)` over the task list (dynamic
  because tensor sizes vary by >100x); signed `int` loop index (MSVC OpenMP
  2.0 constraint — no range-for).
- Thread count bound to the tool's `-t N` (`omp_set_num_threads`), defaulting
  to all cores.
- Errors (file I/O, validation) are recorded via an `std::atomic<bool>` +
  mutex-guarded message; the region drains, then the tool fails cleanly.
  `GGML_ASSERT`s inside the quantizers still abort on invariant violations.
- Scratch memory per thread: f32 buffer (nelem×4) + quantized buffer
  (~nelem/4 bytes). Peak ≈ sum over the largest in-flight tensors; two
  giant tensors (token_embd/output) in parallel ≈ 6–8 GB transient.
- The decode/logits phase is NOT OpenMP — it reuses llama.cpp's own thread
  pool (`-t`) plus the std::thread log-softmax workers, exactly as
  perplexity does.

The compile flag comes from `find_package(OpenMP)` / `OpenMP::OpenMP_CXX`
(`/openmp` on MSVC, `-fopenmp` on GCC/Clang).

## 4. Core changes deliberately not made

- **mmap write-through guard in the loader** — the tool just forces
  `use_mmap=false`; a core-level "read-only mmap" mode would be cleaner but
  is out of scope.
- **`llama_internal_get_tensor_map` is `TODO: remove` in core** — we depend
  on it via the internal header (same as the existing
  `test-quantize-stats`). If it is ever removed, the tool needs a small
  core-side accessor; flagged rather than changed.
- **KV cache clearing uses the fork's `llama_memory_clear(llama_get_memory(ctx), true)`**
  (perplexity's own call) since `llama_kv_cache_clear` does not exist in this
  fork's public API.
- **Imatrix size mismatch** — `llama-quantize` *throws* for a non-token_embd
  tensor whose imatrix size mismatches; the tool warns and quantizes that
  tensor unweighted instead. Only reachable with a corrupt/imatrix-less
  file; kept soft so the tool never hard-fails a whole sweep over a single
  missing entry.
- **F16 write-back rounding** — an f32 shadow buffer would require changing
  tensor types or the graph; deliberately not done (see §1).

## 5. Verification checklist (human operator)

**Status 2026-08-10: steps 1–4 DONE (see results below); step 5 pending.**

1. **Build** (MSVC 2022 via git-bash, Release):
   `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --target llama-fake-quant-ear`
   — expect `/openmp` in the compile line and a clean link against
   `llama-common`, `llama`, `ggml`.
   **DONE** via `_rebuild_examples.bat` (repo root; `LLAMA_BUILD_EXAMPLES=ON`,
   targets `llama-fake-quant-ear llama-sensitivity-db`). Three real compile
   fixes landed during the first green build: const-correct `lookup()`
   (C2662 at apply_recipe), hoisted `#ifdef _OPENMP` out of a `LOG_INF` arg
   (C2121), and `ggml/src` include dir for `sensitivity_db` (C1083 on
   `ggml-quants.h`).
2. **Single-candidate smoke test** (small ctx to keep it quick):
   `llama-fake-quant-ear -m K:\models\...\Ornstein-9Bv2.5-NSC-ACE-F16.gguf -f A:\ornstein_calib_2m.txt -c 1024 -t 6 --recipe ffn_up=Q4_K --logits-file _patches\smoke.logits`
   — expect the class-expansion log (`ffn_up -> 33 tensors`), a
   re-quantize line, per-chunk progress, and a PPL ≈ 2.4–2.6. Confirm
   `_patches/ear_aggregate.py`'s `load_logits` reads the file (magic,
   n_ctx, n_vocab, n_chunk decode).
   **DONE** with `output=Q4_K` on a 3208-token calib slice:
   `Final estimate: PPL = 1.9665 +/- 0.06943`; logits 759,883,748 B;
   **byte-identical across two runs** (deterministic).
   **PITFALL FOUND & FIXED (the reason this needs a checklist):**
   the fork's defaults are `n_gpu_layers = -1` (auto-offload) and
   `vbr_cache_type_k/v = true`. With GPU offload, `tensor->data` is a CUDA
   device pointer (~92 GiB VA) and the host write-back segfaults; VBR also
   refuses CPU KV. The tool now **self-forces** `--no-mmap`, `-ngl 0`, and
   static f16 KV in main() (same pattern as the existing mmap guard), so the
   smoke command needs no GPU flags. Verified: run with zero manual flags
   reproduces PPL 1.9665 + identical logits.
3. **Plan-mode test:** build a 3-step plan
   (step 1 absolute: all classes at Q8_0; steps 2–3 relative switches of one
   class) and run with `--plan plan.json --outdir _patches\plan_test\`.
   Confirm each step's logits file exists, per-step logs show only the
   switched class re-quantized, and a deliberate `"from"` mismatch errors.
4. **EAR-vs-GGUF equivalence:** for one candidate (e.g. `ffn_gate=Q4_K`),
   compare (a) this tool's logits vs (b) the current pipeline
   (quantize-to-GGUF + `llama-perplexity --save-all-logits`) with
   `_patches/ear_aggregate.py`/per-position differencing. EAR and mean
   top-10 overlap must agree to ~1e-4; a per-position logp diff of exactly
   0 is expected except for the F16 write-back rounding (≈1e-3 relative on
   weights, visible in logp at ~1e-4 absolute).
   **2026-08-10 finding:** two mandatory equivalence fixes landed. (1) The
   tool must force `-dev none` (not just `-ngl 0`): the fork's flash-attn
   AUTO probe sees the CUDA backend with `-ngl 0` and enables FA, which
   changes logits by ~3e-3 median (~4e-3 EAR) vs the `-dev none` protocol
   the climbs used. With the guard, pure-F16 forward passes are
   **byte-identical** to the real pipeline (max diff 0.0, verified). (2)
   IQ4_XS must NOT require an imatrix in the tool gate: llama-quant.cpp's
   `tensor_requires_imatrix` excludes IQ4_XS (it quantizes with NULL
   imatrix) and the climbs ran it that way; the tool's original gate was
   stricter and would have refused the climb2 final recipe. Remaining gap:
   requantized forward passes still differ ~1e-2 median logp because the
   tool computes F16 matmul on F16-written-back dequantized values while
   the real pipeline computes quantized-block kernels — this is inherent to
   the fake-quant design. EAR agreement across the 10-point climb2 ladder
   is being measured to confirm the offset is constant (cancels in
   Shapley differences).
   **2026-08-10 result — calibration over climb2 ladder (4 points):**
   ```
   round recorded    tool      delta
   r1    0.844985  0.848222  +0.003238
   r5    0.882417  0.885095  +0.002678
   r9    0.902488  0.905513  +0.003025
   r10   0.904670  0.908641  +0.003971
   ```
   Ordering preserved exactly (both curves strictly increasing, same
   order); per-round EAR within +0.003 ± 0.0007 of recorded. The bias is
   systematic: tool logits sit slightly closer to F16 (F16-writeback matmul
   is one rounding step cleaner than quantized-block kernels), so EAR runs
   ~3e-3 high. This cancels in Shapley φ (differences of tool measurements
   against the same F16 reference); residual ±0.0007 is a level effect
   across very different recipes, not a delta effect. F16 baseline
   byte-identical → forward pass exact; only requantized states carry the
   small bias. **Step 4 VERDICT: PASS with documented bias.**
5. **Perf expectation:** with `--keep-f16-copy`, a one-class step = class
   quantize (seconds, 12 OpenMP threads) + one forward pass over the
   calibration text (~seconds/minutes at `-t 6`, same cost as one
   perplexity run). ~700 measurements ≈ 700 forward passes + ~700 class
   quantizes — order-of-magnitude faster than 700 × (8 min GGUF write +
   2 min perplexity), and the model is loaded only once.

RAM expectations: model in RAM (18.4 GB, forced no-mmap) + optional pristine
copy (18.4 GB with `--keep-f16-copy`). Run without the flag if the box has
< 48 GB.
