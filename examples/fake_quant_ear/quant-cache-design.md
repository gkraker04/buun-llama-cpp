# Quant-Cache Design — llama-fake-quant-ear

Status: DESIGN (2026-08-12). Ordered work: (1) this doc, (2) commit, (3) determinism
proof, then implementation.

## 1. Problem

Algorithm-1 (SLQ arXiv 2605.02404, App A.1) measures EAR per (class, bitwidth) by
walking 10 random permutations per budget. Each permutation runs a 14-step plan
(`plan_for` in `_patches/ear_algo1.py`):

- s1: absolute recipe — `perm[0]` class → budget, all other 13 classes → Q8_0
- s2..s14: one switch per step — class `perm[j]` Q8_0 → budget

Each step re-quantizes the switched class's tensors from **pristine F16** into the
budget type, then runs one forward pass (~8 s). The requantize is the wall:

- Perm ≈ 20.6 min, of which ~50% is CPU-bound `ggml_quantize_chunk` on big
  tensors (token_embd / output: 276 s single-tensor steps at -t 6; the i-quant
  family is 3-5× slower than K-quants, up to 45 min/perm for IQ2_XS)
- The forward is cheap (~8 s); the model-state build is cheap (~15-30 s)

### Redundancy

The same class → same budget quantization is computed **once per step, per perm,
per budget** — but it is byte-deterministic (same pristine bytes + same target
type + same imatrix), so:

- Per budget: ~140 class-quantizations (10 perms × 14 steps) but only **14
  distinct** (class → budget) pairs, plus the Q8_0 state
- Q8_0: the s1 all-classes→Q8_0 quantization is the **same 13 class-quantizes
  in every perm of every budget** — one global set
- Total φ-table: ~1680 step-quantizations vs **~182 distinct** (12×14
  budget-specific + 14 global Q8_0) ≈ **~9× redundancy**

Real, measured costs (IQ4_XS_3, 1235 s): s1 ≈ 3.7 min (13 classes → Q8_0 +
perm[0] → budget), s2+s7 ≈ 4.9 min EACH (1 giant tensor), s3-s6/s8-s14 ≈ 10-50 s.
The big-tensor steps are pure encode; the search-based i-quants make it worse
(IQ2_XS: token_embd 13:11, output 13:07 per perm).

## 2. Cache design

### 2.1 Principle

`requantize_tensor` (fake_quant_ear.cpp:538-618) is a **pure function** of:

```
(pristine bytes, target type, imatrix row, tensor shape)
```

It reads pristine F16 → f32 → `ggml_quantize_chunk` (per expert plane, imatrix
advances by n_per_row) → validate → dequantize → write back into the live
tensor's buffer as the original type (F16/BF32/BF16). It does NOT depend on
model state, permutation order, or step index. So the **final write-back buffer**
(the exact bytes memcpy'd / fp16-converted into `e.t->data`) can be cached and
replayed: a hit is `memcpy(cached, e.t->data, e.nbytes)` — no quantize, no
dequantize, no f32 scratch.

### 2.2 Cache key

Per tensor:

```
key = hash( model_id | tensor_name | target_type | imatrix_id )
```

- `model_id` = the F16 source identity: path + file size (the tool loads one F16
  model per run; if the file changes, the pristine bytes change → cache invalid).
  Use the GGUF `general.name`/file size, or simply the resolved absolute path +
  `nbytes` of the file. Cheap, not cryptographic.
- `tensor_name` = `e.name` (e.g. `blk.7.ffn_up.weight`)
- `target_type` = the budget `ggml_type` (NOT `e.cur` — the cache stores the
  result of quantizing this tensor to this type, independent of live state)
- `imatrix_id` = identity of the imatrix file (path + size + mtime, or a fast
  hash of the file; 5.1 MB → SHA-1 is ~50 ms, fine). Per-tensor imatrix rows are
  a deterministic function of (imatrix file, tensor name, expected size)
  (`imatrix_data::lookup`, :456-475). Unweighted tensors (token_embd, blk.32)
  have no imatrix entry → `lookup` returns nullptr → `unit_imatrix` (constant
  1.0f, :584) is synthesized deterministically for the IQ hard-gate targets.
  Both cases are covered by the imatrix_id in the key.

Key granularity is per (tensor, target). Class-level (14 classes × budgets)
would be coarser but the tool works per-tensor anyway; per-tensor keys reuse
naturally across recipes (a class is 1-33 tensors).

### 2.3 Cache value

One file per (tensor, target):

```
<cache_dir>/<key>/<tensor_name>.<ggml_type_name>.bin
```

Content = the **final write-back buffer** for the live tensor: exactly
`e.nbytes` bytes as written by requantize_tensor's tail (L610-616: F16 →
`ggml_fp32_to_fp16_row`; BF16 → `ggml_fp32_to_bf16_row`; F32 → memcpy of the f32
buffer). A hit is literally `memcpy(cached, e.t->data, e.nbytes)` and update
`e.cur = target`.

File header (small, for validation; not part of the payload):

```
magic "QCE1" | key | tensor_name_len | tensor_name | target_type (u32) |
nbytes (u64) | ne[0..3] (4×i64) | payload(nbytes)
```

On hit, verify tensor_name, target_type, nbytes and ne match the live
`tensor_state`; any mismatch = miss (re-quantize + rewrite).

### 2.4 Cache directory layout / lifecycle

```
<cache_dir>/
  q8_0/      # global: 13 classes → Q8_0 (written once, reused by every budget)
  b_<budget>/  # per-budget: 14 classes → budget (written during that budget)
```

- The tool takes `--quant-cache DIR` (one flag; subdirs implicit).
- **Only one budget is live at a time** (the driver is sequential). Budget cache
  ≈ full model F16 ≈ 18.4 GB; Q8_0 ≈ 18.4 GB; live footprint ≈ 37 GB on disk.
  G: has ~189 GB free — fits with room. The driver deletes `b_<budget>` when the
  budget's 10 perms complete (sequential cleanup), keeping Q8_0 for the run.
  No cross-budget disk growth.
- Cache files live on the **G: SSD** (fast random reads; hits are one seek+read
  per tensor). Never on K:.
- Misses write the cache file AFTER a successful requantize (same transaction
  as the live write-back); a crash mid-write leaves a partial file → validate
  header + nbytes on read; discard partials.
- `--quant-cache` is an opt-in flag. Without it, behavior is byte-identical to
  today.

### 2.5 Implementation surface

- **`fake_quant_ear.cpp`** (~150 L):
  - `quant_cache` class: `open(dir)`, `get(key, tensor_state&) -> bool`,
    `put(key, tensor_state&)`, key derivation (path+size hash), header
    validation, thread-safe (called from the OpenMP requantize loop — use a
    per-file mutex or serialize hits/misses; the write-back into `e.t->data` is
    already per-tensor independent, so only cache-file I/O needs guarding).
  - `requantize_tensor`: at the top, after the restore-target check (L553-560
    restore branch is a memcpy already — no cache needed there), compute key;
    on hit: `memcpy(cached, e.t->data, e.nbytes)`, set `e.cur = target`, return
    true. On miss: existing path, then `put`.
  - CLI: `--quant-cache DIR` (default: off).
- **`_patches/ear_algo1.py`** (~30 L):
  - `tool_cmd` appends `--quant-cache` when the driver flag is set.
  - New driver flag `--quant-cache DIR` (default off; on = the whole φ-table
    run is cache-backed).
  - Pre-warm step (optional, see 2.6): one synthetic 14-step plan per budget
    (each step switches one class → budget, no forwards needed) that fills the
    cache before perm 0. Simpler alternative: let perm 0 fill it naturally.

### 2.6 Expected cost after cache

- Perms 1-9 of a budget: all 14 requantizes hit → perm = 14 forwards ≈ 14×8 s
  ≈ **~2 min** (vs ~20.6 min).
- Perm 0 of a budget: fills the 14 budget entries (≈ one pass of the big
  tensors ≈ 3-5 min) + forwards ≈ **~6 min**.
- First perm of the run (budget 0) also fills Q8_0 once (+3-4 min, amortized
  over all 12 budgets).
- φ-table total ≈ 12 budgets × (6 + 9×2) ≈ **~4.5 h** (vs ~2 d). The 14 global
  Q8_0 quantizes are paid once, not 120×.

Caveat: these are per-budget amortized numbers assuming the tool's measured
~8 s forward and the driver's measured per-class encode times. The i-quant
first-perm fill for IQ2_XS is the worst case (~45 min once per budget).

## 3. Validation plan (ordered)

### 3.1 Determinism proof (BEFORE any cache code — this is the gate)

Prove the cache's core assumption: two fresh quantizes of the same tensor (same
model file, same target, same imatrix) produce **byte-identical write-back
buffers**.

1. Add a tiny debug-only flag to the tool: `--dump-requant <tensor>=<type>:<out>`
   which runs the existing `requantize_tensor` and writes the final `e.t->data`
   buffer to `<out>` (this is exactly the cache value; the flag doubles as the
   reference writer).
2. Pick 3 tensors spanning the shapes/costs: one small (e.g. `blk.0.attn_q`
   ~1-2 MB), one mid (`blk.7.ffn_up` member), one giant (`token_embd.weight`,
   unweighted → exercises the unit-imatrix path), and 2 target types: one
   K-quant (Q4_K) + one i-quant with imatrix (IQ3_S).
3. Run each (tensor,type) pair twice, in **separate fresh processes** (the real
   use pattern), `cmp` the outputs byte-for-byte.
4. Also verify **-t 6 vs -t 12 byte-identity** (the OpenMP loop parallelizes
   over tensors, not within a tensor, so identity is expected — verify, don't
   assume; the existing -t protocol already established this for logits).

Gate: all `cmp`s clean, both thread counts. Any byte difference = the cache is
UNSAFE; stop, investigate (the existing smoke-determinism note says two runs
with identical inputs produced byte-identical 759 MB logits files — strong prior
that this passes).

### 3.2 Cache correctness proof (after implementation)

- Cache-hit vs cache-miss on a real perm: run one perm with `--quant-cache`
  empty (all misses) and again with a warm cache (all hits); **bit-identical**
  `s*.logits` files AND identical EAR per step.
- Cross-check against the existing determinism baseline: a warm-cache perm's
  logits must be byte-identical to a no-cache perm of the same plan.
- Mid-run crash recovery: kill the tool mid-perm with a warm cache; rerun the
  same perm; results identical (partial cache files discarded by header
  validation).

### 3.3 Regression gates

- `--quant-cache` OFF path: byte-identical to the pre-cache build on the same
  plan (the cache must not change anything when disabled).
- Cache file corruption: flip a byte in a `.bin`, rerun → header/nbytes
  validation must force a miss, not a bad memcpy.

## 4. Non-goals / risks

- **Not** a GPU quantize port (that's a multi-day lift; this is the cheap 9×
  win). No ggml core changes. No eval-path changes. No results-format changes.
- Cache size: live ≈ 37 GB on disk (one budget + Q8_0). Budget cleanup is the
  driver's job; if forgotten, G: fills after ~5 budgets (189 GB free). The
  driver MUST rm the previous budget dir before starting the next.
- RAM: cache hits avoid the f32 scratch + q buffer per tensor (~5.6 GB peak for
  token_embd). No additional RAM beyond the existing model load. Do NOT combine
  with `--keep-f16-copy` (measured zero gain; see algo1 driver notes).
- Determinism is the whole bet. Section 3.1 gates it. If the i-quant search is
  ever found non-deterministic (e.g. float reduction order), the cache silently
  returns stale bytes — the header validation can't catch a *wrong but valid*
  buffer; only the 3.1 proof + 3.2 bit-identity can. Keep both in CI-style
  checks for any future change to quantize paths or the tool build.
- Key safety: model_id + imatrix_id must be part of the key, not comments.
  If the imatrix file changes (regeneration, renormalization), stale entries
  would silently poison the table. When the φ-table run's imatrix is
  re-generated, the cache dir must be cleared (or keyed by imatrix hash).

## 5. Open questions

1. Pre-warm: synthetic 14-step fill vs let-perm-0-fill. Synthetic costs ~5 min
   once per budget and makes perm 0 fast too; natural fill needs zero extra
   code. Prefer synthetic if the tool flag is trivial (it is).
2. Cache dir identity for `model_id`: path+size vs embedding `general.name` —
   path+size is enough for a local single-model run; note it in the header.
3. Whether the driver should also cache the anchor (Q8_0 uniform) measurement —
   no: anchor is measured once per run, not per perm; not worth a cache entry.

## 6. Implementation status (2026-08-12, verified on the φ-table run)

- Commit: `3efafffb2` (`examples : fake-quant-ear on-disk quant-cache
  (--quant-cache DIR)`); branch `feature/fake-quant-ear-algo1`.
- §3.1 determinism proof: 4/4 PASS (token_embd@IQ3_S 2,034,237,440 B ×2 in
  fresh processes + ffn_gate@IQ3_S -t 6 vs -t 12 byte-identical).
- §3.2 warm-vs-cold: ssm_alpha recipe cold 21.4 s / warm 20.5 s, logits
  byte-identical; token_embd@IQ3_S cold **6m33s** / warm **22.6s** (cache read
  1.2 s for the 2.03 GB payload + forward), logits byte-identical.
- φ-table continuation (15 remaining perms, killed mid-run worker):
  - IQ3_S_6 (fill, first cache-backed perm): **1389s**, final_ear identical to
    uncached siblings 0.8660145998.
  - IQ3_S_9 (all-hit): **230s** vs ~1390s uncached = **6×**.
  - IQ4_XS_0/1/2 (uniform pre-filled cache): 234/230/230 s, identical 0.8980925679.
  - Q6_K_0 (fill after uniform pre-fill): 475s; Q6_K_1-9 all-hit: 227-256s,
    all identical 0.9315636754 = uniform Q6_K EAR.
  - Budget cleanup (rm `iq3_s`, `iq4_xs`, `q6_k`; keep `q8_0`) fired as designed.
  - Full table 120/120; all cache-backed perms bit-identical to uncached ones.
  - Pre-existing outliers (NOT cache artifacts): IQ2_XS_7, Q2_K_8 (~1e-5 EAR
    spread, from earlier pre-cache passes).
- Notes vs §2.6 predictions: all-hit perms measured 227-256s vs predicted
  ~120s (forward ~8s/step → ~14×8 + encode; the 4 min includes per-step model
  load + forward overhead; still 6× vs uncached). Fill perms 475-1389s (first
  cache-backed perm of a budget costs the full quantize; the uniform
  measurement pre-fills the budget cache when it runs before perms).


