# TURBO4 STATUS — 2026-06-06 (15:10)

**Branch:** `experiments/turbo4-quantize` (commit `473daefd4`)
**Build:** `build_turbo4.bat` — CUDA 13.2, Ninja, MSVC vcvars64, sm_86
**Models:** v6-i3 (current), v7-i4 (imatrix04), v8-hybrid (Q8_0 embed + turbo4 blocks)
**Server:** Port 8082, `test_start_mmvq.bat` (no MTP, f16 cache, --no-warmup)

══════════════════════════════════════════════
CURRENT STATE — MMVQ WORKING ✓
══════════════════════════════════════════════

**MMVQ vec_dot (decode, ne11=1):**
- ✅ **Correct output** — " Paris." for "The capital of France is"
- ✅ **6.51 tok/s** (128-token benchmark, 8-token prefill, 19.7s decode)
- ✅ **No CUDA graph hangs**, no deadlocks
- ✅ Uses `__shfl_xor_sync` with `__activemask()` for warp-level cooperation

**Approach:**
- Register-only iWHT on centroids (s2→butterfly→s1×inv_sqrt)
- Thread pair (2 threads per turbo4 block): each does 64-element butterfly passes 1-6, exchange for pass 7 via warp shuffle
- Float dot product with unrotated q8_1 activations
- No shared memory, no dp4a, no int8 approximation

**Prefill (ne11>1):** Falls back to cuBLAS (3.7–4.2 tok/s)

══════════════════════════════════════════════
SPECIAL TOKEN PROBLEM — 2026-06-05
══════════════════════════════════════════════

**Symptom:** Model generates garbage when `<|im_start|>` (token 248045) appears in prompt.
- Raw prompt "The capital of France is" → `" Paris."` ✓
- ChatML-format prompt with `<|im_start|>user\n...` → `"探索，\n\n..."` ✗ (Chinese gibberish in v6) or `"Search\nSearch..."` ✗ (gibberish in v7)
- Chat completions endpoint (adds special tokens via template) → `"Here is the step-by-step logical deduction..."` ✗ (wrong task entirely)

**Attempted fix:** Regenerated imatrix with `--parse-special` using chat-formatted calibration data (imatrix04.txt containing `<|im_start|>` tokens in context). Re-quantized to v7-i4. **Did not fix it.**

**Analysis:** Not a tokenization issue — `--parse-special` correctly tokenizes the special tokens during calibration. The problem is quantization error. Special tokens have embedding/attention patterns that are outliers relative to the centroid-based turbo4 codec. The quant cannot reconstruct them accurately enough, causing context loss.

**Current workarounds (band-aids, not fixes):**
1. Use `/completion` endpoint with raw prompts — works perfectly
2. Override chat template: `--chat-template "{{ messages[-1].content }}"` — strips special tokens, passes raw content (works but defeats chat format)
3. Avoid `<|im_start|>` tokens in prompt

**Real fix options (codec-level):**
1. Hybrid quant: Q8_0 for embedding/lm_head layers, turbo4 for transformer blocks — **IN PROGRESS (v8-hybrid)**
2. Per-group scale adjustments for outlier token patterns in the quantizer
3. Fused-iWHT quantizer — different approach entirely, might handle outliers differently

**Verification standard:** Output must match Q4_K_M exactly — no special hoops, no template overrides, no workarounds. The fix must make turbo4 a seamless, equally robust part of buun-llama-cpp. Chat templates, special tokens, reasoning mode — all must work without intervention.

══════════════════════════════════════════════
V8-HYBRID MODEL — 2026-06-06
══════════════════════════════════════════════

**Strategy:** Keep `token_embd.weight` and `output.weight` at Q8_0 to preserve special token embeddings. Quantize all transformer blocks with turbo4.

**Build:** `quantize_v8_hybrid.bat` with `--leave-output-tensor --token-embedding-type Q8_0 --imatrix imatrix04.gguf`

**Result:**
- `token_embd.weight` → Q8_0 (1,212.5 MiB) ✅
- `output.weight` → Q8_0 (1,212.5 MiB) ✅
- 504 transformer tensors → TURBO4_0 (type_id=43) ✅
- 360 F32 tensors (norms, biases, ssm_a, conv1d) → F32 ✅
- Total: 14.43 GiB (4.53 BPW)

**Status:** Model built, awaiting GPU availability for testing. imatrix06 generation currently holds GPU.

**Note:** Earlier tensor breakdown script had a bug in type-ID-to-name mapping that incorrectly reported "TURBO3_32". The actual GGUF contains TURBO4_0 (type_id=43) as intended. There is no `TURBO3_32` type in the codebase.

══════════════════════════════════════════════
IMATRIX06 GENERATION — IN PROGRESS
══════════════════════════════════════════════

**Dataset:** Curated balanced dataset for better calibration.
- Top 1,034 conversations from DJLougen parquet by `quality_score` ≥ 0.855 (1.91 MB, 33,745 lines)
- Combined with calibration_datav5.txt (1.64 MB) + groups_merged-enhancedV3.txt (0.27 MB)
- Total: 3.80 MB, 40,514 lines, shuffled with seed=42
- Output: `imatrix06_curated.txt`

**Process:** Running via `generate_imatrix06.bat` (PID 3616)
- Source: Q8_0-MTP model
- Flags: `--process-output --parse-special`
- Target: 1,778 chunks, ETA ~4h from start (14:15)
- Progress: imatrix06.gguf at 14 MB (active, last write 15:08)
- Log: `gen_im6.txt` (tee-buffered, shows chunk 92 but process is further along)

**Purpose:** Better calibration data for future quantizations. Not required for v8-hybrid test (which uses imatrix04).

══════════════════════════════════════════════
HISTORY
══════════════════════════════════════════════

**Stage 1 — The bug (WHT pre-rotation) ✗**
- Applied WHT forward to activations, then dp4a with unrotated int8 centroids
- ~31 tok/s but **completely garbage output** (HTTP 500)
- Root cause: WHT forward and inverse are NOT the same (different s1/s2 sign application)
- WHT_fwd: s1→butterfly→s2×inv_sqrt. WHT_inv: s2→butterfly→s1×inv_sqrt.
- Sign pattern mismatch makes M_ij ≠ M_ji, so Parseval doesn't apply in the simple form

**Stage 2 — Naive float iWHT ✓ (slow)**
- Each thread independently computed ALL 128 iWHT values, then dotted with its 64-element half
- Correct but 1.24 tok/s — each thread did 7-pass 128-element butterfly + 64-element dot

**Stage 3 — Shuffle-optimized ✓ (current)**
- Thread pair: 64-element butterfly each, exchange for pass 7 via __shfl_xor_sync
- 6.51 tok/s — correct, ~5× faster than naive float
- __activemask() ensures deadlock-free across varying iteration counts

**Stage 4 — v7 re-quant with --parse-special ✓ (model exists, didn't fix)**
- New imatrix (imatrix04.gguf) from chat-formatted calibration data with --parse-special
- New model: Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v7-i4.gguf
- Special token corruption persists — codec-level issue, not calibration issue

══════════════════════════════════════════════
PERFORMANCE
══════════════════════════════════════════════

**Decode (MMVQ vec_dot, ne11=1):**
| Variant | Speed | Correct? | Notes |
|---------|-------|----------|-------|
| cuBLAS fallback | ~0.46 tok/s | ✅ | Full dequant→fp16→cublasGemmEx |
| q8_1 vec_dot | ~1.09 tok/s | ✅ | Centroid LUT × norm × q8_1 |
| WHT pre-rot (broken) | ~31 tok/s | ❌ | Wrong sign convention |
| Naive float iWHT | 1.24 tok/s | ✅ | Stage 2: full 128-element per thread |
| Shuffle iWHT | **6.51 tok/s** | ✅ | Stage 3: current best |

**Prefill (cuBLAS fallback, ne11>1):** 3.7–4.2 tok/s
**Q4_K_M baseline (reference):** ~15-18 tok/s decode

══════════════════════════════════════════════
ROOT CAUSE ANALYSIS
══════════════════════════════════════════════

**Why WHT pre-rotation produced garbage:**
The WHT forward and inverse transforms differ in their sign application order:
- Forward: s1 → butterfly → s2 × inv_sqrt
- Inverse: s2 → butterfly → s1 × inv_sqrt

The matrix M_ij = s2[i] × H_ij × s1[j] is NOT symmetric because s1[i] ≠ s2[i]
for many elements. Therefore Σ WHT_inv(C) · A ≠ Σ C · WHT_fwd(A).
The sign mismatch causes element-level corruption.

**Why `<|im_start|>` tokens corrupt output:**
Special tokens (IDs 248044–248046) have embedding patterns that fall into
quantization error gaps in the turbo4 centroid-codebook. The centroid
reconstruction error compounds through attention, causing context loss.
This is a codec-level limitation, not fixable via imatrix calibration alone.

══════════════════════════════════════════════
OUTLOOK
══════════════════════════════════════════════

**Current speed:** 6.51 tok/s decode. Usable but ~2.5× slower than Q4_K_M.

**The bottleneck:** ~1152 float ops per vec_dot call (WHT butterfly + dot product)
vs ~4 int ops for Q4_K_M's dp4a path.

**Priority paths:**
1. Test v8-hybrid for special token fix — **NEXT**
2. Fuse iWHT into quantizer → dp4a vec_dot → ~31 tok/s (biggest speed win)
3. Hybrid quant: Q8_0 for embedding/lm_head, turbo4 for transformer blocks (fixes special token correctness without abandoning turbo4)
4. Dequant turbo4→fp16 at layer load, use fp16 MMVQ (simple, drop-in)
5. Deep-dive buun's uploaded files: notes, scratch files, experiments in fix/ and experiments/ branches — may reveal turbo quant design intent and optimization insights

══════════════════════════════════════════════
KANBAN BOARD — ACTIVE
══════════════════════════════════════════════

**Profiles:**
- `coder` — qwen3.7-plus on opencode-go-anthropic. Edits CUDA/C, builds, tests, commits+pushes.
- `researcher` — deepseek-v4-flash on opencode-go-anthropic. Reads codebase, analyzes, proposes solutions. No code edits.

**Tasks:**
1. Test v8-hybrid for special token fix (coder, blocked on GPU)
2. Deep dive buun's fix/* and experiments/* branches for notes/scratch files (researcher)
3. Analyze special token root cause at codec level (researcher)
4. Implement codec-level fix if v8-hybrid doesn't work (coder, depends on #3)
5. Fuse iWHT into quantizer for dp4a vec_dot speed path (coder)

**Commit policy:** Every working fix committed and pushed immediately to experiments/turbo4-quantize.

══════════════════════════════════════════════
MODIFIED FILES
══════════════════════════════════════════════

- `ggml/src/ggml-cuda/vecdot-turbo4.cuh` — Shuffle-optimized iWHT vec_dot
- `ggml/src/ggml-cuda/mmvq.cu` — WHT pre-rotation removed, clean dispatch
- `ggml/src/ggml-cuda/ggml-cuda.cu` — Non-LIFO pool fix
- `experiments/turbo4-weight-quant/test_start_mmvq.bat` — Added --no-warmup
- `experiments/turbo4-weight-quant/test_start_webui.bat` — Web UI server (--reasoning off, chat-template override)
- `experiments/turbo4-weight-quant/bench_mmvq.py` — Benchmark script
- `experiments/turbo4-weight-quant/test_start_webui.bat` — Working config: --reasoning off --jinja --chat-template override

══════════════════════════════════════════════
WEB UI FIX (BAND-AID)
══════════════════════════════════════════════

Built-in llama.cpp web UI showed garbage in "Reasoning" block because the
chat template adds `<|im_start|>` tokens. Two workarounds exist:

1. Template override (strips special tokens entirely):
   `--jinja --chat-template "{{ messages[-1].content }}" --reasoning off`
   → `/v1/chat/completions` returns correct text via raw content pass-through

2. Use `/completion` endpoint directly (no template involved):
   → Always works, no special handling needed

Neither fixes the root cause — the model should handle special tokens
correctly. See SPECIAL TOKEN PROBLEM section above.

══════════════════════════════════════════════
MODEL CATALOG
══════════════════════════════════════════════

| Version | File | Imatrix | Embed Type | Special Tokens | Notes |
|---------|------|---------|------------|---------------|-------|
| v6-i3 | v6-i3.gguf | imatrix01.gguf (DJLougen, raw text) | turbo4 | ❌ broken | Current working, ~6.5 tok/s |
| v7-i4 | v7-i4.gguf | imatrix04.gguf (chat-format, --parse-special) | turbo4 | ❌ still broken | Didn't fix — codec issue |
| v8-hybrid | v8-hybrid.gguf | imatrix04.gguf | Q8_0 embed + turbo4 blocks | ❓ untested | Awaiting GPU for test |

══════════════════════════════════════════════
MEMORY (personal notes)
══════════════════════════════════════════════
- MMVQ vec_dot working correctly at 6.51 tok/s with shuffle-optimized iWHT
- WHT pre-rotation was fundamentally wrong (s1≠s2 sign breaks Parseval)
- Always match CPU dequant: iWHT on centroids (s2→butterfly→s1), not WHT on activations
- __shfl_xor_sync with __activemask() for deadlock-free warp cooperation
- --no-warmup needed for testing (float iWHT is slow — warmup takes >2min)
- test_start_mmvq.bat at port 8082, api-key dummythicc, flash-attn on, f16 cache
- build_turbo4.bat for builds, _just_ninja.bat for quick iter
- Current branch: experiments/turbo4-quantize on buun/master, commit 473daefd4
- Speed comparison: 31 tok/s dp4a (broken) vs 6.5 tok/s float iWHT (correct)
- ~1152 float ops per turbo4 block per vec_dot call. Bottleneck is compute, not bandwidth
- Fused iWHT quantizer → dp4a vec_dot → ~31 tok/s (needs new quantizer + model)
- Web UI: use --jinja --chat-template override or /completion endpoint to work around special token corruption
- v7-i4 model exists but --parse-special imatrix didn't fix <|im_start|> token corruption
- Codec-level fix needed: hybrid quant (Q8_0 for embed/lm_head) or per-group outlier handling
- v8-hybrid built 2026-06-06: token_embd + output at Q8_0, 504 transformer tensors at TURBO4_0, 14.43 GiB
- imatrix06 generation in progress (PID 3616), curated dataset 40,514 lines, quality-filtered top 1,034 conversations

══════════════════════════════════════════════
MEMORY (your personal notes) [94% — 3,852/4,096 chars]
══════════════════════════════════════════════
icanplaytoo (gkraker04) is technical, detail-obsessed, hands-on operator who builds/tests independently then delegates automation. Values verified claims over guesses, hates shell quoting bugs, silent process deaths, and performative productivity. Prefers CUDA 13.2 for builds (MSVC 19.31 incompatible with CUDA 12.8 — "Host compiler targets unsupported OS"). Clean directory names. Publishes benchmarks on localmaxxing.com. Corrects MTP vs DFlash distinctions, budget-message formatting (leading '. ' matters), and command snippets must be complete. Will call out incorrect assumptions immediately. LOW TOLERANCE for wasted token budget — if a command fails, read the error and fix it immediately rather than retrying blindly.
§
Model load on RTX 3090 takes ~36s for 27B. MTP p_min: 0.3-0.5 sweet spot. Working config: `-c 4096` (262K default exhausts VRAM during graph reservation). `--fit` fits weights but does NOT auto-shrink context.
§
GPU power limit: RTX 3090 capped at 250W in production (stock 350W). Uncapped: ~50% faster decode, ~20-25% faster prompt processing sub-16K. MTP acceptance unchanged by power state. All benchmarks assume 250W cap.
§
Strong preference for building on buun's native turbo codec stack (turbo2/3/4 types) rather than porting external formats like TheTom's TQ types. Wants plans that are thorough enough for weaker local models to execute — writes implementation plans to files that can be pointed at an LLM. Conscious about using his limited "good model requests" wisely. Prefers direct, technical communication with verification. Understands codec details (WHT rotation, block structures, bpw calculations).
§
Branch naming convention: use `experiments/<topic>` prefix for experiment branches (e.g. `experiments/turbo4-quantize`, `experiments/speed`), not flat names. User cares about public GitHub/HF presentation — clean, self-explanatory branches.
§
Demands factual accuracy in documentation — called out bogus "~4x smaller than Q4_K_M" claim. Both are 4-bit, both ~16 GB for 27B. Verify numbers, don't exaggerate.
§
CRITICAL: NEVER use taskkill /F to kill CUDA GPU processes (llama-server). Force-killing corrupts GPU driver state and BSODs Windows. Only safe: taskkill /PID <pid> (no /F sends WM_CLOSE for graceful cleanup). HOWEVER: WM_CLOSE only works on GUI apps, not console-mode servers. The server's own /shutdown API endpoint is the ONLY reliable graceful shutdown. For stuck/dead console servers, the only recovery is a reboot. Build tools (nvcc/cl.exe/ninja) don't hold GPU contexts so /F is safe there.
§
G: drive (463 GB) only ~11 GB free — insufficient for turbo4 quant output (~15.2 GB for 27B). llama-quantize.exe uses mmap I/O and silently truncates when disk fills — exit code 0, no error. File has correct apparent size up to available space but zeros at offset 0. Always verify with `xxd | head -1` (should see 'GGUF' magic). Use A:\models for large file output (1.5 TB free, HF-style structure like G:\models).
§
`build_turbo4.bat` is the ONLY build script to use (`G:\hermes\buun-llama-cpp\build_turbo4.bat`). If it fails, stop and ask the user to look (sometimes the hermes folder gets read-only, which they clear manually). Do NOT try workarounds or other build scripts. The taskkill in build_turbo4.bat is for ONE failed build cleanup — not spammed. Check `build/bin/` for existing binaries before attempting a rebuild.
§
Open WebUI is the primary chat UI; port 11080 Docker backend. /completion works; /v1/chat/completions returns 500 with reasoning_format=AUTO. Misroutes output into reasoning_content (raw floats displayed).
§
Turbo4 special-token corruption: `<|im_start|>` tokens cause garbage. `--parse-special` imatrix doesn't fix it (Jun 2026). `--chat-template` override is diagnostic only — user wants root-cause fixes. `/completion` endpoint works; chat templates trigger the bug. `--reasoning off` keeps output in content field.

══════════════════════════════════════════════
USER PROFILE (who the user is) [94% — 1,941/2,048 chars]
══════════════════════════════════════════════
User wants iterative speed optimization: implement → push → benchmark → iterate. "If it finds any speed it should push and keep looking for speed." Detail-obsessed about system state — watches for zombie processes, overlapping builds. Expects you to track what's actually running. Pushes early wins not perfection.
§
Designated build script is G:\hermes\buun-llama-cpp\build_turbo4.bat — USE IT. Never ad-hoc cmake/ninja commands. Server script is experiments\turbo4-weight-quant\test_start_server.bat. CRITICAL: build_turbo4.bat's zombie-cleanup step (taskkill /F /IM nvcc.exe) kills OUR OWN active builds when stacked. The script also had rmdir /s/q build which forces a 90min+ full rebuild — do NOT add this back. For quick iterative builds without interference: use _just_ninja.bat (MSVC env + ninja, no zombie-kill step).
§
Expects quantize type names to match KV cache parameter conventions — wants both full names (TURBO4_0) and short aliases (turbo4) available. Prefers iterative workflow: edit source first, test/review, then rebuild. Understands compiled-in defaults approach for codebooks (no env vars needed for common case). Hands-on operator who runs commands directly.
§
Turbo4 quantize: G:\models\GestaltLabs\ for source models (F16-MTP), G:\models\gkraker04\ for imatrix/config. Naming: Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v<i>-i3.gguf. Quantize command: --leave-output-tensor --imatrix <imatrix.gguf> --override-kv general.name=str:"<name>" <input.gguf> <output.gguf> turbo4. Use build/bin/llama-quantize.exe (build/ is canonical dir). Pre-computed imatrix at G:\models\gkraker04\imatrix01.gguf (496 entries, DJLougen data). Brain (~14.6GB) + test (~14.6GB) can't coexist on 24GB. Buun MTP n-max ≤ 2, p_min must be 0.0. Proactive branch mgmt/docs.
§
Hates band-aid workarounds — fix root causes. "Always just run it" — execute non-destructive tasks without asking, keep a log. D: drive ccache folder is expendable.
