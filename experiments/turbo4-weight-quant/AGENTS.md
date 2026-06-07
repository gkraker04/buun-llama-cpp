# TURBO4 EXPERIMENTS — 2026-06-05

## Current State — MMVQ working at 6.51 tok/s

The shuffle-optimized iWHT vec_dot is the current best path. MMVQ now produces correct output.
See `G:\hermes\buun-llama-cpp\AGENTS.md` for full status.

### Key Files
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\vecdot-turbo4.cuh` — Shuffle-optimized iWHT vec_dot (working, 6.51 tok/s)
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\mmvq.cu` — Clean MMVQ dispatch (WHT pre-rotation removed)
- `G:\hermes\buun-llama-cpp\ggml\src\ggml-cuda\ggml-cuda.cu` — Non-LIFO pool fix
- `test_start_mmvq.bat` — Port 8082, --no-warmup, f16 cache, flash-attn
- `test_start_webui.bat` — Same + --reasoning off + chat-template override (workaround)
- `bench_mmvq.py` — 128-token benchmark script

### Known Issues
- **Special token corruption**: `<|im_start|>` tokens (ID 248045) corrupt turbo4 quant output. `--parse-special` imatrix didn't fix. Codec-level problem.
- **Speed ceiling**: ~6.5 tok/s vs ~15-18 tok/s for Q4_K_M. Fused-iWHT quantizer → dp4a would hit ~31 tok/s.
- **Prefill**: Falls back to cuBLAS (3.7–4.2 tok/s). MMVQ signature-batched prefill not implemented.

### Server Scripts
| Script | Port | Notes |
|--------|------|-------|
| `test_start_server.bat` | 8082 | --api-key dummythicc, f16 cache |

### Models
| Version | File | Notes |
|---------|------|-------|
| v6-i3 | `Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf` | Working, ~6.5 tok/s |
| v7-i4 | `Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v7-i4.gguf` | --parse-special, same speed, didn't fix special tokens |
