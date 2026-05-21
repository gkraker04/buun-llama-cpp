# Per-Request Speculative Decoding Parameters

**Discovery date:** 2026-05-20
**Source:** `tools/server/server-task.cpp` lines 300-305

## The Find

The server reads speculative decoding parameters from the **chat completion request body**
as flat JSON keys — NOT server startup flags:

```cpp
params.speculative.draft.n_min = json_value(data, "speculative.n_min", defaults.speculative.draft.n_min);
params.speculative.draft.n_max = json_value(data, "speculative.n_max", defaults.speculative.draft.n_max);
params.speculative.draft.p_min = json_value(data, "speculative.p_min", defaults.speculative.draft.p_min);
```

The `json_value()` helper does a flat `body.at(key)` lookup — no dot-notation traversal.
So you pass these as flat keys in the request body:

```json
{
    "model": "qwen",
    "messages": [...],
    "max_tokens": 256,
    "speculative.n_min": 2,
    "speculative.n_max": 8,
    "speculative.p_min": 0.75
}
```

## What This Enables

- **Sweep all 136 (n_min=1..16, n_max=n_min..16) combos** in a single server start
- No server restarts between configs (saves ~40s of model loading each)
- Full 136-config sweep takes ~27 minutes instead of ~2 hours
- p_min can also be swept per-request, same mechanism

## Flag Reference

| Flag | Default | Per-request? | Request Key |
|------|---------|--------------|-------------|
| `--spec-draft-n-max` | 16 | ✅ Yes | `speculative.n_max` |
| `--spec-draft-n-min` | 0 | ✅ Yes | `speculative.n_min` |
| `--spec-draft-p-min` | 0.75 | ✅ Yes | `speculative.p_min` |
| `--spec-type` | none | ✅ Yes | `speculative.type` |
| `--spec-ngram-size-n` | - | ✅ Yes | `speculative.ngram_size_n` |
| `--spec-ngram-size-m` | - | ✅ Yes | `speculative.ngram_size_m` |
| `--spec-ngram-min-hits` | - | ✅ Yes | `speculative.ngram_m_hits` |

## Script

`dflash_nmax_sweep.ps1` performs a full 136-config (n_min, n_max) sweep with p_min=0.75
using per-request params. Run from PowerShell:

```powershell
.\experiments\speed\dflash_nmax_sweep.ps1
```

Results go to `experiments/speed/results/dflash_nmax_sweep_p75.csv`
