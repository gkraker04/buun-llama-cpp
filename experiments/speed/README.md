# Speed Experiments — buun-llama-cpp

This directory contains tooling for measuring and improving token generation speed on the buun-llama-cpp fork.

## Quick Start (Native Windows PowerShell)
```powershell
# 1. Build with experimental optimizations
cmd /c "build_speed.bat"

# 2. Run baseline benchmark (first time only)
.\bench.ps1 --baseline

# 3. Make changes to build config or model settings
#    Update experiments/speed/OPTIMIZATION_PLAN.md as you go

# 4. Rebuild and re-benchmark after each change
cmd /c "build_speed.bat" ; .\bench.ps1 --test "your-test-name"
```

## Files
| File | Purpose |
|------|---------|
| `bench.ps1` | Native PowerShell benchmark (tok/s measurement) |
| `server_control.sh` | Server utilities (PowerShell equivalent: start/stop via PID file) |
| `build_speed.bat` | Build configuration with RTX 3090 optimizations |
| `OPTIMIZATION_PLAN.md` | Master plan tracking progress and findings |
| `results/` | Benchmark results stored as JSON (auto-generated) |
| `logs/` | Server logs from benchmark runs (auto-generated) |

## Workflow
1. Establish baseline with current settings
2. Pick ONE optimization to test  
3. Apply the change
4. Rebuild if needed
5. Run benchmark against baseline
6. Document results and commit
7. Repeat

**Environment:** Native Windows PowerShell, no WSL required. All tools (MSVC, nvcc, Ninja) run natively.

See `OPTIMIZATION_PLAN.md` for the current roadmap.
