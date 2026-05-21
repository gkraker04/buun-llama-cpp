# ============================================================================
# Speed Benchmark Script for buun-llama-cpp speed-experiments branch

# Load quality gate
. "$PSScriptRoot\Test-OutputQuality.ps1"

# Usage: .\bench.ps1 --baseline | .\bench.ps1 --test <name> [extra_server_args...]
# Native Windows PowerShell — no WSL required
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$REPO_ROOT = Resolve-Path "$SCRIPT_DIR\..\.."
$BUILD_DIR = "$REPO_ROOT\build\bin"
$SERVER = "$BUILD_DIR\llama-server.exe"

# Model configuration (matches start_buun_server.bat)
$MODEL_DIR = "G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF"
$MODEL = "$MODEL_DIR\Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf"
$MMPROJ = "$MODEL_DIR\mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"

# Output directory for results
$RESULTS_DIR = "$SCRIPT_DIR\results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null
}

# Default benchmark settings
$PROMPT = "The capital of France is Paris and the largest ocean is the Pacific Ocean. Write a short paragraph about:"
$MAX_TOKENS = 512
$WARMUP_REQUESTS = 2
$BENCHMARK_REQUESTS = 5

Write-Host "[BENCH] Running benchmark: $TEST_NAME"
Write-Host "[BENCH] Model: Ornstein3.6-27B-MTP Q4_K_M"
Write-Host "[BENCH] Tokens per request: $MAX_TOKENS"
Write-Host "[BENCH] Repeats: $BENCHMARK_REQUESTS"

$RESULTS = @()
$TOTAL_TOK = 0
$TIMES = @()

# Warmup requests
Write-Host "[BENCH] Warming up..."
for ($i = 1; $i -le $WARMUP_REQUESTS; $i++) {
    try {
        Invoke-RestMethod -Uri "http://localhost:8081/v1/completions" `
            -Headers @{"Authorization" = "Bearer dummythicc"; "Content-Type" = "application/json"} `
            -Body '{"prompt": "' + $PROMPT + '", "max_tokens": 64, "stream": false}' | Out-Null
    } catch {
        Write-Warning "[WARN] Warmup request $i failed: $_"
    }
}
Write-Host "[OK] Warmup complete"

# Actual benchmark
Write-Host "[BENCH] Starting timed requests..."
for ($i = 1; $i -le $BENCHMARK_REQUESTS; $i++) {
    $STOPWATCH = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $RESPONSE = Invoke-RestMethod -Uri "http://localhost:8081/v1/completions" `
            -Headers @{"Authorization" = "Bearer dummythicc"; "Content-Type" = "application/json"} `
            -Body '{"prompt": "' + $PROMPT + '", "max_tokens": ' + $MAX_TOKENS + ', "stream": false}'

        $GENERATED_TOKS = $response.usage.completion_tokens
        if ($null -eq $GENERATED_TOKS) { $GENERATED_TOKS = 0 }

        # Quality gate — reject gibberish (MTP collapse with low p_min)
        if ($GENERATED_TOKS -gt 0 -and $response.choices -and $response.choices[0].text) {
            if (-not (Test-OutputQuality -Text $response.choices[0].text)) {
                Write-Warning "[QUALITY] FAIL — output is gibberish (MTP collapse). Skipping this run."
                $GENERATED_TOKS = 0
            }
        }
    } catch {
        Write-Warning "[WARN] Request $i failed: $_"
        $GENERATED_TOKS = 0
    }

    $STOPWATCH.Stop()
    $ELAPSED_MS = $STOPWATCH.Elapsed.TotalSeconds

    if ($GENERATED_TOKS -gt 0 -and $ELAPSED_MS -gt 0) {
        $TOKS_PER_SEC = [Math]::Round($GENERATED_TOKS / $ELAPSED_MS, 2)
        $TIMES += $TOKS_PER_SEC
        $TOTAL_TOK += $GENERATED_TOKS
        Write-Host "[BENCH] Request $i : $TOKS_PER_SEC tok/s ($GENERATED_TOKS tokens in ${ELAPSED_MS}s)"
    } else {
        Write-Warning "[WARN] Request $i : Failed or returned no data"
    }

    Start-Sleep -Milliseconds 500  # Small delay between requests
}

# Calculate statistics
if ($TIMES.Count -gt 0) {
    $AVG_TOKS_PER_SEC = [Math]::Round(($TIMES | Measure-Object -Average).Average, 2)
    $MIN_TOKS_PER_SEC = ($TIMES | Sort-Object)[0]
    $MAX_TOKS_PER_SEC = ($TIMES | Sort-Object)[-1]

    # Calculate standard deviation
    $MEAN = $AVG_TOKS_PER_SEC
    $VARIANCE = (($TIMES | ForEach-Object { [Math]::Pow($_ - $MEAN, 2) }) | Measure-Object -Sum).Sum / $TIMES.Count
    $STDDEV = [Math]::Round([Math]::Sqrt($VARIANCE), 2)

    Write-Host ""
    Write-Host "================================================"
    Write-Host "[OK] Benchmark Complete: $TEST_NAME"
    Write-Host "------------------------------------------------"
    Write-Host ("  Requests completed:     {0,-6}" -f $BENCHMARK_REQUESTS)
    Write-Host ("  Average token/s:        {0}" -f $AVG_TOKS_PER_SEC)
    Write-Host ("  Min token/s:            {0}" -f $MIN_TOKS_PER_SEC)
    Write-Host ("  Max token/s:            {0}" -f $MAX_TOKS_PER_SEC)
    Write-Host ("  Std deviation:          {0}" -f $STDDEV)
    Write-Host "================================================"

    # Save results to JSON file
    $TIMESTAMP = Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz"
    $RESULT_FILE = "$RESULTS_DIR\$($TEST_NAME.Replace(' ', '_')).json"

    $RESULTS_OBJ = @{
        test_name          = $TEST_NAME
        timestamp          = $TIMESTAMP
        model              = "Ornstein3.6-27B-MTP-Q4_K_M"
        hardware           = "RTX 3090 (24GB)"
        avg_tok_s          = $AVG_TOKS_PER_SEC
        min_tok_s          = $MIN_TOKS_PER_SEC
        max_tok_s          = $MAX_TOKS_PER_SEC
        stddev             = $STDDEV
        requests_completed = $BENCHMARK_REQUESTS
        warmup_requests    = $WARMUP_REQUESTS
        tokens_per_request = $MAX_TOKENS
        individual_times   = $TIMES
    } | ConvertTo-Json

    $RESULTS_OBJ | Out-File -FilePath $RESULT_FILE -Encoding utf8
    Write-Host "[OK] Results saved to: $RESULT_FILE"

    # Compare against baseline if it exists
    $BASELINE_FILE = "$RESULTS_DIR\baseline.json"
    if (Test-Path $BASELINE_FILE) {
        try {
            $BASELINE_JSON = Get-Content $BASELINE_FILE -Raw | ConvertFrom-Json
            $BASELINE_AVG = $BASELINE_JSON.avg_tok_s

            if ($BASELINE_AVG -gt 0) {
                $DIFF_PCT = [Math]::Round((($AVG_TOKS_PER_SEC - $BASELINE_AVG) / $BASELINE_AVG) * 100, 2)
                Write-Host ""
                if ($DIFF_PCT -gt 0) {
                    Write-Host "[OK] Improvement over baseline: +${DIFF_PCT}%"
                } elseif ($DIFF_PCT -lt 0) {
                    Write-Warning "[WARN] Regression from baseline: ${DIFF_PCT}%"
                } else {
                    Write-Host "[INFO] No significant change from baseline"
                }
            }
        } catch {
            Write-Warning "[WARN] Could not compare against baseline: $_"
        }
    }
} else {
    Write-Error "[ERR] Benchmark failed - no valid results collected"
}

# Cleanup server process if started fresh
if (Test-Path "$RESULTS_DIR\server.pid") {
    try {
        $SERVER_PID = Get-Content "$RESULTS_DIR\server.pid"
        Stop-Process -Id $SERVER_PID -Force -ErrorAction SilentlyContinue | Out-Null
        Write-Host "[BENCH] Server stopped (PID: $SERVER_PID)"
    } catch {
        # Ignore cleanup errors
    }
}
