# ============================================================================
# Batch/Ubatch Sweep Script for buun-llama-cpp speed experiments
# Systematic matrix testing: sweep ubatch upward, double batch when matched
# Native Windows PowerShell — no WSL required
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RESULTS_DIR = "$SCRIPT_DIR\results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null
}

# Configuration for full context preservation
$CTX_SIZE = 262144  # Full context window
$SERVER_BIN = "G:\hermes\buun-llama-cpp\build\bin\llama-server.exe"
$MODEL = "G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF\Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf"
$MMPROJ = "G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF\mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"

# Test matrix
$BATCH_VALUES = @(256, 512, 1024, 2048)
$UBATCH_START = 64
$UBATCH_MULTIPLIER = 2  # Double each step
$MAX_UBATCH_PER_BATCH = $null  # Will be set dynamically per batch level

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Batch/Ubatch Sweep Test" -ForegroundColor Cyan  
Write-Host "Context size: $CTX_SIZE (--fit off)" -ForegroundColor Cyan
Write-Host "Starting sweep..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$RESULTS = @{}

function Start-SweepServer {
    param([int]$batch_size, [int]$ubatch_size)
    
    # Kill any existing test server
    try { Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue } catch {}
    Start-Sleep -Milliseconds 1000

    Write-Host "[SWEEP] Starting server with batch=$batch_size, ubatch=$ubatch_size..."
    
    $SERVER_ARGS = @(
        "--threads", "12",
        "--prio", "3",
        "--n-predict", "32768",
        "--ctx-size", "$CTX_SIZE",
        "--fit", "off",
        "--batch-size", "$batch_size", 
        "--ubatch-size", "$ubatch_size",
        "--flash-attn", "on",
        "--cache-type-k", "turbo4",
        "--cache-type-v", "turbo4",
        "--no-mmap",
        "--n-gpu-layers", "all",
        "--model", $MODEL,
        "--mmproj", $MMPROJ,
        "--no-mmproj-offload",
        "--offline",
        "--host", "0.0.0.0",
        "--port", "8081"
    )

    $START_INFO = Start-Process -FilePath $SERVER_BIN -ArgumentList $SERVER_ARGS -PassThru -WindowStyle Hidden
    $START_INFO.Id | Out-File "$RESULTS_DIR\sweep_server.pid"
    
    # Wait for server startup (max 30s)
    $COUNT = 0
    while ($count -lt 30) {
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:8081/health" -TimeoutSec 2
            Write-Host "[OK] Server ready in ${count}s"
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
            $count++
            if ($count % 5 -eq 0) { Write-Host "[SWEEP] Still starting... (${count}/30s)" }
        }
    }
    
    return $false
}

function Stop-SweepServer {
    try { 
        $PID = Get-Content "$RESULTS_DIR\sweep_server.pid"
        Stop-Process -Id $PID -Force -ErrorAction SilentlyContinue | Out-Null  
        Write-Host "[SWEEP] Server stopped (PID: $PID)"
    } catch { /* Ignore cleanup errors */ }
}

function Run-Benchmark {
    param([int]$batch_size, [int]$ubatch_size)
    
    Write-Host "[BENCH] Running benchmark for batch=$batch_size, ubatch=$ubatch_size..."
    
    $STOPWATCH = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        # Simple completion request for timing
        $RESPONSE = Invoke-RestMethod -Uri "http://localhost:8081/v1/completions" `
            -Headers @{"Authorization" = "Bearer dummythicc"; "Content-Type" = "application/json"} `
            -Body '{"prompt": "The capital of France is Paris. Continue:", "max_tokens": 512, "stream": false}'

        $TOKS_GENERATED = $response.usage.completion_tokens
        if ($null -eq $TOKS_GENERATED) { $TOKS_GENERATED = 0 }
    } catch {
        Write-Host "[ERR] Request failed: $_"
        return $null
    }
    
    $STOPWATCH.Stop()
    $ELAPSED_SEC = $STOPWATCH.Elapsed.TotalSeconds
    
    if ($TOKS_GENERATED -gt 0 -and $ELAPSED_SEC -gt 0) {
        $TOKS_PER_SEC = [Math]::Round($TOKS_GENERATED / $ELAPSED_SEC, 2)
        return @{tok_s=$TOKS_PER_SEC; tokens=$TOKS_GENERATED; time_s=$ELAPSED_SEC}
    }
    
    return $null
}

# Main sweep loop
$SWEEP_INDEX = 0
foreach ($BATCH_VAL in $BATCH_VALUES) {
    Write-Host "--- Sweep Level: batch=$BATCH_VAL ---" -ForegroundColor Yellow
    
    $UBATCH_VAL = $UBATCH_START
    while ($true) {
        $SWEEP_INDEX++
        $TEST_NAME = "batch${BATCH_VAL}_ubatch${UBATCH_VAL}"
        
        Write-Host "[TEST] #$SWEEP_INDEX : batch=$BATCH_VAL, ubatch=$UBATCH_VAL"
        
        if (Start-SweepServer -batch_size $BATCH_VAL -ubatch_size $UBATCH_VAL) {
            Start-Sleep -Milliseconds 1000  # Let server stabilize
            
            $RESULT = Run-Benchmark -batch_size $BATCH_VAL -ubatch_size $UBATCH_VAL
            
            if ($RESULT) {
                Write-Host "[OK] Result: ${RESULT.tok_s} tok/s (${RESULT.tokens} tokens in ${RESULT.time_s}s)"
                
                # Save individual result
                $RESULTS_OBJ = @{
                    sweep_index      = $SWEEP_INDEX
                    timestamp        = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
                    batch_size       = $BATCH_VAL
                    ubatch_size      = $UBATCH_VAL
                    ctx_size         = $CTX_SIZE
                    tok_s            = $RESULT.tok_s
                    tokens_generated = $RESULT.tokens
                    time_seconds     = $RESULT.time_s
                } | ConvertTo-Json
                
                $RESULTS_OBJ | Out-File "$RESULTS_DIR\$TEST_NAME.json" -Encoding utf8
                $RESULTS["$TEST_NAME"] = $RESULT
            } else {
                Write-Host "[WARN] Benchmark failed for this config"
            }
            
            Stop-SweepServer
        } else {
            Write-Host "[ERR] Server failed to start with batch=$BATCH_VAL, ubatch=$UBATCH_VAL — stopping sweep"
            break
        }
        
        # Double ubatch for next iteration
        $UBATCH_VAL = $UBATCH_VAL * $UBATCH_MULTIPLIER
        
        # Safety limit — don't go beyond reasonable VRAM expectations
        if ($UBATCH_VAL -gt 8192) {
            Write-Host "[SWEEP] Reached max ubatch safety limit (8192), moving to next batch level"
            break
        }
    }
}

# Summary report
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan  
Write-Host "Sweep Complete — Results Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

if ($RESULTS.Count -gt 0) {
    $BEST = $RESULTS.GetEnumerator() | Sort-Object { [double]$_.Value.tok_s } -Descending | Select-Object -First 1
    Write-Host "[OK] Best result: ${BEST.Key} → ${BEST.Value.tok_s} tok/s" -ForegroundColor Green
    
    # Save combined results
    $COMBINED = @()
    foreach ($KEY in $RESULTS.Keys) {
        $VAL = $RESULTS[$KEY]
        $OBJ = @{test=$KEY; tok_s=$VAL.tok_s; tokens_generated=$VAL.tokens}
        $COMBINED += $OBJ
    }
    
    $COMBINED_JSON = $COMBINED | ConvertTo-Json
    $COMBINED_JSON | Out-File "$RESULTS_DIR\sweep_combined_results.json" -Encoding utf8
    
    Write-Host "[OK] All results saved to results/sweep_combined_results.json" -ForegroundColor Green
} else {
    Write-Host "[WARN] No valid results collected during sweep" -ForegroundColor Yellow
}

Stop-SweepServer
Write-Host "[SWEEP] Done."
