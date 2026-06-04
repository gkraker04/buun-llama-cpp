param(
    [int]$TestPort = 8082,
    [string]$ModelPath = "G:\models\gkraker04\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-GGUF\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v5-i3.gguf",
    [string]$ServerPath = "G:\hermes\buun-llama-cpp\build\bin\llama-server.exe",
    [string]$LogDir = "G:\models\gkraker04\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-GGUF\logs",
    [string]$JinjaPath = "G:\models\spiritbuun\Qwen3.6-chat_template\chat_template.jinja"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $LogDir "turbo4_test_${timestamp}.log"
$responseFile = Join-Path $LogDir "turbo4_response.txt"

# GLOBAL TIMEOUT: kill everything after 2 minutes
$timeoutSeconds = 120
$deadline = [datetime]::Now.AddSeconds($timeoutSeconds)

function Check-Timeout {
    if ([datetime]::Now -gt $deadline) {
        Write-Host "TIMEOUT after ${timeoutSeconds}s - killing all servers"
        Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
        # Restore brain server even on timeout
        Start-Process -FilePath "G:\hermes\start_llama_server.bat"
        exit 1
    }
}

function Kill-All-Servers {
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
}

Write-Host "========================================"
Write-Host "TURBO4 TEST HARNESS (${timeoutSeconds}s timeout)"
Write-Host "========================================"

# Step 1: Kill existing servers
Write-Host "[1/5] Killing existing servers..."
Kill-All-Servers
Check-Timeout

# Step 2: Start turbo4 test server
Write-Host "[2/5] Starting turbo4 test server on port $TestPort..."
$serverProcess = Start-Process -FilePath $ServerPath -ArgumentList @(
    "--threads", "12",
    "--ctx-size", "2048",
    "--n-predict", "32768",
    "--batch-size", "256",
    "--ubatch-size", "64",
    "--flash-attn", "off",
    "--cache-type-k", "f16",
    "--cache-type-v", "f16",
    "--n-gpu-layers", "all",
    "--fit-target", "0",
    "--model", $ModelPath,
    "--log-file", $logFile,
    "--offline",
    "--mmap",
    "--log-verbosity", "4",
    "--log-prefix",
    "--log-timestamps",
    "--cache-type-k-draft", "f16",
    "--cache-type-v-draft", "f16",
    "--temperature", "0.6",
    "--top-k", "20",
    "--top-p", "0.95",
    "--min-p", "0.01",
    "--repeat-penalty", "1.0",
    "--presence-penalty", "1.5",
    "--ctx-checkpoints", "0",
    "--cache-ram", "0",
    "--kv-unified",
    "--parallel", "1",
    "--alias", "Qwen3.6-27B-turbo4-test",
    "--host", "0.0.0.0",
    "--port", $TestPort,
    "--api-key", "dummythicc",
    "--props",
    "--ubatch-size", "2",  # Force multi-token batch to hit cuBLAS path
    "--spec-type", "mtp",
    "--spec-draft-n-max", "8",
    "--spec-draft-p-min", "0.3",
    "--jinja",
    "--chat-template-file", $JinjaPath,
    "--no-warmup"
) -PassThru

Write-Host "Server process started with PID: $($serverProcess.Id)"
Check-Timeout

# Step 3: Wait for server to be ready
Write-Host "[3/5] Waiting for server to load (max 60 seconds)..."
$maxWait = 60
$waited = 0
$ready = $false

while ($waited -lt $maxWait) {
    Check-Timeout
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${TestPort}/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "Server is ready after ${waited} seconds!"
            $ready = $true
            break
        }
    } catch {
        # Server not ready yet, wait and try again
    }
    Start-Sleep -Seconds 5
    $waited += 5
}

if (-not $ready) {
    Write-Host "Server failed to start within ${maxWait} seconds."
    if (Test-Path $logFile) {
        Get-Content $logFile | Add-Content -Path $responseFile
    }
    goto :RestoreBrain
}

# Step 4: Send test prompt
Write-Host "[4/5] Sending test prompt..."
$body = @{
    model = "Qwen3.6-27B-turbo4-test"
    messages = @(
        @{ role = "user"; content = "Say hello in under 10 words." }
    )
    max_tokens = 50
    stream = $false
} | ConvertTo-Json -Compress

try {
    $response = Invoke-WebRequest -Uri "http://localhost:${TestPort}/v1/chat/completions" -Method POST -Body $body -ContentType "application/json" -Headers @{"Authorization" = "Bearer dummythicc"}
    $response.Content | Out-File -FilePath $responseFile
    Write-Host "Response saved to: $responseFile"
    Write-Host "Content:"
    Get-Content $responseFile
} catch {
    Write-Host "Request failed: $_"
    $_.Exception.Message | Out-File -FilePath $responseFile
}
Check-Timeout

:RestoreBrain
# Step 5: Kill test server and restore brain server
Write-Host "[5/5] Killing test server and restoring brain server..."
Kill-All-Servers

Write-Host "Starting main brain server on port 8081..."
Start-Process -FilePath "G:\hermes\start_llama_server.bat"

Write-Host "========================================"
Write-Host "TEST COMPLETE - Brain server starting..."
Write-Host "Log: $logFile"
Write-Host "Response: $responseFile"  
Write-Host "========================================"
