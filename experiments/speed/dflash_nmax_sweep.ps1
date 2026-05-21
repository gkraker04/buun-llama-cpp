# DFlash n_min × n_max sweep — leverages per-request speculative params
# Discovery: speculative.n_min, speculative.n_max, speculative.p_min are passed
# as flat JSON keys in the chat completions request body (server-task.cpp:300-302)
# This allows sweeping all 136 (n_min,n_max) combos in ONE server start.

param([string]$ResultsFile = "")

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogDir = "$ScriptDir\logs"
$ResultsDir = "$ScriptDir\results"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null }
if (-not $ResultsFile) { $ResultsFile = "$ResultsDir\dflash_nmax_sweep_p75.csv" }
$ServerLog = "$LogDir\dflash_nmax_sweep_p75.log"

$RepoRoot = Resolve-Path "$ScriptDir\..\.."
$Server = "$RepoRoot\build\bin\llama-server.exe"
$ModelDir = "G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF"
$Model = "Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf"
$MmpRoj = "mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"
$DraftDir = "G:\models\gkraker04\dflash-drafter-3.6"
$Draft = "$DraftDir\dflash-draft-3.6-Q4_K_M.gguf"

$ApiKey = "dummythicc"
$BaseUrl = "http://localhost:8081"
$HealthUrl = "$BaseUrl/health"
$ChatUrl = "$BaseUrl/v1/chat/completions"

# Server args from test_start_dflash.bat
$ServerArgs = @(
    "--threads", "12", "--prio", "3", "--n-predict", "32768",
    "--model", "$ModelDir\$Model",
    "--spec-draft-model", $Draft,
    "--spec-type", "dflash",
    "-ngl", "99", "-ngld", "99",
    "-cd", "256", "-b", "256", "-ub", "64",
    "-fa", "on", "-np", "1",
    "--fit-target", "5120",
    "--no-mmap",
    "--cache-type-k", "turbo2_tcq",
    "--cache-type-v", "turbo2_tcq",
    "--cache-type-k-draft", "turbo2_tcq",
    "--cache-type-v-draft", "turbo2_tcq",
    "--mmproj", "$ModelDir\$MmpRoj",
    "--no-mmproj-offload",
    "--host", "0.0.0.0", "--port", "8081",
    "--api-key", $ApiKey,
    "--offline",
    "--jinja", "--reasoning", "off",
    "--log-file", $ServerLog
)

function Invoke-Curl {
    param([string]$Method, [string]$Url, [string]$Body, [int]$Timeout = 120)
    $argList = @("-s", "-X", $Method, $Url)
    if ($Body) {
        $argList += @("-H", "Content-Type: application/json", "-H", "Authorization: Bearer $ApiKey", "-d", $Body)
    }
    return & curl.exe @argList 2>$null
}

function Wait-Server {
    param([int]$Timeout = 120)
    for ($i = 0; $i -lt $Timeout; $i++) {
        Start-Sleep -Seconds 1
        $resp = Invoke-Curl -Method "GET" -Url $HealthUrl
        if ($resp -eq "200" -or $resp -eq "{}") {
            return $i + 1
        }
    }
    return $null
}

function Write-Csv {
    param([string]$Line)
    Add-Content -Path $ResultsFile -Value $Line -Encoding utf8
}

# Cleanup function
function Stop-Server {
    & taskkill /f /im llama-server.exe 2>$null
    Start-Sleep -Seconds 2
}

# Header
"n_min,n_max,tokens,tok_s,error" | Out-File -FilePath $ResultsFile -Encoding utf8 -Force

# Kill any prior server
Stop-Server

# Start server
Write-Host "Starting server..."
$proc = Start-Process -FilePath $Server -ArgumentList $ServerArgs -NoNewWindow -PassThru -RedirectStandardOutput "nul" -RedirectStandardError "nul"
$pid = $proc.Id
Write-Host "PID: $pid"

$ready = Wait-Server
if (-not $ready) {
    Write-Host "FAILED - server did not start"
    exit 1
}
Write-Host "Server ready after ${ready}s"

# Warmup
Write-Host "Warming up..."
$warmup = '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":5,"temperature":0.1,"stream":false}'
Invoke-Curl -Method "POST" -Url $ChatUrl -Body $warmup | Out-Null
Start-Sleep -Seconds 1

# Build benchmark body template
$BenchBody = @{
    model = "qwen"
    messages = @(
        @{role = "system"; content = "You are a helpful assistant."},
        @{role = "user"; content = "Write a short essay about the history of computer processors."}
    )
    max_tokens = 256
    temperature = 0.6
    stream = $false
}

$TotalConfigs = 0
for ($nm = 1; $nm -le 16; $nm++) {
    for ($nx = $nm; $nx -le 16; $nx++) {
        $TotalConfigs++
    }
}

$StartTime = Get-Date
$Completed = 0

for ($n_min = 1; $n_min -le 16; $n_min++) {
    for ($n_max = $n_min; $n_max -le 16; $n_max++) {
        $Completed++
        $t0 = Get-Date
        
        # Add per-request spec params
        $body = $BenchBody.Clone()
        $body["speculative.n_min"] = $n_min
        $body["speculative.n_max"] = $n_max
        $body["speculative.p_min"] = 0.75
        $bodyJson = $body | ConvertTo-Json -Compress
        
        $resp = Invoke-Curl -Method "POST" -Url $ChatUrl -Body $bodyJson
        
        if (-not $resp) {
            Write-Host "[$Completed/$TotalConfigs] n_min=$n_min, n_max=${n_max}: CURL FAILED"
            Write-Csv "$n_min,$n_max,0,0,curl_failed"
            continue
        }
        
        try {
            $data = $resp | ConvertFrom-Json
            $ct = $data.usage.completion_tokens
            $elapsed = ((Get-Date) - $t0).TotalSeconds
            if ($elapsed -gt 0) { $tok_s = [math]::Round($ct / $elapsed, 2) } else { $tok_s = 0 }
            
            $eta_remaining = ((Get-Date) - $StartTime).TotalSeconds / $Completed * ($TotalConfigs - $Completed)
            $eta_m = [math]::Round($eta_remaining / 60, 0)
            
            Write-Host "[$Completed/$TotalConfigs] n_min=$n_min, n_max=${n_max}: ${ct}tok @ ${tok_s}tok/s (ETA ${eta_m}m)"
            Write-Csv "${n_min},${n_max},${ct},${tok_s},"
        } catch {
            Write-Host "[$Completed/$TotalConfigs] n_min=$n_min, n_max=${n_max}: PARSE ERROR"
            Write-Csv "$n_min,$n_max,0,0,parse_error"
        }
    }
}

$TotalTime = ((Get-Date) - $StartTime).TotalMinutes
Write-Host "`nDONE in ${TotalTime:F1}m"
Write-Host "Results: $ResultsFile"

# Print top configs
$results = Import-Csv $ResultsFile | Where-Object { $_.tok_s -gt 0 } | Sort-Object tok_s -Descending
Write-Host "`n=== TOP 10 ==="
$results | Select-Object -First 10 | Format-Table n_min, n_max, tok_s, tokens

# Kill server
Stop-Server
Write-Host "Server stopped"
