# MTP Depth Sweep — tests n_max=1,2,3 at winning batch/ubatch
param([string]$ResultsFile = "")

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogDir = "$ScriptDir\logs"
$ResultsDir = "$ScriptDir\results"
$RepoRoot = Resolve-Path "$ScriptDir\..\.."

$Server = "$RepoRoot\build\bin\llama-server.exe"
$ModelDir = "G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF"
$Model = "Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf"
$MmpRoj = "mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null }
if (-not $ResultsFile) { $ResultsFile = "$ResultsDir\mtp_depth_sweep.csv" }

. "$ScriptDir\Test-OutputQuality.ps1"

$csvHeader = "n_max,batch_size,ubatch_size,decode_tok_s,prompt_tok_s,gen_tokens,vram_mb,quality_passes,status"
$csvHeader | Out-File -FilePath $ResultsFile -Encoding utf8 -Force

$BenchmarkPrompt = '{"prompt":"Explain the fundamental differences between classical cryptography and quantum cryptography.","max_tokens":256,"stream":false}'
$WarmupPrompt   = '{"prompt":"Write a short sentence.","max_tokens":64,"stream":false}'

$Batch = 256
$Ubatch = 128

function Get-VRAM {
    $smi = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($smi) {
        if ($smi -is [array]) { return $smi[0].Trim() }
        return $smi.Trim()
    }
    return 0
}

function Ensure-ServerDead {
    taskkill /IM llama-server.exe /F 2>$null
    Start-Sleep -Milliseconds 500
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:8081/health" -UseBasicParsing -TimeoutSec 1
        taskkill /IM llama-server.exe /F 2>$null
        Start-Sleep -Seconds 1
    } catch { }
}

function Build-CmdLine {
    param($nMax, $logfile)
    $parts = @()
    $parts += "`"$Server`""
    $parts += "--threads 12 --prio 3 --n-predict 32768"
    $parts += "--batch-size $Batch --ubatch-size $Ubatch"
    $parts += "--flash-attn on"
    $parts += "--cache-type-k turbo3_tcq --cache-type-v turbo2_tcq"
    $parts += "--no-mmap --n-gpu-layers all --fit off --ctx-size 262144"
    $parts += "--model `"$ModelDir\$Model`""
    $parts += "--log-file `"$logfile`""
    $parts += "--offline --log-prefix --log-timestamps"
    $parts += "--cache-type-k-draft turbo3_tcq --cache-type-v-draft turbo2_tcq"
    $parts += "--temperature 0.6 --top-p 0.95 --min-p 0.01"
    $parts += "--repeat-penalty 1.0 --presence-penalty 0.0"
    $parts += "--ctx-checkpoints 8 --cache-ram 8192 --kv-unified --parallel 1"
    $parts += "--mmproj `"$ModelDir\$MmpRoj`" --no-mmproj-offload --image-min-tokens 1024"
    $parts += "--alias Qwen3.6-27B --host 0.0.0.0 --port 8081 --api-key dummythicc"
    $parts += "--props"

    
    
    $parts += "--spec-type draft-mtp --spec-draft-n-max $nMax --spec-draft-p-min 0.75"
    return ($parts -join " ")
}

function Test-NMax {
    param($NMax)
    Write-Host "=== TESTING: n_max=$NMax ==="
    Ensure-ServerDead
    $TS = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $LogOut = "$LogDir\mtp_n${NMax}_${TS}.log"
    $CmdLine = Build-CmdLine -nMax $NMax -logfile $LogOut
    Write-Host "[CMD] $CmdLine"
    $Cwd = (Get-Location).Path
    $Proc = Start-Process -FilePath "wt" -ArgumentList "new-tab", "-d", $Cwd, "cmd", "/c", "`"$CmdLine`" & echo Server exited & pause" -PassThru -WindowStyle Normal
    $Ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        try {
            $test = Invoke-WebRequest -Uri "http://localhost:8081/health" -UseBasicParsing -TimeoutSec 5
            if ($test.StatusCode -eq 200) { $Ready = $true; Write-Host "[OK] Server ready after ${i}s"; break }
        } catch { }
    }
    if (-not $Ready) {
        Write-Host "[FAIL] Server did not start"
        if (-not $Proc.HasExited) { $Proc.Kill() }
        return $null
    }
    Write-Host "[VRAM] After load: $(Get-VRAM)MB"
    for ($w = 1; $w -le 2; $w++) {
        try { $null = Invoke-WebRequest -Uri "http://localhost:8081/v1/completions" -Method Post -Headers @{"Authorization"="Bearer dummythicc"; "Content-Type"="application/json"} -Body $WarmupPrompt -UseBasicParsing -TimeoutSec 60; Write-Host "  Warmup $w OK" } catch { Write-Warning "  Warmup $w failed" }
    }
    Write-Host "[BENCH] 5 requests..."
    $DecodeScores = @()
    $PromptScores = @()
    $GenToks = @()
    $QualityPass = 0
    $QualityFail = 0
    for ($b = 1; $b -le 5; $b++) {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:8081/v1/completions" -Method Post -Headers @{"Authorization"="Bearer dummythicc"; "Content-Type"="application/json"} -Body $BenchmarkPrompt -UseBasicParsing -TimeoutSec 120
            $json = $resp.Content | ConvertFrom-Json
            $timings = $json.timings
            $genText = $json.choices[0].text
            if (Test-OutputQuality -Text $genText) {
                $QualityPass++
                $PromptScores += $timings.prompt_per_second
                $DecodeScores += $timings.predicted_per_second
                $GenToks += $json.usage.completion_tokens
                Write-Host "  [$b] decode: $([math]::Round($timings.predicted_per_second,1)) tok/s | quality OK"
            } else {
                $QualityFail++
                Write-Host "  [$b] decode: $([math]::Round($timings.predicted_per_second,1)) tok/s | quality FAIL"
            }
        } catch { Write-Warning "  Request $b failed: $_" }
        Start-Sleep -Milliseconds 200
    }
    if (-not $Proc.HasExited) { $Proc.Kill() }
    taskkill /IM llama-server.exe /F 2>$null
    if ($DecodeScores.Count -gt 0) {
        $avgDecode = ($DecodeScores | Measure-Object -Average).Average
        $avgPrompt = ($PromptScores | Measure-Object -Average).Average
        $avgGen = ($GenToks | Measure-Object -Average).Average
        Write-Host "[RESULT] n_max=$NMax => decode: $([math]::Round($avgDecode,2)) tok/s | quality: $QualityPass/5"
        return @{nMax=$NMax; decode=$avgDecode; prompt=$avgPrompt; genToks=$avgGen; vram=(Get-VRAM); qPass=$QualityPass; qFail=$QualityFail; status="ok"}
    } else {
        Write-Host "[FAIL] n_max=$NMax no valid results"
        return $null
    }
}

foreach ($NMax in @(1, 2, 3)) {
    $result = Test-NMax -NMax $NMax
    if ($result) {
        "$($result.nMax),$Batch,$Ubatch,$([math]::Round($result.decode,3)),$([math]::Round($result.prompt,3)),$([math]::Round($result.genToks,1)),$($result.vram),$($result.qPass)/5,$($result.status)" | Out-File -FilePath $ResultsFile -Encoding utf8 -Append
    } else {
        # n_max might be $null if the result object failed
        $nm = if ($result -and $result.nMax) { $result.nMax } else { $NMax }
        "$nm,$Batch,$Ubatch,0,0,0,0,0/5,fail" | Out-File -FilePath $ResultsFile -Encoding utf8 -Append
    }
}

Write-Host "MTP DEPTH SWEEP COMPLETE"
