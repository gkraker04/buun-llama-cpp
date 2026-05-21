# n_max × p_min sweep — tests n_max=1..4 × p_min=0.0..1.0
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
if (-not $ResultsFile) { $ResultsFile = "$ResultsDir\nmax_pmin_sweep.csv" }

. "$ScriptDir\Test-OutputQuality.ps1"

"n_max,p_min,batch,ubatch,decode_tok_s,prompt_tok_s,gen_tokens,vram_mb,quality_pass,quality_fail,status" | Out-File -FilePath $ResultsFile -Encoding utf8 -Force

$BenchmarkPrompt = '{"prompt":"Explain the fundamental differences between classical cryptography and quantum cryptography.","max_tokens":256,"stream":false}'
$WarmupPrompt   = '{"prompt":"Write a short sentence.","max_tokens":64,"stream":false}'
$Batch = 256; $Ubatch = 128

function Get-VRAM {
    $smi = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($smi) { if ($smi -is [array]) { return $smi[0].Trim() }; return $smi.Trim() }
    return 0
}
function Ensure-Dead {
    taskkill /IM llama-server.exe /F 2>$null; Start-Sleep 1
    try { $null = Invoke-WebRequest -Uri "http://localhost:8081/health" -UseBasicParsing -TimeoutSec 1; taskkill /IM llama-server.exe /F 2>$null; Start-Sleep 1 } catch { }
}
function Build-Cmd {
    param($nMax, $pMin, $logfile)
    $parts = @()
    $parts += "`"$Server`""; $parts += "--threads 12 --prio 3 --n-predict 32768"
    $parts += "--batch-size $Batch --ubatch-size $Ubatch --flash-attn on"
    $parts += "--cache-type-k turbo3_tcq --cache-type-v turbo2_tcq"
    $parts += "--no-mmap --n-gpu-layers all --fit off --ctx-size 262144"
    $parts += "--model `"$ModelDir\$Model`" --log-file `"$logfile`""
    $parts += "--offline --log-prefix --log-timestamps"
    $parts += "--cache-type-k-draft turbo3_tcq --cache-type-v-draft turbo2_tcq"
    $parts += "--temp 0.6 --top-p 0.95 --min-p 0.01 --repeat-penalty 1.0 --presence-penalty 0.0"
    $parts += "--ctx-checkpoints 8 --cache-ram 8192 --kv-unified --parallel 1"
    $parts += "--mmproj `"$ModelDir\$MmpRoj`" --no-mmproj-offload --image-min-tokens 1024"
    $parts += "--host 0.0.0.0 --port 8081 --api-key dummythicc"
    $parts += "--spec-type draft-mtp --spec-draft-n-max $nMax --spec-draft-p-min $pMin"
    return ($parts -join " ")
}

function Test-Config {
    param($NMax, $PMin)
    Write-Host "n_max=$NMax p_min=$PMin"
    Ensure-Dead
    $TS = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $LogOut = "$LogDir\n${NMax}_p$([string]$PMin).Replace('.','_')_$TS.log"
    $CmdLine = Build-Cmd -nMax $NMax -pMin $PMin -logfile $LogOut
    $Cwd = (Get-Location).Path
    $Proc = Start-Process -FilePath "wt" -ArgumentList "new-tab", "-d", $Cwd, "cmd", "/c", "`"$CmdLine`" & pause" -PassThru -WindowStyle Normal
    $Ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Seconds 1
        try { $test = Invoke-WebRequest -Uri "http://localhost:8081/health" -UseBasicParsing -TimeoutSec 5; if ($test.StatusCode -eq 200) { $Ready = $true; break } } catch { }
    }
    if (-not $Ready) { if (-not $Proc.HasExited) { $Proc.Kill() }; return $null }
    $vram = Get-VRAM
    for ($w = 1; $w -le 2; $w++) { try { $null = Invoke-WebRequest -Uri "http://localhost:8081/v1/completions" -Method Post -Headers @{"Authorization"="Bearer dummythicc"; "Content-Type"="application/json"} -Body $WarmupPrompt -UseBasicParsing -TimeoutSec 60 } catch { } }
    $dScores = @(); $pScores = @(); $gToks = @(); $qPass = 0; $qFail = 0
    for ($b = 1; $b -le 3; $b++) {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:8081/v1/completions" -Method Post -Headers @{"Authorization"="Bearer dummythicc"; "Content-Type"="application/json"} -Body $BenchmarkPrompt -UseBasicParsing -TimeoutSec 120
            $json = $resp.Content | ConvertFrom-Json; $t = $json.timings; $txt = $json.choices[0].text
            if (Test-OutputQuality -Text $txt) { $qPass++; $dScores += $t.predicted_per_second; $pScores += $t.prompt_per_second; $gToks += $json.usage.completion_tokens; Write-Host "  [$b] $([math]::Round($t.predicted_per_second,1)) tok/s OK" }
            else { $qFail++; Write-Host "  [$b] $([math]::Round($t.predicted_per_second,1)) tok/s FAIL" }
        } catch { }
        Start-Sleep -Milliseconds 200
    }
    if (-not $Proc.HasExited) { $Proc.Kill() }; taskkill /IM llama-server.exe /F 2>$null
    if ($dScores.Count -gt 0) {
        $avgD = ($dScores | Measure-Object -Average).Average; $avgP = ($pScores | Measure-Object -Average).Average; $avgG = ($gToks | Measure-Object -Average).Average
        Write-Host "  => $([math]::Round($avgD,2)) tok/s quality: $qPass/$($qPass+$qFail)"
        return @{nMax=$NMax; pMin=$PMin; decode=$avgD; prompt=$avgP; genToks=$avgG; vram=$vram; qP=$qPass; qF=$qFail; status="ok"}
    }
    return @{nMax=$NMax; pMin=$PMin; decode=0; prompt=0; genToks=0; vram=0; qP=$qPass; qF=$qFail; status="fail"}
}

foreach ($nMax in @(1, 2, 3, 4)) {
    foreach ($pMin in @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) {
        $r = Test-Config -NMax $nMax -PMin $pMin
        if ($r) { "$($r.nMax),$($r.pMin),$Batch,$Ubatch,$([math]::Round($r.decode,3)),$([math]::Round($r.prompt,3)),$([math]::Round($r.genToks,1)),$($r.vram),$($r.qP),$($r.qF),$($r.status)" | Out-File -FilePath $ResultsFile -Encoding utf8 -Append }
        else { "$nMax,$pMin,$Batch,$Ubatch,0,0,0,0,0,0,fail" | Out-File -FilePath $ResultsFile -Encoding utf8 -Append }
    }
}

Write-Host "`nSWEEP COMPLETE: $ResultsFile"
