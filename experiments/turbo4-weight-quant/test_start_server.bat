@echo off
setlocal
rem windows line endings fixed

REM === Kill any existing server on port 8082 before starting ===
echo Killing any existing server on port 8082...
REM Try by image name first (broad kill)
taskkill /F /IM llama-server.exe 2>nul
timeout /t 1 /nobreak >nul
REM Then kill anything still holding port 8082
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8082"') do (
    if not "%%a"=="" taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak >nul
echo.

if "%1"=="" set "NMAX=3"
if not "%1"=="" set "NMAX=%1"
if "%2"=="" set "PMIN=0.0"
if not "%2"=="" set "PMIN=%2"
if "%3"=="" set "BATCH=2048"
if not "%3"=="" set "BATCH=%3"
if "%4"=="" set "UBATCH=512"
if not "%4"=="" set "UBATCH=%4"
if "%5"=="" set "K_CACHE=turbo2_tcq"
if not "%5"=="" set "K_CACHE=%5"
if "%6"=="" set "V_CACHE=turbo2_tcq"
if not "%6"=="" set "V_CACHE=%6"

echo Ornstein MTP: n_max=%NMAX%, p_min=%PMIN%, batch=%BATCH%, ubatch=%UBATCH%, K=%K_CACHE%, V=%V_CACHE%

REM === Start llama-server for Hermes Agent on Windows ===
REM Model: Ornstein3.6-27B-MTP Q4_K_M
REM Port: 8081
REM Uses Buun build for in-graph MTP (depth=3 chain predictions, p_min=0 optimal)

set "SERVER=G:\hermes\buun-llama-cpp\build\bin\llama-server.exe"
REM set "BUUN=1"

if not exist "%SERVER%" (
  echo WARNING: buun-llama-server.exe not found!
  set "SERVER=G:\hermes\llama.cpp\build_ninja\bin\llama-server.exe"
  set "BUUN=0"
)

echo Using: %SERVER%

set "MODEL_DIR=G:\models\gkraker04\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-GGUF"
set "MODEL=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf"
set "MMPROJ=%MODEL_DIR%\mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"

set "LOG_DIR=%MODEL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Build per-second timestamp (2026-05-17_22-18-45 format)
set "TS_YEAR=%DATE:~-4%"
set "TS_MONTH=%DATE:~4,2%"
set "TS_DAY=%DATE:~7,2%"
set "TS_HOUR=%TIME:~0,2%"
set "TS_MIN=%TIME:~3,2%"
set "TS_SEC=%TIME:~6,2%"
if "%TS_HOUR:~0,1%"==" " set "TS_HOUR=0%TS_HOUR:~1,1%"
set "LOG=%LOG_DIR%\server_%TS_YEAR%-%TS_MONTH%-%TS_DAY%_%TS_HOUR%-%TS_MIN%-%TS_SEC%.log"
set "JINJA=G:\models\spiritbuun\Qwen3.6-chat_template\chat_template.jinja"

if "%BUUN%" equ "0" (
    set "NMAX=3"
    set "PMIN=0.75"
    set "K_CACHE=q4_0"
    set "V_CACHE=q4_0"
)

echo Starting buun-llama-server...
"%SERVER%" ^
    --threads 12 ^
    --prio 2 ^
    --ctx-size 2048 ^
    --n-predict 32768 ^
    --batch-size 256 ^
    --ubatch-size 64 ^
    --flash-attn on ^
    --cache-type-k %K_CACHE% ^
    --cache-type-v %V_CACHE% ^
    --n-gpu-layers all ^
    --fit-target 0 ^
    --model "%MODEL%" ^
    --log-file "%LOG%" ^
    --offline ^
    --no-mmap ^
    --log-verbosity 4 ^
    --log-prefix ^
    --log-timestamps ^
    --cache-type-k-draft %K_CACHE% ^
    --cache-type-v-draft %V_CACHE% ^
    --temperature 0.7 ^
    --top-k 20 ^
    --top-p 0.8 ^
    --min-p 0.00 ^
    --repeat-penalty 1.0 ^
    --presence-penalty 1.5 ^
    --ctx-checkpoints 4 ^
    --cache-ram 4096 ^
    --kv-unified ^
    --parallel 1 ^
    --alias Qwen3.6-27B-turbo4-test ^
    --host 0.0.0.0 ^
    --port 8082 ^
    --api-key dummythicc ^
    --props ^
    --spec-type draft-mtp,ngram-mod,ngram-map-k4v ^
    --spec-draft-n-max %NMAX% ^
    --spec-draft-p-min %PMIN% ^
    --spec-ngram-mod-n-match 16 ^
    --spec-ngram-mod-n-min 24 ^
    --spec-ngram-mod-n-max 48 ^
    --spec-ngram-map-k4v-size-n 12 ^
    --spec-ngram-map-k4v-size-m 48 ^
    --spec-ngram-map-k4v-min-hits 1 ^
    --jinja ^
    --chat-template-file "%JINJA%"

if "%ERRORLEVEL%"=="0" (
  echo Server started successfully on port 8081
) else (
  echo Server exited with code %ERRORLEVEL%.
)

REM    --mmproj "%MMPROJ%" ^
REM    --no-mmproj-offload ^
REM    --image-min-tokens 1024 ^
