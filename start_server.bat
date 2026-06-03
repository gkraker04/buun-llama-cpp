@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM start_server.bat — Start llama-server with KNOWN GOOD defaults
REM ============================================================================
REM DEFAULTS (LOCKED — don't touch without a good reason):
REM   model=v6-i3            Model version (v4-i3, v5-i3, v6-i3)
REM   port=8081               Server port
REM   ctx=2048                Context size
REM   batch=256               Batch size
REM   ubatch=64               Ubatch size
REM   kmem=4096               KV cache RAM limit (MiB)
REM   mtp_n=3                 MTP draft max
REM   mtp_p=0.0               MTP draft p_min
REM   flash=on                Flash attention
REM   threads=6               CPU threads
REM   kv=turbo2_tcq           KV cache type
REM   mmproj=off              Multimodal projector
REM ============================================================================
REM Usage:
REM   start_server                       — v6, port 8081, defaults
REM   start_server model=v5-i3 port=8082 — v5 on alt port
REM   start_server mtp_n=0               — no MTP speculation
REM ===========================================================================

:: --- Parse named parameters ---
set "MODEL=v6-i3"
set "PORT=8081"
set "CTX=2048"
set "BATCH=256"
set "UBATCH=64"
set "KMEM=4096"
set "MTP_N=3"
set "MTP_P=0.0"
set "FLASH=on"
set "THREADS=6"
set "KV=turbo2_tcq"
set "MMPROJ=off"

for %%A in (%*) do (
    for /f "tokens=1,2 delims==" %%a in ("%%A") do (
        if /i "%%a"=="model"    set "MODEL=%%b"
        if /i "%%a"=="port"     set "PORT=%%b"
        if /i "%%a"=="ctx"      set "CTX=%%b"
        if /i "%%a"=="batch"    set "BATCH=%%b"
        if /i "%%a"=="ubatch"   set "UBATCH=%%b"
        if /i "%%a"=="kmem"     set "KMEM=%%b"
        if /i "%%a"=="mtp_n"    set "MTP_N=%%b"
        if /i "%%a"=="mtp_p"    set "MTP_P=%%b"
        if /i "%%a"=="flash"    set "FLASH=%%b"
        if /i "%%a"=="threads"  set "THREADS=%%b"
        if /i "%%a"=="kv"       set "KV=%%b"
        if /i "%%a"=="mmproj"   set "MMPROJ=%%b"
    )
)

echo === Starting server: model=%MODEL% port=%PORT% ctx=%CTX% batch=%BATCH% ubatch=%UBATCH% ===
echo === MTP: n=%MTP_N% p=%MTP_P% | flash=%FLASH% | kv=%KV% ===

:: --- Kill anything on our port ---
echo Cleaning up port %PORT%...
taskkill /F /IM llama-server.exe 2>nul
for /l %%i in (1,1,3) do @echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%"') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)
for /l %%i in (1,1,3) do @echo.

:: --- Resolve model path ---
set "MODEL_DIR=G:\models\gkraker04\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-GGUF"
set "MODEL_PATH=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf"
if /i "%MODEL%"=="v6-i3" set "MODEL_PATH=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf"
if /i "%MODEL%"=="v5-i3" set "MODEL_PATH=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v5-i3.gguf"
if /i "%MODEL%"=="v4-i3" set "MODEL_PATH=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v4-i3.gguf"
if /i "%MODEL%"=="v3"    set "MODEL_PATH=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v3.gguf"

:: --- Server binary ---
set "SERVER=G:\hermes\buun-llama-cpp\build\bin\llama-server.exe"
if not exist "%SERVER%" (
    echo ERROR: %SERVER% not found — run build_llama first
    pause
    exit /b 1
)

:: --- Build command ---
set "CMD=%SERVER%"
set "CMD=%CMD% --threads %THREADS%"
set "CMD=%CMD% --ctx-size %CTX%"
set "CMD=%CMD% --batch-size %BATCH%"
set "CMD=%CMD% --ubatch-size %UBATCH%"
set "CMD=%CMD% --n-gpu-layers all"
set "CMD=%CMD% --no-mmap"
set "CMD=%CMD% --flash-attn %FLASH%"
set "CMD=%CMD% --cache-ram %KMEM%"
set "CMD=%CMD% --kv-unified"
set "CMD=%CMD% --parallel 1"
set "CMD=%CMD% --host 0.0.0.0"
set "CMD=%CMD% --port %PORT%"
set "CMD=%CMD% --spec-type draft-mtp"
set "CMD=%CMD% --spec-draft-n-max %MTP_N%"
set "CMD=%CMD% --spec-draft-p-min %MTP_P%"
set "CMD=%CMD% --cache-type-k %KV%"
set "CMD=%CMD% --cache-type-v %KV%"
set "CMD=%CMD% --cache-type-k-draft %KV%"
set "CMD=%CMD% --cache-type-v-draft %KV%"
set "CMD=%CMD% --model %MODEL_PATH%"

:: --- Logging ---
set "LOG_DIR=%MODEL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "DT=%%a"
set "TS=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%_%DT:~8,2%-%DT:~10,2%-%DT:~12,2%"
set "LOG=%LOG_DIR%\server_%TS%.log"

echo Starting server... log: %LOG%
echo Command: %CMD%
echo %CMD% > "%LOG%"
echo. >> "%LOG%"

%CMD% >> "%LOG%" 2>&1
echo Server exited with code %ERRORLEVEL% >> "%LOG%" 2>&1
echo === Server stopped (exit code %ERRORLEVEL%) ===
pause
