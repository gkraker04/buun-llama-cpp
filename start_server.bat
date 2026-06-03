@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM start_server.bat — Start llama-server with KNOWN GOOD defaults
REM ============================================================================
REM DEFAULTS:
REM   model=v6-i3  port=8081  ctx=2048  batch=256  ubatch=64
REM   kmem=4096  mtp_n=3  mtp_p=0.0  flash=on  threads=6
REM ============================================================================
REM Usage: start_server [model=v6-i3] [port=8081] [mtp_n=3] [mtp_p=0.0]
REM ===========================================================================

set model=v6-i3
set port=8081
set ctx=2048
set batch=256
set ubatch=64
set kmem=4096
set mtp_n=3
set mtp_p=0.0
set threads=6

for %%A in (%*) do (
    for /f "tokens=1,2 delims==" %%a in ("%%A") do (
        if /i "%%a"=="model"   set "model=%%b"
        if /i "%%a"=="port"    set "port=%%b"
        if /i "%%a"=="mtp_n"   set "mtp_n=%%b"
        if /i "%%a"=="mtp_p"   set "mtp_p=%%b"
        if /i "%%a"=="threads" set "threads=%%b"
    )
)

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul

taskkill /F /IM llama-server.exe >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%port%"') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)

set MODEL_DIR=G:\models\gkraker04\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-GGUF
if "%model%"=="v6-i3" set MODEL=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v6-i3.gguf
if "%model%"=="v5-i3" set MODEL=%MODEL_DIR%\Ornstein3.6-27B-MTP-NSC-ACE-SABER-turbo4-MTP-v5-i3.gguf

echo === Starting %model% on port %port% ===

G:\hermes\buun-llama-cpp\build\bin\llama-server.exe ^
    --threads %threads% ^
    --ctx-size %ctx% ^
    --batch-size %batch% ^
    --ubatch-size %ubatch% ^
    --n-gpu-layers all ^
    --no-mmap ^
    --flash-attn on ^
    --cache-ram %kmem% ^
    --kv-unified ^
    --host 0.0.0.0 ^
    --port %port% ^
    --spec-type draft-mtp ^
    --spec-draft-n-max %mtp_n% ^
    --spec-draft-p-min %mtp_p% ^
    --model "%MODEL%"

echo Server exited with code %ERRORLEVEL%
pause
