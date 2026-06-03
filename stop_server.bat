@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM stop_server.bat — Kill llama-server and reclaim GPU memory
REM ============================================================================
REM Usage:
REM   stop_server              — Kill all llama-server processes + GPU cleanup
REM   stop_server port=8082    — Also kill zombies on port 8082
REM ===========================================================================

set "PORT=8081"

for %%A in (%*) do (
    for /f "tokens=1,2 delims==" %%a in ("%%A") do (
        if /i "%%a"=="port" set "PORT=%%b"
    )
)

echo === Stopping server on port %PORT% ===

:: --- Kill llama-server by image name ---
taskkill /F /IM llama-server.exe 2>nul && echo Killed llama-server.exe || echo No llama-server running

:: --- Kill any zombies holding our port ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%"') do (
    if not "%%a"=="" (
        taskkill /F /PID %%a 2>nul && echo Killed PID %%a || echo Could not kill PID %%a
    )
)

:: --- Kill zombie nvcc/cl/ninja processes ---
taskkill /F /IM nvcc.exe 2>nul
taskkill /F /IM cl.exe 2>nul
taskkill /F /IM ninja.exe 2>nul
taskkill /F /IM cmake.exe 2>nul

echo === Cleanup complete. Check GPU memory: ===
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>nul
