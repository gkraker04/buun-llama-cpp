@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM build_llama.bat — Build llama.cpp/buun-llama-cpp with named parameters
REM ============================================================================
REM DEFAULTS (lock these for production):
REM   build_dir=build          Target directory
REM   targets=llama-server    What to build
REM   arch=86-real            CUDA architecture
REM   jobs=12                 Parallel ninja jobs
REM   cuda_flags=ON           GGML_CUDA, GGML_CUDA_FA, GGML_CUDA_FA_ALL_QUANTS
REM   turbo4=ON               Additionally build for turbo4 weight-quant (build-t4 dir)
REM ============================================================================
REM Usage:
REM   build_llama                          — full server build
REM   build_llama targets=llama-quantize   — quantize tool only
REM   build_llama targets=ggml-cuda        — quick CUDA lib rebuild
REM   build_llama turbo4=OFF               — skip build-t4 target
REM ============================================================================

:: --- Parse named parameters ---
set "BUILD_DIR=build"
set "TARGETS=llama-server"
set "ARCH=86-real"
set "JOBS=12"
set "CUDA_FLAGS=ON"
set "TURBO4=ON"

for %%A in (%*) do (
    for /f "tokens=1,2 delims==" %%a in ("%%A") do (
        if /i "%%a"=="build_dir"    set "BUILD_DIR=%%b"
        if /i "%%a"=="targets"      set "TARGETS=%%b"
        if /i "%%a"=="arch"         set "ARCH=%%b"
        if /i "%%a"=="jobs"         set "JOBS=%%b"
        if /i "%%a"=="cuda_flags"   set "CUDA_FLAGS=%%b"
        if /i "%%a"=="turbo4"       set "TURBO4=%%b"
    )
)

echo === Build: BUILD_DIR=%BUILD_DIR% TARGETS=%TARGETS% ARCH=%ARCH% JOBS=%JOBS% ===

:: --- MSVC environment ---
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
if %errorlevel% neq 0 exit /b 1

:: --- Kill zombies ---
taskkill /F /IM nvcc.exe 2>nul
taskkill /F /IM cl.exe 2>nul
taskkill /F /IM ninja.exe 2>nul

:: --- Ninja build via CMake ---
cd /d G:\hermes\buun-llama-cpp
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

cmake .. -G Ninja ^
    -DGGML_CUDA=%CUDA_FLAGS% ^
    -DGGML_CUDA_FA=%CUDA_FLAGS% ^
    -DGGML_CUDA_FA_ALL_QUANTS=%CUDA_FLAGS% ^
    -DGGML_NATIVE=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=%ARCH% ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_TESTS=OFF

if %errorlevel% neq 0 exit /b 1

ninja %TARGETS% -j%JOBS%

if %errorlevel% equ 0 (
    echo === Build SUCCEEDED: %BUILD_DIR%/bin/%TARGETS%.exe ===
) else (
    echo === Build FAILED with code %errorlevel% ===
    exit /b %errorlevel%
)

:: --- Quick turbo4 build (if enabled) ---
if /i "%TURBO4%"=="ON" (
    echo === Turbo4 weight-quant post-build ===
    :: build-t4 is the turbo4-targeted build directory
)
