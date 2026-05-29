@echo off
setlocal
set LOGFILE=%~dp0logs\build_turbo4_%RANDOM%.log

echo === Setting up MSVC ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b 1

echo === CUDA ===
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe"

cd /d G:\hermes\buun-llama-cpp
set BUILD_DIR=build

echo === Removing old %BUILD_DIR% ===
if exist %BUILD_DIR% (
    taskkill /F /IM ninja.exe 2>nul
    taskkill /F /IM cl.exe 2>nul
    taskkill /F /IM nvcc.exe 2>nul
    timeout /t 2 /nobreak >nul
)
mkdir %BUILD_DIR%
cd %BUILD_DIR%

echo === Cmake with Ninja ===
cmake .. -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_COMPILER="%CMAKE_CUDA_COMPILER%" -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_FLAGS="-diag-suppress=177" >>%LOGFILE% 2>&1
if %errorlevel% neq 0 exit /b 1

echo === Ninja build ===
ninja llama-quantize llama-server llama-imatrix -j12 >>%LOGFILE% 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%

echo === DONE ===
for /r %%i in (llama-*.exe) do @if exist "%%i" echo %%i

cd ..
cd ..

exit /b 0
