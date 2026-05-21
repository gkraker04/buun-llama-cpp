# build_server.ps1 - Build llama.cpp server via PowerShell
# Replaces test_build_server.bat with same flags as real build

$ErrorActionPreference = "Stop"

# Setup MSVC environment
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
$vcvars_result = $?
if (-not $vcvars_result) {
    Write-Host "ERROR: Failed to initialize MSVC environment"
    exit 1
}

# Clean old build
if (Test-Path "build") {
    Write-Host "=== Clean old build ==="
    Remove-Item -Path "build" -Recurse -Force
}
New-Item -Path "build" -ItemType Directory | Out-Null

# Configure with CMake
Write-Host "=== Cmake with Ninja ==="
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_CUDA_FA=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DCMAKE_CUDA_ARCHITECTURES=86-real `
    -DGGML_NATIVE=ON `
    -DGGML_CCACHE=ON `
    -DCMAKE_CUDA_FLAGS="-Xcompiler=-O2,-Ob2"

cmake --build build --config Release -j6

Write-Host "=== Build complete ==="
