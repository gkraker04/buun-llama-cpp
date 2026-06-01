turbo4 weight quantization: GGML_TYPE_TURBO4_0 for model weights

Adds turbo4 weight quantization (GGML_TYPE_TURBO4_0) to buun-llama-cpp, extending the existing turbo KV cache types to support quantized model weights.

## Changes

### New files
- ggml/src/ggml-cuda/turbo-matmul.cu -- CUDA matmul kernel with three code paths:
  - Single-token decode (direct turbo4->fp16 dequant)
  - Multi-token decode (<=8 tokens)
  - Large prefill (turbo4->fp16 dequant + cuBLAS GEMM)
- ggml/src/ggml-cuda/turbo-matmul.cuh -- declaration for ggml_cuda_mul_mat_turbo()

### Modified files
- ggml/src/ggml-cuda/ggml-cuda.cu -- turbo dispatch in ggml_cuda_mul_mat(), guard in ggml_cuda_mul_mat_id() and fused GLU path
- ggml/src/ggml-cuda/turbo-quant-cuda.cuh -- turbo4_dequant_element() and turbo4_dequant_block_to_half() device functions
- ggml/src/ggml-cuda/convert.cu -- turbo types in ggml_get_to_fp16_cuda() switch
- ggml/src/ggml-turbo-quant.c -- WHT butterfly rotation in quantize/dequantize (matching GPU matmul rotation, was QR matrix)
- ggml/src/ggml-quants.c -- ggml_validate_row_data() entries for turbo types
- ggml/src/ggml-cpu/ops.cpp -- GET_ROWS support for turbo types
- src/llama-model-loader.cpp -- ftype mapping and type_max entries
- include/llama.h -- LLAMA_FTYPE_MOSTLY_TURBO{4,3,2}_0 ftype enums
- tools/quantize/quantize.cpp -- turbo{4,3,2} quant options
- src/llama-quant.cpp -- ftype->ggml_type mapping

## Implementation details

- Activations pre-rotated via k_turbo_wht (WHT butterfly) to match the weight rotation applied during quantization
- Only affects turbo4 weight quantization -- existing turbo2/turbo3 KV cache types are unchanged
- Tested on Ornstein3.6-27B-MTP (27B) with MTP speculative decoding -- correct inference output confirmed

## Status

- Correct inference (no gibberish) with WHT butterfly rotation
- MTP speculative decoding compatible
- ~12.6 tok/s on RTX 3090, ~30% slower than Q4_K_M -- CUDA matmul kernels are naive, no MMQ/MMVQ tuning
- Kernel tuning to close performance gap is next

## Build

Tested on Windows with Visual Studio 2022 + CUDA 12.8 + Ninja + RTX 3090 (SM 86).

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"

mkdir build && cd build
cmake .. -G Ninja ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_NATIVE=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86-real ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TESTS=OFF

ninja -j12
```

To quantize a model to turbo4:
```bat
llama-quantize --type turbo4 model.gguf model-turbo4.gguf
```

To serve with MTP:
```bat
llama-server -m model-turbo4.gguf --no-mmap -ngl 99 -c 4096 --spec-type draft-mtp
```
