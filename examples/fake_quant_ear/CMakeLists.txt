# llama-fake-quant-ear: in-memory fake-quantization engine for Algorithm-1
# Multi-Bitwidth Shapley Estimation. Links the full llama library (forward
# pass + logits) and ggml (ggml_quantize_chunk / dequantize_row_*).
#
# Needs OpenMP for the parallel per-tensor quantize loop (/openmp on MSVC,
# -fopenmp on GCC/Clang).

set(TARGET llama-fake-quant-ear)
add_executable(${TARGET} fake_quant_ear.cpp)
install(TARGETS ${TARGET} RUNTIME)
# internal headers used deliberately: llama_internal_get_tensor_map
# (src/llama-model.h, same header tests/test-quantize-stats.cpp includes)
# and ggml-quants.h (dequantize_row_* / block types)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

find_package(OpenMP REQUIRED)
target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
