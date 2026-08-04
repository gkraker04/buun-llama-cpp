cmake_minimum_required(VERSION 3.16)

get_filename_component(TESTS_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(ROOT_DIR "${TESTS_DIR}" DIRECTORY)

set(STAGE_SOURCE "${ROOT_DIR}/src/llama-vbr-artifact-stage.cpp")
set(STAGE_HEADER "${ROOT_DIR}/src/llama-vbr-artifact-stage.h")
set(PUBLIC_HEADER "${ROOT_DIR}/include/llama.h")

foreach(path IN LISTS STAGE_SOURCE STAGE_HEADER PUBLIC_HEADER)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "F4.2a-1 contract input missing: ${path}")
    endif()
endforeach()

file(READ "${STAGE_SOURCE}" source)
set(stage_marker "vbr_adopt_stage_result vbr_stage_validated_manifest(")
string(FIND "${source}" "${stage_marker}" stage_start)
if(stage_start EQUAL -1)
    message(FATAL_ERROR "staging implementation marker missing")
endif()
string(SUBSTRING "${source}" ${stage_start} -1 staging_region)

# The same TU also owns the direction adapter, which legitimately spells the
# H2D primitive above this marker. The staging region itself is capability
# construction only and must not acquire any target-write vocabulary.
set(FORBIDDEN_STAGE_TOKENS
    "ggml_backend_tensor_set"
    "vbr_vmm_try_map"
    "vbr_vmm_unmap"
    "seq_rm("
    "state_read("
    "tracker_install"
    "ownership_index")

foreach(token IN LISTS FORBIDDEN_STAGE_TOKENS)
    string(FIND "${staging_region}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR
            "F4.2a-1 staging reached target-write token '${token}'")
    endif()
endforeach()

# Negative control: the scanner must catch an injected target write in the
# exact region it protects.
set(negative "${staging_region}\nggml_backend_tensor_set(target, data, 0, n);")
string(FIND "${negative}" "ggml_backend_tensor_set" negative_found)
if(negative_found EQUAL -1)
    message(FATAL_ERROR "staging target-write negative control did not fire")
endif()

file(READ "${PUBLIC_HEADER}" public_header)
foreach(token
        "vbr_staged_payloads"
        "vbr_stage_validated_manifest"
        "vbr_h2d_chunk_ring")
    string(FIND "${public_header}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "internal F4.2a-1 type leaked into public llama.h: ${token}")
    endif()
endforeach()

file(READ "${ROOT_DIR}/src/llama-cache-accounting.h" accounting_header)
string(FIND "${accounting_header}"
    "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2" accounting_schema)
if(accounting_schema EQUAL -1)
    message(FATAL_ERROR "F4.2a-1 must not move accounting schema 2")
endif()

file(READ "${ROOT_DIR}/common/common-cache-plan.h" plan_header)
string(FIND "${plan_header}"
    "COMMON_CACHE_PLAN_SCHEMA_VERSION = 5" plan_schema)
if(plan_schema EQUAL -1)
    message(FATAL_ERROR "F4.2a-1 contracts require reviewed cache-plan schema 5")
endif()

message(STATUS "F4.2a-1 staging target-purity contract passed")
