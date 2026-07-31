cmake_minimum_required(VERSION 3.16)

get_filename_component(TESTS_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(ROOT_DIR "${TESTS_DIR}" DIRECTORY)

set(ADOPT_SOURCE "${ROOT_DIR}/src/llama-vbr-artifact-adopt.cpp")
set(ADOPT_HEADER "${ROOT_DIR}/src/llama-vbr-artifact-adopt.h")
set(PUBLIC_HEADER "${ROOT_DIR}/include/llama.h")
foreach(path IN LISTS ADOPT_SOURCE ADOPT_HEADER PUBLIC_HEADER)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "F4.2a-2 contract input missing: ${path}")
    endif()
endforeach()

file(READ "${ADOPT_SOURCE}" source)
file(READ "${ADOPT_HEADER}" header)
file(READ "${ROOT_DIR}/src/llama-vbr-generation.cpp" generation_source)
file(READ "${ROOT_DIR}/src/llama-memory-recurrent.cpp" recurrent_source)

foreach(required IN ITEMS
        "vbr_adopt_empty_manifest("
        "vbr_operation_kind::state_import"
        "vbr_recovery_mint("
        "vbr_recovery_pending_for_except("
        "vbr_operation_registry_quiescent_for_except("
        "accounting_serial_after_prepare()"
        "adoption_materialize_claims()"
        "prepare_import_image("
        "install_import_image_swap("
        "vbr_parse_recurrent_companion("
        "vbr_recurrent_companion_adoption_provider("
        "required_companion_unavailable"
        "vbr_adopt_check_complete_tree("
        "target_empty"
        "BEGIN VBR_IMPORT_NOFAIL_PUBLISH"
        "END VBR_IMPORT_NOFAIL_PUBLISH")
    string(FIND "${source}${header}" "${required}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "F4.2a-2 adoption contract missing '${required}'")
    endif()
endforeach()

# The model-free G2 matrix substitutes only target/backend doors in the real
# phase driver. Its single injection pointer is null by default, and no production
# translation unit may arm or implement the test-only seam.
foreach(required IN ITEMS
        "class vbr_adopt_test_seam"
        "struct vbr_adopt_test_control"
        "const vbr_adopt_test_control * test = nullptr")
    string(FIND "${header}" "${required}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR
            "F4.2a-2 G2 inertness contract missing '${required}'")
    endif()
endforeach()

file(GLOB_RECURSE production_candidates
    "${ROOT_DIR}/src/*.cpp" "${ROOT_DIR}/src/*.h"
    "${ROOT_DIR}/common/*.cpp" "${ROOT_DIR}/common/*.h"
    "${ROOT_DIR}/tools/server/*.cpp" "${ROOT_DIR}/tools/server/*.h")
set(production_test_seam_hits "")
foreach(path IN LISTS production_candidates)
    if(path STREQUAL ADOPT_SOURCE OR path STREQUAL ADOPT_HEADER)
        continue()
    endif()
    file(READ "${path}" candidate)
    if(candidate MATCHES
            "vbr_adopt_test_seam|vbr_adopt_test_control|\\.test[ \t]*=[ \t]*&")
        list(APPEND production_test_seam_hits "${path}")
    endif()
endforeach()
if(production_test_seam_hits)
    message(FATAL_ERROR
        "F4.2a-2 test injection escaped into production: ${production_test_seam_hits}")
endif()

function(g2_scan_text candidate output)
    if(candidate MATCHES
            "vbr_adopt_test_seam|vbr_adopt_test_control|\\.test[ \t]*=[ \t]*&")
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()
g2_scan_text("vbr_composite_publish_hooks hooks; hooks.test = &fake;"
    negative_g2_hit)
if(NOT negative_g2_hit)
    message(FATAL_ERROR
        "F4.2a-2 G2 production-inertness negative control did not trip")
endif()

# Phase 12 publication through close is no-fail. There must be no post-phase
# 12 injection call even though the test-only policy has a generic field.
foreach(forbidden IN ITEMS
        "fault_after(server_hooks, vbr_adopt_phase::composite_publish)"
        "fault_before(server_hooks, vbr_adopt_phase::close)"
        "fault_after(server_hooks, vbr_adopt_phase::close)")
    string(FIND "${source}" "${forbidden}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR
            "F4.2a-2 no-fail region grew a fault seam: ${forbidden}")
    endif()
endforeach()

function(find_publish_forbidden text output)
    set(hits "")
    foreach(token IN ITEMS
            "ggml_backend_"
            "vmm_pool_map"
            "vmm_pool_unmap"
            "new "
            "make_unique"
            "push_back"
            "resize("
            "reserve("
            "state_read("
            "publish_unit("
            "LLAMA_LOG"
            "return false")
        string(FIND "${text}" "${token}" found)
        if(NOT found EQUAL -1)
            list(APPEND hits "${token}")
        endif()
    endforeach()
    set(${output} "${hits}" PARENT_SCOPE)
endfunction()

string(FIND "${source}" "// BEGIN VBR_IMPORT_NOFAIL_PUBLISH" begin)
string(FIND "${source}" "// END VBR_IMPORT_NOFAIL_PUBLISH" end)
if(begin EQUAL -1 OR end EQUAL -1 OR end LESS_EQUAL begin)
    message(FATAL_ERROR "F4.2a-2 no-fail publish markers are malformed")
endif()
math(EXPR length "${end}-${begin}")
string(SUBSTRING "${source}" ${begin} ${length} publish_region)
find_publish_forbidden("${publish_region}" publish_hits)
if(publish_hits)
    message(FATAL_ERROR
        "F4.2a-2 phase-12 region grew a fallible token: ${publish_hits}")
endif()

function(check_metadata_region region_source region_begin_marker region_end_marker)
    string(FIND "${region_source}" "${region_begin_marker}" region_begin)
    string(FIND "${region_source}" "${region_end_marker}" region_end)
    if(region_begin EQUAL -1 OR region_end EQUAL -1 OR
       region_end LESS_EQUAL region_begin)
        message(FATAL_ERROR
            "F4.2a-2 metadata-swap markers are malformed: ${region_begin_marker}")
    endif()
    math(EXPR region_length "${region_end}-${region_begin}")
    string(SUBSTRING "${region_source}" ${region_begin}
        ${region_length} checked_region)
    find_publish_forbidden("${checked_region}" checked_hits)
    if(checked_hits)
        message(FATAL_ERROR
            "F4.2a-2 metadata swap grew a fallible token: ${checked_hits}")
    endif()
endfunction()
check_metadata_region("${source}"
    "// BEGIN VBR_IMPORT_KV_METADATA_SWAP"
    "// END VBR_IMPORT_KV_METADATA_SWAP")
check_metadata_region("${generation_source}"
    "// BEGIN VBR_IMPORT_TRACKER_METADATA_SWAP"
    "// END VBR_IMPORT_TRACKER_METADATA_SWAP")
check_metadata_region("${recurrent_source}"
    "// BEGIN VBR_IMPORT_RECURRENT_METADATA_SWAP"
    "// END VBR_IMPORT_RECURRENT_METADATA_SWAP")

set(negative "${publish_region}\nggml_backend_tensor_set(dst, src, 0, n);")
find_publish_forbidden("${negative}" negative_hits)
if(NOT negative_hits)
    message(FATAL_ERROR "F4.2a-2 phase-12 negative control did not trip")
endif()

file(READ "${PUBLIC_HEADER}" public_header)
foreach(token IN ITEMS
        "vbr_adopt_empty_manifest"
        "vbr_staged_payloads"
        "vbr_tracker_import_image")
    string(FIND "${public_header}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "internal F4.2a-2 type leaked into public llama.h: ${token}")
    endif()
endforeach()

file(READ "${ROOT_DIR}/src/llama-cache-accounting.h" accounting_header)
string(FIND "${accounting_header}"
    "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2" accounting_schema)
if(accounting_schema EQUAL -1)
    message(FATAL_ERROR "F4.2a-2 must not move accounting schema 2")
endif()
file(READ "${ROOT_DIR}/common/common-cache-plan.h" plan_header)
string(FIND "${plan_header}"
    "COMMON_CACHE_PLAN_SCHEMA_VERSION = 4" plan_schema)
if(plan_schema EQUAL -1)
    message(FATAL_ERROR "F4.2a-2 must not move cache-plan schema 4")
endif()

message(STATUS "F4.2a-2 atomic adoption contracts passed")
