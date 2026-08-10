if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(GLOB_RECURSE production_files
    "${SOURCE_ROOT}/src/*.cpp" "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/server/*.cpp" "${SOURCE_ROOT}/tools/server/*.h")
set(production_text "")
set(non_owner_text "")
set(generation_header_path "${SOURCE_ROOT}/src/llama-vbr-generation.h")
set(generation_source_path "${SOURCE_ROOT}/src/llama-vbr-generation.cpp")
set(kv_header_path "${SOURCE_ROOT}/src/llama-kv-cache.h")
set(kv_source_path "${SOURCE_ROOT}/src/llama-kv-cache.cpp")
foreach(path IN LISTS production_files)
    file(READ "${path}" text)
    string(APPEND production_text "${text}\n")
    if (NOT path STREQUAL generation_header_path AND
        NOT path STREQUAL generation_source_path AND
        NOT path STREQUAL kv_header_path AND
        NOT path STREQUAL kv_source_path)
        string(APPEND non_owner_text "${text}\n")
    endif()
endforeach()
file(READ "${generation_header_path}" generation_header)
file(READ "${generation_source_path}" generation_source)
file(READ "${kv_header_path}" kv_header)
file(READ "${kv_source_path}" kv_source)

function(vbr_generation_orphans_absent text output)
    foreach(orphan IN ITEMS
            common_checkpoint_shadow
            common_shadow_checkpoint
            server_shadow_global_state
            llama_vbr_checkpoint_shadow
            checkpoint_vbr_eligibility
            vbr_generation_oracle)
        string(FIND "${text}" "${orphan}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

function(vbr_generation_boundary_valid
        generation_header generation_source kv_header kv_source non_owner output)
    count_literal("${generation_source}"
        "VBR_GENERATION_MUTATION_DISPATCH_EXHAUSTIVE" dispatch_markers)
    count_literal("${generation_header}"
        "bool vbr_generation_capture_stream(" stream_declarations)
    count_literal("${generation_source}"
        "bool vbr_generation_capture_stream(" stream_definitions)
    count_literal("${kv_source}"
        "vbr_generation_capture_stream(" stream_kv_uses)
    count_literal("${generation_header}"
        "bool vbr_generation_capture_controller(" controller_declarations)
    count_literal("${generation_source}"
        "bool vbr_generation_capture_controller(" controller_definitions)
    count_literal("${kv_source}"
        "vbr_generation_capture_controller(" controller_kv_uses)
    count_literal("${kv_header}"
        "bool vbr_generation_capture_live_guarded(" live_declarations)
    count_literal("${kv_source}"
        "vbr_generation_capture_live_guarded(" live_kv_sites)

    foreach(symbol IN ITEMS
            VBR_GENERATION_MUTATION_DISPATCH_EXHAUSTIVE
            vbr_generation_capture_stream
            vbr_generation_capture_controller
            vbr_generation_capture_live_guarded)
        string(FIND "${non_owner}" "${symbol}" escaped)
        if (NOT escaped EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()

    if (NOT dispatch_markers EQUAL 1 OR
        NOT stream_declarations EQUAL 1 OR
        NOT stream_definitions EQUAL 1 OR
        NOT stream_kv_uses EQUAL 1 OR
        NOT controller_declarations EQUAL 1 OR
        NOT controller_definitions EQUAL 1 OR
        NOT controller_kv_uses EQUAL 1 OR
        NOT live_declarations EQUAL 1 OR
        NOT live_kv_sites EQUAL 3)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

vbr_generation_orphans_absent("${production_text}" orphans_absent)
vbr_generation_boundary_valid(
    "${generation_header}" "${generation_source}"
    "${kv_header}" "${kv_source}" "${non_owner_text}" boundary_ok)
if (NOT orphans_absent OR NOT boundary_ok)
    message(FATAL_ERROR
        "VBR generation tracker/artifact boundary is missing or orphan shadow authority remains")
endif()

# Negative control: reintroducing any retired shadow owner must trip the same validator.
vbr_generation_orphans_absent(
    "${production_text}\nstruct common_checkpoint_shadow {};" orphan_mutation_ok)
if (orphan_mutation_ok)
    message(FATAL_ERROR "VBR generation orphan negative control did not bite")
endif()

# A helper name in a comment cannot replace its owned declaration/definition.
string(REPLACE "bool vbr_generation_capture_stream("
    "bool removed_generation_capture_stream("
    comment_only_generation_source "${generation_source}")
string(APPEND comment_only_generation_source
    "\n// vbr_generation_capture_stream\n")
vbr_generation_boundary_valid(
    "${generation_header}" "${comment_only_generation_source}"
    "${kv_header}" "${kv_source}" "${non_owner_text}" comment_only_ok)
if (comment_only_ok)
    message(FATAL_ERROR "VBR generation comment-only negative control did not bite")
endif()

# A new call outside the two owner translation units must fail the same allowlist.
set(misplaced_call_text
    "${non_owner_text}\nvbr_generation_capture_stream(tracker, stream, seq, frontier, cells, output);")
vbr_generation_boundary_valid(
    "${generation_header}" "${generation_source}"
    "${kv_header}" "${kv_source}" "${misplaced_call_text}" misplaced_call_ok)
if (misplaced_call_ok)
    message(FATAL_ERROR "VBR generation misplaced-call negative control did not bite")
endif()

# Process-local tracker and generation records never cross the public/server wire surface.
foreach(path IN ITEMS
        "${SOURCE_ROOT}/include/llama.h"
        "${SOURCE_ROOT}/tools/server/server-task.h"
        "${SOURCE_ROOT}/tools/server/server-task.cpp")
    file(READ "${path}" text)
    foreach(token IN ITEMS vbr_generation_tracker vbr_checkpoint_generation_record)
        string(FIND "${text}" "${token}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR
                "process-local generation state reached API/wire surface: ${path} (${token})")
        endif()
    endforeach()
endforeach()

message(STATUS "VBR generation tracker/artifact isolation invariants PASS")
