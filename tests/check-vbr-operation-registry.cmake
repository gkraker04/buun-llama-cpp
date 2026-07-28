if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(GLOB_RECURSE vbr_sources
    "${SOURCE_ROOT}/src/*.cpp"
    "${SOURCE_ROOT}/src/*.h")
set(all_vbr_source "")
foreach(path IN LISTS vbr_sources)
    file(READ "${path}" text)
    string(APPEND all_vbr_source "${text}")
endforeach()

# One allocator definition and one mint site for the whole process.
count_literal("${all_vbr_source}" "VBR_OPERATION_ALLOCATOR_DEFINITION" allocator_defs)
if (NOT allocator_defs EQUAL 1)
    message(FATAL_ERROR "expected exactly one VBR operation allocator definition, got ${allocator_defs}")
endif()
count_literal("${all_vbr_source}" "VBR_OPERATION_MINT_SITE" mint_sites)
if (NOT mint_sites EQUAL 1)
    message(FATAL_ERROR "expected exactly one VBR operation-ID mint site, got ${mint_sites}")
endif()

# Composite/leaf memories may validate and forward an ID, but must never mint one. Keep the
# allowlist deliberately closed so a new memory implementation cannot silently become an allocator.
foreach(path IN LISTS vbr_sources)
    if (path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-operation.cpp" OR
        path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-operation.h" OR
        path STREQUAL "${SOURCE_ROOT}/src/llama-context.cpp")
        continue()
    endif()
    file(READ "${path}" text)
    string(FIND "${text}" "vbr_operation_registry_begin(" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "operation-ID minting outside the allocator/top-level wrapper is forbidden: ${path}")
    endif()
endforeach()

# iSWA must forward the same named value to both armed children. Runtime equality is additionally
# asserted by test-vbr-representation-epoch.
file(READ "${SOURCE_ROOT}/src/llama-kv-cache-iswa.cpp" iswa_source)
count_literal("${iswa_source}" "vbr_retier_freeze_begin(owner, operation_id)" iswa_forwards)
if (NOT iswa_forwards EQUAL 2)
    message(FATAL_ERROR "iSWA must forward one identical operation_id to both children")
endif()
string(FIND "${iswa_source}" "void llama_kv_cache_iswa::vbr_retier_freeze_end(" iswa_end_begin)
string(FIND "${iswa_source}" "llama_memory_vbr_preflight_data llama_kv_cache_iswa::vbr_retier_preflight("
    iswa_end_finish)
if (iswa_end_begin EQUAL -1 OR iswa_end_finish EQUAL -1 OR
    iswa_end_finish LESS_EQUAL iswa_end_begin)
    message(FATAL_ERROR "could not isolate the iSWA retier-freeze end implementation")
endif()
math(EXPR iswa_end_length "${iswa_end_finish} - ${iswa_end_begin}")
string(SUBSTRING "${iswa_source}" ${iswa_end_begin} ${iswa_end_length} iswa_end_source)
string(FIND "${iswa_end_source}" "vbr_operation_armed()" mutable_end_guard)
if (NOT mutable_end_guard EQUAL -1)
    message(FATAL_ERROR "iSWA freeze end must pair from its immutable begin record, not armed policy")
endif()
foreach(path IN ITEMS
        "${SOURCE_ROOT}/src/llama-memory-hybrid.h"
        "${SOURCE_ROOT}/src/llama-memory-hybrid-iswa.h")
    file(READ "${path}" text)
    count_literal("${text}" "vbr_retier_freeze_begin(owner, operation_id)" hybrid_forwards)
    if (NOT hybrid_forwards EQUAL 1)
        message(FATAL_ERROR "hybrid wrapper must forward its caller-provided operation_id: ${path}")
    endif()
endforeach()

# The strong process-local identity must not cross known state/checkpoint/server envelopes.
set(serialization_files
    "${SOURCE_ROOT}/include/llama.h"
    "${SOURCE_ROOT}/common/common.h"
    "${SOURCE_ROOT}/common/common.cpp"
    "${SOURCE_ROOT}/src/llama-io.h"
    "${SOURCE_ROOT}/src/llama-io.cpp"
    "${SOURCE_ROOT}/tools/server/server-common.h"
    "${SOURCE_ROOT}/tools/server/server-common.cpp"
    "${SOURCE_ROOT}/tools/server/server-task.h"
    "${SOURCE_ROOT}/tools/server/server-task.cpp")
foreach(path IN LISTS serialization_files)
    if (EXISTS "${path}")
        file(READ "${path}" text)
        foreach(process_local_symbol IN ITEMS vbr_operation_id vbr_recovery_capability)
            string(FIND "${text}" "${process_local_symbol}" found)
            if (NOT found EQUAL -1)
                message(FATAL_ERROR
                    "process-local ${process_local_symbol} reached serialization/API surface: ${path}")
            endif()
        endforeach()
    endif()
endforeach()

# The compiler performs the real exhaustive checks; these markers make accidental removal visible
# even in source-only CI jobs.
count_literal("${all_vbr_source}" "VBR_MUTATION_INVENTORY_EXHAUSTIVE" inventory_checks)
count_literal("${all_vbr_source}" "VBR_OPERATION_KIND_EXHAUSTIVE" kind_checks)
count_literal("${all_vbr_source}" "VBR_STABLE_READ_INVENTORY_EXHAUSTIVE" stable_read_checks)
if (NOT inventory_checks EQUAL 1 OR NOT kind_checks EQUAL 1 OR NOT stable_read_checks EQUAL 1)
    message(FATAL_ERROR "closed VBR enum/registry exhaustive checks are missing")
endif()

message(STATUS "VBR operation registry source invariants PASS")
