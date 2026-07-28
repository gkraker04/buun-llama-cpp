# B0/C0 source-contract scans [P2]: one closed reason enum in one header, no enum or
# name-table replicas, pinned band starts, policy-free C0 leaf header, and the wired-once
# observer surfaces. Mechanical greps in the spirit of check-vbr-generation-isolation.cmake;
# each scan is negative-controlled in the gate by mutating a file COPY and expecting failure.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(GLOB_RECURSE contract_files
    "${SOURCE_ROOT}/src/*.cpp"    "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/*.cpp"  "${SOURCE_ROOT}/tools/*.h"
    "${SOURCE_ROOT}/tests/*.cpp")

set(all_source "")
foreach(path IN LISTS contract_files)
    file(READ "${path}" text)
    string(APPEND all_source "${text}")
endforeach()

# --- one closed reason enum, defined exactly once, in common/common-cache-plan.h ---
count_literal("${all_source}" "enum common_cache_plan_reason : uint16_t" reason_defs)
if (NOT reason_defs EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one common_cache_plan_reason definition, found ${reason_defs}")
endif()

# (numeric band-start values are pinned at COMPILE TIME by exact-value static_asserts in
# common-cache-plan.h — stronger than any source-text scan, so no grep pin here)

# --- the other B0/C0 closed enums are each defined exactly once ---
foreach(def
        "enum class common_cache_plan_disposition : uint8_t"
        "enum class common_cache_plan_provider : uint8_t"
        "enum class common_cache_plan_outcome : uint8_t"
        "enum class common_cache_plan_selection : uint8_t"
        "enum class common_cache_plan_inventory_state : uint8_t"
        "enum class common_cache_plan_planner_status : uint8_t"
        "enum class llama_cache_acct_category : uint8_t"
        "enum class llama_cache_acct_residency : uint8_t"
        "enum class llama_cache_acct_measure : uint8_t"
        "enum class llama_cache_acct_known : uint8_t"
        "enum class llama_cache_acct_unit : uint8_t"
        "enum class llama_cache_acct_cost_kind : uint8_t"
        "enum class llama_cache_acct_txn_state : uint8_t")
    count_literal("${all_source}" "${def}" def_count)
    if (NOT def_count EQUAL 1)
        message(FATAL_ERROR "expected exactly one definition of '${def}', found ${def_count}")
    endif()
endforeach()

# --- name spellings are SINGULAR: every reason name is extracted mechanically from the
# X-macro list (its one authoritative spelling) and any second quoted occurrence anywhere in
# the tree is a shadow replica. "none" is excluded — it is a legitimate name in other closed
# vocabularies (e.g. the A2 tombstone table). ---
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" plan_header)
string(REGEX MATCHALL "X\\([A-Z_0-9]+, +\"[a-z_0-9]+\"" reason_entries "${plan_header}")
list(LENGTH reason_entries reason_entry_count)
if (reason_entry_count LESS 30)
    message(FATAL_ERROR
        "expected at least 30 X-macro reason entries, found ${reason_entry_count}")
endif()
foreach(entry IN LISTS reason_entries)
    string(REGEX REPLACE ".*\"([a-z_0-9]+)\"" "\\1" name "${entry}")
    if (name STREQUAL "none")
        continue()
    endif()
    count_literal("${all_source}" "\"${name}\"" name_count)
    if (NOT name_count EQUAL 1)
        message(FATAL_ERROR "name-table replica: \"${name}\" spelled ${name_count} times")
    endif()
endforeach()

# non-reason closed names keep the representative replica ban
foreach(name "\"valid_not_chosen_cost\"" "\"restore_failed_fell_back_cold\"" "\"rolling_window_tape\"")
    count_literal("${all_source}" "${name}" name_count)
    if (NOT name_count EQUAL 1)
        message(FATAL_ERROR "name-table replica: ${name} spelled ${name_count} times")
    endif()
endforeach()

# --- C0 leaf header stays policy-free: no name strings, no JSON, no server includes ---
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" acct_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.cpp" acct_source)
foreach(banned "nlohmann" "#include \"server" "const char *")
    string(FIND "${acct_header}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "llama-cache-accounting.h must stay policy/presentation-free (found '${banned}')")
    endif()
    string(FIND "${acct_source}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "llama-cache-accounting.cpp must stay policy/presentation-free (found '${banned}')")
    endif()
endforeach()

# (once-only finalization is a RUNTIME invariant: cache_plan_finalize early-returns and
# fault-counts on an already-finalized record — outcome != unknown is the finalized state)

# VBR env census completeness (D-pins r4): every getenv("VBR_*") in the tree must appear
# in COMMON_CACHE_PLAN_VBR_ENV_LIST, so an override can never silently escape the
# calibration regime identity.
file(READ "${SOURCE_ROOT}/common/common-cache-plan-estimate.h" census_src)
set(vbr_env_names "")
foreach(dir src common tools ggml/src)
    file(GLOB_RECURSE dir_files "${SOURCE_ROOT}/${dir}/*.c" "${SOURCE_ROOT}/${dir}/*.cpp"
                                "${SOURCE_ROOT}/${dir}/*.cu" "${SOURCE_ROOT}/${dir}/*.h"
                                "${SOURCE_ROOT}/${dir}/*.cuh")
    foreach(f ${dir_files})
        file(READ "${f}" body)
        string(REGEX MATCHALL "getenv\\(\"VBR_[A-Z_0-9]+\"" hits "${body}")
        foreach(hit ${hits})
            string(REGEX REPLACE "^getenv\\(\"" "" name "${hit}")
            string(REGEX REPLACE "\"$" "" name "${name}")
            list(APPEND vbr_env_names "${name}")
        endforeach()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES vbr_env_names)
foreach(name ${vbr_env_names})
    string(FIND "${census_src}" "X(\"${name}\"" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "VBR env override ${name} is used in the tree but missing from "
                            "COMMON_CACHE_PLAN_VBR_ENV_LIST — the calibration regime identity "
                            "would not see it")
    endif()
endforeach()
list(LENGTH vbr_env_names n_vbr_env)
message(STATUS "vbr env census covers ${n_vbr_env} overrides")

message(STATUS "cache-plan/accounting contract scans passed")
