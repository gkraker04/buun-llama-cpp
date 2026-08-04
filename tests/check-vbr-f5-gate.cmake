# Milestone-F acceptance harness contract. The retained gate may drive public
# routes and test binaries, but it must not arm a production-only nudge.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/src/llama-vbr-downward.h" downward_header)
file(READ "${SOURCE_ROOT}/src/llama-vbr-downward.cpp" downward_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact-stage.h" stage_header)
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" kv_cache_source)
file(READ "${SOURCE_ROOT}/tools/server/server-vbr-artifact-store.h" store_header)
file(READ "${SOURCE_ROOT}/tests/test-vbr-artifact-adopt.cpp" adopt_test)
file(READ "${SOURCE_ROOT}/tools/server/bench/f5-state-matrix.py" gate_script)

foreach(pin IN ITEMS
        "not_attempted = 0"
        "case vbr_downward_reserve_status::not_attempted: return \"not_attempted\""
        "vbr_downward_reserve_status::not_attempted")
    string(FIND
        "${downward_header}${downward_source}${stage_header}${store_header}"
        "${pin}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "F5 downward not-attempted pin is missing '${pin}'")
    endif()
endforeach()

string(FIND "${kv_cache_source}"
    "vbr_f5_preserve_empty_tiers_ = vbr_freeze_ &&" freeze_guard)
if (freeze_guard EQUAL -1)
    message(FATAL_ERROR
        "F5 empty-tier preservation must remain subordinate to VBR_FREEZE")
endif()

foreach(pin IN ITEMS
        "--f5-cuda"
        "types.size() > 1"
        "require_straddled"
        "target_memory->seq_rm(0, -1, -1)"
        "make_construction_empty_preserve_tiers"
        "vbr_import_decision::native_import")
    string(FIND "${adopt_test}" "${pin}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "F5 retained CUDA matrix is missing '${pin}'")
    endif()
endforeach()

foreach(pin IN ITEMS
        "-ctk"
        "-ctv"
        "vbr"
        "--cache-lifecycle"
        "VBR_FREEZE"
        "VBR_BUDGET_MIB"
        "VBR_F5_PRESERVE_EMPTY_TIERS"
        "action={action}"
        "downward_rebase"
        "live_rebased"
        "not_attempted"
        "wrong-tenant"
        "1.0e-4")
    string(FIND "${gate_script}" "${pin}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "F5 server matrix is missing '${pin}'")
    endif()
endforeach()

# The one reviewed production-side token is a freeze-only empty-boundary latch
# for the routed downward cell. Strip that exact spelling, then reject every
# other F5 nudge/control environment variable.
file(GLOB_RECURSE production_sources
    "${SOURCE_ROOT}/src/*.cpp"
    "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp"
    "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/ggml/src/*.c"
    "${SOURCE_ROOT}/ggml/src/*.cpp"
    "${SOURCE_ROOT}/ggml/src/*.cu"
    "${SOURCE_ROOT}/ggml/src/*.cuh"
    "${SOURCE_ROOT}/ggml/include/*.h"
    "${SOURCE_ROOT}/tools/server/*.cpp"
    "${SOURCE_ROOT}/tools/server/*.h")
foreach(path IN LISTS production_sources)
    file(READ "${path}" text)
    string(REPLACE "VBR_F5_PRESERVE_EMPTY_TIERS" "" text "${text}")
    string(FIND "${text}" "VBR_F5_" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "F5 production-inertness violated by gate-only token in ${path}")
    endif()
endforeach()
set(negative "${downward_source}\nconst char * bad = \"VBR_F5_FORCE_TIER\";")
string(REPLACE "VBR_F5_PRESERVE_EMPTY_TIERS" "" negative "${negative}")
string(FIND "${negative}" "VBR_F5_" negative_found)
if (negative_found EQUAL -1)
    message(FATAL_ERROR "F5 production-inertness negative control did not trip")
endif()

message(STATUS "F5 state-matrix harness contracts passed")

# Case-sensitivity gap closure: the lowercase member must appear at EXACTLY its
# four sanctioned sites in llama-kv-cache.cpp (declaration, freeze-subordinate
# latch, and the two empty-boundary full-reset guards); any new consultation
# elsewhere fails this count.
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" kv_source_f5)
string(REGEX MATCHALL "vbr_f5_preserve_empty_tiers_" latch_member_hits "${kv_source_f5}")
list(LENGTH latch_member_hits latch_member_count)
if (NOT latch_member_count EQUAL 4)
    message(FATAL_ERROR
        "F5 latch member consulted at ${latch_member_count} sites in llama-kv-cache.cpp (expect 4)")
endif()
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.h" kv_header_f5)
string(REGEX MATCHALL "vbr_f5_preserve_empty_tiers_" latch_header_hits "${kv_header_f5}")
list(LENGTH latch_header_hits latch_header_count)
if (NOT latch_header_count EQUAL 1)
    message(FATAL_ERROR
        "F5 latch member declared ${latch_header_count} times in llama-kv-cache.h (expect 1)")
endif()
