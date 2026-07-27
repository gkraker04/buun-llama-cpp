if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

function(count_literal text needle output)
    string(LENGTH "${text}" before)
    string(REPLACE "${needle}" "" stripped "${text}")
    string(LENGTH "${stripped}" after)
    string(LENGTH "${needle}" needle_length)
    math(EXPR count "(${before} - ${after}) / ${needle_length}")
    set(${output} "${count}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE implementation_sources
    "${SOURCE_ROOT}/src/*.cpp"
    "${SOURCE_ROOT}/common/*.cpp"
    "${SOURCE_ROOT}/tools/server/*.cpp")

set(all_implementation_source "")
foreach(path IN LISTS implementation_sources)
    file(READ "${path}" text)
    string(APPEND all_implementation_source "${text}")
endforeach()

# The closed A0 registrant inventory must have one exhaustive A1 dispatch classification.
count_literal("${all_implementation_source}"
    "VBR_GENERATION_MUTATION_DISPATCH_EXHAUSTIVE" dispatch_checks)
if (NOT dispatch_checks EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one exhaustive A1 mutation-dispatch classification")
endif()

# Raw generation fields may be compared only by the one eligibility authority. Mutation code uses
# tracker methods; the disabled oracle independently uses canonical bytes plus covered masks.
# C2 hardening: the scan covers headers as well as sources; the only exemptions are the files
# where these identifiers ARE the struct/tracker definitions themselves.
file(GLOB_RECURSE raw_scan_files
    "${SOURCE_ROOT}/src/*.cpp" "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/server/*.cpp" "${SOURCE_ROOT}/tools/server/*.h")
set(raw_generation_symbols
    "captured_page_gen"
    "page_event_gen"
    "page_last_destructive_gen"
    "page_last_import_gen"
    "cell_last_dependency_gen"
    "cell_last_membership_gen"
    "cell_dependency_provenance"
    "cell_membership_provenance"
    "global_generation_"
    "repr_gen")
foreach(path IN LISTS raw_scan_files)
    if (path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-generation.cpp" OR
        path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-generation.h" OR
        path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-generation-types.h")
        continue()
    endif()
    file(READ "${path}" text)
    foreach(symbol IN LISTS raw_generation_symbols)
        string(FIND "${text}" "${symbol}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR
                "raw VBR generation access outside checkpoint_vbr_eligibility: ${path} (${symbol})")
        endif()
    endforeach()
endforeach()

file(READ "${SOURCE_ROOT}/src/llama-vbr-generation.cpp" generation_source)
count_literal("${generation_source}"
    "VBR_GENERATION_ELIGIBILITY_AUTHORITY" evaluator_markers)
if (NOT evaluator_markers EQUAL 1)
    message(FATAL_ERROR "the sole VBR generation eligibility authority marker is missing")
endif()

# C2: three separately counted narrow authority regions — (a) §5.5 classifier, (b) capture
# helpers, (c) evaluator. The raw-comparison ban re-runs over ALL text outside the three
# regions, so tracker mutation code between them cannot silently grow record-comparison logic.
set(generation_outside "${generation_source}")
foreach(region CLASSIFIER CAPTURE EVALUATOR)
    set(begin_marker "VBR_GENERATION_${region}_REGION_BEGIN")
    set(end_marker   "VBR_GENERATION_${region}_REGION_END")
    count_literal("${generation_outside}" "${begin_marker}" begin_count)
    count_literal("${generation_outside}" "${end_marker}"   end_count)
    if (NOT begin_count EQUAL 1 OR NOT end_count EQUAL 1)
        message(FATAL_ERROR
            "expected exactly one ${region} authority region (begin=${begin_count} end=${end_count})")
    endif()
    string(FIND "${generation_outside}" "${begin_marker}" begin_pos)
    string(FIND "${generation_outside}" "${end_marker}"   end_pos)
    if (begin_pos EQUAL -1 OR end_pos LESS begin_pos)
        message(FATAL_ERROR "malformed ${region} authority region markers")
    endif()
    string(SUBSTRING "${generation_outside}" 0 ${begin_pos} region_before)
    string(LENGTH "${end_marker}" end_marker_length)
    math(EXPR after_pos "${end_pos} + ${end_marker_length}")
    string(SUBSTRING "${generation_outside}" ${after_pos} -1 region_after)
    set(generation_outside "${region_before}${region_after}")
endforeach()
foreach(banned IN ITEMS "captured_page_gen" "classify_expected_tombstone")
    string(FIND "${generation_outside}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "record-comparison symbol '${banned}' escaped the narrow authority regions in llama-vbr-generation.cpp")
    endif()
endforeach()
string(FIND "${generation_source}" "vbr_generation_oracle_" oracle_in_evaluator)
if (NOT oracle_in_evaluator EQUAL -1)
    message(FATAL_ERROR "the debug byte oracle must never become an admission input")
endif()

# Separate trust domain: the oracle may compare its independently reconstructed set to the
# production record, but cannot import tracker/index/count/mask-builder helpers.
file(READ "${SOURCE_ROOT}/src/llama-vbr-generation-oracle.cpp" oracle_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-generation-oracle.h" oracle_header)
string(FIND "${oracle_header}" "\"llama-vbr-generation.h\"" production_header_import)
if (NOT production_header_import EQUAL -1)
    message(FATAL_ERROR
        "VBR generation oracle imported the production tracker/helper header")
endif()
foreach(forbidden IN ITEMS
        "vbr_generation_tracker"
        "vbr_generation_capture_stream"
        "exact_dependency_count"
        "dependency_generation("
        "page_generation(")
    string(FIND "${oracle_source}" "${forbidden}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "VBR generation oracle imported a production dependency helper: ${forbidden}")
    endif()
endforeach()

# A1 stamps, tracker state, and process-local stability/provenance fields do not cross any current
# C/state/checkpoint/server envelope. A future lineage envelope needs a separately reviewed wire
# type rather than serializing these live structs.
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
        foreach(process_local_symbol IN ITEMS
                vbr_generation_tracker
                vbr_generation_event
                vbr_checkpoint_generation_record
                vbr_unit_generation
                vbr_extent_ref
                vbr_extent_handle
                vbr_extent_store
                vbr_recovery_capability
                vbr_failed_operation_record)
            string(FIND "${text}" "${process_local_symbol}" found)
            if (NOT found EQUAL -1)
                message(FATAL_ERROR
                    "process-local VBR generation state reached serialization/API surface: ${path} (${process_local_symbol})")
            endif()
        endforeach()
    endif()
endforeach()

message(STATUS "VBR generation isolation source invariants PASS")

# --- A2 additions ----------------------------------------------------------------------------

# The §5.5 tombstone classifier is evaluator-private: exactly one definition and no callers
# outside the sole evaluator translation unit.
file(GLOB_RECURSE a2_scan_files
    "${SOURCE_ROOT}/src/*.cpp" "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/server/*.cpp" "${SOURCE_ROOT}/tools/server/*.h")
foreach(path IN LISTS a2_scan_files)
    file(READ "${path}" text)
    string(FIND "${text}" "classify_expected_tombstone" found)
    if (NOT found EQUAL -1 AND NOT path MATCHES "llama-vbr-generation\.cpp$")
        message(FATAL_ERROR "tombstone classification escaped the evaluator TU: ${path}")
    endif()
    # Committed-extent admission lookups stay inside the evaluator/store/tests trust domain.
    string(FIND "${text}" "lookup_committed" found)
    if (NOT found EQUAL -1 AND NOT path MATCHES "llama-vbr-(generation|extent)\.(cpp|h)$")
        message(FATAL_ERROR "committed-extent admission lookup escaped its trust domain: ${path}")
    endif()
    # Recovery capabilities are registry-minted only.
    string(FIND "${text}" "vbr_recovery_mint" found)
    if (NOT found EQUAL -1 AND NOT path MATCHES "llama-vbr-operation\.(cpp|h)$|llama-kv-cache\.cpp$")
        message(FATAL_ERROR "recovery capability minted outside the registry trust domain: ${path}")
    endif()
endforeach()

# C4 (v3.2): exactly two awaiting_ack transition sites — the fail-closed capability destructor
# and explicit resolve_quarantined. Both retain the record + manifest targets; ONLY the
# tokenized ack (single site) reclaims the slot after the owning tracker's invalidation.
file(READ "${SOURCE_ROOT}/src/llama-vbr-operation.cpp" op_text)
string(REGEX MATCHALL "state[ ]*=[ ]*vbr_recovery_state::awaiting_ack" quarantine_sites "${op_text}")
list(LENGTH quarantine_sites quarantine_count)
if (NOT quarantine_count EQUAL 3)
    message(FATAL_ERROR "expected exactly three awaiting_ack transition sites (destructor + resolve + recorded-advancement), found ${quarantine_count}")
endif()
string(REGEX MATCHALL "vbr_recovery_ack_quarantine" ack_sites "${op_text}")
list(LENGTH ack_sites ack_count)
if (ack_count GREATER 2)  # declaration-adjacent + one definition
    message(FATAL_ERROR "quarantine ack must have exactly one implementation site, found ${ack_count} mentions")
endif()

message(STATUS "A2 extent/recovery/classifier isolation invariants PASS")

# --- A2 commit-2 additions (checkpoint shadow bridge) -----------------------------------------

# same implementation tree as the raw-symbol scan, plus the public API, the tests, and every
# .inc fragment (the compose fragment carries the record type and must stay allowlisted)
set(c2_scan_files ${raw_scan_files})
file(GLOB_RECURSE c2_extra_files
    "${SOURCE_ROOT}/include/*.h"
    "${SOURCE_ROOT}/src/*.inc"
    "${SOURCE_ROOT}/tests/*.cpp" "${SOURCE_ROOT}/tests/*.h")
list(APPEND c2_scan_files ${c2_extra_files})

# The opaque bridge handle (and its free-function prefix) may be named ONLY by the two bridge
# TUs, the bridge header, and the two named test files. Exact files — no directory exemptions.
set(c2_handle_allowed
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint.h"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint.cpp"
    "${SOURCE_ROOT}/common/common-checkpoint-shadow.cpp"
    "${SOURCE_ROOT}/tests/test-vbr-representation-epoch.cpp"
    "${SOURCE_ROOT}/tests/test-checkpoint-shadow-lifecycle.cpp")

# The composite generation record may be named ONLY where the record vocabulary is defined, in
# the two record-typed implementation TUs, and in the two named test files.
set(c2_record_allowed
    "${SOURCE_ROOT}/src/llama-vbr-generation-types.h"
    "${SOURCE_ROOT}/src/llama-vbr-generation.h"
    "${SOURCE_ROOT}/src/llama-vbr-generation.cpp"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint.cpp"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint-compose.inc"
    "${SOURCE_ROOT}/tests/test-vbr-representation-epoch.cpp"
    "${SOURCE_ROOT}/tests/test-checkpoint-shadow-lifecycle.cpp")

foreach(path IN LISTS c2_scan_files)
    file(READ "${path}" text)
    string(FIND "${text}" "llama_vbr_checkpoint_shadow" found)
    if (NOT found EQUAL -1)
        list(FIND c2_handle_allowed "${path}" allowed)
        if (allowed EQUAL -1)
            message(FATAL_ERROR "opaque checkpoint bridge handle escaped its allowlist: ${path}")
        endif()
    endif()
    string(FIND "${text}" "vbr_checkpoint_generation_record" found)
    if (NOT found EQUAL -1)
        list(FIND c2_record_allowed "${path}" allowed)
        if (allowed EQUAL -1)
            message(FATAL_ERROR "composite generation record escaped its allowlist: ${path}")
        endif()
    endif()
endforeach()

# F4 (verify round): the opaque bridge header must stay record-free — it may not include any
# generation header (the record would become transitively visible to the common holder TU).
file(READ "${SOURCE_ROOT}/src/llama-vbr-checkpoint.h" c2_bridge_header)
string(FIND "${c2_bridge_header}" "llama-vbr-generation" bridge_generation_include)
if (NOT bridge_generation_include EQUAL -1)
    message(FATAL_ERROR "the opaque bridge header imported generation vocabulary")
endif()

# F3 (verify round): the canonical oracle observer region in llama-kv-cache.cpp is scanned for
# forbidden production-index inputs — the observation builder must stay a direct cell scan.
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" c2_kv_source)
foreach(marker IN ITEMS "VBR_GENERATION_ORACLE_OBSERVER_REGION_BEGIN" "VBR_GENERATION_ORACLE_OBSERVER_REGION_END")
    count_literal("${c2_kv_source}" "${marker}" observer_marker_count)
    if (NOT observer_marker_count EQUAL 1)
        message(FATAL_ERROR "expected exactly one ${marker}")
    endif()
endforeach()
string(FIND "${c2_kv_source}" "VBR_GENERATION_ORACLE_OBSERVER_REGION_BEGIN" observer_begin)
string(FIND "${c2_kv_source}" "VBR_GENERATION_ORACLE_OBSERVER_REGION_END"   observer_end)
if (observer_end LESS observer_begin)
    message(FATAL_ERROR "malformed oracle observer region markers")
endif()
math(EXPR observer_length "${observer_end} - ${observer_begin}")
string(SUBSTRING "${c2_kv_source}" ${observer_begin} ${observer_length} observer_region)
foreach(banned IN ITEMS "vbr_ownership_" "rank_below" "enumerate_owned")
    string(FIND "${observer_region}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "production ownership-index input reached the oracle observer region: ${banned}")
    endif()
endforeach()

# common.h names the opaque holder exactly twice (forward declaration + owning member) and
# never the raw bridge handle.
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
count_literal("${common_header}" "common_checkpoint_shadow" holder_mentions)
if (NOT holder_mentions EQUAL 2)
    message(FATAL_ERROR
        "common.h must name the checkpoint shadow holder exactly twice, found ${holder_mentions}")
endif()

# F12/ODR guard: the handle struct definition is mirrored token-identically between the bridge
# TU and the test-only factory in the named lifecycle test.
set(c2_handle_definition
    "struct llama_vbr_checkpoint_shadow {\n    vbr_checkpoint_generation_record record;\n};")
foreach(path IN ITEMS
        "${SOURCE_ROOT}/src/llama-vbr-checkpoint.cpp"
        "${SOURCE_ROOT}/tests/test-checkpoint-shadow-lifecycle.cpp")
    file(READ "${path}" text)
    string(FIND "${text}" "${c2_handle_definition}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR
            "canonical llama_vbr_checkpoint_shadow definition missing or diverged in ${path}")
    endif()
endforeach()

message(STATUS "A2 commit-2 checkpoint shadow bridge invariants PASS")
