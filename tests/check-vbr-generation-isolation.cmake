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
foreach(path IN LISTS implementation_sources)
    if (path STREQUAL "${SOURCE_ROOT}/src/llama-vbr-generation.cpp")
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
