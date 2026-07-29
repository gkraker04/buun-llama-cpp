# D-S4 closed destruction-inventory scan.
# Usage: cmake -DSOURCE_ROOT=<repo root> -P tests/check-cache-destruction-inventory.cmake

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-lifecycle.h" inventory_header)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_context)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" server_task)

# Parse the one X-macro rather than maintaining a second function allowlist in CI.
string(REGEX MATCHALL
    "X\\([a-z_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z0-9_]+\\)"
    inventory_entries "${inventory_header}")
list(LENGTH inventory_entries inventory_count)
if (NOT inventory_count EQUAL 6)
    message(FATAL_ERROR
        "D-S4 inventory must contain exactly six classes, found ${inventory_count}")
endif()

set(expected_classes
    slot_drop
    live_range_drop
    host_artifact_drop
    checkpoint_drop
    token_ledger_truncate
    mandatory_recovery_reset)
set(allowed_functions "")
set(admission_owners "")
foreach(entry IN LISTS inventory_entries)
    string(REGEX REPLACE
        "X\\(([a-z_]+),[ \t]+([a-z0-9_]+),[ \t]+([a-z0-9_]+)\\)"
        "\\1;\\2;\\3" parsed "${entry}")
    list(GET parsed 0 cls)
    list(GET parsed 1 fn)
    list(GET parsed 2 admission_owner)
    list(FIND expected_classes "${cls}" cls_index)
    if (cls_index EQUAL -1)
        message(FATAL_ERROR "unexpected D-S4 class in inventory: ${cls}")
    endif()
    list(REMOVE_ITEM expected_classes "${cls}")
    list(APPEND allowed_functions "${fn}")
    list(APPEND admission_owners "${admission_owner}")
endforeach()
if (expected_classes)
    message(FATAL_ERROR "D-S4 inventory omitted classes: ${expected_classes}")
endif()

# Speculative/backup removals are deliberately not lease subjects. This is the one exact
# server-side exclusion seam; library-internal exclusions are censused below.
list(APPEND allowed_functions
    server_cache_transient_seq_rm_impl
    server_cache_transient_token_truncate_impl)

set(raw_re
    "(^|[^a-zA-Z0-9_])(common_context_seq_rm|llama_memory_seq_rm|llama_memory_seq_rm_attn|states\\.(erase|pop_front)|prompt\\.checkpoints\\.(erase|clear)|prompt\\.tokens\\.(clear|keep_first)|prompt\\.clear)[ \t]*\\(")

# Function-aware lexical scan. A raw primitive is legal only while the scanner is inside a
# function whose symbol came from the X-macro (or the one closed transient exclusion). The
# negative controls call this same function on mutated source copies.
function(ds4_scan_translation_unit source output)
    string(REPLACE ";" "\\;" escaped_source "${source}")
    string(REPLACE "\n" ";" lines "${escaped_source}")
    set(active "")
    set(pending "")
    set(depth 0)
    set(errors "")
    set(line_no 0)
    foreach(raw_line IN LISTS lines)
        math(EXPR line_no "${line_no} + 1")
        string(REGEX REPLACE "//.*$" "" line "${raw_line}")

        if (active STREQUAL "")
            if (pending STREQUAL "")
                foreach(fn IN LISTS allowed_functions)
                    if (line MATCHES "(^|[^a-zA-Z0-9_])${fn}[ \t]*\\(")
                        set(pending "${fn}")
                        break()
                    endif()
                endforeach()
            endif()
            if (NOT pending STREQUAL "")
                string(FIND "${line}" "{" open_pos)
                string(FIND "${line}" ";" semicolon_pos)
                if (NOT open_pos EQUAL -1 AND
                    (semicolon_pos EQUAL -1 OR open_pos LESS semicolon_pos))
                    set(active "${pending}")
                    set(pending "")
                    set(depth 0)
                elseif (NOT semicolon_pos EQUAL -1)
                    # It was a call/declaration, not a definition.
                    set(pending "")
                endif()
            endif()
        endif()

        if (line MATCHES "${raw_re}" AND active STREQUAL "")
            list(APPEND errors "line ${line_no}: ${raw_line}")
        endif()

        if (NOT active STREQUAL "")
            string(REGEX MATCHALL "\\{" opens "${line}")
            string(REGEX MATCHALL "\\}" closes "${line}")
            list(LENGTH opens n_open)
            list(LENGTH closes n_close)
            math(EXPR depth "${depth} + ${n_open} - ${n_close}")
            if (depth LESS_EQUAL 0)
                set(active "")
                set(depth 0)
            endif()
        endif()
    endforeach()
    set(${output} "${errors}" PARENT_SCOPE)
endfunction()

ds4_scan_translation_unit("${server_context}" context_errors)
ds4_scan_translation_unit("${server_task}" task_errors)
if (context_errors OR task_errors)
    message(FATAL_ERROR
        "raw destructive primitive escaped the D-S4 inventory:\n"
        "${context_errors}\n${task_errors}")
endif()

# Every inventory-mapped logical admission owner reaches the single API exactly once in source;
# two full-slot classes intentionally share observe_full_slot. Physical _impl and transient paths
# do not recursively admit.
list(REMOVE_DUPLICATES admission_owners)
list(LENGTH admission_owners admission_owner_count)
set(server_retention_source "${server_context}\n${server_task}")
count_literal("${server_retention_source}" "server_cache_retention_admit(" admission_calls)
if (NOT admission_calls EQUAL admission_owner_count)
    message(FATAL_ERROR
        "expected one D-S4 API call per mapped admission owner (${admission_owner_count}), "
        "found ${admission_calls}")
endif()

# Exact library-internal exclusion census. These are mandatory recovery, training-only, public
# forwarding, or draft/speculative internals — none is a server retention-policy seam. Exact
# counts are intentional merge tripwires: upstream movement in these destructive families must
# trigger a human re-audit rather than silently expanding the exclusion set.
file(READ "${SOURCE_ROOT}/src/llama-context.cpp" llama_context)
file(READ "${SOURCE_ROOT}/common/speculative.cpp" speculative)
file(READ "${SOURCE_ROOT}/common/common.cpp" common_source)
count_literal("${llama_context}" "memory->clear(true)" context_full_clears)
count_literal("${llama_context}" "bool llama_memory_seq_rm(" seq_rm_exports)
count_literal("${llama_context}" "bool llama_memory_seq_rm_attn(" seq_rm_attn_exports)
count_literal("${speculative}" "llama_memory_seq_rm(" speculative_seq_rm)
count_literal("${speculative}" "llama_memory_clear(" speculative_clear)
count_literal("${common_source}" "llama_memory_clear(" common_clear)
count_literal("${common_source}" "llama_memory_seq_rm(" common_seq_rm)
if (NOT context_full_clears EQUAL 2 OR
    NOT seq_rm_exports EQUAL 1 OR NOT seq_rm_attn_exports EQUAL 1 OR
    NOT speculative_seq_rm EQUAL 7 OR NOT speculative_clear EQUAL 1 OR
    NOT common_clear EQUAL 1 OR NOT common_seq_rm EQUAL 1)
    message(FATAL_ERROR
        "D-S4 mandatory/transient library exclusion census drifted: "
        "context_clear=${context_full_clears} exports=${seq_rm_exports}/${seq_rm_attn_exports} "
        "spec=${speculative_seq_rm}/${speculative_clear} "
        "common=${common_seq_rm}/${common_clear}")
endif()

# Negative control A: a raw call outside an allowed function must fail the real scanner.
set(raw_negative
    "${server_task}\nvoid unregistered_destroy() { cache.states.erase(cache.states.begin()); }")
ds4_scan_translation_unit("${raw_negative}" raw_negative_errors)
if (NOT raw_negative_errors)
    message(FATAL_ERROR "D-S4 raw-call negative control did not trip")
endif()

# Negative control B: removing one class from the one X-macro must fail exact cardinality.
string(REPLACE
    "    X(checkpoint_drop,          server_cache_checkpoint_drop_impl,          checkpoint_drop) \\\n"
    "" inventory_negative "${inventory_header}")
string(REGEX MATCHALL
    "X\\([a-z_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z0-9_]+\\)"
    negative_entries "${inventory_negative}")
list(LENGTH negative_entries negative_count)
if (negative_count EQUAL 6)
    message(FATAL_ERROR "D-S4 missing-class negative control did not trip")
endif()

message(STATUS
    "D-S4 destruction inventory scan passed: 6 classes, ${admission_owner_count} admission owners, "
    "raw-call + missing-class negative controls")
