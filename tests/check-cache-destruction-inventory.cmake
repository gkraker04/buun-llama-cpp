# D-S4 closed destruction-inventory scan.
# Usage: cmake -DSOURCE_ROOT=<repo root> -P tests/check-cache-destruction-inventory.cmake

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-lifecycle.h" inventory_header)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_context)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" server_task)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.h" quote_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.cpp" quote_source)

# Parse the one X-macro rather than maintaining a second function allowlist in CI.
string(REGEX MATCHALL
    "X\\([a-z_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z_]+\\)"
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
set(release_owners "")
foreach(entry IN LISTS inventory_entries)
    string(REGEX REPLACE
        "X\\(([a-z_]+),[ \t]+([a-z0-9_]+),[ \t]+([a-z0-9_]+),[ \t]+([a-z_]+)\\)"
        "\\1;\\2;\\3;\\4" parsed "${entry}")
    list(GET parsed 0 cls)
    list(GET parsed 1 fn)
    list(GET parsed 2 admission_owner)
    list(GET parsed 3 release_owner)
    list(FIND expected_classes "${cls}" cls_index)
    if (cls_index EQUAL -1)
        message(FATAL_ERROR "unexpected D-S4 class in inventory: ${cls}")
    endif()
    list(REMOVE_ITEM expected_classes "${cls}")
    list(APPEND allowed_functions "${fn}")
    list(APPEND admission_owners "${admission_owner}")
    list(APPEND release_owners "${cls}:${release_owner}")
endforeach()
if (expected_classes)
    message(FATAL_ERROR "D-S4 inventory omitted classes: ${expected_classes}")
endif()

# D-A0b wiring: policy objects and lease inspection are live under debug OR
# lifecycle, while the cache-plan serialization observer remains debug-only.
function(da0b_lifecycle_wiring_valid source output)
    string(FIND "${source}" "if (params_base.cache_debug) {\n            cache_plan_obs =" debug_emit_gate)
    string(FIND "${source}" "if (params_base.cache_debug || params_base.cache_lifecycle) {" lifecycle_gate)
    string(FIND "${source}" "cache_authority->destruction.lease_evaluator =" lease_wiring)
    if (debug_emit_gate EQUAL -1 OR lifecycle_gate EQUAL -1 OR
        lease_wiring LESS_EQUAL lifecycle_gate)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

da0b_lifecycle_wiring_valid("${server_context}" lifecycle_wiring_valid)
if (NOT lifecycle_wiring_valid)
    message(FATAL_ERROR "D-A0b debug-or-lifecycle lease wiring contract drifted")
endif()

# Attribution gates: unified-KV idle purge and the durable idle-save purge are
# idle_reclaim, while a genuine LoRA/slot rebind retains slot_rebind default.
contract_extract_region(
    "${server_context}"
    "bool try_clear_idle_slots()"
    "bool launch_slot_with_task("
    idle_body idle_found)
if (NOT idle_found)
    message(FATAL_ERROR "D-A0b idle-reclaim attribution seam missing")
endif()
string(FIND "${idle_body}" "server_cache_destruction_reason::idle_reclaim" idle_reason)
string(FIND "${server_context}" "// [TAG_IDLE_SLOT_CLEAR]" tagged_idle)
string(FIND "${server_context}" "clearing cache for lora change." true_rebind)
if (tagged_idle GREATER_EQUAL 0)
    string(SUBSTRING "${server_context}" ${tagged_idle} 240 tagged_body)
else()
    set(tagged_body "")
endif()
if (true_rebind GREATER_EQUAL 0)
    string(SUBSTRING "${server_context}" ${true_rebind} 320 rebind_body)
else()
    set(rebind_body "")
endif()
string(FIND "${tagged_body}" "server_cache_destruction_reason::idle_reclaim" tagged_reason)
string(FIND "${rebind_body}" "slot.prompt_clear();" rebind_default)
if (idle_reason EQUAL -1 OR tagged_idle EQUAL -1 OR
    tagged_reason EQUAL -1 OR true_rebind EQUAL -1 OR
    rebind_default EQUAL -1)
    message(FATAL_ERROR "D-A0b destruction-reason attribution drifted")
endif()
string(REPLACE
    "if (params_base.cache_debug || params_base.cache_lifecycle) {"
    "if (params_base.cache_debug) {"
    lifecycle_negative "${server_context}")
da0b_lifecycle_wiring_valid("${lifecycle_negative}" lifecycle_negative_valid)
if (lifecycle_negative_valid)
    message(FATAL_ERROR "D-A0b lifecycle-without-debug negative control did not trip")
endif()

set(expected_release_owners
    "slot_drop:none;live_range_drop:none;host_artifact_drop:legacy_wrapper_or_capability;checkpoint_drop:none;token_ledger_truncate:none;mandatory_recovery_reset:none")
if (NOT release_owners STREQUAL expected_release_owners)
    message(FATAL_ERROR
        "D-A0b accounting release-owner inventory drifted: ${release_owners}")
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
    "    X(checkpoint_drop,          server_cache_checkpoint_drop_impl,          checkpoint_drop,                     none) \\\n"
    "" inventory_negative "${inventory_header}")
string(REGEX MATCHALL
    "X\\([a-z_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z0-9_]+,[ \t]+[a-z_]+\\)"
    negative_entries "${inventory_negative}")
list(LENGTH negative_entries negative_count)
if (negative_count EQUAL 6)
    message(FATAL_ERROR "D-S4 missing-class negative control did not trip")
endif()

# D-A0b release ownership: raw authority primitives are physical-only. The
# lifecycle-off wrapper has the one legacy release terminal; capability commit
# becomes the alternate terminal only after a D-A ratchet flips.
function(da0b_host_raw_release_free source output)
    contract_extract_region(
        "${source}"
        "static server_prompt_cache::iterator server_prompt_cache_destroy_entry_impl("
        "server_prompt_cache::iterator server_prompt_cache::destroy_entry("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    contract_find_forbidden(
        "${body}" forbidden "acct_release_entry" "->release(")
    if (NOT forbidden)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

da0b_host_raw_release_free("${server_task}" raw_release_free)
if (NOT raw_release_free)
    message(FATAL_ERROR "D-A0b raw host primitive contains an accounting release")
endif()
string(FIND "${server_task}" "auto next = server_prompt_cache_destroy_entry_impl(*this, it);" raw_call_pos)
string(FIND "${server_task}" "(void) acct->release(op);" legacy_release_pos)
if (raw_call_pos EQUAL -1 OR legacy_release_pos LESS_EQUAL raw_call_pos)
    message(FATAL_ERROR "D-A0b legacy wrapper does not own the post-mutation release terminal")
endif()
count_literal(
    "${server_task}" "server_prompt_cache_destroy_entry_impl(" raw_impl_sites)
if (NOT raw_impl_sites EQUAL 2)
    message(FATAL_ERROR
        "D-A0b raw host primitive must have one definition and one censused wrapper call; found ${raw_impl_sites}")
endif()

# Negative control C: injecting an internal release into the raw primitive must
# fail the same body scan.
string(REPLACE
    "    return cache.states.erase(it);"
    "    cache.acct_release_entry(*it); return cache.states.erase(it);"
    internal_release_negative "${server_task}")
da0b_host_raw_release_free("${internal_release_negative}" internal_negative_valid)
if (internal_negative_valid)
    message(FATAL_ERROR "D-A0b internal-release negative control did not trip")
endif()

# Prepared release carries no ledger-writing callback through the boundary.
# It binds the scheduler thread at prepare and checks it at the sole commit.
foreach(pin
        "server_cache_recovery_pin"
        "server_cache_prepare_release_set("
        "scheduler_owner_ = std::this_thread::get_id();"
        "scheduler_owner_ == std::this_thread::get_id();"
        "release_.commit()")
    string(FIND "${quote_header}${quote_source}" "${pin}" pin_pos)
    if (pin_pos EQUAL -1)
        message(FATAL_ERROR "D-A0b prepared-release concurrency pin missing: ${pin}")
    endif()
endforeach()
function(da0b_commit_region_valid source output)
    contract_extract_region(
        "${source}"
        "server_cache_prepared_release_capability::commit("
        "server_cache_prepare_release_result server_cache_prepare_release_set("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    contract_find_forbidden(
        "${body}" forbidden
        "std::function"
        "callback"
        "gauge_set("
        "reserve("
        "stage("
        "preview_release_set(")
    if (forbidden)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

da0b_commit_region_valid("${quote_source}" commit_region_valid)
if (NOT commit_region_valid)
    message(FATAL_ERROR "D-A0b prepare-to-commit region admits a ledger producer/callback")
endif()
string(REPLACE
    "switch (release_.commit()) {"
    "ledger.gauge_set(); switch (release_.commit()) {"
    callback_negative "${quote_source}")
da0b_commit_region_valid("${callback_negative}" callback_negative_valid)
if (callback_negative_valid)
    message(FATAL_ERROR "D-A0b ledger-callback negative control did not trip")
endif()

# D-A1 is the first live capability terminal. The host raw erase and the
# prepared C commit are adjacent on the scheduler owner thread: no callback,
# sidecar retirement, or ledger producer may enter this physical-mutation gap.
function(da1_host_commit_gap_valid source output)
    contract_extract_region(
        "${source}"
        "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
        "const auto release_status = prepared.commit();"
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    contract_find_forbidden(
        "${body}" forbidden
        "gauge_set("
        "reserve("
        "stage("
        "preview_release_set("
        "release("
        "clone("
        "retire("
        "server_cache_retention_admit("
        "server_fault("
        "std::function")
    if (forbidden)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

da1_host_commit_gap_valid("${server_task}" host_commit_gap_valid)
string(FIND "${server_task}"
    "GGML_ASSERT(scheduler_owner == std::this_thread::get_id());"
    da1_owner_assert)
if (NOT host_commit_gap_valid OR da1_owner_assert EQUAL -1)
    message(FATAL_ERROR
        "D-A1 host prepare-to-commit owner-thread/no-callback contract drifted")
endif()
string(REPLACE
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it); acct->gauge_set();"
    da1_gap_negative "${server_task}")
da1_host_commit_gap_valid("${da1_gap_negative}" da1_gap_negative_valid)
if (da1_gap_negative_valid)
    message(FATAL_ERROR "D-A1 host commit-gap negative control did not trip")
endif()
string(REPLACE
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it); acct->release({});"
    da1_release_negative "${server_task}")
da1_host_commit_gap_valid(
    "${da1_release_negative}" da1_release_negative_valid)
if (da1_release_negative_valid)
    message(FATAL_ERROR
        "D-A1 host commit-gap release negative control did not trip")
endif()

# CACHE_HOST_LIFECYCLE is debug evidence, not an Observed-template signal:
# B authority also supplies a record when --cache-debug is absent.
string(FIND "${server_context}"
    "prompt_cache->debug_observability = params_base.cache_debug;"
    da1_debug_wiring)
contract_extract_region(
    "${server_task}"
    "void server_prompt_cache::commit_restore_delivery("
    "// Lifecycle-off is the historical move/rebind/erase terminal verbatim."
    da1_restore_commit da1_restore_commit_found)
string(FIND "${da1_restore_commit}"
    "if (debug_observability) {" da1_debug_emit_gate)
string(FIND "${da1_restore_commit}"
    "debug_lifecycle_emissions++;" da1_debug_emit_count)
string(FIND "${da1_restore_commit}"
    "CACHE_HOST_LIFECYCLE" da1_debug_emit)
if (da1_debug_wiring EQUAL -1 OR NOT da1_restore_commit_found OR
    da1_debug_emit_gate EQUAL -1 OR
    da1_debug_emit_count LESS da1_debug_emit_gate OR
    da1_debug_emit LESS da1_debug_emit_count)
    message(FATAL_ERROR "D-A1 debug-only lifecycle evidence gate drifted")
endif()

message(STATUS
    "D-S4 destruction inventory scan passed: 6 classes, ${admission_owner_count} admission owners, "
    "raw-call + missing-class + D-A0b release-owner + D-A1 commit-gap/debug-emission negative controls")
