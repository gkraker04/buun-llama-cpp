# D-S4 closed destruction-inventory scan.
# Usage: cmake -DSOURCE_ROOT=<repo root> -P tests/check-cache-destruction-inventory.cmake

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-lifecycle.h" inventory_header)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_context)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" server_task)
file(READ "${SOURCE_ROOT}/tests/test-server-prompt-cache.cpp" server_task_test)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.h" quote_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.cpp" quote_source)
file(READ "${SOURCE_ROOT}/tools/server/server-retention-sidecar.cpp" sidecar_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-authority.cpp" authority_source)

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
    string(FIND "${source}" "if (params_base.cache_optimizer.cache_debug) {\n            cache_plan_obs =" debug_emit_gate)
    string(FIND "${source}" "if (params_base.cache_optimizer.cache_debug ||\n            params_base.cache_optimizer.cache_lifecycle) {" lifecycle_gate)
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
    "if (params_base.cache_optimizer.cache_debug ||\n            params_base.cache_optimizer.cache_lifecycle) {"
    "if (params_base.cache_optimizer.cache_debug) {"
    lifecycle_negative "${server_context}")
da0b_lifecycle_wiring_valid("${lifecycle_negative}" lifecycle_negative_valid)
if (lifecycle_negative_valid)
    message(FATAL_ERROR "D-A0b lifecycle-without-debug negative control did not trip")
endif()

set(expected_release_owners
    "slot_drop:legacy_wrapper_or_capability;live_range_drop:none;host_artifact_drop:legacy_wrapper_or_capability;checkpoint_drop:legacy_wrapper_or_capability;token_ledger_truncate:none;mandatory_recovery_reset:none")
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
set(server_retention_source
    "${server_context}\n${server_task}\n${authority_source}")
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
    "    X(checkpoint_drop,          server_cache_checkpoint_drop_impl,          observe_checkpoint_drop,             legacy_wrapper_or_capability) \\\n"
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
if (NOT raw_impl_sites EQUAL 4)
    message(FATAL_ERROR
        "D-A/ZC raw host primitive must have one definition and three censused capability/wrapper calls; found ${raw_impl_sites}")
endif()

# ZC's narrow prepared-release adapter owns a third censused mutation edge.
# Its immutable-inventory caller may advance after a refusal, but once this
# edge erases a node the exact accounting commit must be adjacent and fatal on
# drift, matching the already-landed capability terminals.
function(zc_host_commit_gap_valid source output)
    contract_extract_region(
        "${source}"
        "(void) server_prompt_cache_destroy_entry_impl(*this, it);"
        "const auto commit_status = prepared.commit();"
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
        "->release("
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

zc_host_commit_gap_valid("${server_task}" zc_host_commit_gap_ok)
if (NOT zc_host_commit_gap_ok)
    message(FATAL_ERROR "ZC host prepared-release commit gap drifted")
endif()
string(REPLACE
    "(void) server_prompt_cache_destroy_entry_impl(*this, it);"
    "(void) server_prompt_cache_destroy_entry_impl(*this, it); acct->release({});"
    zc_gap_negative "${server_task}")
zc_host_commit_gap_valid("${zc_gap_negative}" zc_gap_negative_ok)
if (zc_gap_negative_ok)
    message(FATAL_ERROR "ZC host prepared-release negative control did not trip")
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
        "} else if (capability_ready) {"
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
string(FIND "${server_task}"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    da1_raw_erase)
string(FIND "${server_task}"
    "} else if (capability_ready) {"
    da1_capability_arm)
if (NOT host_commit_gap_valid OR da1_owner_assert EQUAL -1)
    message(FATAL_ERROR
        "D-A1 host prepare-to-commit owner-thread/no-callback contract drifted")
endif()
string(REPLACE
    "} else if (capability_ready) {"
    "} else if (capability_ready) { acct->gauge_set();"
    da1_gap_negative "${server_task}")
da1_host_commit_gap_valid("${da1_gap_negative}" da1_gap_negative_valid)
if (da1_gap_negative_valid)
    message(FATAL_ERROR "D-A1 host commit-gap negative control did not trip")
endif()
string(REPLACE
    "} else if (capability_ready) {"
    "} else if (capability_ready) { acct->release({});"
    da1_release_negative "${server_task}")
da1_host_commit_gap_valid(
    "${da1_release_negative}" da1_release_negative_valid)
if (da1_release_negative_valid)
    message(FATAL_ERROR
        "D-A1 host commit-gap release negative control did not trip")
endif()

# D-A2 and D-A3 share one certified commit terminal. The helper itself and
# each raw-erase -> helper-call edge are scanned: moving any ledger producer
# into either side of that boundary must fail this contract.
contract_extract_region(
    "${server_task}"
    "void commit_certified_host_destruction("
    "certified.capability.commit(certified.pin);"
    certified_commit_helper certified_commit_helper_found)
contract_find_forbidden(
    "${certified_commit_helper}" certified_helper_forbidden
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
count_literal(
    "${server_task}"
    "commit_certified_host_destruction(" certified_commit_sites)
if (NOT certified_commit_helper_found OR certified_helper_forbidden OR
    NOT certified_commit_sites EQUAL 3)
    message(FATAL_ERROR
        "D-A2/D-A3 shared certified-destruction terminal drifted")
endif()
string(REPLACE
    "    GGML_ASSERT(certified.ready);"
    "    cache.acct->release({}); GGML_ASSERT(certified.ready);"
    certified_helper_negative "${server_task}")
contract_extract_region(
    "${certified_helper_negative}"
    "void commit_certified_host_destruction("
    "certified.capability.commit(certified.pin);"
    certified_negative_region certified_negative_found)
contract_find_forbidden(
    "${certified_negative_region}" certified_negative_forbidden "release(")
if (NOT certified_negative_found OR NOT certified_negative_forbidden)
    message(FATAL_ERROR
        "D-A2/D-A3 shared-terminal negative control did not trip")
endif()

# D-A2 opens only the exact host_dedup class. Its disjoint survivor pin is
# acquired before the raw erase, and the shared capability terminal is the
# first operation after the erase on that control-flow arm. Every other reason
# continues through D-A1's legacy/capability behavior.
contract_extract_region(
    "${server_task}"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    "commit_certified_host_destruction(\n            *this, redundant, scheduler_owner, nullptr,"
    da2_commit_gap da2_commit_gap_found)
contract_find_forbidden(
    "${da2_commit_gap}" da2_gap_forbidden
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
string(FIND "${server_task}"
    "reason == server_cache_destruction_reason::host_dedup &&"
    da2_reason_gate)
string(FIND "${quote_source}"
    "!recovery_pin.disjoint(victims, current_ops)" da2_prepare_disjoint)
if (NOT da2_commit_gap_found OR da2_gap_forbidden OR
    da2_reason_gate EQUAL -1 OR da2_prepare_disjoint EQUAL -1 OR
    da1_raw_erase GREATER_EQUAL da1_capability_arm)
    message(FATAL_ERROR
        "D-A2 exact-host-only/disjoint-pin/commit-gap contract drifted")
endif()
string(REPLACE
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it); acct->release({});"
    da2_gap_negative "${server_task}")
contract_extract_region(
    "${da2_gap_negative}"
    "auto next = server_prompt_cache_destroy_entry_impl(*this, it);"
    "commit_certified_host_destruction(\n            *this, redundant, scheduler_owner, nullptr,"
    da2_negative_gap da2_negative_found)
contract_find_forbidden(
    "${da2_negative_gap}" da2_negative_forbidden "release(")
if (NOT da2_negative_found OR NOT da2_negative_forbidden)
    message(FATAL_ERROR "D-A2 commit-gap negative control did not trip")
endif()

# CACHE_HOST_LIFECYCLE is debug evidence, not an Observed-template signal:
# B authority also supplies a record when --cache-debug is absent.
string(FIND "${server_context}"
    "prompt_cache->debug_observability = params_base.cache_optimizer.cache_debug;"
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

# D-A2 maintenance receipts use the same explicit --cache-debug view. The
# lifecycle/authority substrate remains active with debug off, but performs no
# JSON construction and emits no CACHE_HOST_DESTRUCTION line.
contract_extract_region(
    "${server_task}"
    "void server_prompt_cache_observe_host_destruction("
    "host_destruction_certification certify_host_destruction("
    da2_debug_emit_region da2_debug_emit_region_found)
string(FIND "${da2_debug_emit_region}"
    "if (!cache.debug_observability) {" da2_debug_emit_gate)
string(FIND "${da2_debug_emit_region}"
    "debug_destruction_emissions++;" da2_debug_emit_count)
string(FIND "${da2_debug_emit_region}"
    "CACHE_HOST_DESTRUCTION" da2_debug_emit)
if (NOT da2_debug_emit_region_found OR da2_debug_emit_gate EQUAL -1 OR
    da2_debug_emit_count LESS da2_debug_emit_gate OR
    da2_debug_emit LESS da2_debug_emit_count)
    message(FATAL_ERROR "D-A2 debug-only maintenance evidence gate drifted")
endif()

# E1 debug-plane observability: a recovery-pinned host artifact is excluded
# before victim quoting, and the same pressure attempt reports its eventual
# floor terminal. These are CACHE_HOST_DESTRUCTION metadata only; neither key
# may migrate into the public cache-control serializer.
function(e1_pin_exclusion_observability_valid source output)
    foreach(needle
            "void emit_recovery_pin_excluded("
            "if (!cache.debug_observability) {"
            "payload[\"evidence_event\"] = \"recovery_pin_excluded\";"
            "payload[\"recovery_pin_excluded\"] = {"
            "payload[\"floor_outcome\"] = \"pending\";"
            "void emit_host_pressure_floor_outcome("
            "payload[\"evidence_event\"] = \"floor_outcome\";"
            "payload[\"floor_outcome\"] = outcome;"
            "if (it->recovery_pins != 0) {"
            "emit_recovery_pin_excluded(*this, *it);"
            "emit_host_pressure_floor_outcome(")
        string(FIND "${source}" "${needle}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

e1_pin_exclusion_observability_valid(
    "${server_task}" e1_pin_exclusion_observability)
if (NOT e1_pin_exclusion_observability)
    message(FATAL_ERROR
        "E1 recovery-pin exclusion/floor debug evidence contract drifted")
endif()
string(REPLACE
    "payload[\"recovery_pin_excluded\"] = {"
    "payload[\"recovery_pin_hidden\"] = {"
    e1_pin_exclusion_negative "${server_task}")
e1_pin_exclusion_observability_valid(
    "${e1_pin_exclusion_negative}" e1_pin_exclusion_negative_valid)
if (e1_pin_exclusion_negative_valid)
    message(FATAL_ERROR
        "E1 recovery-pin exclusion evidence negative control did not trip")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-cache-control-wire.cpp"
    cache_control_wire)
foreach(private_key "recovery_pin_excluded" "floor_outcome")
    string(FIND "${cache_control_wire}" "${private_key}" private_key_leak)
    if (NOT private_key_leak EQUAL -1)
        message(FATAL_ERROR
            "E1 debug destruction metadata leaked into cache-control wire: ${private_key}")
    endif()
endforeach()

# D-A3 routes both bounded host-pressure loops through one pressure terminal,
# which owns the fitted-price attempt and the historical FIFO all-refused
# fallback. A successful trade uses the same shared capability terminal as
# D-A2 and emits ranking evidence only through the debug-gated D-A2 line.
count_literal(
    "${server_task}" "destroy_priced_host_entry(" da3_priced_entry_sites)
count_literal(
    "${server_task}" "evict_front_under_pressure(" da3_pressure_entry_sites)
function(da3_hard_floor_valid source output)
    foreach(needle
            "candidate.lease_known && !candidate.hard_leased"
            "candidate.victim->recovery_pins == 0"
            "if (!zc_pressure_checked && !update_impl(self)) {"
            "note_host_trade_publication_skip()")
        string(FIND "${source}" "${needle}" needle_pos)
        if (needle_pos EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
da3_hard_floor_valid("${server_task}" da3_hard_floor_ok)
contract_extract_region(
    "${server_task}"
    "host_destruction_certification certify_host_destruction("
    "server_prompt_cache::iterator server_prompt_cache::destroy_entry_impl("
    da3_trade_body da3_trade_body_found)
contract_extract_region(
    "${da3_trade_body}"
    "server_prompt_cache_destroy_entry_impl(*this, chosen->victim);"
    "commit_certified_host_destruction(\n            *this, certified, scheduler_owner, &chosen->ranking,"
    da3_commit_gap da3_commit_gap_found)
contract_find_forbidden(
    "${da3_commit_gap}" da3_gap_forbidden
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
foreach(pin
        "common_cache_plan_calib_find("
        "server_cache_host_retention_price_us("
        "authority.host_retention_weight("
        "evidence.pin.binds_exact("
        "candidate.ranking.zero_destruction"
        "COMMON_CACHE_PLAN_TIE_REL_FLOOR"
        "host_trade_legacy_fallbacks++"
        "candidate.hard_leased"
        "candidate.victim->recovery_pins == 0")
    string(FIND "${da3_trade_body}" "${pin}" da3_pin)
    if (da3_pin EQUAL -1)
        message(FATAL_ERROR "D-A3 priced-host trade pin missing: ${pin}")
    endif()
endforeach()
foreach(pin
        "note_host_trade_veto()"
        "note_host_trade_publication_skip()"
        "host_trade_hard_lease_vetoes++"
        "host_trade_publication_skips++")
    string(FIND "${server_task}${inventory_header}" "${pin}" da3_floor_pin)
    if (da3_floor_pin EQUAL -1)
        message(FATAL_ERROR "D-A3 hard-floor evidence pin missing: ${pin}")
    endif()
endforeach()
foreach(pin
        "server_cache_destruction_reason::host_capacity"
        "server_cache_destruction_reason::host_token_limit")
    string(FIND "${server_task}" "evict_front_under_pressure(\n                    ${pin}," da3_route)
    if (da3_route EQUAL -1)
        message(FATAL_ERROR "D-A3 host-pressure entry point drifted: ${pin}")
    endif()
endforeach()
foreach(pin
        "price_us"
        "retention_weight_milli"
        "rank_ordinal"
        "victim_source_id"
        "victim_artifact_id"
        "zero_destruction_tie_break")
    string(FIND "${da2_debug_emit_region}" "${pin}" da3_debug_pin)
    if (da3_debug_pin EQUAL -1)
        message(FATAL_ERROR "D-A3 ranking evidence field missing: ${pin}")
    endif()
endforeach()
if (NOT da3_priced_entry_sites EQUAL 2 OR
    NOT da3_pressure_entry_sites EQUAL 3 OR
    NOT da3_hard_floor_ok OR
    NOT da3_trade_body_found OR NOT da3_commit_gap_found OR
    da3_gap_forbidden)
    message(FATAL_ERROR
        "D-A3 priced-host routing/capability-adjacency contract drifted")
endif()
string(REPLACE
    "server_prompt_cache_destroy_entry_impl(*this, chosen->victim);"
    "server_prompt_cache_destroy_entry_impl(*this, chosen->victim); acct->release({});"
    da3_gap_negative "${server_task}")
contract_extract_region(
    "${da3_gap_negative}"
    "server_prompt_cache_destroy_entry_impl(*this, chosen->victim);"
    "commit_certified_host_destruction(\n            *this, certified, scheduler_owner, &chosen->ranking,"
    da3_negative_gap da3_negative_found)
contract_find_forbidden(
    "${da3_negative_gap}" da3_negative_forbidden "release(")
if (NOT da3_negative_found OR NOT da3_negative_forbidden)
    message(FATAL_ERROR "D-A3 commit-gap negative control did not trip")
endif()
string(REPLACE
    "candidate.lease_known && !candidate.hard_leased"
    "candidate.lease_known"
    da3_floor_negative "${server_task}")
da3_hard_floor_valid("${da3_floor_negative}" da3_floor_negative_valid)
if (da3_floor_negative_valid)
    message(FATAL_ERROR "D-A3 hard-floor negative control did not trip")
endif()
foreach(test_pin
        "test_host_trade_hard_lease_veto"
        "test_host_trade_all_hard_skips_publication"
        "test_host_trade_floor_skips_recovery_pin")
    string(FIND "${server_task_test}" "${test_pin}" da3_floor_test)
    if (da3_floor_test EQUAL -1)
        message(FATAL_ERROR "D-A3 hard-floor regression missing: ${test_pin}")
    endif()
endforeach()

# D-A4 independently owns only live checkpoint payloads. The physical eraser
# stays release-free; a certified member erase is immediately followed by the
# same prepared-capability terminal used by D-A2/3, while cloned host
# checkpoints retain aggregate ownership (clone never copies release_ops).
contract_extract_region(
    "${server_context}"
    "checkpoint_iterator server_cache_checkpoint_drop_impl("
    "checkpoint_iterator checkpoint_drop_joined_impl("
    da4_raw_checkpoint da4_raw_checkpoint_found)
contract_find_forbidden(
    "${da4_raw_checkpoint}" da4_raw_checkpoint_forbidden
    "retention_obs"
    "release("
    "retire("
    "server_cache_retention_admit(")

# A1 extraction: policy/certification lives in server-task.cpp,
# while the only typed raw adapter must still terminate at the censused slot
# _impl door. This is deliberately narrower than a general callback seam.
function(da_a1_checkpoint_adapter_valid source output)
    contract_extract_region(
        "${source}"
        "static checkpoint_iterator checkpoint_drop_authority_adapter("
        "server_cache_checkpoint_authority_context checkpoint_authority_context()"
        adapter_region adapter_found)
    string(FIND "${adapter_region}"
        "server_cache_checkpoint_drop_impl(first, last);" adapter_raw)
    contract_extract_region(
        "${source}"
        "server_cache_checkpoint_authority_context checkpoint_authority_context()"
        "void checkpoint_ring_changed()"
        context_region context_found)
    string(FIND "${context_region}"
        "checkpoint_drop_authority_adapter," context_binding)
    if (adapter_found AND context_found AND
        NOT adapter_raw EQUAL -1 AND NOT context_binding EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

da_a1_checkpoint_adapter_valid("${server_context}" da_a1_adapter_valid)
if (NOT da_a1_adapter_valid)
    message(FATAL_ERROR
        "A1 checkpoint authority adapter escaped the censused raw _impl door")
endif()
string(REPLACE
    "server_cache_checkpoint_drop_impl(first, last);"
    "checkpoint_drop_joined_impl(first, last);"
    da_a1_adapter_negative "${server_context}")
da_a1_checkpoint_adapter_valid(
    "${da_a1_adapter_negative}" da_a1_adapter_negative_valid)
if (da_a1_adapter_negative_valid)
    message(FATAL_ERROR "A1 checkpoint adapter negative control did not trip")
endif()

contract_extract_region(
    "${server_task}"
    "next = context.raw_drop("
    "prepared.capability.commit(retained_pin);"
    da4_commit_gap da4_commit_gap_found)
contract_find_forbidden(
    "${da4_commit_gap}" da4_commit_gap_forbidden
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
contract_extract_region(
    "${sidecar_source}"
    "bool server_retention_sidecar_store::clone("
    "bool server_retention_sidecar_store::rebind("
    da4_clone_region da4_clone_region_found)
string(FIND "${da4_clone_region}" "release_ops" da4_clone_ops)
foreach(pin
        "admit_live_checkpoint("
        "attach_release_ops("
        "checkpoint_admission_artifact("
        "checkpoint_inventory("
        "acquire_recovery_pin("
        "retire_after_committed_release("
        "server_cache_plan_checkpoint_thinning("
        "server_cache_checkpoint_bounded_replay("
        "server_cache_multiply_retention_weight("
        "server_cache_weighted_price_us("
        "server_cache_destruction_receipt_json("
        "server_cache_plan_evaluate_checkpoint("
        "checkpoint_frontier_is_current("
        "CACHE_HOST_DESTRUCTION")
    string(FIND
        "${server_context}${server_task}${sidecar_source}${authority_source}"
        "${pin}" da4_pin)
    if (da4_pin EQUAL -1)
        message(FATAL_ERROR "D-A4 checkpoint authority pin missing: ${pin}")
    endif()
endforeach()
foreach(test_pin
        "test_checkpoint_thinning_policy"
        "seam_heuristic_protected = true"
        "server_cache_checkpoint_protection::seam_heuristic"
        "recovery_available = false"
        "hard_leased = true"
        "test_checkpoint_capacity_floor"
        "test_checkpoint_attempt_latch_rearms_on_ring_change"
        "test_checkpoint_effect_matrix_consistency"
        "test_live_checkpoint_batch_admission"
        "test_lifecycle_restore_batch_timing"
        "test_checkpoint_creation_churn_timing"
        "test_checkpoint_bounded_publication_skip_predicate"
        "test_consuming_rebind_mints_checkpoint_ownership"
        "retire_after_committed_release(victim_key)")
    string(FIND "${server_task_test}" "${test_pin}" da4_test_pin)
    if (da4_test_pin EQUAL -1)
        message(FATAL_ERROR "D-A4 checkpoint regression missing: ${test_pin}")
    endif()
endforeach()
if (NOT da4_raw_checkpoint_found OR da4_raw_checkpoint_forbidden OR
    NOT da4_commit_gap_found OR da4_commit_gap_forbidden OR
    NOT da4_clone_region_found OR NOT da4_clone_ops EQUAL -1)
    message(FATAL_ERROR
        "D-A4 live-checkpoint ownership/commit-gap/host-clone contract drifted")
endif()
string(FIND "${server_context}"
    "bool optional_thinning_attempt = !zc_retention &&\n                            slot.lifecycle_authority &&\n                            slot.checkpoint_thinning_attempt_begin(false);"
    da4_attempt_lifecycle_gate)
string(FIND "${server_task}"
    "if (!attempt_claimed &&\n        !server_cache_checkpoint_thinning_attempt_begin(context, capacity_mode)) {\n        return false;\n    }\n    context.thinning_refusal"
    da4_priced_attempt_early_out)
string(FIND "${server_task}"
    "if (!context.attempts.begin(\n            server_cache_checkpoint_attempt_lane::capacity_floor)) {\n        return false;\n    }\n    refusal = common_cache_plan_destruction_reason::mandatory_anchor;"
    da4_floor_attempt_early_out)
if (da4_attempt_lifecycle_gate EQUAL -1 OR
    da4_priced_attempt_early_out EQUAL -1 OR
    da4_floor_attempt_early_out EQUAL -1)
    message(FATAL_ERROR
        "D-A4 generation latch escaped the pre-inventory policy boundary")
endif()
string(FIND "${server_context}"
    "slot.checkpoint_publication_skipped(\n                                    slot.checkpoint_thinning_refusal);"
    da4_bounded_publication_skip)
if (da4_bounded_publication_skip EQUAL -1)
    message(FATAL_ERROR
        "D-A4 failed optional thin no longer suppresses bounded redundant publication")
endif()
contract_extract_region(
    "${server_task}"
    "bool server_cache_checkpoint_thin_priced("
    "bool server_cache_checkpoint_capacity_floor("
    da4_priced_inventory da4_priced_inventory_found)
contract_extract_region(
    "${server_task}"
    "bool server_cache_checkpoint_capacity_floor("
    "void server_cache_checkpoint_publication_skipped("
    da4_floor_inventory da4_floor_inventory_found)
contract_find_forbidden(
    "${da4_priced_inventory}${da4_floor_inventory}"
    da4_identity_rebuild_forbidden
    "server_cache_lease_build_identity("
    "media_content_identity("
    "candidate_for_instance("
    "lora_config_identity(")
if (NOT da4_priced_inventory_found OR NOT da4_floor_inventory_found OR
    da4_identity_rebuild_forbidden)
    message(FATAL_ERROR
        "D-A4 creation inventory rebuilt immutable checkpoint identity")
endif()
count_literal(
    "${authority_source}${server_context}${server_task}"
    "admit_live_checkpoint("
    da4_admit_live_checkpoint_count)
count_literal(
    "${authority_source}${server_context}${server_task}"
    "admit_live_checkpoints("
    da4_admit_live_checkpoints_count)
if (NOT da4_admit_live_checkpoint_count EQUAL 3 OR
    NOT da4_admit_live_checkpoints_count EQUAL 3)
    message(FATAL_ERROR
        "D-A4/ZC1 live-checkpoint ownership admission census drifted: expected single definition + detached ZC adapter + historical creation adapter and batch definition + single adapter + restore, found single=${da4_admit_live_checkpoint_count} batch=${da4_admit_live_checkpoints_count}")
endif()
string(REPLACE
    "next = context.raw_drop("
    "next = context.raw_drop( authority.ledger.release({});"
    da4_gap_negative "${server_task}")
contract_extract_region(
    "${da4_gap_negative}"
    "next = context.raw_drop("
    "prepared.capability.commit(retained_pin);"
    da4_negative_gap da4_negative_found)
contract_find_forbidden(
    "${da4_negative_gap}" da4_negative_forbidden "release(")
if (NOT da4_negative_found OR NOT da4_negative_forbidden)
    message(FATAL_ERROR "D-A4 commit-gap negative control did not trip")
endif()

# D-A5 is the sole B-candidate live-slot capability terminal. The raw slot
# mutation must be immediately followed by the conditional release commit;
# autonomous VBR/unified-KV reclaim keeps using the legacy wrapper and may not
# call this door.
contract_extract_region(
    "${server_context}"
    "server_cache_slot_drop_impl(false);"
    "capability.commit(\n            cache_plan_destruction_recovery_pin);"
    da5_commit_gap da5_commit_gap_found)
contract_find_forbidden(
    "${da5_commit_gap}" da5_gap_forbidden
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
count_literal("${server_context}" "prompt_clear_certified(" da5_certified_sites)
if (NOT da5_commit_gap_found OR da5_gap_forbidden OR
    NOT da5_certified_sites EQUAL 2)
    message(FATAL_ERROR
        "D-A5 live-displacement single-site/commit-gap contract drifted")
endif()
string(REPLACE
    "server_cache_slot_drop_impl(false);"
    "server_cache_slot_drop_impl(false); cache_authority->ledger.release({});"
    da5_gap_negative "${server_context}")
contract_extract_region(
    "${da5_gap_negative}"
    "server_cache_slot_drop_impl(false);"
    "capability.commit(\n            cache_plan_destruction_recovery_pin);"
    da5_negative_gap da5_negative_found)
contract_find_forbidden(
    "${da5_negative_gap}" da5_negative_forbidden "release(")
if (NOT da5_negative_found OR NOT da5_negative_forbidden)
    message(FATAL_ERROR "D-A5 commit-gap negative control did not trip")
endif()

# The capability becomes live inside certify, before control returns to the B
# authorization seam. Prove the whole window in two pieces: the certify tail
# after prepare is ledger-quiet, and the caller reaches the raw terminal with
# no writer. The sole prompt_save spelling in the caller is structurally
# guarded by !displacement.ready and therefore belongs only to the
# uncertified fallback arm.
contract_extract_region(
    "${server_context}"
    "const auto fresh = cache_authority->ledger.snapshot();\n        auto prepared = server_cache_prepare_release_set("
    "out.ready = true;"
    da5_prepare_tail da5_prepare_tail_found)
contract_find_forbidden(
    "${da5_prepare_tail}" da5_prepare_tail_forbidden
    "gauge_set("
    "reserve("
    "stage("
    "preview_release_set("
    "release("
    "clone("
    "retire("
    "prompt_save("
    "update("
    "server_cache_retention_admit("
    "server_fault("
    "std::function")
contract_extract_region(
    "${server_context}"
    "displacement = cache_plan_certify_live_displacement("
    "planned_ret->prompt_clear_certified("
    da5_authorize_window da5_authorize_window_found)
contract_find_forbidden(
    "${da5_authorize_window}" da5_authorize_window_forbidden
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
count_literal(
    "${da5_authorize_window}"
    "planned_ret != legacy_ret &&\n                        !displacement.ready)"
    da5_uncertified_save_guard)
count_literal(
    "${da5_authorize_window}" "legacy_ret->prompt_save(" da5_fallback_saves)
if (NOT da5_prepare_tail_found OR da5_prepare_tail_forbidden OR
    NOT da5_authorize_window_found OR da5_authorize_window_forbidden OR
    NOT da5_uncertified_save_guard EQUAL 1 OR
    NOT da5_fallback_saves EQUAL 1)
    message(FATAL_ERROR
        "D-A5 full prepared-capability window contract drifted")
endif()
string(REPLACE
    "out.capability = std::move(prepared.capability);"
    "out.capability = std::move(prepared.capability); cache_authority->ledger.release({});"
    da5_window_negative "${server_context}")
contract_extract_region(
    "${da5_window_negative}"
    "const auto fresh = cache_authority->ledger.snapshot();\n        auto prepared = server_cache_prepare_release_set("
    "out.ready = true;"
    da5_negative_prepare_tail da5_negative_prepare_tail_found)
contract_find_forbidden(
    "${da5_negative_prepare_tail}"
    da5_negative_prepare_tail_forbidden "release(")
if (NOT da5_negative_prepare_tail_found OR
    NOT da5_negative_prepare_tail_forbidden)
    message(FATAL_ERROR
        "D-A5 full-window negative control did not trip")
endif()
foreach(legacy_vbr "vbr_clear_idle_slots" "vbr_reclaim_before_degrade" "try_clear_idle_slots")
    contract_extract_region(
        "${server_context}" "${legacy_vbr}(" "}" vbr_region vbr_found)
    if (vbr_found)
        string(FIND "${vbr_region}" "prompt_clear_certified(" vbr_certified)
        if (NOT vbr_certified EQUAL -1)
            message(FATAL_ERROR
                "D-A5 must not absorb autonomous ${legacy_vbr}")
        endif()
    endif()
endforeach()

message(STATUS
    "D-S4 destruction inventory scan passed: 6 classes, ${admission_owner_count} admission owners, "
    "raw-call + missing-class + D-A0b release-owner + D-A1/D-A2/D-A3/D-A4/D-A5 commit-gap/debug-emission negative controls")
