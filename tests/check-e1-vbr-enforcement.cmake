if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${SOURCE_ROOT}/tests/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" kv_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-hard-seal.cpp" seal_source)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.cpp" lease_source)
file(READ "${SOURCE_ROOT}/tools/server/server-common.cpp" common_source)
file(READ "${SOURCE_ROOT}/tools/server/server-common.h" common_header)
file(READ "${SOURCE_ROOT}/include/llama.h" public_header)

function(enforcement_contract kv server output)
    contract_extract_region("${kv}" "llama_kv_cache::vbr_degrade_result llama_kv_cache::vbr_degrade_next"
        "void llama_kv_cache::vbr_transcode_anchor_test" degrade_region degrade_found)
    contract_extract_region("${kv}" "bool llama_kv_cache::vbr_tx_hard_seal_allowed"
        "bool llama_kv_cache::vbr_tx_prepare_commit" tree_region tree_found)
    contract_extract_region("${server}" "int vbr_clear_idle_slots"
        "void vbr_reclaim_before_degrade" clear_region clear_found)
    contract_extract_region("${server}" "int vbr_reset_on_low_lcp"
        "void recurrent_shrink_for_prefill" reset_region reset_found)
    contract_extract_region("${server}" "bool try_clear_idle_slots"
        "std::vector<common_adapter_lora_info> construct_lora_list" purge_region purge_found)
    if (NOT degrade_found OR NOT tree_found OR NOT clear_found OR
        NOT reset_found OR NOT purge_found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    string(FIND "${degrade_region}" "vbr_hard_seal_step_blocked(" degrade_guard)
    string(FIND "${degrade_region}" "hard_lease_blocked" degrade_status)
    string(FIND "${tree_region}" "vbr_hard_seal_step_blocked(" tree_guard)
    string(FIND "${clear_region}" "hard_lease_blocks_live_prefix()" clear_guard)
    string(FIND "${reset_region}" "hard_lease_blocks_live_range(" reset_guard)
    string(FIND "${purge_region}" "hard_lease_blocks_live_prefix()" purge_guard)
    string(FIND "${server}" "vbr_hard_seal_blocked_take(ret != 0)" terminal_take)
    string(FIND "${server}" "server_cache_has_hard_lease(" any_hard_gate)
    string(FIND "${kv}" "vbr_hard_seal_defer_jumped_steps(" tx_defer)
    string(FIND "${kv}" "GGML_ASSERT(session.verdict == verdict);" uniform_assert)
    string(FIND "${kv}" "plan_slots() can refuse before prepare_with_slots()" plan_reset)
    string(FIND "${kv}" "vbr_hard_seal_evidence_record(order_ordinal);" step_record)
    string(FIND "${kv}" "vbr_hard_seal_evidence_record(step.order_idx);" tree_record)
    string(FIND "${server}" "SRV_INF(\"CACHE_VBR_HARD_SEAL %s\\n\"" seal_evidence)
    string(FIND "${server}" "ERROR_TYPE_HARD_LEASE_BLOCKED" typed_http)
    string(FIND "${server}" "vbr_hard_seal_evidence_take(hard_seal_evidence);" evidence_take)
    string(FIND "${server}" "if (params_base.cache_debug) {\n            for (const auto & step : hard_seal_evidence)" debug_gate)
    if (degrade_guard EQUAL -1 OR degrade_status EQUAL -1 OR
        tree_guard EQUAL -1 OR clear_guard EQUAL -1 OR
        reset_guard EQUAL -1 OR purge_guard EQUAL -1 OR
        terminal_take EQUAL -1 OR any_hard_gate EQUAL -1 OR
        tx_defer EQUAL -1 OR uniform_assert EQUAL -1 OR
        plan_reset EQUAL -1 OR step_record EQUAL -1 OR tree_record EQUAL -1 OR
        seal_evidence EQUAL -1 OR typed_http EQUAL -1 OR evidence_take EQUAL -1 OR
        debug_gate EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

enforcement_contract("${kv_source}" "${server_source}" contract_ok)
if (NOT contract_ok)
    message(FATAL_ERROR "E1.1c VBR enforcement/guard census failed")
endif()

function(typed_http_contract common header output)
    string(FIND "${header}" "ERROR_TYPE_HARD_LEASE_BLOCKED" enum_token)
    string(FIND "${common}" "case ERROR_TYPE_HARD_LEASE_BLOCKED:" case_token)
    string(FIND "${common}" "type_str = \"hard_lease_blocked\";" wire_token)
    if (enum_token EQUAL -1 OR case_token EQUAL -1 OR wire_token EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

typed_http_contract("${common_source}" "${common_header}" typed_http_ok)
if (NOT typed_http_ok)
    message(FATAL_ERROR "E1.1c typed HTTP terminal census failed")
endif()

function(real_vbr_grant_contract server output)
    contract_extract_region("${server}"
        "case SERVER_TASK_TYPE_CACHE_HOLDER_CREATE:"
        "case SERVER_TASK_TYPE_CACHE_CAPTURE:"
        scheduler_region scheduler_found)
    if (NOT scheduler_found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    string(FIND "${scheduler_region}" "params_base.vbr_enabled()" old_refusal)
    if (NOT old_refusal EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

real_vbr_grant_contract("${server_source}" real_vbr_grant_ok)
if (NOT real_vbr_grant_ok)
    message(FATAL_ERROR "E1.1c real VBR hard-grant path regressed")
endif()

contract_require_token("${lease_source}" "leases->inspect_range("
    "E1.1c range-qualified lease consult")
contract_extract_region("${server_source}" "bool hard_lease_blocks_live_range("
    "bool hard_lease_blocks_live_prefix()" range_region range_found)
if (NOT range_found)
    message(FATAL_ERROR "E1.1c range-qualification helper missing")
endif()
contract_require_token("${range_region}" "server_cache_hard_lease_blocks_range("
    "E1.1c range-qualified helper door")
contract_forbid_token("${range_region}" "lease_obs->inspect("
    "E1.1c unqualified live lease consult")
contract_require_token("${server_source}" "llama_get_memory(ctx_tgt)->vbr_hard_seal_guard_set("
    "E1.1c controller callback installation")
contract_require_token("${server_source}" "server_cache_has_hard_lease("
    "E1.1c zero-hard-lease fast gate")
contract_require_token("${server_source}" "constexpr const char * refusal = \"hard_lease_blocked\";"
    "E1.1c typed no-stall refusal")
contract_require_token("${server_source}" "SRV_INF(\"CACHE_VBR_HARD_SEAL %s\\n\""
    "E1.1c cache-debug sealed-step evidence")
foreach(token
        "{\"order_ordinal\", step.order_ordinal}"
        "{\"layer\", step.il}"
        "{\"side\", step.is_v ? \"v\" : \"k\"}")
    contract_require_token("${server_source}" "${token}"
        "E1.1c sealed-unit coordinate evidence")
endforeach()
contract_require_token("${server_source}" "send_error(slot, refusal, ERROR_TYPE_HARD_LEASE_BLOCKED);"
    "E1.1c typed HTTP terminal")
contract_require_token("${common_header}" "ERROR_TYPE_HARD_LEASE_BLOCKED"
    "E1.1c typed HTTP error census")
contract_require_token("${common_source}" "type_str = \"hard_lease_blocked\";"
    "E1.1c typed HTTP serialization")
contract_require_token("${server_source}" "vbr_hard_seal_blocked_take(ret != 0)"
    "E1.1c consume-and-clear decode terminal")
contract_require_token("${kv_source}" "vbr_hard_seal_defer_jumped_steps("
    "E1.1c tree-shed jumped-step preservation")
contract_require_token("${kv_source}" "plan_slots() can refuse before prepare_with_slots()"
    "E1.1c pre-admission latch reset")
contract_require_token("${kv_source}" "GGML_ASSERT(session.verdict == verdict);"
    "E1.1c subject-uniform guard assertion")
contract_forbid_token("${server_source}"
    "E1.1c owns VBR controller enforcement"
    "E1.1c pre-enforcement route refusal")
foreach(token "hard_seal" "cache_control" "server_cache_lease")
    contract_forbid_token("${public_header}" "${token}" "E1.1c public llama.h boundary")
endforeach()

# The approved legacy doors are read-only skip guards. They may consult the
# lease helper but must never grow a D-A capability/prepare/commit path.
contract_extract_region("${server_source}" "int vbr_clear_idle_slots"
    "void vbr_reclaim_before_degrade" clear_region clear_found)
contract_extract_region("${server_source}" "int vbr_reset_on_low_lcp"
    "void recurrent_shrink_for_prefill" reset_region reset_found)
contract_extract_region("${server_source}" "bool try_clear_idle_slots"
    "std::vector<common_adapter_lora_info> construct_lora_list" purge_region purge_found)
if (NOT clear_found OR NOT reset_found OR NOT purge_found)
    message(FATAL_ERROR "E1.1c legacy guard regions missing")
endif()
foreach(region "${clear_region}" "${reset_region}" "${purge_region}")
    foreach(token "prepare_release_set" "certify_" "commit_certified")
        contract_forbid_token("${region}" "${token}" "E1.1c legacy guard D-A isolation")
    endforeach()
endforeach()

# House-standard negative control: remove one guard and rerun the complete
# census, rather than testing string replacement itself.
string(REPLACE "if (s.hard_lease_blocks_live_prefix())"
    "if (false)" guard_negative "${server_source}")
enforcement_contract("${kv_source}" "${guard_negative}" guard_negative_ok)
if (guard_negative_ok)
    message(FATAL_ERROR "E1.1c reclaim-guard negative control did not trip")
endif()

string(REPLACE "vbr_hard_seal_step_blocked(order_ordinal, seal_session)"
    "false" classifier_negative "${kv_source}")
enforcement_contract("${classifier_negative}" "${server_source}" classifier_negative_ok)
if (classifier_negative_ok)
    message(FATAL_ERROR "E1.1c classifier negative control did not trip")
endif()

string(REPLACE "vbr_hard_seal_blocked_take(ret != 0)"
    "vbr_hard_seal_blocked_take(false)" terminal_negative "${server_source}")
enforcement_contract("${kv_source}" "${terminal_negative}" terminal_negative_ok)
if (terminal_negative_ok)
    message(FATAL_ERROR "E1.1c terminal-consumption negative control did not trip")
endif()

string(REPLACE "SRV_INF(\"CACHE_VBR_HARD_SEAL %s\\n\""
    "SRV_DBG(\"CACHE_VBR_HARD_SEAL %s\\n\""
    evidence_negative "${server_source}")
enforcement_contract("${kv_source}" "${evidence_negative}" evidence_negative_ok)
if (evidence_negative_ok)
    message(FATAL_ERROR "E1.1c sealed-step evidence negative control did not trip")
endif()

string(REPLACE
    "if (params_base.cache_debug) {\n            for (const auto & step : hard_seal_evidence)"
    "if (true) {\n            for (const auto & step : hard_seal_evidence)"
    debug_gate_negative "${server_source}")
enforcement_contract("${kv_source}" "${debug_gate_negative}" debug_gate_negative_ok)
if (debug_gate_negative_ok)
    message(FATAL_ERROR "E1.1c debug-only evidence negative control did not trip")
endif()

string(REPLACE "vbr_hard_seal_evidence_record(order_ordinal);"
    "/* evidence removed */" evidence_record_negative "${kv_source}")
enforcement_contract("${evidence_record_negative}" "${server_source}" evidence_record_negative_ok)
if (evidence_record_negative_ok)
    message(FATAL_ERROR "E1.1c sealed-unit recording negative control did not trip")
endif()

string(REPLACE "type_str = \"hard_lease_blocked\";"
    "type_str = \"server_error\";" typed_http_negative "${common_source}")
typed_http_contract("${typed_http_negative}" "${common_header}" typed_http_negative_ok)
if (typed_http_negative_ok)
    message(FATAL_ERROR "E1.1c typed-HTTP negative control did not trip")
endif()

string(REPLACE "vbr_hard_seal_defer_jumped_steps("
    "vbr_hard_seal_retire_step(" tx_defer_negative "${kv_source}")
enforcement_contract("${tx_defer_negative}" "${server_source}" tx_defer_negative_ok)
if (tx_defer_negative_ok)
    message(FATAL_ERROR "E1.1c tx-defer negative control did not trip")
endif()

string(REPLACE "server_cache_has_hard_lease("
    "server_cache_no_hard_lease(" any_hard_negative "${server_source}")
enforcement_contract("${kv_source}" "${any_hard_negative}" any_hard_negative_ok)
if (any_hard_negative_ok)
    message(FATAL_ERROR "E1.1c zero-hard fast-gate negative control did not trip")
endif()

string(REPLACE "GGML_ASSERT(session.verdict == verdict);"
    "GGML_ASSERT(true);" uniform_negative "${kv_source}")
enforcement_contract("${uniform_negative}" "${server_source}" uniform_negative_ok)
if (uniform_negative_ok)
    message(FATAL_ERROR "E1.1c subject-uniform negative control did not trip")
endif()

string(REPLACE "plan_slots() can refuse before prepare_with_slots()"
    "pre-admission reset removed" plan_reset_negative "${kv_source}")
enforcement_contract("${plan_reset_negative}" "${server_source}" plan_reset_negative_ok)
if (plan_reset_negative_ok)
    message(FATAL_ERROR "E1.1c pre-admission reset negative control did not trip")
endif()

string(REPLACE
    "server_cache_control_status selector_status ="
    "if (params_base.vbr_enabled()) {}\n                            server_cache_control_status selector_status ="
    old_refusal_negative "${server_source}")
real_vbr_grant_contract("${old_refusal_negative}" old_refusal_negative_ok)
if (old_refusal_negative_ok)
    message(FATAL_ERROR "E1.1c obsolete-refusal negative control did not trip")
endif()

message(STATUS "E1.1c VBR enforcement contracts passed")
