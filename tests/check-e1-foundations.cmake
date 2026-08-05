if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/common/common-cache-family.h" family_header)
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" task_header)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-authority.h" authority_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.h" lease_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.cpp" lease_source)
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" kv_source)
file(READ "${SOURCE_ROOT}/src/llama-context.cpp" llama_context_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-hard-seal.cpp" seal_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-hard-seal.h" seal_header)
file(READ "${SOURCE_ROOT}/include/llama.h" public_header)
file(READ "${SOURCE_ROOT}/common/arg.cpp" arg_source)
file(READ "${SOURCE_ROOT}/tools/server/server.cpp" server_source)

file(GLOB server_production_tus "${SOURCE_ROOT}/tools/server/*.cpp")
set(server_production_source "")
set(pre_family_production_source "")
set(proof_callsite_source "")
foreach(path IN LISTS server_production_tus)
    file(READ "${path}" source)
    string(APPEND server_production_source "\n${source}")
    get_filename_component(name "${path}" NAME)
    if (NOT name STREQUAL "server-cache-control.cpp")
        string(APPEND pre_family_production_source "\n${source}")
    endif()
    if (NOT name STREQUAL "server-cache-lease.cpp" AND
        NOT name STREQUAL "server-cache-retention-proof.cpp" AND
        NOT name STREQUAL "server-cache-vbr-proof.cpp")
        string(APPEND proof_callsite_source "\n${source}")
    endif()
endforeach()

function(require_token source token label)
    string(FIND "${source}" "${token}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "E1.0 ${label} missing '${token}'")
    endif()
endfunction()

function(forbid_token source token label)
    string(FIND "${source}" "${token}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "E1.0 ${label} admitted forbidden '${token}'")
    endif()
endfunction()

function(family_contract_valid family task authority context task_cpp output)
    foreach(token
            "struct common_cache_family_id"
            "enum class common_cache_family_role"
            "struct common_cache_family_binding"
            "common_cache_family_main_family("
            "common_cache_family_allows_additional_weight(")
        string(FIND "${family}" "${token}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    string(FIND "${task}"
        "server_cache_control_token cache_family_binding_token;" task_field)
    string(FIND "${authority}"
        "common_cache_family_binding cache_family;" policy_field)
    string(REGEX MATCH
        "common_cache_family_main_family\\([^;]*cache_family[^;]*\\)"
        checkpoint_resolver "${context}")
    string(REGEX MATCH
        "common_cache_family_allows_additional_weight\\([^;]*victim->cache_family[^;]*\\)"
        callback_guard "${task_cpp}")
    if (task_field EQUAL -1 OR policy_field EQUAL -1 OR
        checkpoint_resolver STREQUAL "" OR callback_guard STREQUAL "")
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

function(no_declared_family_construction source output)
    foreach(token
            "common_cache_family_role::"
            ".cache_family.family"
            ".cache_family.role")
        string(FIND "${source}" "${token}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

function(no_proof_adapter_calls source output)
    string(FIND "${source}" "fallback_proof_for_test(" found)
    if (found EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(no_classifier_callers source output)
    string(FIND "${source}" "vbr_hard_seal_classify(" method_call)
    string(FIND "${source}" "vbr_classify_hard_seal(" kernel_call)
    if (method_call EQUAL -1 AND kernel_call EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Strong types, optional task/slot/host/policy carriers, and one resolver.
family_contract_valid(
    "${family_header}" "${task_header}" "${authority_header}"
    "${context_source}" "${task_source}" family_valid)
if (NOT family_valid)
    message(FATAL_ERROR "E1.0 family foundation contract failed")
endif()
require_token("${common_header}" "common-cache-family.h"
    "checkpoint family include")
require_token("${common_header}" "common_cache_family_binding cache_family;"
    "checkpoint family carrier")
no_declared_family_construction(
    "${pre_family_production_source}" no_declared_construction)
if (NOT no_declared_construction)
    message(FATAL_ERROR "E1.0 production constructed a declared family")
endif()

# The existing lease table remains the one authority and the hard door owns a
# move-only proof. No boolean verdict/evaluator may reappear beside it.
require_token("${lease_header}" "class server_cache_lease_table"
    "one-table census")
require_token("${lease_header}" "class server_cache_durable_fallback_proof"
    "proof owner")
require_token("${lease_header}" "server_cache_durable_fallback_proof acquire("
    "proof provider")
forbid_token("${lease_header}" "fallback_state preflight("
    "second hard evaluator")
require_token("${lease_source}" "auto proof = fallback->acquire(subject, identity);"
    "single hard acquire door")
file(GLOB_RECURSE server_cpp_headers
    "${SOURCE_ROOT}/tools/server/*.cpp"
    "${SOURCE_ROOT}/tools/server/*.h")
set(table_definitions 0)
foreach(path IN LISTS server_cpp_headers)
    file(READ "${path}" source)
    string(REGEX MATCHALL "class server_cache_lease_table \\{" matches "${source}")
    list(LENGTH matches count)
    math(EXPR table_definitions "${table_definitions} + ${count}")
endforeach()
if (NOT table_definitions EQUAL 1)
    message(FATAL_ERROR
        "E1.0 one-table census found ${table_definitions} lease authorities")
endif()

# The private adapter definitions are the only production-TU exceptions.
no_proof_adapter_calls("${proof_callsite_source}" proof_calls_valid)
if (NOT proof_calls_valid)
    message(FATAL_ERROR "E1.0 production called a private proof adapter")
endif()

# The classifier is read-only and has no context/server caller until E1.1c.
require_token("${seal_source}" "uint8_t seal_tier"
    "caller-owned seal tier")
require_token("${seal_header}" "VBR_HARD_SEAL_DEFAULT_FLOOR"
    "frozen default floor")
string(REGEX MATCHALL "vbr_hard_seal_classify\\(" seal_hooks "${kv_source}")
list(LENGTH seal_hooks seal_hook_count)
if (NOT seal_hook_count EQUAL 1)
    message(FATAL_ERROR
        "E1.0 controller must expose exactly one read-only hook definition, found ${seal_hook_count}")
endif()
set(forbidden_classifier_callers
    "${llama_context_source}\n${server_production_source}")
no_classifier_callers(
    "${forbidden_classifier_callers}" classifier_callers_valid)
if (NOT classifier_callers_valid)
    message(FATAL_ERROR "E1.0 hard-seal classifier gained a premature caller")
endif()

# E1.1a/E1.1b deliberately add scheduler-only control/family tasks. Routes,
# CLI flags, VBR enforcement, and public llama.h remain outside these units.
function(surface_contract_valid source output)
    foreach(token
            "/cache/lease"
            "/cache/family"
            "--cache-family")
        string(FIND "${source}" "${token}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
set(surface_sources "${task_header}\n${server_source}\n${arg_source}")
surface_contract_valid("${surface_sources}" surface_valid)
if (NOT surface_valid)
    message(FATAL_ERROR "E1.0 task/route/flag surface boundary failed")
endif()
foreach(token "cache_family" "durable_fallback" "hard_seal")
    forbid_token("${public_header}" "${token}" "public llama.h boundary")
endforeach()

# House-standard negative controls rerun the actual predicates.
string(REPLACE
    "common_cache_family_allows_additional_weight("
    "common_cache_family_additional_weight_removed("
    callback_negative "${task_source}")
family_contract_valid(
    "${family_header}" "${task_header}" "${authority_header}"
    "${context_source}" "${callback_negative}" family_negative_valid)
if (family_negative_valid)
    message(FATAL_ERROR "E1.0 callback-neutrality negative control did not trip")
endif()

set(declared_negative
    "${pre_family_production_source}\ncommon_cache_family_role::main")
no_declared_family_construction(
    "${declared_negative}" declared_negative_valid)
if (declared_negative_valid)
    message(FATAL_ERROR "E1.0 declared-construction negative control did not trip")
endif()

set(proof_negative
    "${proof_callsite_source}\nserver_cache_vbr_fallback_proof_for_test(package)")
no_proof_adapter_calls("${proof_negative}" proof_negative_valid)
if (proof_negative_valid)
    message(FATAL_ERROR "E1.0 proof-callsite negative control did not trip")
endif()

set(classifier_negative
    "${forbidden_classifier_callers}\ncache.vbr_hard_seal_classify(out)")
no_classifier_callers("${classifier_negative}" classifier_negative_valid)
if (classifier_negative_valid)
    message(FATAL_ERROR "E1.0 classifier-caller negative control did not trip")
endif()

set(surface_negative "${surface_sources}\nPOST /cache/lease")
surface_contract_valid("${surface_negative}" surface_negative_valid)
if (surface_negative_valid)
    message(FATAL_ERROR "E1.0 route negative control did not trip")
endif()

message(STATUS "E1.0 behavior-neutral foundation contracts passed")
