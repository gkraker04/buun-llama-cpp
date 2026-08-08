cmake_minimum_required(VERSION 3.14)

include(${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake)

set(root "${CMAKE_CURRENT_LIST_DIR}/..")
file(READ "${root}/common/common.h" common_h)
file(READ "${root}/common/arg.cpp" arg_cpp)
file(READ "${root}/common/common-cache-optimizer.cpp" optimizer_cpp)
file(READ "${root}/tools/server/server.cpp" server_cpp)
file(READ "${root}/tools/server/server-http.cpp" http_cpp)
file(READ "${root}/tools/server/server-context.cpp" context_cpp)

function(assert_no_runtime_raw_reads text label)
    string(REGEX MATCHALL
        "params(_base)?\\.(cache_debug|cache_lifecycle|cache_plan_authority|cache_plan_preflight|cache_control_api)"
        raw_hits "${text}")
    if (raw_hits)
        message(FATAL_ERROR "${label}: raw cache-control reads escaped resolver: ${raw_hits}")
    endif()
endfunction()

assert_no_runtime_raw_reads("${server_cpp}" "server startup")
assert_no_runtime_raw_reads("${http_cpp}" "HTTP middleware")
assert_no_runtime_raw_reads("${context_cpp}" "server context")

# Negative control: the scanner must reject the most tempting reintroduction.
set(mutated "${context_cpp}\nif (params_base.cache_lifecycle) {}\n")
string(REGEX MATCHALL
    "params(_base)?\\.(cache_debug|cache_lifecycle|cache_plan_authority|cache_plan_preflight|cache_control_api)"
    mutation_hits "${mutated}")
if (NOT mutation_hits)
    message(FATAL_ERROR "raw-read negative control did not trip")
endif()

foreach(pin IN ITEMS
        "params.cache_optimizer_mode_explicit = true;"
        "params.cache_debug_explicit = true;"
        "params.cache_plan_preflight_explicit = true;"
        "params.cache_control_api_explicit = true;"
        "params.cache_plan_authority_explicit = true;"
        "params.cache_lifecycle_explicit = true;")
    contract_require_token("${arg_cpp}" "${pin}" "parser explicitness")
endforeach()

contract_require_token("${arg_cpp}"
    "common_cache_optimizer_resolve_params(params, &cache_optimizer_error)"
    "post-parse resolver")
contract_require_token("${server_cpp}"
    "common_cache_optimizer_resolve_params(params, &cache_optimizer_error)"
    "post-preset resolver")

# Until ZC6 the absent default is deliberately off. This exact declaration is
# a mutation-sensitive pin: changing it to auto before the default ratchet must
# fail the foundation gate.
contract_require_token("${common_h}"
    "common_cache_optimizer_mode cache_optimizer_mode = common_cache_optimizer_mode::off;"
    "absent optimizer default")
set(default_mutation "${common_h}")
string(REPLACE
    "common_cache_optimizer_mode cache_optimizer_mode = common_cache_optimizer_mode::off;"
    "common_cache_optimizer_mode cache_optimizer_mode = common_cache_optimizer_mode::auto_mode;"
    default_mutation "${default_mutation}")
string(FIND "${default_mutation}"
    "common_cache_optimizer_mode cache_optimizer_mode = common_cache_optimizer_mode::off;"
    old_default)
if (NOT old_default EQUAL -1)
    message(FATAL_ERROR "absent-default negative control did not trip")
endif()

# ZC1 consumes retention_policy at its host/checkpoint adapters, the one
# save-time lineage resolver, record/preflight serialization, and
# initialization wiring. ZC2/ZC3 consume observer_store_enabled exactly at
# loader descriptor capture and observer construction. The authority ceiling
# remains descriptive until its own ratchet.
function(assert_no_future_policy_reads text label)
    string(REGEX MATCHALL
        "(\\.|->)local_authority_ceiling"
        future_hits "${text}")
    if (future_hits)
        message(FATAL_ERROR
            "${label}: ZC0b descriptive policy field acquired a production read: ${future_hits}")
    endif()
endfunction()

file(GLOB_RECURSE production_cpp
    "${root}/tools/server/*.cpp"
    "${root}/tools/server/*.h")
foreach(path IN LISTS production_cpp)
    file(READ "${path}" text)
    assert_no_future_policy_reads("${text}" "${path}")
endforeach()

set(all_production "")
foreach(path IN LISTS production_cpp)
    file(READ "${path}" text)
    string(APPEND all_production "${text}")
endforeach()

set(expected_observer_store_enabled_reads 8)
count_literal("${all_production}" ".observer_store_enabled"
    observer_store_enabled_reads)
if (NOT observer_store_enabled_reads EQUAL expected_observer_store_enabled_reads)
    message(FATAL_ERROR
        "ZC2 observer-store consumer census drifted: ${observer_store_enabled_reads}")
endif()
count_literal("${all_production}" ".retention_policy" retention_policy_reads)
if (NOT retention_policy_reads EQUAL 6)
    message(FATAL_ERROR
        "ZC1 retention-policy consumer census drifted: ${retention_policy_reads}")
endif()

# Negative controls: future member reads remain forbidden, and an extra
# retention-policy consumer must fail the exact census.
set(future_mutation "void probe(const auto & effective) { (void) effective.local_authority_ceiling; }")
string(REGEX MATCHALL
    "(\\.|->)local_authority_ceiling"
    future_mutation_hits "${future_mutation}")
if (NOT future_mutation_hits)
    message(FATAL_ERROR "future-policy member-read negative control did not trip")
endif()
set(observer_mutation "${all_production}\nvoid probe(const auto & effective) { (void) effective.observer_store_enabled; }")
count_literal("${observer_mutation}" ".observer_store_enabled"
    observer_mutation_reads)
math(EXPR expected_observer_mutation_reads
    "${expected_observer_store_enabled_reads} + 1")
if (NOT observer_mutation_reads EQUAL expected_observer_mutation_reads)
    message(FATAL_ERROR "observer-store consumer negative control did not trip")
endif()
set(retention_mutation "${all_production}\nvoid probe(const auto & effective) { (void) effective.retention_policy; }")
count_literal("${retention_mutation}" ".retention_policy" retention_mutation_reads)
if (retention_mutation_reads EQUAL 6)
    message(FATAL_ERROR "retention-policy consumer negative control did not trip")
endif()

contract_require_token("${optimizer_cpp}"
    "out.landed_authority_level = raw.cache_plan_authority;"
    "off-mode landed authority identity")
contract_require_token("${optimizer_cpp}"
    "out.landed_authority_level = common_cache_plan_authority_level::off;"
    "non-off landed authority suppression")

message(STATUS "cache optimizer effective-config contract scan passed")
