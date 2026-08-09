cmake_minimum_required(VERSION 3.14)

include(${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake)

set(root "${CMAKE_CURRENT_LIST_DIR}/..")
file(READ "${root}/common/common.h" common_h)
file(READ "${root}/common/arg.cpp" arg_cpp)
file(READ "${root}/common/common-cache-optimizer.cpp" optimizer_cpp)
file(READ "${root}/tools/server/server.cpp" server_cpp)
file(READ "${root}/tools/server/server-http.cpp" http_cpp)
file(READ "${root}/tools/server/server-context.cpp" context_cpp)
file(READ "${root}/tools/server/README.md" server_readme)

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
foreach(pin IN ITEMS
        "cache optimizer: mode=%s (%s), retention=%s, calibration=%s"
        "params.cache_optimizer_mode_explicit ? \"explicit\" : \"default\""
        "common_cache_optimizer_mode_name(cache_optimizer.mode)"
        "cache_optimizer.observer_store_enabled ? \"enabled\" : \"disabled\""
        "cache_optimizer.local_authority_ceiling")
    contract_require_token("${server_cpp}" "${pin}" "ZC6 startup summary")
endforeach()

# Learned authority remains opt-in. The raw CLI carrier and effective resolver
# both map omission to the historical off path and preserve explicit modes.
contract_require_token("${optimizer_cpp}"
    "const auto mode = raw.mode_explicit"
    "absent optimizer default owner")
contract_require_token("${optimizer_cpp}"
    ": common_cache_optimizer_mode::off;"
    "opt-in optimizer default")
contract_require_token("${common_h}"
    "common_cache_optimizer_mode cache_optimizer_mode = common_cache_optimizer_mode::off;"
    "raw absent-mode default")
contract_require_token("${arg_cpp}"
    "default: off; use auto to enable learned cache authority"
    "opt-in CLI default guidance")
contract_require_token("${server_readme}"
    "The cache optimizer defaults to `off`, preserving the historical policy"
    "opt-in public default guidance")
contract_forbid_token("${server_readme}"
    "The cache optimizer defaults to `auto`"
    "stale automatic-default guidance")
set(default_mutation "${optimizer_cpp}")
string(REPLACE
    "const auto mode = raw.mode_explicit"
    "const auto mode = true"
    default_mutation "${default_mutation}")
string(FIND "${default_mutation}"
    "const auto mode = raw.mode_explicit"
    old_default)
if (NOT old_default EQUAL -1)
    message(FATAL_ERROR "absent-default negative control did not trip")
endif()

# ZC1 consumes retention_policy at its host/checkpoint adapters, the one
# save-time lineage resolver, record/preflight serialization, and
# initialization wiring. ZC2/ZC3 consume observer_store_enabled exactly at
# loader descriptor capture and observer construction. ZC5a consumes the
# local authority ceiling at four scoped startup/planning doors in context.

file(GLOB_RECURSE production_cpp
    "${root}/tools/server/*.cpp"
    "${root}/tools/server/*.h")

set(all_production "")
foreach(path IN LISTS production_cpp)
    file(READ "${path}" text)
    string(APPEND all_production "${text}")
endforeach()

set(expected_observer_store_enabled_reads 9)
count_literal("${all_production}" ".observer_store_enabled"
    observer_store_enabled_reads)
if (NOT observer_store_enabled_reads EQUAL expected_observer_store_enabled_reads)
    message(FATAL_ERROR
        "ZC2 observer-store consumer census drifted: ${observer_store_enabled_reads}")
endif()
count_literal("${all_production}" ".retention_policy" retention_policy_reads)
if (NOT retention_policy_reads EQUAL 7)
    message(FATAL_ERROR
        "ZC1 retention-policy consumer census drifted: ${retention_policy_reads}")
endif()
count_literal("${context_cpp}" ".local_authority_ceiling"
    local_authority_ceiling_reads)
if (NOT local_authority_ceiling_reads EQUAL 4)
    message(FATAL_ERROR
        "ZC5a local-authority ceiling consumer census drifted: ${local_authority_ceiling_reads}")
endif()

# Negative controls: an extra authority or retention consumer must fail its
# exact census.
set(future_mutation "${context_cpp}\nvoid probe(const auto & effective) { (void) effective.local_authority_ceiling; }")
count_literal("${future_mutation}" ".local_authority_ceiling"
    future_mutation_hits)
if (future_mutation_hits EQUAL 4)
    message(FATAL_ERROR "authority-policy member-read negative control did not trip")
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
if (retention_mutation_reads EQUAL 7)
    message(FATAL_ERROR "retention-policy consumer negative control did not trip")
endif()

contract_require_token("${optimizer_cpp}"
    "out.landed_authority_level = raw.cache_plan_authority;"
    "off-mode landed authority identity")
contract_require_token("${optimizer_cpp}"
    "out.landed_authority_level = common_cache_plan_authority_level::off;"
    "non-off landed authority suppression")

message(STATUS "cache optimizer effective-config contract scan passed")
