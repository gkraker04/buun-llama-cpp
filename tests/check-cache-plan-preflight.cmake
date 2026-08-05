if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-plan-preflight.cpp" preflight_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)

function(extract_between source begin_token end_token output ok)
    string(FIND "${source}" "${begin_token}" begin)
    string(FIND "${source}" "${end_token}" end)
    if (begin EQUAL -1 OR end LESS_EQUAL begin)
        set(${output} "" PARENT_SCOPE)
        set(${ok} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} body)
    set(${output} "${body}" PARENT_SCOPE)
    set(${ok} TRUE PARENT_SCOPE)
endfunction()

function(preflight_contract_valid source output)
    string(FIND "${source}"
        "server_cache_plan_preflight_view cache_plan_preflight(" begin)
    string(FIND "${source}" "bool build_capture_request(" end)
    if (begin EQUAL -1 OR end LESS_EQUAL begin)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} body)
    foreach(required
            "assert_scheduler_thread("
            "cache_plan_preflight_scheduler_thread,"
            "cache_plan_stage1_mode_for(plan_authority, true)"
            "server_cache_plan_preflight_build_view(")
        string(FIND "${body}" "${required}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    foreach(forbidden
            "destruction_quote_sequence"
            "quote_samples"
            "cache_plan_obs"
            "server_cache_plan_disarm_unlaunched("
            "finalize_execution("
            "authorize("
            "server_cache_prepare_release_set("
            "server_cache_prepared_release_capability"
            "server_cache_recovery_pin::acquire("
            "prompt_save("
            "prompt_load("
            "prompt_clear")
        string(FIND "${body}" "${forbidden}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

preflight_contract_valid("${context_source}" contract_valid)
if (NOT contract_valid)
    message(FATAL_ERROR "E0.1 read-only preflight contract failed")
endif()

string(REPLACE
    "server_cache_plan_preflight_view view;"
    "server_cache_plan_preflight_view view; cache_plan_authority->finalize_execution(*(common_cache_plan_record *) nullptr);"
    mutation_negative "${context_source}")
preflight_contract_valid("${mutation_negative}" mutation_negative_valid)
if (mutation_negative_valid)
    message(FATAL_ERROR "E0.1 mutation negative control did not trip")
endif()

function(stage1_mode_wiring_valid source output)
    extract_between("${source}"
        "void cache_plan_inventory_and_plan_before_mutation("
        "server_slot * get_available_slot("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(required
            "common_cache_plan_destruction_counters throwaway_destruction_counters;"
            "auto * destruction_counters = mode.preflight\n            ? &throwaway_destruction_counters"
            "!mode.preflight && cache_authority\n                ? &cache_authority->destruction_quote_sequence"
            "const bool quote_lifecycle_available = mode.preflight\n            ? preview_lifecycle_available\n            : true;"
            "if (destruction_counters &&"
            "mode.preflight,"
            "*destruction_counters, &source_registry")
        string(FIND "${body}" "${required}" found_pin)
        if (found_pin EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

stage1_mode_wiring_valid("${context_source}" wiring_valid)
if (NOT wiring_valid)
    message(FATAL_ERROR "E0.1 throwaway/sequence mode wiring drifted")
endif()
string(REPLACE
    "? preview_lifecycle_available\n            : true;"
    "? preview_lifecycle_available\n            : preview_lifecycle_available;"
    lifecycle_negative "${context_source}")
stage1_mode_wiring_valid("${lifecycle_negative}" lifecycle_negative_valid)
if (lifecycle_negative_valid)
    message(FATAL_ERROR "E0.1 real-path lifecycle literal negative control did not trip")
endif()
string(REPLACE
    "? &throwaway_destruction_counters"
    "? &cache_authority->destruction_counters"
    counter_negative "${context_source}")
stage1_mode_wiring_valid("${counter_negative}" counter_negative_valid)
if (counter_negative_valid)
    message(FATAL_ERROR "E0.1 production-counter use-site negative control did not trip")
endif()

function(shared_kernel_read_only source output)
    extract_between("${source}"
        "bool cache_plan_inventory_live_rows("
        "server_slot * get_available_slot("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(forbidden
            "cache_plan_begin_inventory("
            "prompt_cache->update("
            "prompt_cache->states.erase"
            "prompt_cache->states.emplace"
            "prompt_save("
            "prompt_load("
            "prompt_clear("
            "server_prompt_cache_destroy_entry"
            "cache_plan_source_id =")
        string(FIND "${body}" "${forbidden}" found_forbidden)
        if (NOT found_forbidden EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

shared_kernel_read_only("${context_source}" kernel_read_only)
if (NOT kernel_read_only)
    message(FATAL_ERROR "E0.1 shared stage-1 kernel gained a prompt-cache mutation")
endif()
string(REPLACE
    "GGML_ASSERT(mode.plan_authority);"
    "GGML_ASSERT(mode.plan_authority); prompt_cache->cache_plan_begin_inventory();"
    prompt_mutation_negative "${context_source}")
shared_kernel_read_only("${prompt_mutation_negative}" prompt_mutation_valid)
if (prompt_mutation_valid)
    message(FATAL_ERROR "E0.1 prompt-cache mutation negative control did not trip")
endif()

foreach(registry_pin
        "source_registry.get("
        "find_host_source("
        "server_cache_plan_local_source_registry local_;")
    string(FIND "${context_source}" "${registry_pin}" registry_pin_pos)
    if (registry_pin_pos EQUAL -1)
        message(FATAL_ERROR
            "E0.1 source registry missed a consumer: '${registry_pin}'")
    endif()
endforeach()

foreach(unminted_pin
        "bool preview_unminted = false;"
        "(options.admission_sequence == 0 && !options.preview_unminted)"
        "quote.receipt.admission_sequence == 0 ||")
    string(FIND "${context_source}${preflight_source}" "${unminted_pin}"
        unminted_pos)
    if (unminted_pos EQUAL -1)
        file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.h"
            quote_header)
        file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.cpp"
            quote_source)
        string(FIND "${quote_header}${quote_source}" "${unminted_pin}"
            unminted_pos)
    endif()
    if (unminted_pos EQUAL -1)
        message(FATAL_ERROR "E0.1 unminted receipt contract lost '${unminted_pin}'")
    endif()
endforeach()

foreach(forbidden_tu
        "server_cache_prepare_release_set"
        "server_cache_prepared_release_capability"
        "server_cache_recovery_pin"
        "prompt_save"
        "prompt_load"
        "prompt_clear"
        "finalize_execution"
        "destruction_quote_sequence")
    string(FIND "${preflight_source}" "${forbidden_tu}" forbidden_pos)
    if (NOT forbidden_pos EQUAL -1)
        message(FATAL_ERROR
            "E0.1 pure projection linked mutation/capability symbol '${forbidden_tu}'")
    endif()
endforeach()

string(FIND "${task_source}"
    "json server_task_result_cache_plan_preflight::to_json()" result_json)
string(FIND "${task_source}" "return json();" null_json)
if (result_json EQUAL -1 OR null_json LESS_EQUAL result_json)
    message(FATAL_ERROR "E0.1 internal result accidentally gained a wire serializer")
endif()

message(STATUS "E0.1 cache-plan preflight contracts passed")
