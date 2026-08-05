# B-A pre-mutation/authority contract. The source ordering pins are deliberate:
# moving the planner below the first cache mutation silently turns it back into a
# post-hoc observer. The negative control swaps the two calls in-memory.
if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" task_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-plan-authority.h" authority_header)
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" plan_header)

# The landed behavior ceiling moves one ratchet at a time. B-A4 is the final
# declared level, so this pin certifies the complete graduated ladder.
string(REGEX MATCH
    "SERVER_CACHE_PLAN_IMPLEMENTED_AUTHORITY_LEVEL[ \t\r\n]*=[ \t\r\n]*common_cache_plan_authority_level::lru;"
    implemented_ceiling "${authority_header}")
if (NOT implemented_ceiling)
    message(FATAL_ERROR "B-A implemented authority ceiling is not lru")
endif()

function(cache_plan_authority_order_valid source output)
    string(FIND "${source}" "task, *ret, incoming_adapter, *plan_rec);" inventory_pos)
    string(FIND "${source}" "cache_plan_authority->plan_before_mutation(" planner_pos)
    string(FIND "${source}" "cache_plan_authority->authorize(" authority_pos)
    string(FIND "${source}" "recurrent_shrink_for_prefill(\"before prompt cache save/load\")" mutation_pos)
    if (inventory_pos GREATER_EQUAL 0 AND
        planner_pos GREATER inventory_pos AND
        authority_pos GREATER planner_pos AND
        mutation_pos GREATER authority_pos)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

cache_plan_authority_order_valid("${server_source}" order_valid)
if (NOT order_valid)
    message(FATAL_ERROR "B-A planner is not staged before the first cache mutation")
endif()

string(REPLACE
    "cache_plan_authority->authorize("
    "recurrent_shrink_for_prefill(\"before prompt cache save/load\"); cache_plan_authority->authorize("
    order_negative "${server_source}")
cache_plan_authority_order_valid("${order_negative}" negative_valid)
if (negative_valid)
    message(FATAL_ERROR "B-A pre-mutation ordering negative control did not trip")
endif()

function(cache_plan_complete_inventory_valid source output)
    string(FIND "${source}" "bool cache_plan_inventory_live_rows(" inventory_begin)
    string(FIND "${source}" "server_slot * get_available_slot(" inventory_end)
    if (inventory_begin EQUAL -1 OR inventory_end LESS_EQUAL inventory_begin)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR inventory_length "${inventory_end} - ${inventory_begin}")
    string(SUBSTRING "${source}" ${inventory_begin} ${inventory_length} inventory_body)
    string(FIND "${inventory_body}" "std::find_if" short_circuit_pos)
    if (short_circuit_pos EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

cache_plan_complete_inventory_valid("${server_source}" complete_valid)
if (NOT complete_valid)
    message(FATAL_ERROR "B-A planner inventory contains a short-circuit scan")
endif()
string(REPLACE
    "rec.authority_inventory_complete = true;"
    "std::find_if(); rec.authority_inventory_complete = true;"
    complete_negative "${server_source}")
cache_plan_complete_inventory_valid("${complete_negative}" complete_negative_valid)
if (complete_negative_valid)
    message(FATAL_ERROR "B-A complete-inventory negative control did not trip")
endif()

# Authority must retain the shipped save-before-load side effect. The exact
# selected-source restore follows it, so any save-time dedup is revalidated
# before the authority execution branch.
string(FIND "${server_source}" "ret->prompt_save(*prompt_cache);" save_pos)
string(FIND "${server_source}" "if (authority_exec) {" execution_pos)
if (save_pos EQUAL -1 OR execution_pos LESS_EQUAL save_pos)
    message(FATAL_ERROR "B-A authority bypasses displaced-state prompt_save")
endif()

string(REGEX MATCHALL "server_cache_plan_disarm_unlaunched\\(" disarm_sites "${server_source}")
list(LENGTH disarm_sites n_disarm_sites)
string(REGEX MATCHALL
    "server_cache_plan_disarm_unlaunched\\([^\\)]*cache_plan_destruction_recovery_pin\\)"
    disarm_pin_sites "${server_source}")
list(LENGTH disarm_pin_sites n_disarm_pin_sites)
if (n_disarm_sites LESS 5 OR NOT n_disarm_pin_sites EQUAL n_disarm_sites)
    message(FATAL_ERROR "B-A launch-failure exits are not fully disarmed")
endif()

string(FIND "${server_source}" "cache_plan_override_checkpoint(" override_begin)
string(FIND "${server_source}" "server_slot * get_available_slot(" override_end)
if (override_begin EQUAL -1 OR override_end LESS_EQUAL override_begin)
    message(FATAL_ERROR "B-A checkpoint override seam is missing")
endif()
math(EXPR override_length "${override_end} - ${override_begin}")
string(SUBSTRING "${server_source}" ${override_begin} ${override_length} override_body)
string(FIND "${override_body}" "return fallback;" override_fallback)
string(FIND "${override_body}" ".rend()" override_cold)
if (override_fallback EQUAL -1 OR NOT override_cold EQUAL -1)
    message(FATAL_ERROR "B-A checkpoint override can synthesize cold on drift")
endif()

foreach(pin
        "common_cache_plan_authority_level cache_plan_authority{};"
        "bool planner_precomputed = false;"
        "bool authority_prequalified = false;"
        "bool inventory_saturated() const noexcept"
        "server_cache_plan_execution authorize("
        "required_source_id >= 0"
        "server_cache_plan_demote_for_coverage_recovery("
        "server_cache_plan_revalidate_checkpoint_execution("
        "server_cache_plan_disarm_unlaunched("
        "GGML_ASSERT(rec != nullptr || required_source_id < 0);"
        "int32_t cache_plan_source_id = -1;"
        "COMMON_CACHE_PLAN_SCHEMA_VERSION = 6")
    string(FIND "${common_header}${plan_header}${server_source}${task_source}${task_header}${authority_header}" "${pin}" pin_pos)
    if (pin_pos EQUAL -1)
        message(FATAL_ERROR "B-A authority contract pin missing: ${pin}")
    endif()
endforeach()

message(STATUS "B-A pre-mutation authority contract passed")
