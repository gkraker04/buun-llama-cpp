# B-A0b pre-mutation/dual-run contract. The source ordering pins are deliberate:
# moving the planner below the first cache mutation silently turns it back into a
# post-hoc observer. The negative control swaps the two calls in-memory.
if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_source)
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" plan_header)

function(cache_plan_authority_order_valid source output)
    string(FIND "${source}" "task, *ret, incoming_adapter, *plan_rec);" inventory_pos)
    string(FIND "${source}" "cache_plan_authority->plan_before_mutation(" planner_pos)
    string(FIND "${source}" "recurrent_shrink_for_prefill(\"before prompt cache save/load\")" mutation_pos)
    if (inventory_pos GREATER_EQUAL 0 AND
        planner_pos GREATER inventory_pos AND
        mutation_pos GREATER planner_pos)
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
    "cache_plan_authority->plan_before_mutation("
    "recurrent_shrink_for_prefill(\"before prompt cache save/load\"); cache_plan_authority->plan_before_mutation("
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

foreach(pin
        "common_cache_plan_authority_level cache_plan_authority{};"
        "bool planner_precomputed = false;"
        "bool authority_prequalified = false;"
        "bool inventory_saturated() const noexcept"
        "COMMON_CACHE_PLAN_SCHEMA_VERSION = 5")
    string(FIND "${common_header}${plan_header}" "${pin}" pin_pos)
    if (pin_pos EQUAL -1)
        message(FATAL_ERROR "B-A0b contract pin missing: ${pin}")
    endif()
endforeach()

message(STATUS "B-A0b pre-mutation authority contract passed")
