if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.cpp" POLICY_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.h" POLICY_H)

function(policy_kernel_is_inert TEXT OUT)
    string(FIND "${TEXT}" "server_cache_plan_retention_set(" POLICY_CALL_POS)
    string(FIND "${TEXT}" "server_cache_simulate_retention(" SIM_CALL_POS)
    if (POLICY_CALL_POS EQUAL -1 AND SIM_CALL_POS EQUAL -1)
        set(${OUT} TRUE PARENT_SCOPE)
    else()
        set(${OUT} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(require_policy_kernel_inert TEXT LABEL)
    policy_kernel_is_inert("${TEXT}" INERT)
    if (NOT INERT)
        message(FATAL_ERROR
            "ZC1 behavior-neutral kernel is called from production region: ${LABEL}")
    endif()
endfunction()

function(policy_has_forbidden TEXT OUT)
    set(FOUND FALSE)
    foreach(FORBIDDEN IN ITEMS
            "checkpoint_drop("
            "prepare_release_set("
            "acct_release_entry("
            "prompt.checkpoints"
            "destruction_counters"
            "cache_plan_authority")
        string(FIND "${TEXT}" "${FORBIDDEN}" POS)
        if (NOT POS EQUAL -1)
            set(FOUND TRUE)
        endif()
    endforeach()
    set(${OUT} ${FOUND} PARENT_SCOPE)
endfunction()

file(GLOB SERVER_PRODUCTION_TUS
    "${SOURCE_ROOT}/tools/server/*.cpp"
    "${SOURCE_ROOT}/tools/server/*.h")
list(REMOVE_ITEM SERVER_PRODUCTION_TUS
    "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.cpp"
    "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.h")
foreach(SOURCE_FILE IN LISTS SERVER_PRODUCTION_TUS)
    file(READ "${SOURCE_FILE}" SOURCE_TEXT)
    get_filename_component(FILE_NAME "${SOURCE_FILE}" NAME)
    require_policy_kernel_inert("${SOURCE_TEXT}" "${FILE_NAME}")
endforeach()

policy_has_forbidden("${POLICY_CPP}" HAS_FORBIDDEN)
if (HAS_FORBIDDEN)
    message(FATAL_ERROR
        "ZC1 pure retention policy acquired a mutation/authority token")
endif()

foreach(REQUIRED IN ITEMS
        "minimum_historical"
        "protected_over_capacity"
        "incomplete_evidence"
        "capacity_unavailable"
        "historical_bucket("
        "stable_id == 0")
    string(FIND "${POLICY_CPP}${POLICY_H}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC1 pure retention policy lost required contract token: ${REQUIRED}")
    endif()
endforeach()

# Negative controls prove both central checks bite.
set(MUTATED_CALL "void probe() { server_cache_simulate_retention({}, {}); }")
policy_kernel_is_inert("${MUTATED_CALL}" MUTATED_INERT)
if (MUTATED_INERT)
    message(FATAL_ERROR "ZC1 caller-census negative control did not trip")
endif()

set(MUTATED_POLICY "${POLICY_CPP}\nvoid probe() { checkpoint_drop(); }")
policy_has_forbidden("${MUTATED_POLICY}" MUTATED_HAS_FORBIDDEN)
if (NOT MUTATED_HAS_FORBIDDEN)
    message(FATAL_ERROR "ZC1 mutation negative control did not trip")
endif()

message(STATUS "ZC1 behavior-neutral retention-policy contract checks passed")
