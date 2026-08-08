# D-S2 source contract: every accounting category has exactly one capacity-participation /
# transactional classification, and process-local budget types stay out of wire
# and shipped artifact structs. Negative controls mutate source copies in memory.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" acct_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-budget.h" budget_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" plan_header)
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" server_task_header)

function(cache_budget_classification_valid table_header output)
    string(REGEX MATCH
        "enum class llama_cache_acct_category[^\\{]*\\{([^}]*)\\}"
        category_match "${acct_header}")
    if (category_match STREQUAL "")
        message(FATAL_ERROR "could not parse llama_cache_acct_category")
    endif()
    set(category_body "${CMAKE_MATCH_1}")
    string(REGEX REPLACE "//[^\n]*" "" category_body "${category_body}")
    string(REGEX MATCHALL
        "[A-Za-z_][A-Za-z0-9_]*[ \t]*(=[^,\n]+)?[ \t]*,"
        category_tokens "${category_body}")

    set(categories "")
    foreach(token IN LISTS category_tokens)
        string(REGEX REPLACE "[ \t]*=.*" "" name "${token}")
        string(REGEX REPLACE "[ \t]*,$" "" name "${name}")
        string(STRIP "${name}" name)
        if (NOT name STREQUAL "_count")
            list(APPEND categories "${name}")
        endif()
    endforeach()

    string(REGEX MATCHALL
        "X\\([A-Za-z_][A-Za-z0-9_]*,[ \t]*(participating|excluded),[ \t]*(direct_gauge|transactional),[ \t]*(device|host|by_domain|none)\\)"
        table_rows "${table_header}")
    set(classified "")
    foreach(row IN LISTS table_rows)
        string(REGEX REPLACE "^X\\(([A-Za-z_][A-Za-z0-9_]*),.*" "\\1"
            name "${row}")
        list(APPEND classified "${name}")
    endforeach()

    set(valid TRUE)
    foreach(name IN LISTS categories)
        set(hits 0)
        foreach(candidate IN LISTS classified)
            if (candidate STREQUAL name)
                math(EXPR hits "${hits} + 1")
            endif()
        endforeach()
        if (NOT hits EQUAL 1)
            set(valid FALSE)
        endif()
    endforeach()
    foreach(name IN LISTS classified)
        if (NOT name IN_LIST categories)
            set(valid FALSE)
        endif()
    endforeach()
    list(LENGTH categories category_count)
    list(LENGTH classified classified_count)
    if (NOT category_count EQUAL classified_count)
        set(valid FALSE)
    endif()
    set(${output} "${valid}" PARENT_SCOPE)
endfunction()

cache_budget_classification_valid("${budget_header}" table_valid)
if (NOT table_valid)
    message(FATAL_ERROR
        "D-S2 category classification is not an exact exhaustive census")
endif()

string(REGEX REPLACE
    "[ \t]*X\\(container_overhead,[^\n]*\n" ""
    classification_negative "${budget_header}")
cache_budget_classification_valid("${classification_negative}" negative_valid)
if (negative_valid)
    message(FATAL_ERROR
        "D-S2 category-classification negative control did not trip")
endif()

# Budget rows are a process-local observer surface. They must not leak into the
# cache-plan wire record or the three shipped artifact structs.
foreach(source IN ITEMS plan_header common_header server_task_header)
    string(FIND "${${source}}" "llama_cache_budget_" budget_leak)
    if (NOT budget_leak EQUAL -1)
        message(FATAL_ERROR
            "D-S2 process-local budget type leaked into ${source}")
    endif()
endforeach()
set(purity_negative "${plan_header}\nllama_cache_budget_result forbidden;")
string(FIND "${purity_negative}" "llama_cache_budget_" negative_leak)
if (negative_leak EQUAL -1)
    message(FATAL_ERROR "D-S2 process-local purity negative control did not trip")
endif()

count_literal("${acct_header}"
    "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2" acct_schema_pins)
count_literal("${plan_header}"
    "COMMON_CACHE_PLAN_SCHEMA_VERSION = 7" plan_schema_pins)
if (NOT acct_schema_pins EQUAL 1 OR NOT plan_schema_pins EQUAL 1)
    message(FATAL_ERROR
        "accounting/cache-plan schema pin drifted "
        "(found C=${acct_schema_pins}, plan=${plan_schema_pins})")
endif()

message(STATUS "D-S2 cache-budget classification/purity contracts passed")
