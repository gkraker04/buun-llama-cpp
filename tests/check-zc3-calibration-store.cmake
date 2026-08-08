if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-store.cpp" STORE_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-store.h" STORE_H)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp" OBSERVER_CPP)

foreach(REQUIRED IN ITEMS
        "ROOT_PAYLOAD_LIMIT = 64 * 1024"
        "PROFILE_PAYLOAD_LIMIT = 1024 * 1024"
        "MAX_PROFILES = 16"
        "BUUNCAL1"
        "buun-cache-calibration-v1"
        "O_EXCL"
        "O_NOFOLLOW"
        "openat(directory_descriptor_"
        "fstatat(directory_descriptor_"
        "validated_regular_stat"
        "STORE_ENTRY_LIMIT = 64"
        "STORE_BYTE_LIMIT = 32 * 1024 * 1024"
        "static_assert(STORE_COMMIT_HIGH_WATER < STORE_BYTE_LIMIT)"
        "LOCK_EX | LOCK_NB"
        "manifest_.next_boot_claim_ordinal++"
        "manifest_.next_profile_generation_ordinal++"
        "manifest_.next_immutable_file_ordinal++"
        "manifest_.next_persisted_prune_epoch++"
        "positive_definite(instance.v"
        "load_referenced_profile(*existing"
        "server_cache_calibration_commit_ack"
        "server_cache_calibration_profile_currency"
        "ack.profile_identity_digest"
        "ack.profile_generation_ordinal !="
        "ack.profile_file_generation <="
        "server_cache_calibration_validate_profile("
        "resume_validation_pending = true"
        "complete_resume_validation()"
        "enqueue_one_cached_dirty()"
        "Slot reuse requires immutable-snapshot acceptance"
        "committed_ack_count_.load(std::memory_order_acquire)"
        "drain_latest_for_shutdown("
        "fchmod(fd, 0600)"
        "static_assert(std::is_standard_layout_v<server_cache_calibration_profile_snapshot>)"
        "static_assert(sizeof(server_cache_calibration_profile_snapshot) <= 1024 * 1024)")
    contract_require_token("${STORE_CPP}${STORE_H}" "${REQUIRED}"
        "ZC3b bounded/capability/dirty-currency contract")
endforeach()

foreach(REQUIRED IN ITEMS
        "cache_calibration->start(calibration_dir)"
        "cache_calibration->resolve_load("
        "cache_calibration->lifecycle(*cache_optimizer_observations)"
        "cache_calibration->flush_latest(*cache_optimizer_observations)")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC3b coordinator integration contract")
endforeach()

contract_find_forbidden("${STORE_CPP}${STORE_H}" FORBIDDEN
    "cache_plan_authority"
    "authorize("
    "finalize_execution"
    "SERVER_TASK_TYPE"
    "CACHE_PLAN")
if (FORBIDDEN)
    message(FATAL_ERROR "ZC3b persistence acquired planner/task authority: ${FORBIDDEN}")
endif()

contract_find_forbidden("${CONTEXT_CPP}" CONTEXT_IO
    "manifest.bcal"
    "profile_name("
    "openat("
    "fsync("
    "renameat("
    "write_file_exclusive")
if (CONTEXT_IO)
    message(FATAL_ERROR "ZC3b scheduler context acquired file I/O: ${CONTEXT_IO}")
endif()

count_literal("${CONTEXT_CPP}" "cache_calibration->start(calibration_dir)"
    COORDINATOR_START_COUNT)
if (NOT COORDINATOR_START_COUNT EQUAL 1)
    message(FATAL_ERROR "ZC3b coordinator must have one production start; got ${COORDINATOR_START_COUNT}")
endif()

function(zc3b_validate_observer_gate TEXT OUT)
    contract_extract_region("${TEXT}"
        "if (params_base.cache_optimizer.observer_store_enabled) {"
        "if (params_base.cache_optimizer.cache_debug ||"
        OBSERVER_REGION OBSERVER_REGION_FOUND)
    if (NOT OBSERVER_REGION_FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    count_literal("${OBSERVER_REGION}"
        "cache_calibration->start(calibration_dir)" REGION_START_COUNT)
    if (NOT REGION_START_COUNT EQUAL 1)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    string(REPLACE "${OBSERVER_REGION}" "" OUTSIDE_REGION "${TEXT}")
    string(FIND "${OUTSIDE_REGION}"
        "cache_calibration->start(calibration_dir)" OUTSIDE_START)
    if (NOT OUTSIDE_START EQUAL -1)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()

zc3b_validate_observer_gate("${CONTEXT_CPP}" OBSERVER_GATE_VALID)
if (NOT OBSERVER_GATE_VALID)
    message(FATAL_ERROR "ZC3b coordinator escaped the observer-only construction gate")
endif()

count_literal("${OBSERVER_CPP}" "increment_saturating(mutation_generation_);"
    MUTATION_COUNT)
if (NOT MUTATION_COUNT EQUAL 2)
    message(FATAL_ERROR "ZC3b authority-currency generation census drifted: ${MUTATION_COUNT}")
endif()

file(GLOB SERVER_PRODUCTION "${SOURCE_ROOT}/tools/server/*.cpp")
set(PRODUCTION_TEXT "")
foreach(FILE IN LISTS SERVER_PRODUCTION)
    if (NOT FILE MATCHES "server-cache-calibration-store\\.cpp$")
        file(READ "${FILE}" FILE_TEXT)
        string(APPEND PRODUCTION_TEXT "${FILE_TEXT}")
    endif()
endforeach()
string(FIND "${PRODUCTION_TEXT}" "server_cache_calibration_set_test_fault(" FAULT_CALL)
if (NOT FAULT_CALL EQUAL -1)
    message(FATAL_ERROR "ZC3b test fault door escaped into production")
endif()

# Mutation controls exercise the same scoped validators/censuses.
set(MUTATED_CONTEXT "${CONTEXT_CPP}\ncache_calibration->start(calibration_dir);")
count_literal("${MUTATED_CONTEXT}" "cache_calibration->start(calibration_dir)"
    MUTATED_START_COUNT)
if (MUTATED_START_COUNT EQUAL 1)
    message(FATAL_ERROR "ZC3b coordinator-start negative control did not trip")
endif()

set(MUTATED_GATE "${CONTEXT_CPP}")
string(REPLACE
    "if (params_base.cache_optimizer.observer_store_enabled) {"
    "if (true) {"
    MUTATED_GATE "${MUTATED_GATE}")
zc3b_validate_observer_gate("${MUTATED_GATE}" MUTATED_GATE_VALID)
if (MUTATED_GATE_VALID)
    message(FATAL_ERROR "ZC3b observer-gate negative control did not trip")
endif()

set(MUTATED_OBSERVER "${OBSERVER_CPP}")
string(REPLACE "increment_saturating(mutation_generation_);" ""
    MUTATED_OBSERVER "${MUTATED_OBSERVER}")
count_literal("${MUTATED_OBSERVER}" "increment_saturating(mutation_generation_);"
    MUTATED_MUTATION_COUNT)
if (MUTATED_MUTATION_COUNT EQUAL 2)
    message(FATAL_ERROR "ZC3b currency-mutation negative control did not trip")
endif()

set(MUTATED_STORE "${STORE_CPP}")
string(REPLACE " | O_NOFOLLOW" "" MUTATED_STORE "${MUTATED_STORE}")
count_literal("${STORE_CPP}" "O_NOFOLLOW" ORIGINAL_NOFOLLOW_COUNT)
count_literal("${MUTATED_STORE}" "O_NOFOLLOW" MUTATED_NOFOLLOW_COUNT)
if (NOT MUTATED_NOFOLLOW_COUNT LESS ORIGINAL_NOFOLLOW_COUNT)
    message(FATAL_ERROR "ZC3b no-follow negative control did not trip")
endif()

set(MUTATED_STORE_MODE "${STORE_CPP}")
string(REPLACE "fchmod(fd, 0600)" "true"
    MUTATED_STORE_MODE "${MUTATED_STORE_MODE}")
string(FIND "${MUTATED_STORE_MODE}" "fchmod(fd, 0600)"
    MUTATED_STORE_MODE_FOUND)
if (NOT MUTATED_STORE_MODE_FOUND EQUAL -1)
    message(FATAL_ERROR "ZC3b created-file-mode negative control did not trip")
endif()

set(MUTATED_PROFILE_SWITCH "${STORE_CPP}")
string(REPLACE "observer.set_execution_fingerprint(fingerprint);"
    "/* removed atomic profile transition */"
    MUTATED_PROFILE_SWITCH "${MUTATED_PROFILE_SWITCH}")
count_literal("${STORE_CPP}" "observer.set_execution_fingerprint(fingerprint);"
    PROFILE_SWITCH_COUNT)
count_literal("${MUTATED_PROFILE_SWITCH}"
    "observer.set_execution_fingerprint(fingerprint);"
    MUTATED_PROFILE_SWITCH_COUNT)
if (NOT MUTATED_PROFILE_SWITCH_COUNT LESS PROFILE_SWITCH_COUNT)
    message(FATAL_ERROR "ZC3b profile-transition negative control did not trip")
endif()

message(STATUS "ZC3b calibration-store contract checks passed")
