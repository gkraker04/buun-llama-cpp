if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.cpp" POLICY_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-retention-policy.h" POLICY_H)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" SERVER_TASK_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" SERVER_CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-authority.cpp" CACHE_AUTHORITY_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-authority.h" CACHE_AUTHORITY_H)
file(READ "${SOURCE_ROOT}/tools/server/server-retention-sidecar.cpp" RETENTION_SIDECAR_CPP)
file(READ "${SOURCE_ROOT}/common/common.h" COMMON_H)
file(READ "${SOURCE_ROOT}/common/common-checkpoint-shadow.cpp" CHECKPOINT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/bench/zc1-retention-shapes.py" SHAPES_PY)

function(policy_kernel_is_inert TEXT OUT)
    string(FIND "${TEXT}" "server_cache_plan_retention_set(" POLICY_CALL_POS)
    string(FIND "${TEXT}" "server_cache_simulate_retention(" SIM_CALL_POS)
    string(FIND "${TEXT}" "server_cache_plan_host_retention_victim(" HOST_CALL_POS)
    if (POLICY_CALL_POS EQUAL -1 AND SIM_CALL_POS EQUAL -1 AND
        HOST_CALL_POS EQUAL -1)
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

function(checkpoint_boundary_complete CONTEXT TASK AUTHORITY SIDECAR COMMON CHECKPOINT OUT)
    set(COMPLETE TRUE)
    foreach(REQUIRED IN ITEMS
            "it->id_task_referenced = slot.task->id"
            "server_cache_checkpoint_task_policy::"
            "current_reference"
            "server_retention_anchor_policy::"
            "checkpoint_desired_set")
        string(FIND "${CONTEXT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    string(REGEX MATCHALL
        "server_cache_checkpoint_task_protected\\("
        CONTEXT_TASK_PROTECTION_CALLS "${CONTEXT}")
    list(LENGTH CONTEXT_TASK_PROTECTION_CALLS CONTEXT_TASK_PROTECTION_COUNT)
    string(REGEX MATCHALL
        "server_cache_checkpoint_task_protected\\("
        TASK_PROTECTION_CALLS "${TASK}")
    list(LENGTH TASK_PROTECTION_CALLS TASK_PROTECTION_COUNT)
    foreach(REQUIRED IN ITEMS
            "checkpoint.id_task_referenced == checkpoint_task_id"
            "checkpoint.id_task == checkpoint_task_id")
        string(FIND "${AUTHORITY}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    if (NOT CONTEXT_TASK_PROTECTION_COUNT EQUAL 1 OR
        NOT TASK_PROTECTION_COUNT EQUAL 2)
        set(COMPLETE FALSE)
    endif()
    foreach(REQUIRED IN ITEMS
            "int id_task_referenced = -1"
            "id_task_referenced(other.id_task_referenced)"
            "id_task_referenced = -1")
        string(FIND "${COMMON}${CHECKPOINT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    foreach(REQUIRED IN ITEMS
            "anchor_policy =="
            "server_retention_anchor_policy::checkpoint_desired_set"
            "key.kind != common_retention_artifact_kind::checkpoint"
            "record.stamp.mandatory_anchor = false")
        string(FIND "${SIDECAR}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    set(${OUT} ${COMPLETE} PARENT_SCOPE)
endfunction()

function(checkpoint_evidence_fallback_complete CONTEXT OUT)
    set(COMPLETE TRUE)
    foreach(REQUIRED IN ITEMS
            "zc_fallback_to_historical("
            "if (!zc_retention)"
            "zc_retention = false"
            "checkpoint_publication_allowed = true"
            "common_cache_retention_outcome::deferred"
            "common_cache_retention_reason::checkpoint_staging_unavailable")
        string(FIND "${CONTEXT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    string(REGEX MATCHALL
        "server_cache_retention_status_is_evidence_unavailable\\("
        EVIDENCE_PREDICATE_CALLS "${CONTEXT}")
    list(LENGTH EVIDENCE_PREDICATE_CALLS EVIDENCE_PREDICATE_CALL_COUNT)
    if (NOT EVIDENCE_PREDICATE_CALL_COUNT EQUAL 2)
        set(COMPLETE FALSE)
    endif()
    set(${OUT} ${COMPLETE} PARENT_SCOPE)
endfunction()

function(shapes_gate_modes_complete TEXT OUT)
    set(COMPLETE TRUE)
    foreach(REQUIRED IN ITEMS
            "choices=(1, 3)"
            "max_workers=args.workers"
            "\"workers\": args.workers")
        string(FIND "${TEXT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    set(${OUT} ${COMPLETE} PARENT_SCOPE)
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
    if (FILE_NAME STREQUAL "server-task.cpp")
        # ZC1's host adapter owns the one host selector call.
        string(REPLACE "server_cache_plan_host_retention_victim(" "allowed_host_policy(" CLEANED "${SOURCE_TEXT}")
        require_policy_kernel_inert("${CLEANED}" "${FILE_NAME}")
    elseif(FILE_NAME STREQUAL "server-context.cpp")
        # ZC1's checkpoint adapter calls the same selector once for ordinary
        # publication and once for configuration shrink.
        string(REPLACE "server_cache_plan_retention_set(" "allowed_checkpoint_policy(" CLEANED "${SOURCE_TEXT}")
        require_policy_kernel_inert("${CLEANED}" "${FILE_NAME}")
    else()
        require_policy_kernel_inert("${SOURCE_TEXT}" "${FILE_NAME}")
    endif()
endforeach()

policy_has_forbidden("${POLICY_CPP}" HAS_FORBIDDEN)
if (HAS_FORBIDDEN)
    message(FATAL_ERROR
        "ZC1 pure retention policy acquired a mutation/authority token")
endif()

string(REGEX MATCHALL "server_cache_plan_retention_set\\(" CHECKPOINT_POLICY_CALLS "${SERVER_CONTEXT_CPP}")
list(LENGTH CHECKPOINT_POLICY_CALLS CHECKPOINT_POLICY_CALL_COUNT)
if (NOT CHECKPOINT_POLICY_CALL_COUNT EQUAL 2)
    message(FATAL_ERROR
        "ZC1 checkpoint policy must have exactly two production calls; got ${CHECKPOINT_POLICY_CALL_COUNT}")
endif()

foreach(REQUIRED IN ITEMS
        "server_cache_retention_replay_cap("
        "server_cache_checkpoint_drop_stale("
        "slot.prompt.checkpoints.size() > capacity"
        "publication_skipped_shrink_pending"
        "shrink_blocked_protected"
        "shrink_blocked_recovery_unavailable"
        "slot.retention_obs->reserve_stamp("
        "it->id_task_referenced = slot.task->id"
        "server_cache_checkpoint_task_protected("
        "server_cache_checkpoint_task_policy::"
        "current_reference"
        "server_retention_anchor_policy::"
        "checkpoint_desired_set"
        "last.computation_frontier == ckpt_frontier")
    string(FIND "${SERVER_CONTEXT_CPP}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC1 checkpoint adapter lost totality/safety token: ${REQUIRED}")
    endif()
endforeach()

checkpoint_boundary_complete(
    "${SERVER_CONTEXT_CPP}" "${SERVER_TASK_CPP}"
    "${CACHE_AUTHORITY_CPP}${CACHE_AUTHORITY_H}"
    "${RETENTION_SIDECAR_CPP}"
    "${COMMON_H}" "${CHECKPOINT_CPP}" CHECKPOINT_BOUNDARY_OK)
if (NOT CHECKPOINT_BOUNDARY_OK)
    message(FATAL_ERROR
        "ZC1 checkpoint reference/lane-anchor boundary is incomplete")
endif()

checkpoint_evidence_fallback_complete(
    "${SERVER_CONTEXT_CPP}" CHECKPOINT_FALLBACK_OK)
if (NOT CHECKPOINT_FALLBACK_OK)
    message(FATAL_ERROR
        "ZC1 checkpoint evidence-unavailable historical fallback is incomplete")
endif()

shapes_gate_modes_complete("${SHAPES_PY}" SHAPES_GATE_MODES_OK)
if (NOT SHAPES_GATE_MODES_OK)
    message(FATAL_ERROR
        "ZC1 live driver lost concurrent/serialized gate separation")
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

function(host_adapter_has_safety TEXT OUT)
    set(SAFE TRUE)
    foreach(REQUIRED IN ITEMS
            "!artifact.mandatory_anchor"
            "!server_cache_lease_is_hard(artifact.candidate.lease)"
            "it->recovery_pins == 0"
            "try_destroy_entry_prepared(")
        string(FIND "${TEXT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(SAFE FALSE)
        endif()
    endforeach()
    set(${OUT} ${SAFE} PARENT_SCOPE)
endfunction()

host_adapter_has_safety("${SERVER_TASK_CPP}" ADAPTER_SAFE)
if (NOT ADAPTER_SAFE)
    message(FATAL_ERROR "ZC1 host adapter safety conjunction is incomplete")
endif()

string(REGEX MATCHALL "server_cache_plan_host_retention_victim\\(" HOST_POLICY_CALLS "${SERVER_TASK_CPP}")
list(LENGTH HOST_POLICY_CALLS HOST_POLICY_CALL_COUNT)
if (NOT HOST_POLICY_CALL_COUNT EQUAL 1)
    message(FATAL_ERROR
        "ZC1 host policy must have exactly one production call; got ${HOST_POLICY_CALL_COUNT}")
endif()

foreach(REQUIRED IN ITEMS
        "retention_policy =="
        "common_cache_optimizer_retention_policy::intentional_baseline"
        "!artifact.mandatory_anchor"
        "!server_cache_lease_is_hard(artifact.candidate.lease)"
        "it->recovery_pins == 0"
        "try_destroy_entry_prepared("
        "retention_identity_failure ="
        "retention_epoch_exhausted"
        "retention_id_exhausted")
    string(FIND "${SERVER_TASK_CPP}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC1 host adapter lost safety/exact-release token: ${REQUIRED}")
    endif()
endforeach()

foreach(REQUIRED IN ITEMS
        "prompt_cache->retention_policy ="
        "params_base.cache_optimizer.retention_policy"
        "prompt_cache->retention_event_begin()"
        "prompt_cache->retention_policy !="
        "common_cache_optimizer_retention_policy::"
        "historical_legacy")
    string(FIND "${SERVER_CONTEXT_CPP}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC1 effective-config/event wiring lost token: ${REQUIRED}")
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

set(MUTATED_REFERENCE "${SERVER_CONTEXT_CPP}")
string(REPLACE "current_reference" "historical_affinity"
    MUTATED_REFERENCE "${MUTATED_REFERENCE}")
checkpoint_boundary_complete(
    "${MUTATED_REFERENCE}" "${SERVER_TASK_CPP}"
    "${CACHE_AUTHORITY_CPP}${CACHE_AUTHORITY_H}"
    "${RETENTION_SIDECAR_CPP}"
    "${COMMON_H}" "${CHECKPOINT_CPP}" MUTATED_REFERENCE_OK)
if (MUTATED_REFERENCE_OK)
    message(FATAL_ERROR "ZC1 current-reference negative control did not trip")
endif()

set(MUTATED_ANCHOR "${RETENTION_SIDECAR_CPP}")
string(REPLACE
    "server_retention_anchor_policy::checkpoint_desired_set"
    "server_retention_anchor_policy::scored"
    MUTATED_ANCHOR "${MUTATED_ANCHOR}")
checkpoint_boundary_complete(
    "${SERVER_CONTEXT_CPP}" "${SERVER_TASK_CPP}"
    "${CACHE_AUTHORITY_CPP}${CACHE_AUTHORITY_H}"
    "${MUTATED_ANCHOR}"
    "${COMMON_H}" "${CHECKPOINT_CPP}" MUTATED_ANCHOR_OK)
if (MUTATED_ANCHOR_OK)
    message(FATAL_ERROR "ZC1 anchor-boundary negative control did not trip")
endif()

set(MUTATED_FALLBACK "${SERVER_CONTEXT_CPP}")
string(REPLACE
    "server_cache_retention_status_is_evidence_unavailable("
    "server_cache_retention_status_unchecked("
    MUTATED_FALLBACK "${MUTATED_FALLBACK}")
checkpoint_evidence_fallback_complete(
    "${MUTATED_FALLBACK}" MUTATED_FALLBACK_OK)
if (MUTATED_FALLBACK_OK)
    message(FATAL_ERROR "ZC1 historical-fallback negative control did not trip")
endif()

set(MUTATED_FALLBACK_IDEMPOTENCE "${SERVER_CONTEXT_CPP}")
string(REPLACE "if (!zc_retention)" "if (false)"
    MUTATED_FALLBACK_IDEMPOTENCE "${MUTATED_FALLBACK_IDEMPOTENCE}")
checkpoint_evidence_fallback_complete(
    "${MUTATED_FALLBACK_IDEMPOTENCE}" MUTATED_FALLBACK_IDEMPOTENCE_OK)
if (MUTATED_FALLBACK_IDEMPOTENCE_OK)
    message(FATAL_ERROR "ZC1 historical-fallback idempotence control did not trip")
endif()

set(MUTATED_SHAPES "${SHAPES_PY}")
string(REPLACE "max_workers=args.workers" "max_workers=3"
    MUTATED_SHAPES "${MUTATED_SHAPES}")
shapes_gate_modes_complete("${MUTATED_SHAPES}" MUTATED_SHAPES_OK)
if (MUTATED_SHAPES_OK)
    message(FATAL_ERROR "ZC1 serialized-driver negative control did not trip")
endif()


# Adapter negative controls: removing any hard/pin/mandatory conjunct or the
# prepared-release requirement must make the contract fail.
set(MUTATED_ADAPTER "${SERVER_TASK_CPP}")
string(REPLACE "it->recovery_pins == 0" "true" MUTATED_ADAPTER "${MUTATED_ADAPTER}")
host_adapter_has_safety("${MUTATED_ADAPTER}" MUTATED_SAFE)
if (MUTATED_SAFE)
    message(FATAL_ERROR "ZC1 recovery-pin negative control did not trip")
endif()

set(MUTATED_PREPARED_DOOR "${SERVER_TASK_CPP}")
string(REPLACE "try_destroy_entry_prepared("
    "destroy_entry(" MUTATED_PREPARED_DOOR "${MUTATED_PREPARED_DOOR}")
host_adapter_has_safety("${MUTATED_PREPARED_DOOR}" MUTATED_PREPARED_SAFE)
if (MUTATED_PREPARED_SAFE)
    message(FATAL_ERROR "ZC1 prepared-release-door negative control did not trip")
endif()

set(MUTATED_OFF_GATE "${SERVER_CONTEXT_CPP}")
string(REPLACE "prompt_cache->retention_policy !="
    "prompt_cache->retention_policy =="
    MUTATED_OFF_GATE "${MUTATED_OFF_GATE}")
string(FIND "${MUTATED_OFF_GATE}" "prompt_cache->retention_policy !="
    MUTATED_OFF_GATE_POS)
if (NOT MUTATED_OFF_GATE_POS EQUAL -1)
    message(FATAL_ERROR "ZC1 optimizer-off retention-event gate control did not trip")
endif()

message(STATUS "ZC1 retention-policy and host-adapter contract checks passed")
