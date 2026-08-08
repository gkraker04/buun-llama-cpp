if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.h" OBSERVER_H)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp" OBSERVER_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" TASK_CPP)
file(READ "${SOURCE_ROOT}/src/llama-context.cpp" LLAMA_CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/src/llama-ext.h" LLAMA_EXT_H)
file(READ "${SOURCE_ROOT}/include/llama.h" PUBLIC_LLAMA_H)

function(observer_pure TEXT OUT)
    set(PURE TRUE)
    foreach(FORBIDDEN IN ITEMS
            "cache_plan_authority"
            "common_cache_plan_calib_find"
            "authorize("
            "std::filesystem"
            "llama_synchronize(")
        string(FIND "${TEXT}" "${FORBIDDEN}" POS)
        if (NOT POS EQUAL -1)
            set(PURE FALSE)
        endif()
    endforeach()
    set(${OUT} ${PURE} PARENT_SCOPE)
endfunction()

function(server_observer_wiring_complete CONTEXT LLAMA OUT)
    set(COMPLETE TRUE)
    foreach(REQUIRED IN ITEMS
            "params_base.cache_optimizer.observer_store_enabled"
            "llama_set_sync_fence_observer(ctx_tgt, true)"
            "cache_observation_note_submission(batch_view)"
            "cache_observation_close_fence()"
            "slot.cache_observation_epoch.latch_fence("
            "llama_arm_sync_fence_observer(ctx_tgt)"
            "llama_get_sync_fence_info(ctx)")
        string(FIND "${CONTEXT}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    foreach(REQUIRED IN ITEMS
            "ggml_backend_sched_synchronize(sched.get())"
            "sync_fence_observer_enabled_ && sync_fence_observer_armed_"
            "n_queued_tokens > 0"
            "sync_fence_info_.completed_us = ggml_time_us()"
            "++sync_fence_info_.serial"
            "sync_fence_observer_armed_ = false")
        string(FIND "${LLAMA}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    set(${OUT} ${COMPLETE} PARENT_SCOPE)
endfunction()

function(observer_multi_submission_complete HEADER IMPL OUT)
    set(COMPLETE TRUE)
    foreach(REQUIRED IN ITEMS
            "uint64_t backend_service_us_ = 0"
            "uint32_t fenced_submissions_ = 0"
            "bool submission_pending_ = false")
        string(FIND "${HEADER}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    foreach(REQUIRED IN ITEMS
            "tokens_ += value.tokens"
            "server_cache_observation_replay_chain_geometry("
            "server_cache_observation_same_chain_geometry(first, value)"
            "backend_service_us_ += uint64_t("
            "!terminal_ready_"
            "fenced_submissions_ != submissions_")
        string(FIND "${IMPL}" "${REQUIRED}" POS)
        if (POS EQUAL -1)
            set(COMPLETE FALSE)
        endif()
    endforeach()
    set(${OUT} ${COMPLETE} PARENT_SCOPE)
endfunction()

observer_pure("${OBSERVER_H}${OBSERVER_CPP}" PURE)
if (NOT PURE)
    message(FATAL_ERROR "ZC2 observer acquired authority, persistence, fitting, or synchronization")
endif()

observer_multi_submission_complete(
    "${OBSERVER_H}" "${OBSERVER_CPP}" MULTI_COMPLETE)
if (NOT MULTI_COMPLETE)
    message(FATAL_ERROR
        "ZC2 multi-submission attribution lost per-fence aggregation")
endif()

server_observer_wiring_complete("${CONTEXT_CPP}" "${LLAMA_CONTEXT_CPP}" COMPLETE)
if (NOT COMPLETE)
    message(FATAL_ERROR "ZC2 passive submission/fence wiring is incomplete")
endif()

contract_extract_region("${CONTEXT_CPP}"
    "void cache_observation_finish(server_slot & slot)"
    "void cache_authority_config_failed(bool mirror_to_shadow)"
    TERMINAL_REGION TERMINAL_FOUND)
if (NOT TERMINAL_FOUND)
    message(FATAL_ERROR "ZC2 operation-terminal fence region is missing")
endif()
foreach(REQUIRED IN ITEMS
        "slot.cache_observation_epoch.latch_fence("
        "cache_observation_fence(ctx_tgt)"
        "slot.cache_observation_epoch.mark_operation_terminal()")
    string(FIND "${TERMINAL_REGION}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC2 release-before-outer-close protection is incomplete: ${REQUIRED}")
    endif()
endforeach()

contract_extract_region("${CONTEXT_CPP}"
    "if (ok) {"
    "// try again with the updated n_batch"
    DECODE_RETRY_OBSERVER_REGION DECODE_RETRY_OBSERVER_FOUND)
if (NOT DECODE_RETRY_OBSERVER_FOUND)
    message(FATAL_ERROR "ZC2 decode-retry observer terminal is missing")
endif()
foreach(REQUIRED IN ITEMS
        "cache_observation_abandon("
        "cache_optimizer_observations->slot_batch_tokens("
        "server_cache_observation_reason::operation_failed")
    string(FIND "${DECODE_RETRY_OBSERVER_REGION}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC2 failed decode retry can contaminate a success sample: ${REQUIRED}")
    endif()
endforeach()
string(FIND "${TERMINAL_REGION}" "llama_synchronize(" TERMINAL_SYNC)
if (NOT TERMINAL_SYNC EQUAL -1)
    message(FATAL_ERROR "ZC2 operation terminal added a synchronization call")
endif()

foreach(INTERNAL_NAME IN ITEMS
        "llama_sync_fence_info"
        "llama_get_sync_fence_info"
        "llama_set_sync_fence_observer")
    string(FIND "${PUBLIC_LLAMA_H}" "${INTERNAL_NAME}" PUBLIC_POS)
    if (NOT PUBLIC_POS EQUAL -1)
        message(FATAL_ERROR
            "ZC2 internal fence telemetry leaked into installed llama.h: ${INTERNAL_NAME}")
    endif()
    string(FIND "${LLAMA_EXT_H}" "${INTERNAL_NAME}" INTERNAL_POS)
    if (INTERNAL_POS EQUAL -1)
        message(FATAL_ERROR
            "ZC2 internal fence telemetry missing from llama-ext.h: ${INTERNAL_NAME}")
    endif()
endforeach()

count_literal("${OBSERVER_CPP}${CONTEXT_CPP}${TASK_CPP}"
    "CACHE_OPTIMIZER_OBSERVATION" OBSERVATION_LOG_OWNERS)
if (NOT OBSERVATION_LOG_OWNERS EQUAL 1)
    message(FATAL_ERROR
        "ZC2 observation JSON must have one noexcept serializer owner; got ${OBSERVATION_LOG_OWNERS}")
endif()

contract_extract_region("${CONTEXT_CPP}"
    "void cache_observation_note_submission("
    "void cache_observation_close_fence()"
    SUBMISSION_REGION SUBMISSION_FOUND)
if (NOT SUBMISSION_FOUND)
    message(FATAL_ERROR "ZC2 submission attribution region is missing")
endif()
count_literal("${SUBMISSION_REGION}"
    "for (int32_t i = 0; i < batch_view.n_tokens; ++i) {"
    TOKEN_SCAN_COUNT)
if (NOT TOKEN_SCAN_COUNT EQUAL 1)
    message(FATAL_ERROR
        "ZC2 batch attribution must scan prompt tokens exactly once; got ${TOKEN_SCAN_COUNT}")
endif()
string(FIND "${SUBMISSION_REGION}" "llama_synchronize(" EXTRA_SYNC)
if (NOT EXTRA_SYNC EQUAL -1)
    message(FATAL_ERROR "ZC2 observation added a synchronization call")
endif()

foreach(REQUIRED IN ITEMS
        "template <bool Observed, bool Measure>"
        "if constexpr (Measure)"
        "load_impl<false, false>"
        "load_impl<false, true>"
        "load_impl<true, false>"
        "load_impl<true, true>")
    string(FIND "${TASK_CPP}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR
            "ZC2 restore attribution lost its compile-time off-path split: ${REQUIRED}")
    endif()
endforeach()

string(REGEX MATCHALL
    "server_prompt_cache_observe_cpu_operation\\("
    CPU_OBSERVATION_CALLS "${TASK_CPP}")
list(LENGTH CPU_OBSERVATION_CALLS CPU_OBSERVATION_COUNT)
if (NOT CPU_OBSERVATION_COUNT EQUAL 8)
    message(FATAL_ERROR
        "ZC2 prompt-cache preparation/apply census changed: ${CPU_OBSERVATION_COUNT}")
endif()
string(REGEX MATCHALL "observe_checkpoint_cpu_operation\\("
    CHECKPOINT_OBSERVATION_CALLS "${TASK_CPP}")
list(LENGTH CHECKPOINT_OBSERVATION_CALLS CHECKPOINT_OBSERVATION_COUNT)
if (NOT CHECKPOINT_OBSERVATION_COUNT EQUAL 4)
    message(FATAL_ERROR
        "ZC2 checkpoint preparation/apply census changed: ${CHECKPOINT_OBSERVATION_COUNT}")
endif()
string(REGEX MATCHALL "observe_cache_cpu_operation\\("
    SLOT_OBSERVATION_CALLS "${CONTEXT_CPP}")
list(LENGTH SLOT_OBSERVATION_CALLS SLOT_OBSERVATION_COUNT)
if (NOT SLOT_OBSERVATION_COUNT EQUAL 4)
    message(FATAL_ERROR
        "ZC2 live-slot preparation/apply census changed: ${SLOT_OBSERVATION_COUNT}")
endif()
foreach(REQUIRED IN ITEMS
        "server_cache_observation_operation::durability_prepare"
        "server_cache_observation_operation::destruction_apply"
        "server_cache_observation_cpu_start"
        "server_cache_observation_capture_cpu_start(")
    string(FIND "${TASK_CPP}${CONTEXT_CPP}" "${REQUIRED}" POS)
    if (POS EQUAL -1)
        message(FATAL_ERROR "ZC2 CPU-only seam missing: ${REQUIRED}")
    endif()
endforeach()

# House-standard negative controls: mutate each protected property and require
# the same predicate that guards production to reject it.
observer_pure("${OBSERVER_CPP}\ncache_plan_authority->authorize();" MUTATED_PURE)
if (MUTATED_PURE)
    message(FATAL_ERROR "ZC2 observer-purity negative control did not trip")
endif()

set(MUTATED_CONTEXT "${CONTEXT_CPP}")
string(REPLACE
    "const auto fence_before = cache_observation_fence(ctx_tgt);"
    "llama_synchronize(ctx_tgt);\n        const auto fence_before = cache_observation_fence(ctx_tgt);"
    MUTATED_CONTEXT "${MUTATED_CONTEXT}")
contract_extract_region("${MUTATED_CONTEXT}"
    "void cache_observation_note_submission("
    "void cache_observation_close_fence()"
    MUTATED_SUBMISSION_REGION MUTATED_SUBMISSION_FOUND)
string(FIND "${MUTATED_SUBMISSION_REGION}" "llama_synchronize(" MUTATED_SYNC)
if (MUTATED_SYNC EQUAL -1)
    message(FATAL_ERROR "ZC2 added-sync negative control did not trip")
endif()

set(MUTATED_CONTEXT "${CONTEXT_CPP}")
string(REPLACE
    "for (int32_t i = 0; i < batch_view.n_tokens; ++i) {"
    "for (int32_t i = 0; i < batch_view.n_tokens; ++i) {}\n        for (int32_t i = 0; i < batch_view.n_tokens; ++i) {"
    MUTATED_CONTEXT "${MUTATED_CONTEXT}")
contract_extract_region("${MUTATED_CONTEXT}"
    "void cache_observation_note_submission("
    "void cache_observation_close_fence()"
    MUTATED_SUBMISSION_REGION MUTATED_SUBMISSION_FOUND)
count_literal("${MUTATED_SUBMISSION_REGION}"
    "for (int32_t i = 0; i < batch_view.n_tokens; ++i) {"
    MUTATED_TOKEN_SCAN_COUNT)
if (MUTATED_TOKEN_SCAN_COUNT EQUAL 1)
    message(FATAL_ERROR "ZC2 long-prompt rescan negative control did not trip")
endif()

set(MUTATED_LLAMA "${LLAMA_CONTEXT_CPP}")
string(REPLACE
    "sync_fence_observer_armed_ = false;"
    "/* missing one-shot disarm */"
    MUTATED_LLAMA "${MUTATED_LLAMA}")
server_observer_wiring_complete("${CONTEXT_CPP}" "${MUTATED_LLAMA}" MUTATED_COMPLETE)
if (MUTATED_COMPLETE)
    message(FATAL_ERROR "ZC2 fence-update negative control did not trip")
endif()

set(MUTATED_OBSERVER "${OBSERVER_CPP}")
string(REPLACE
    "backend_service_us_ += uint64_t("
    "/* deleted separately-fenced backend accumulation */ uint64_t("
    MUTATED_OBSERVER "${MUTATED_OBSERVER}")
observer_multi_submission_complete(
    "${OBSERVER_H}" "${MUTATED_OBSERVER}" MUTATED_MULTI_COMPLETE)
if (MUTATED_MULTI_COMPLETE)
    message(FATAL_ERROR
        "ZC2 multi-submission aggregation negative control did not trip")
endif()

set(MUTATED_CHAIN_GEOMETRY "${OBSERVER_CPP}")
string(REPLACE
    "server_cache_observation_same_chain_geometry(first, value)"
    "true /* deleted frozen chain geometry */"
    MUTATED_CHAIN_GEOMETRY "${MUTATED_CHAIN_GEOMETRY}")
observer_multi_submission_complete(
    "${OBSERVER_H}" "${MUTATED_CHAIN_GEOMETRY}"
    MUTATED_CHAIN_GEOMETRY_COMPLETE)
if (MUTATED_CHAIN_GEOMETRY_COMPLETE)
    message(FATAL_ERROR
        "ZC2 chain-geometry negative control did not trip")
endif()

set(MUTATED_TERMINAL_OWNER "${OBSERVER_CPP}")
string(REPLACE "!active_ || !terminal_ready_ || submissions_ == 0"
    "!active_ || submissions_ == 0"
    MUTATED_TERMINAL_OWNER "${MUTATED_TERMINAL_OWNER}")
observer_multi_submission_complete(
    "${OBSERVER_H}" "${MUTATED_TERMINAL_OWNER}"
    MUTATED_TERMINAL_OWNER_COMPLETE)
if (MUTATED_TERMINAL_OWNER_COMPLETE)
    message(FATAL_ERROR
        "ZC2 epoch-owner terminal negative control did not trip")
endif()

set(MUTATED_DECODE_RETRY "${DECODE_RETRY_OBSERVER_REGION}")
string(REPLACE "server_cache_observation_reason::operation_failed"
    "server_cache_observation_reason::none"
    MUTATED_DECODE_RETRY "${MUTATED_DECODE_RETRY}")
string(FIND "${MUTATED_DECODE_RETRY}"
    "server_cache_observation_reason::operation_failed"
    MUTATED_DECODE_RETRY_TERMINAL)
if (NOT MUTATED_DECODE_RETRY_TERMINAL EQUAL -1)
    message(FATAL_ERROR
        "ZC2 decode-retry terminal negative control did not trip")
endif()

set(MUTATED_CONTEXT "${CONTEXT_CPP}")
string(REPLACE
    "slot.cache_observation_epoch.latch_fence("
    "slot.cache_observation_epoch.deleted_terminal_latch("
    MUTATED_CONTEXT "${MUTATED_CONTEXT}")
contract_extract_region("${MUTATED_CONTEXT}"
    "void cache_observation_finish(server_slot & slot)"
    "void cache_authority_config_failed(bool mirror_to_shadow)"
    MUTATED_TERMINAL_REGION MUTATED_TERMINAL_FOUND)
string(FIND "${MUTATED_TERMINAL_REGION}"
    "slot.cache_observation_epoch.latch_fence(" MUTATED_TERMINAL_LATCH)
if (NOT MUTATED_TERMINAL_LATCH EQUAL -1)
    message(FATAL_ERROR "ZC2 terminal-latch deletion negative control did not trip")
endif()

set(MUTATED_TASK "${TASK_CPP}")
string(REPLACE
    "server_prompt_cache_observe_cpu_operation("
    "server_prompt_cache_observation_deleted("
    MUTATED_TASK "${MUTATED_TASK}")
string(REGEX MATCHALL "server_prompt_cache_observe_cpu_operation\\("
    MUTATED_CPU_CALLS "${MUTATED_TASK}")
list(LENGTH MUTATED_CPU_CALLS MUTATED_CPU_COUNT)
if (MUTATED_CPU_COUNT EQUAL 8)
    message(FATAL_ERROR "ZC2 D-A2 census deletion negative control did not trip")
endif()

message(STATUS "ZC2 observer contract checks passed")
