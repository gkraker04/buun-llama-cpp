if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-model.h" MODEL_H)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-model.cpp" MODEL_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp" OBSERVER_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.h" OBSERVER_H)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lifecycle.h" LIFECYCLE_H)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-store.h" STORE_H)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-store.cpp" STORE_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" TASK_CPP)

foreach(REQUIRED IN ITEMS
        "server_cache_calibration_predict("
        "server_cache_calibration_state("
        "server_cache_calibration_update("
        "server_cache_calibration_bound_direct_difference("
        "VALIDATION_LAMBDAS = {"
        "1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0"
        "CONFIDENCE_ERROR_SYSTEM = 1e-3"
        "DRIFT_FALSE_ALARM_SYSTEM = 1e-3"
        "CONDITION_LIMIT = 1e8"
        "server_cache_calibration_validation_assignment("
        "qualified_execution_ordinal % 8 == mixed % 8"
        "record.capped_service_us"
        "groups[group_index].delta[d] += scalar * term.feature[d]"
        "server_cache_calibration_authority_terminal::tail_exceeded"
        "server_cache_calibration_authority_terminal::drifted"
        "server_cache_calibration_arena_layout::total_size"
        "std::nothrow")
    contract_require_token("${MODEL_H}${MODEL_CPP}" "${REQUIRED}"
        "ZC4 bounded estimator procedure")
endforeach()
count_literal("${MODEL_H}${MODEL_CPP}${OBSERVER_CPP}"
    "server_cache_calibration_validation_assignment("
    VALIDATION_SCHEDULE_USE_COUNT)
if (NOT VALIDATION_SCHEDULE_USE_COUNT EQUAL 4)
    message(FATAL_ERROR
        "ZC4 validation schedule owner/caller census changed: ${VALIDATION_SCHEDULE_USE_COUNT}")
endif()

contract_extract_region("${OBSERVER_CPP}"
    "bool server_cache_observation_store::observe("
    "server_cache_observation_key server_cache_observation_cpu_key("
    OBSERVE_ADMISSION_REGION OBSERVE_ADMISSION_FOUND)
if (NOT OBSERVE_ADMISSION_FOUND)
    message(FATAL_ERROR "ZC4 observer admission-clock region is missing")
endif()
foreach(REQUIRED IN ITEMS
        "record.admission_clock.steady_us"
        "record.admission_clock.unix_ms")
    contract_require_token("${OBSERVE_ADMISSION_REGION}" "${REQUIRED}"
        "ZC4 pre-outcome admission clock")
endforeach()
contract_find_forbidden("${OBSERVE_ADMISSION_REGION}"
    OBSERVE_COMPLETION_CLOCKS "steady_clock::now" "system_clock::now")
if (OBSERVE_COMPLETION_CLOCKS)
    message(FATAL_ERROR
        "ZC4 completion-side clock controls estimator admission: ${OBSERVE_COMPLETION_CLOCKS}")
endif()
contract_extract_region("${OBSERVER_CPP}"
    "void server_cache_calibration_epoch::arm("
    "void server_cache_calibration_epoch::bind_provider("
    EPOCH_ARM_REGION EPOCH_ARM_FOUND)
if (NOT EPOCH_ARM_FOUND)
    message(FATAL_ERROR "ZC4 epoch admission-clock arm is missing")
endif()
contract_require_token("${EPOCH_ARM_REGION}" "admission_clock_ = admission_clock;"
    "ZC4 missing admission clock stays fail-closed")
contract_find_forbidden("${EPOCH_ARM_REGION}" EPOCH_ARM_CLOCK_RETRY
    "server_cache_observation_capture_admission_clock(")
if (EPOCH_ARM_CLOCK_RETRY)
    message(FATAL_ERROR
        "ZC4 epoch arm retries a missing pre-outcome clock after provider work")
endif()
function(zc4_validate_preoutcome_clock TASK CONTEXT OBSERVER OUT)
    foreach(REQUIRED IN ITEMS
            "observation->admission_clock ="
            "load_observation.admission_clock"
            "checkpoint_observation_admission_clock"
            "admission_clock_ = admission_clock;"
            "record.admission_clock = admission_clock_")
        string(FIND "${TASK}${CONTEXT}${OBSERVER}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    count_literal("${TASK}${CONTEXT}"
        "server_cache_observation_capture_cpu_start(" CPU_START_COUNT)
    if (NOT CPU_START_COUNT EQUAL 9)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    contract_extract_region("${CONTEXT}"
        "void cache_observation_begin_provider_cpu(server_slot & slot)"
        "void cache_observation_abandon("
        REPLAY_CLOCK_REGION REPLAY_CLOCK_FOUND)
    if (NOT REPLAY_CLOCK_FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(REQUIRED IN ITEMS
            "slot.cache_observation_epoch.arm("
            "server_cache_observation_capture_admission_clock())")
        string(FIND "${REPLAY_CLOCK_REGION}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    contract_extract_region("${CONTEXT}"
        "// Capture the checkpoint provider's admission currency before"
        "// [WS-6] A live_rebased PARTIAL_ONLY restore"
        CHECKPOINT_CLOCK_REGION CHECKPOINT_CLOCK_FOUND)
    if (NOT CHECKPOINT_CLOCK_FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    string(FIND "${CHECKPOINT_CLOCK_REGION}"
        "if (cache_optimizer_observations) {" CHECKPOINT_CLOCK_GATE)
    string(FIND "${CHECKPOINT_CLOCK_REGION}"
        "server_cache_observation_capture_admission_clock()"
        CHECKPOINT_CLOCK_CAPTURE)
    if (CHECKPOINT_CLOCK_GATE EQUAL -1 OR CHECKPOINT_CLOCK_CAPTURE EQUAL -1 OR
        CHECKPOINT_CLOCK_GATE GREATER CHECKPOINT_CLOCK_CAPTURE)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_preoutcome_clock(
    "${TASK_CPP}" "${CONTEXT_CPP}" "${OBSERVER_CPP}"
    PREOUTCOME_CLOCK_VALID)
if (NOT PREOUTCOME_CLOCK_VALID)
    message(FATAL_ERROR "ZC4 provider admission-clock coverage changed")
endif()
function(zc4_validate_validation_schedule TEXT OUT)
    foreach(REQUIRED IN ITEMS
            "UINT64_C(0x9e3779b97f4a7c15)"
            "mixed >> 30"
            "UINT64_C(0xbf58476d1ce4e5b9)"
            "mixed >> 27"
            "UINT64_C(0x94d049bb133111eb)"
            "mixed >> 31"
            "qualified_execution_ordinal % 8 == mixed % 8")
        string(FIND "${TEXT}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_validation_schedule("${MODEL_CPP}" VALIDATION_SCHEDULE_VALID)
if (NOT VALIDATION_SCHEDULE_VALID)
    message(FATAL_ERROR "ZC4 frozen validation schedule changed")
endif()
function(zc4_no_obsolete_opportunity_cutoff TEXT OUT)
    string(FIND "${TEXT}"
        "replay_tokens > uint64_t(params_base.n_batch)" FOUND)
    if (FOUND EQUAL -1)
        set(${OUT} TRUE PARENT_SCOPE)
    else()
        set(${OUT} FALSE PARENT_SCOPE)
    endif()
endfunction()
zc4_no_obsolete_opportunity_cutoff("${CONTEXT_CPP}" NO_OLD_CUTOFF)
if (NOT NO_OLD_CUTOFF)
    message(FATAL_ERROR "ZC4 obsolete single-submission opportunity cutoff")
endif()

foreach(REQUIRED IN ITEMS
        "slot_scratch_capacity = 4096"
        "prepare_slot_scratch(size_t count)"
        "reset_slot_scratch()"
        "note_slot_submission("
        "slot_batch_tokens_"
        "slot_first_positions_")
    contract_require_token("${OBSERVER_H}${CONTEXT_CPP}" "${REQUIRED}"
        "ZC4 arena-owned per-slot attribution scratch")
endforeach()
function(zc4_slot_scratch_bounded TEXT OUT)
    contract_find_forbidden("${TEXT}" FOUND
        "std::vector<uint32_t> cache_observation_batch_tokens"
        "std::vector<llama_pos> cache_observation_first_pos")
    if (FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
    else()
        set(${OUT} TRUE PARENT_SCOPE)
    endif()
endfunction()
zc4_slot_scratch_bounded("${CONTEXT_CPP}" SLOT_SCRATCH_BOUNDED)
if (NOT SLOT_SCRATCH_BOUNDED)
    message(FATAL_ERROR "ZC4 per-slot observer scratch escaped the fixed arena")
endif()

function(zc4_validate_destruction_typing TEXT OUT)
    foreach(REQUIRED IN ITEMS
            "server_cache_destruction_census_valid()"
            "server_cache_destruction_census[size_t(destruction_class)]"
            "server_cache_destruction_class destruction_class"
            "server_cache_destruction_release_owner release_owner")
        string(FIND "${TEXT}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_destruction_typing("${LIFECYCLE_H}${MODEL_H}${MODEL_CPP}"
    DESTRUCTION_TYPING_VALID)
if (NOT DESTRUCTION_TYPING_VALID)
    message(FATAL_ERROR "ZC4 destruction-action census typing is incomplete")
endif()

function(zc4_validate_opportunity_hook TEXT OUT)
    contract_extract_region("${TEXT}"
        "void cache_plan_inventory_and_plan_before_mutation("
        "server_slot * get_available_slot("
        OPPORTUNITY_REGION OPPORTUNITY_REGION_FOUND)
    if (NOT OPPORTUNITY_REGION_FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(REQUIRED IN ITEMS
            "if (!mode.preflight && cache_optimizer_observations)"
            "out.observations.emplace()"
            "out.observations ? &*out.observations : nullptr"
            "note_safe_measurable_opportunity("
            "!rec.inventory_saturated()")
        string(FIND "${OPPORTUNITY_REGION}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    count_literal("${OPPORTUNITY_REGION}"
        "note_safe_measurable_opportunity(" OPPORTUNITY_USE_COUNT)
    if (NOT OPPORTUNITY_USE_COUNT EQUAL 1)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_opportunity_hook("${CONTEXT_CPP}" OPPORTUNITY_HOOK_VALID)
if (NOT OPPORTUNITY_HOOK_VALID)
    message(FATAL_ERROR "ZC4 complete-inventory opportunity hook is incomplete")
endif()
foreach(REQUIRED IN ITEMS
        "server_cache_observation_replay_chain_geometry("
        "std::min<uint64_t>("
        "replay_tokens, uint64_t(params_base.n_batch)")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC4 measurable multi-submission opportunity geometry")
endforeach()

function(zc4_validate_profile_use_hook CONTEXT STORE OUT)
    contract_extract_region("${CONTEXT}"
        "server_slot * get_available_slot("
        "server_cache_control_status cache_control_resolve"
        PROFILE_USE_REGION PROFILE_USE_REGION_FOUND)
    if (NOT PROFILE_USE_REGION_FOUND)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(REQUIRED IN ITEMS
            "task.type == SERVER_TASK_TYPE_COMPLETION"
            "task.id != cache_calibration_last_profile_use_task"
            "cache_calibration->note_profile_use()")
        string(FIND "${PROFILE_USE_REGION}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    count_literal("${CONTEXT}" "cache_calibration->note_profile_use()"
        PROFILE_USE_COUNT)
    string(FIND "${STORE}"
        "void server_cache_calibration_coordinator::note_profile_use() noexcept"
        PROFILE_USE_OWNER)
    if (NOT PROFILE_USE_COUNT EQUAL 1 OR PROFILE_USE_OWNER EQUAL -1)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_profile_use_hook("${CONTEXT_CPP}" "${STORE_CPP}"
    PROFILE_USE_HOOK_VALID)
if (NOT PROFILE_USE_HOOK_VALID)
    message(FATAL_ERROR "ZC4 request-only profile-use hook is incomplete")
endif()

function(zc4_validate_source_and_resume_attribution CONTEXT TASK OBSERVER STORE OUT)
    foreach(REQUIRED IN ITEMS
            "it_best->adapter_application_digest"
            "it_best->adapter_application_complete")
        string(FIND "${TASK}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    foreach(REQUIRED IN ITEMS
            "checkpoint_observation_adapter_digest ="
            "it->adapter_application_digest"
            "it->adapter_application_complete")
        string(FIND "${CONTEXT}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(${OUT} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    string(FIND "${OBSERVER}"
        "resume_validation_pending_[instance_slot]" OBSERVER_SLOT)
    string(FIND "${STORE}"
        "resume_outcomes[i].estimator_slot" STORE_SLOT)
    if (OBSERVER_SLOT EQUAL -1 OR STORE_SLOT EQUAL -1)
        set(${OUT} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${OUT} TRUE PARENT_SCOPE)
endfunction()
zc4_validate_source_and_resume_attribution(
    "${CONTEXT_CPP}" "${TASK_CPP}" "${OBSERVER_CPP}" "${STORE_CPP}"
    SOURCE_RESUME_ATTRIBUTION_VALID)
if (NOT SOURCE_RESUME_ATTRIBUTION_VALID)
    message(FATAL_ERROR "ZC4 source/resume attribution contract is incomplete")
endif()

foreach(REQUIRED IN ITEMS
        "server_cache_calibration_preassign("
        "server_cache_calibration_complete("
        "server_cache_calibration_abandon("
        "server_cache_calibration_state("
        "context.force_validation = resume_validation_pending_[instance_slot]"
        "principal_cell->fit_rows < 4")
    contract_require_token("${OBSERVER_CPP}" "${REQUIRED}"
        "ZC4 observer-to-estimator join")
endforeach()

foreach(REQUIRED IN ITEMS
        "construct<server_cache_observation_store>"
        "construct<server_cache_fingerprint_worker>"
        "construct<server_cache_calibration_coordinator>"
        "server_cache_calibration_snapshot_workspace>"
        "server_cache_calibration_arena_layout::global_tables_begin"
        "server_cache_calibration_arena_layout::fingerprint_begin"
        "server_cache_calibration_arena_layout::profile_slots_begin"
        "server_cache_calibration_arena_layout::snapshots_begin"
        "server_cache_calibration_arena_layout::codec_scratch_begin")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC4 fixed-arena ownership")
endforeach()
foreach(REQUIRED IN ITEMS
        "sizeof(server_cache_calibration_snapshot_workspace) =="
        "server_cache_calibration_store store(codec_scratch_, codec_scratch_size_)"
        "encode_profile_streamed(")
    contract_require_token("${STORE_H}${STORE_CPP}" "${REQUIRED}"
        "ZC4 bounded snapshot/codec workspace")
endforeach()

contract_find_forbidden("${MODEL_CPP}${MODEL_H}" MODEL_FORBIDDEN
    "cache_plan_authority"
    "authorize("
    "finalize_execution"
    "SERVER_TASK_TYPE"
    "std::filesystem"
    "std::mutex"
    "std::vector"
    "llama_decode("
    "llama_synchronize(")
if (MODEL_FORBIDDEN)
    message(FATAL_ERROR "ZC4 numerical model acquired scheduler/I/O/authority work: ${MODEL_FORBIDDEN}")
endif()

# ZC4 remains shadow-only. The actual mutation owners live in context/task, so
# scan every production TU rather than three nominal authority files. Only the
# numerical owner and observer annotation owner may consume predictions.
function(zc4_has_model_authority_read TEXT OUT)
    set(FOUND_ANY FALSE)
    foreach(TOKEN IN ITEMS
            "server_cache_calibration_predict("
            "server_cache_calibration_state("
            "server_cache_calibration_bound_direct_difference(")
        string(FIND "${TEXT}" "${TOKEN}" FOUND)
        if (NOT FOUND EQUAL -1)
            set(FOUND_ANY TRUE)
        endif()
    endforeach()
    set(${OUT} ${FOUND_ANY} PARENT_SCOPE)
endfunction()

file(GLOB SERVER_CPP "${SOURCE_ROOT}/tools/server/*.cpp")
set(NON_SHADOW_MODEL_READERS "")
foreach(PATH IN LISTS SERVER_CPP)
    if (PATH MATCHES "server-cache-calibration-model\\.cpp$" OR
        PATH MATCHES "server-cache-observer\\.cpp$")
        continue()
    endif()
    file(READ "${PATH}" TEXT)
    zc4_has_model_authority_read("${TEXT}" HAS_MODEL_READ)
    if (HAS_MODEL_READ)
        string(APPEND NON_SHADOW_MODEL_READERS "${PATH};")
    endif()
endforeach()
if (NON_SHADOW_MODEL_READERS)
    message(FATAL_ERROR
        "ZC4 local fit escaped into production authority: ${NON_SHADOW_MODEL_READERS}")
endif()

count_literal("${CONTEXT_CPP}" "cache_calibration_arena->construct<" ARENA_OWNER_COUNT)
if (NOT ARENA_OWNER_COUNT EQUAL 4)
    message(FATAL_ERROR "ZC4 arena owner census changed: ${ARENA_OWNER_COUNT}")
endif()

# House-standard negative controls exercise the same checks as production.
set(MUTATED_MODEL "${MODEL_CPP}\nvoid zc4_bad() { llama_synchronize(nullptr); }")
contract_find_forbidden("${MUTATED_MODEL}${MODEL_H}" MUTATED_FORBIDDEN
    "cache_plan_authority" "authorize(" "finalize_execution" "SERVER_TASK_TYPE"
    "std::filesystem" "std::mutex" "std::vector" "llama_decode("
    "llama_synchronize(")
if (NOT MUTATED_FORBIDDEN)
    message(FATAL_ERROR "ZC4 model-purity negative control did not trip")
endif()

set(MUTATED_SLOT_SCRATCH "${CONTEXT_CPP}\nstd::vector<uint32_t> cache_observation_batch_tokens;")
zc4_slot_scratch_bounded("${MUTATED_SLOT_SCRATCH}" MUTATED_SLOT_BOUNDED)
if (MUTATED_SLOT_BOUNDED)
    message(FATAL_ERROR
        "ZC4 per-slot scratch heap-escape negative control did not trip")
endif()

set(MUTATED_CONTEXT "${CONTEXT_CPP}")
string(REPLACE "construct<server_cache_observation_store>"
    "make_unbounded_observation_store" MUTATED_CONTEXT "${MUTATED_CONTEXT}")
count_literal("${MUTATED_CONTEXT}" "cache_calibration_arena->construct<"
    MUTATED_ARENA_OWNER_COUNT)
if (MUTATED_ARENA_OWNER_COUNT EQUAL 4)
    message(FATAL_ERROR "ZC4 arena-owner negative control did not trip")
endif()

set(MUTATED_OBSERVER "${OBSERVER_CPP}")
string(REPLACE "server_cache_calibration_preassign("
    "zc4_preassign_removed(" MUTATED_OBSERVER "${MUTATED_OBSERVER}")
string(FIND "${MUTATED_OBSERVER}"
    "server_cache_calibration_preassign(" MUTATED_PREASSIGN)
if (NOT MUTATED_PREASSIGN EQUAL -1)
    message(FATAL_ERROR "ZC4 estimator-join negative control did not trip")
endif()

set(MUTATED_VALIDATION_SCHEDULE "${MODEL_H}${MODEL_CPP}${OBSERVER_CPP}")
string(REPLACE "server_cache_calibration_validation_assignment("
    "zc4_fixed_validation_residue("
    MUTATED_VALIDATION_SCHEDULE "${MUTATED_VALIDATION_SCHEDULE}")
count_literal("${MUTATED_VALIDATION_SCHEDULE}"
    "server_cache_calibration_validation_assignment("
    MUTATED_VALIDATION_SCHEDULE_COUNT)
if (MUTATED_VALIDATION_SCHEDULE_COUNT EQUAL 4)
    message(FATAL_ERROR
        "ZC4 validation-schedule negative control did not trip")
endif()

set(MUTATED_ADMISSION_CLOCK "${OBSERVE_ADMISSION_REGION}\nsteady_clock::now();")
contract_find_forbidden("${MUTATED_ADMISSION_CLOCK}"
    MUTATED_COMPLETION_CLOCK "steady_clock::now" "system_clock::now")
if (NOT MUTATED_COMPLETION_CLOCK)
    message(FATAL_ERROR
        "ZC4 admission-clock negative control did not trip")
endif()

set(MUTATED_EPOCH_ARM "${EPOCH_ARM_REGION}\nserver_cache_observation_capture_admission_clock();")
contract_find_forbidden("${MUTATED_EPOCH_ARM}" MUTATED_EPOCH_ARM_RETRY
    "server_cache_observation_capture_admission_clock(")
if (NOT MUTATED_EPOCH_ARM_RETRY)
    message(FATAL_ERROR
        "ZC4 epoch-arm clock-retry negative control did not trip")
endif()

set(MUTATED_CLOCK_TASK "${TASK_CPP}")
string(REPLACE "observation->admission_clock ="
    "observation->completion_clock ="
    MUTATED_CLOCK_TASK "${MUTATED_CLOCK_TASK}")
zc4_validate_preoutcome_clock(
    "${MUTATED_CLOCK_TASK}" "${CONTEXT_CPP}" "${OBSERVER_CPP}"
    MUTATED_PREOUTCOME_CLOCK_VALID)
if (MUTATED_PREOUTCOME_CLOCK_VALID)
    message(FATAL_ERROR
        "ZC4 provider admission-clock negative control did not trip")
endif()

set(MUTATED_REPLAY_CLOCK_CONTEXT "${CONTEXT_CPP}")
string(REPLACE "server_cache_observation_capture_admission_clock())"
    "server_cache_observation_admission_clock{})"
    MUTATED_REPLAY_CLOCK_CONTEXT "${MUTATED_REPLAY_CLOCK_CONTEXT}")
zc4_validate_preoutcome_clock(
    "${TASK_CPP}" "${MUTATED_REPLAY_CLOCK_CONTEXT}" "${OBSERVER_CPP}"
    MUTATED_REPLAY_CLOCK_VALID)
if (MUTATED_REPLAY_CLOCK_VALID)
    message(FATAL_ERROR
        "ZC4 live/cold replay clock negative control did not trip")
endif()

set(MUTATED_CHECKPOINT_CLOCK_CONTEXT "${CONTEXT_CPP}")
string(REPLACE
    "if (cache_optimizer_observations) {\n                                            checkpoint_observation_admission_clock ="
    "if (true) {\n                                            checkpoint_observation_admission_clock ="
    MUTATED_CHECKPOINT_CLOCK_CONTEXT "${MUTATED_CHECKPOINT_CLOCK_CONTEXT}")
zc4_validate_preoutcome_clock(
    "${TASK_CPP}" "${MUTATED_CHECKPOINT_CLOCK_CONTEXT}" "${OBSERVER_CPP}"
    MUTATED_CHECKPOINT_CLOCK_VALID)
if (MUTATED_CHECKPOINT_CLOCK_VALID)
    message(FATAL_ERROR
        "ZC4 checkpoint clock observer-gate negative control did not trip")
endif()

set(MUTATED_VALIDATION_CONSTANT "${MODEL_CPP}")
string(REPLACE "UINT64_C(0x9e3779b97f4a7c15)"
    "UINT64_C(0x9e3779b97f4a7c16)"
    MUTATED_VALIDATION_CONSTANT "${MUTATED_VALIDATION_CONSTANT}")
zc4_validate_validation_schedule(
    "${MUTATED_VALIDATION_CONSTANT}" MUTATED_VALIDATION_CONSTANT_VALID)
if (MUTATED_VALIDATION_CONSTANT_VALID)
    message(FATAL_ERROR
        "ZC4 validation-schedule constant negative control did not trip")
endif()

set(MUTATED_CONTEXT_AUTHORITY
    "${CONTEXT_CPP}\nvoid zc4_bad_authority() { server_cache_calibration_predict(); }")
zc4_has_model_authority_read("${MUTATED_CONTEXT_AUTHORITY}"
    MUTATED_AUTHORITY_READ)
if (NOT MUTATED_AUTHORITY_READ)
    message(FATAL_ERROR "ZC4 shadow-only authority negative control did not trip")
endif()

set(MUTATED_OPPORTUNITY "${CONTEXT_CPP}")
string(REPLACE "note_safe_measurable_opportunity("
    "zc4_opportunity_removed(" MUTATED_OPPORTUNITY "${MUTATED_OPPORTUNITY}")
zc4_validate_opportunity_hook("${MUTATED_OPPORTUNITY}"
    MUTATED_OPPORTUNITY_VALID)
if (MUTATED_OPPORTUNITY_VALID)
    message(FATAL_ERROR "ZC4 opportunity-hook negative control did not trip")
endif()

set(MUTATED_OBSERVATION_DOOR "${CONTEXT_CPP}")
string(REPLACE "if (!mode.preflight && cache_optimizer_observations)"
    "if (!mode.preflight)" MUTATED_OBSERVATION_DOOR
    "${MUTATED_OBSERVATION_DOOR}")
zc4_validate_opportunity_hook("${MUTATED_OBSERVATION_DOOR}"
    MUTATED_OBSERVATION_DOOR_VALID)
if (MUTATED_OBSERVATION_DOOR_VALID)
    message(FATAL_ERROR
        "ZC4 null-observer/preflight gate negative control did not trip")
endif()

set(MUTATED_OLD_OPPORTUNITY
    "${CONTEXT_CPP}\nreplay_tokens > uint64_t(params_base.n_batch)")
zc4_no_obsolete_opportunity_cutoff(
    "${MUTATED_OLD_OPPORTUNITY}" MUTATED_NO_OLD_CUTOFF)
if (MUTATED_NO_OLD_CUTOFF)
    message(FATAL_ERROR
        "ZC4 obsolete opportunity-cutoff negative control did not trip")
endif()

set(MUTATED_PROFILE_USE "${CONTEXT_CPP}")
string(REPLACE "cache_calibration->note_profile_use()"
    "zc4_profile_use_removed()" MUTATED_PROFILE_USE "${MUTATED_PROFILE_USE}")
zc4_validate_profile_use_hook("${MUTATED_PROFILE_USE}" "${STORE_CPP}"
    MUTATED_PROFILE_USE_VALID)
if (MUTATED_PROFILE_USE_VALID)
    message(FATAL_ERROR "ZC4 profile-use negative control did not trip")
endif()

set(MUTATED_SOURCE_TASK "${TASK_CPP}")
string(REPLACE "it_best->adapter_application_digest"
    "incoming_adapter_application_digest" MUTATED_SOURCE_TASK
    "${MUTATED_SOURCE_TASK}")
zc4_validate_source_and_resume_attribution(
    "${CONTEXT_CPP}" "${MUTATED_SOURCE_TASK}" "${OBSERVER_CPP}" "${STORE_CPP}"
    MUTATED_SOURCE_ATTRIBUTION_VALID)
if (MUTATED_SOURCE_ATTRIBUTION_VALID)
    message(FATAL_ERROR "ZC4 source-attribution negative control did not trip")
endif()

set(MUTATED_RESUME_STORE "${STORE_CPP}")
string(REPLACE "resume_outcomes[i].estimator_slot"
    "0" MUTATED_RESUME_STORE "${MUTATED_RESUME_STORE}")
zc4_validate_source_and_resume_attribution(
    "${CONTEXT_CPP}" "${TASK_CPP}" "${OBSERVER_CPP}" "${MUTATED_RESUME_STORE}"
    MUTATED_RESUME_ATTRIBUTION_VALID)
if (MUTATED_RESUME_ATTRIBUTION_VALID)
    message(FATAL_ERROR "ZC4 resume-attribution negative control did not trip")
endif()

set(MUTATED_DESTRUCTION_TYPING "${LIFECYCLE_H}${MODEL_H}${MODEL_CPP}")
string(REPLACE "server_cache_destruction_class destruction_class"
    "uint16_t destruction_class" MUTATED_DESTRUCTION_TYPING
    "${MUTATED_DESTRUCTION_TYPING}")
zc4_validate_destruction_typing("${MUTATED_DESTRUCTION_TYPING}"
    MUTATED_DESTRUCTION_TYPING_VALID)
if (MUTATED_DESTRUCTION_TYPING_VALID)
    message(FATAL_ERROR "ZC4 destruction-typing negative control did not trip")
endif()

message(STATUS "ZC4 calibration-model contract checks passed")
