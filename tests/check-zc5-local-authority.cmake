if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/common/common-cache-optimizer.cpp" OPTIMIZER_CPP)
file(READ "${SOURCE_ROOT}/common/arg.cpp" ARG_CPP)
file(READ "${SOURCE_ROOT}/ggml/include/ggml-backend.h" BACKEND_H)
file(READ "${SOURCE_ROOT}/ggml/src/ggml-cuda/ggml-cuda.cu" CUDA_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-fingerprint.cpp" FINGERPRINT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp" OBSERVER_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-calibration-model.cpp" MODEL_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-plan-authority.cpp" AUTHORITY_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-plan-authority.h" AUTHORITY_H)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/src/llama-model.cpp" MODEL_LOADER_CPP)

foreach(REQUIRED IN ITEMS
        "GGML_BACKEND_DEVICE_IDENTITY_V1_PROC"
        "GGML_BACKEND_DEVICE_LINK_V1_PROC"
        "cudaDriverGetVersion("
        "cudaRuntimeGetVersion("
        "cudaDeviceGetP2PAttribute("
        "resolve_canonical_devices("
        "cpu_backend_identity_v1("
        "pci_domain_bus_device_function")
    contract_require_token("${BACKEND_H}${CUDA_CPP}${FINGERPRINT_CPP}"
        "${REQUIRED}" "ZC5a exact hardware identity")
endforeach()
foreach(REQUIRED IN ITEMS
        "while (params.tensor_buft_overrides.size() < ntbo)"
        "has_active_tensor_buft_override(params)"
        "row.pattern != nullptr || row.buft != nullptr")
    contract_require_token("${ARG_CPP}${FINGERPRINT_CPP}" "${REQUIRED}"
        "ZC5a null-padded tensor-override exactness")
endforeach()
count_literal("${FINGERPRINT_CPP}"
    "has_active_tensor_buft_override(params)" ACTIVE_OVERRIDE_USES)
if (NOT ACTIVE_OVERRIDE_USES EQUAL 2)
    message(FATAL_ERROR
        "ZC5a active tensor override predicate needs exactly two codec uses")
endif()
foreach(REQUIRED IN ITEMS
        "effective_tensor_split"
        "pimpl->effective_tensor_split[i] = splits[i] - prior_split;"
        "effective_tensor_split_count"
        "params_base, effective_split, effective_split_count")
    contract_require_token("${MODEL_LOADER_CPP}${CONTEXT_CPP}"
        "${REQUIRED}" "ZC5a effective placement identity")
endforeach()
contract_find_forbidden("${CONTEXT_CPP}" PLACEMENT_DEEP_COPY
    "common_params fingerprint_params = params_base;")
if (PLACEMENT_DEEP_COPY)
    message(FATAL_ERROR
        "ZC5a effective placement reintroduced a deep common_params copy")
endif()
foreach(REQUIRED IN ITEMS
        "if (!gpu_identities.empty())"
        "ggml_backend_dev_type(device) !="
        "GGML_BACKEND_DEVICE_TYPE_GPU"
        "unmanifested GPU device=%s")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC5a CPU-host versus runtime-GPU budget attribution")
endforeach()
foreach(REQUIRED IN ITEMS
        "if (strcmp(name, GGML_BACKEND_DEVICE_IDENTITY_V1_PROC) == 0)"
        "return (void *) ggml_backend_cuda_device_identity_v1;"
        "if (strcmp(name, GGML_BACKEND_DEVICE_LINK_V1_PROC) == 0)"
        "return (void *) ggml_backend_cuda_device_link_v1;")
    contract_require_token("${CUDA_CPP}" "${REQUIRED}"
        "ZC5a CUDA hardware proc registration")
endforeach()

foreach(REQUIRED IN ITEMS
        "server_cache_calibration_capture_snapshot("
        "server_cache_calibration_snapshot_current("
        "server_cache_calibration_snapshot_lookup_exact("
        "authority_currency_serial_"
        "note_authority_mutation();")
    contract_require_token("${OBSERVER_CPP}${MODEL_CPP}"
        "${REQUIRED}" "ZC5a immutable local snapshot currency")
endforeach()
foreach(REQUIRED IN ITEMS
        "PROVISIONAL_AUTHORITY_VALIDATION_FLOOR = 4"
        "PROVISIONAL_AUTHORITY_RESIDUAL_FLOOR = 4"
        "PROVISIONAL_AUTHORITY_ENVELOPE_MULTIPLIER = 3.0"
        "PROVISIONAL_AUTHORITY_RELATIVE_FLOOR = 0.01"
        "PROVISIONAL_AUTHORITY_ABSOLUTE_FLOOR_US = 500.0"
        "provisional_authority_radius("
        "instance.response_reservoir"
        "const double absolute_residual = std::fabs("
        "instance.n_validation < PROVISIONAL_AUTHORITY_VALIDATION_FLOOR")
    contract_require_token("${MODEL_CPP}" "${REQUIRED}"
        "ZC5a provisional recent-error envelope")
endforeach()

contract_extract_region("${AUTHORITY_CPP}"
    "void server_cache_plan_authority::plan_local_before_mutation("
    "void server_cache_plan_authority::fail_closed("
    LOCAL_PLAN LOCAL_PLAN_FOUND)
if (NOT LOCAL_PLAN_FOUND)
    message(FATAL_ERROR "ZC5a local planner owner is missing")
endif()
foreach(REQUIRED IN ITEMS
        "server_cache_calibration_capture_snapshot("
        "server_cache_calibration_snapshot_lookup_exact("
        "common_cache_plan_choose_preestimated("
        "server_cache_calibration_bound_direct_difference("
        "bound.benefit_lower_us > 0.0"
        "candidate_evidence.measurable"
        "row.viable() && row.is_chain()"
        "common_cache_plan_compose_preestimated_chains("
        "price_consequences(i, row)"
        "candidate_evidence.requires_d_consequences"
        "consequence_points_complete[size_t(challenger)]"
        "most favorable"
        "rec.inventory[size_t(legacy_plan_candidate)].is_chain()"
        "common_cache_optimizer_disposition::certified_improvement"
        "rec.authority_prequalified = true;")
    contract_require_token("${LOCAL_PLAN}" "${REQUIRED}"
        "ZC5a certified-improvement planner")
endforeach()
contract_find_forbidden("${LOCAL_PLAN}" LOCAL_MIXED_SOURCE
    "common_cache_plan_calib_find(" "common_cache_plan_run_planner(")
if (LOCAL_MIXED_SOURCE)
    message(FATAL_ERROR
        "ZC5a local authority mixed checked-in coefficients: ${LOCAL_MIXED_SOURCE}")
endif()

foreach(REQUIRED IN ITEMS
        "out.local_authority_ceiling = raw.cache_plan_authority_explicit"
        "common_cache_plan_authority_level::route_home")
    contract_require_token("${OPTIMIZER_CPP}" "${REQUIRED}"
        "ZC5c route-home ceiling resolver")
endforeach()
foreach(REQUIRED IN ITEMS
        "const auto decision_level = server_cache_plan_level_of(rec.selection);"
        "!server_cache_plan_level_enabled(configured_level, decision_level)")
    contract_require_token("${LOCAL_PLAN}" "${REQUIRED}"
        "ZC5b graduated local planner")
endforeach()
foreach(REQUIRED IN ITEMS
        "out->local.candidates[index].feature"
        "mode.plan_authority->plan_local_before_mutation("
        "cache_plan_authority->local_currency_current(*record)"
        "local_online_authority")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC5a production authority wiring")
endforeach()
foreach(REQUIRED IN ITEMS
        "if (!local_currency_current(rec))"
        "rec.optimizer.local_authority.certified_once"
        "latch.certified_for(rec.optimizer.local_authority)"
        "common_cache_optimizer_authority_state::executed")
    contract_require_token("${AUTHORITY_CPP}" "${REQUIRED}"
        "ZC5a local latch terminal")
endforeach()
foreach(REQUIRED IN ITEMS
        "other.reset();"
        "fallback_legacy(rec, reason, latch);"
        "common_cache_plan_authority_fallback::internal_fault, latch"
        "&execution.local_authority);"
        "server_cache_observation_apply_restore_geometry(")
    contract_require_token("${AUTHORITY_CPP}${CONTEXT_CPP}${OBSERVER_CPP}"
        "${REQUIRED}" "ZC5a consuming latch and exact recovery geometry")
endforeach()
contract_find_forbidden("${LOCAL_PLAN}" FLEET_WIDE_CONFIDENCE
    "reduction.all_active")
if (FLEET_WIDE_CONFIDENCE)
    message(FATAL_ERROR
        "ZC5a reintroduced confidence maturity for unselected candidates")
endif()
contract_extract_region("${AUTHORITY_CPP}"
    "struct local_profile_reduction {"
    "void server_cache_plan_authority::plan_local_before_mutation("
    PROFILE_REDUCTION PROFILE_REDUCTION_FOUND)
if (NOT PROFILE_REDUCTION_FOUND)
    message(FATAL_ERROR "ZC5a profile-reduction owner is missing")
endif()
contract_find_forbidden("${PROFILE_REDUCTION}" FLEET_WIDE_TERMINAL
    "reason = common_cache_optimizer_fallback_reason::drifted"
    "reason = common_cache_optimizer_fallback_reason::out_of_coverage")
if (FLEET_WIDE_TERMINAL)
    message(FATAL_ERROR
        "ZC5a reintroduced unselected confidence/coverage veto: ${FLEET_WIDE_TERMINAL}")
endif()
foreach(REQUIRED IN ITEMS
        "static_assert(!std::is_copy_constructible_v<"
        "std::move(*stage1_inventory.local_authority)"
        "profile_display_label("
        "server_cache_calibration_secure_random(profile_display_salt)")
    contract_require_token("${AUTHORITY_H}${AUTHORITY_CPP}${CONTEXT_CPP}"
        "${REQUIRED}" "ZC5a move-only local capability and display identity")
endforeach()
contract_find_forbidden("${LOCAL_PLAN}" RAW_PROFILE_ROOT
    "rec.optimizer.profile_identity = common_cache_plan_sha256_hex_digest")
if (RAW_PROFILE_ROOT)
    message(FATAL_ERROR
        "ZC5a receipt profile identity leaks the durable execution root")
endif()
foreach(REQUIRED IN ITEMS
        "!cache_plan_authority->local_currency_current(*record)"
        "!cache_plan_authority->local_currency_current(*slot.cache_plan)")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC5a final provider currency fences")
endforeach()

# Same-validator negative controls for the two load-bearing authority gates.
set(MUTATED_LOCAL "${LOCAL_PLAN}")
string(REPLACE "bound.benefit_lower_us > 0.0"
               "bound.benefit_lower_us >= 0.0"
               MUTATED_LOCAL "${MUTATED_LOCAL}")
string(FIND "${MUTATED_LOCAL}" "bound.benefit_lower_us > 0.0" STRICT_MARGIN)
if (NOT STRICT_MARGIN EQUAL -1)
    message(FATAL_ERROR "ZC5a strict-margin negative control did not trip")
endif()
set(MUTATED_OPTIMISTIC_D "${LOCAL_PLAN}")
string(REPLACE "if (!consequence_points_complete[size_t(challenger)])"
               "if (false)"
               MUTATED_OPTIMISTIC_D "${MUTATED_OPTIMISTIC_D}")
string(FIND "${MUTATED_OPTIMISTIC_D}"
    "consequence_points_complete[size_t(challenger)]" OPTIMISTIC_D_GATE)
if (NOT OPTIMISTIC_D_GATE EQUAL -1)
    message(FATAL_ERROR
        "ZC5b optimistic missing-D selection negative control did not trip")
endif()
set(MUTATED_ENVELOPE "${MODEL_CPP}")
string(REPLACE "PROVISIONAL_AUTHORITY_ENVELOPE_MULTIPLIER = 3.0"
               "PROVISIONAL_AUTHORITY_ENVELOPE_MULTIPLIER = 1.0"
               MUTATED_ENVELOPE "${MUTATED_ENVELOPE}")
string(FIND "${MUTATED_ENVELOPE}"
    "PROVISIONAL_AUTHORITY_ENVELOPE_MULTIPLIER = 3.0"
    ENVELOPE_MULTIPLIER_PIN)
if (NOT ENVELOPE_MULTIPLIER_PIN EQUAL -1)
    message(FATAL_ERROR
        "ZC5a provisional-envelope negative control did not trip")
endif()
set(MUTATED_CONTEXT "${CONTEXT_CPP}")
string(REPLACE "cache_plan_authority->local_currency_current(*record)"
               "true"
               MUTATED_CONTEXT "${MUTATED_CONTEXT}")
string(FIND "${MUTATED_CONTEXT}"
    "cache_plan_authority->local_currency_current(*record)" CURRENCY_GATE)
if (NOT CURRENCY_GATE EQUAL -1)
    message(FATAL_ERROR "ZC5a currency negative control did not trip")
endif()
set(MUTATED_FINGERPRINT "${FINGERPRINT_CPP}")
string(REPLACE "has_active_tensor_buft_override(params)"
               "!params.tensor_buft_overrides.empty()"
               MUTATED_FINGERPRINT "${MUTATED_FINGERPRINT}")
count_literal("${MUTATED_FINGERPRINT}"
    "has_active_tensor_buft_override(params)" MUTATED_OVERRIDE_USES)
if (NOT MUTATED_OVERRIDE_USES EQUAL 0)
    message(FATAL_ERROR
        "ZC5a tensor-override exactness negative control did not trip")
endif()
set(MUTATED_CONTEXT_BUDGET "${CONTEXT_CPP}")
string(REPLACE
    "if (!device || ggml_backend_dev_type(device) !=\n                            GGML_BACKEND_DEVICE_TYPE_GPU)"
    "if (!device)"
    MUTATED_CONTEXT_BUDGET "${MUTATED_CONTEXT_BUDGET}")
string(FIND "${MUTATED_CONTEXT_BUDGET}"
    "ggml_backend_dev_type(device) !=\n                            GGML_BACKEND_DEVICE_TYPE_GPU"
    MUTATED_CPU_CLASSIFIER)
if (NOT MUTATED_CPU_CLASSIFIER EQUAL -1)
    message(FATAL_ERROR
        "ZC5a CPU budget-classifier negative control did not trip")
endif()
set(MUTATED_LATCH "${AUTHORITY_CPP}")
string(REPLACE "other.reset();" "" MUTATED_LATCH "${MUTATED_LATCH}")
string(FIND "${MUTATED_LATCH}" "other.reset();" MOVED_FROM_RESET)
if (NOT MOVED_FROM_RESET EQUAL -1)
    message(FATAL_ERROR
        "ZC5a destructive-move negative control did not trip")
endif()
set(MUTATED_REDUCTION "${PROFILE_REDUCTION}")
string(REPLACE
    "coverage = common_cache_optimizer_coverage_class::out_of_coverage;"
    "coverage = common_cache_optimizer_coverage_class::out_of_coverage;\n            reason = common_cache_optimizer_fallback_reason::out_of_coverage;"
    MUTATED_REDUCTION "${MUTATED_REDUCTION}")
contract_find_forbidden("${MUTATED_REDUCTION}" MUTATED_FLEET_TERMINAL
    "reason = common_cache_optimizer_fallback_reason::out_of_coverage")
if (NOT MUTATED_FLEET_TERMINAL)
    message(FATAL_ERROR
        "ZC5a unselected-terminal negative control did not trip")
endif()

message(STATUS "ZC5a-c local-authority contract checks passed")
