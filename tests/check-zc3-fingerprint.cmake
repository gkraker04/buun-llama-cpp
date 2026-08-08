if (NOT DEFINED SOURCE_ROOT)
    get_filename_component(SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-cache-fingerprint.h" FINGERPRINT_H)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-fingerprint.cpp" FINGERPRINT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp" OBSERVER_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" CONTEXT_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" SERVER_TASK_CPP)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" SERVER_TASK_H)
file(READ "${SOURCE_ROOT}/common/common.h" COMMON_H)
file(READ "${SOURCE_ROOT}/common/common-checkpoint-shadow.cpp" CHECKPOINT_CPP)
file(READ "${SOURCE_ROOT}/src/llama.cpp" LLAMA_CPP)
file(READ "${SOURCE_ROOT}/src/llama-mmap.cpp" LLAMA_MMAP_CPP)
file(READ "${SOURCE_ROOT}/src/llama-model.cpp" LLAMA_MODEL_CPP)
file(READ "${SOURCE_ROOT}/src/llama-ext.h" LLAMA_EXT_H)
file(READ "${SOURCE_ROOT}/include/llama.h" PUBLIC_LLAMA_H)
file(READ "${SOURCE_ROOT}/tests/test-server-cache-fingerprint.cpp" TEST_CPP)

foreach(NAME IN ITEMS
        "server_cache_execution_fingerprint"
        "server_cache_fingerprint_worker"
        "llama_model_artifact_descriptor")
    contract_forbid_token("${PUBLIC_LLAMA_H}" "${NAME}"
        "ZC3a internal fingerprint leaked into installed llama.h")
endforeach()

foreach(DOMAIN IN ITEMS
        "buun-zc-artifacts-v1"
        "buun-zc-config-v1"
        "buun-zc-exec-v1"
        "buun-zc-adapter-application-v1")
    count_literal("${FINGERPRINT_CPP}" "${DOMAIN}" DOMAIN_COUNT)
    if (NOT DOMAIN_COUNT EQUAL 1)
        message(FATAL_ERROR
            "ZC3a domain '${DOMAIN}' must have one codec owner; got ${DOMAIN_COUNT}")
    endif()
endforeach()

foreach(REQUIRED IN ITEMS
        "constexpr std::array<server_cache_fingerprint_field_type, 32> FIELD_TYPES"
        "sizeof(ARTIFACT_DOMAIN)"
        "sizeof(CONFIG_DOMAIN)"
        "sizeof(EXEC_DOMAIN)"
        "HASH_CHUNK_BYTES = 1024 * 1024"
        "HASH_RATE_BYTES_PER_SECOND = 32ULL * 1024 * 1024"
        "scheduler_demand_.load"
        "IOPRIO_CLASS_IDLE"
        "pread(fd, data, size, off_t(offset))")
    contract_require_token("${FINGERPRINT_CPP}" "${REQUIRED}"
        "ZC3a canonical/worker contract")
endforeach()

foreach(REQUIRED IN ITEMS
        "adapter_application_digest(other.adapter_application_digest)"
        "adapter_application_complete(other.adapter_application_complete)"
        "adapter_application_digest = {};"
        "adapter_application_complete = false;")
    contract_require_token("${CHECKPOINT_CPP}" "${REQUIRED}"
        "ZC3a checkpoint adapter-identity copy/clear contract")
endforeach()
foreach(REQUIRED IN ITEMS
        "entry.adapter_application_digest = adapter_application_digest;"
        "entry.adapter_application_complete = adapter_application_complete;"
        "next.adapter_application_complete ="
        "victim->adapter_application_digest"
        "victim_state->adapter_application_digest")
    contract_require_token("${CONTEXT_CPP}${SERVER_TASK_CPP}" "${REQUIRED}"
        "ZC3a immutable host/checkpoint adapter carrier")
endforeach()
set(COMMON_H_WITH_SENTINEL "${COMMON_H}\n// ZC3A_COMMON_HEADER_EOF")
contract_extract_region("${COMMON_H_WITH_SENTINEL}"
    "struct common_prompt_checkpoint {"
    "// ZC3A_COMMON_HEADER_EOF"
    CHECKPOINT_CARRIER CHECKPOINT_CARRIER_FOUND)
contract_extract_region("${SERVER_TASK_H}"
    "struct server_prompt_cache_state {"
    "struct server_prompt_cache_load_observation {"
    HOST_CARRIER HOST_CARRIER_FOUND)
contract_extract_region("${SERVER_TASK_H}"
    "struct server_prompt_cache_load_observation {"
    "inline void server_prompt_cache_apply_retention_lineage("
    LOAD_OBSERVATION_CARRIER LOAD_OBSERVATION_CARRIER_FOUND)
if (NOT CHECKPOINT_CARRIER_FOUND OR NOT HOST_CARRIER_FOUND OR
    NOT LOAD_OBSERVATION_CARRIER_FOUND)
    message(FATAL_ERROR "ZC3a adapter carrier regions are incomplete")
endif()
foreach(CARRIER IN ITEMS CHECKPOINT_CARRIER HOST_CARRIER LOAD_OBSERVATION_CARRIER)
    count_literal("${${CARRIER}}"
        "std::array<uint8_t, 32> adapter_application_digest = {};"
        ADAPTER_CARRIER_COUNT)
    if (NOT ADAPTER_CARRIER_COUNT EQUAL 1)
        message(FATAL_ERROR
            "ZC3a ${CARRIER} needs exactly one adapter identity carrier; got ${ADAPTER_CARRIER_COUNT}")
    endif()
endforeach()
set(MUTATED_CHECKPOINT "${CHECKPOINT_CPP}")
string(REPLACE
    "adapter_application_complete(other.adapter_application_complete),"
    ""
    MUTATED_CHECKPOINT "${MUTATED_CHECKPOINT}")
string(FIND "${MUTATED_CHECKPOINT}"
    "adapter_application_complete(other.adapter_application_complete)"
    MUTATED_CHECKPOINT_COPY)
if (NOT MUTATED_CHECKPOINT_COPY EQUAL -1)
    message(FATAL_ERROR "ZC3a checkpoint-copy negative control did not trip")
endif()
foreach(REQUIRED IN ITEMS
        "llama_model_artifact_capture_enabled() &&"
        "llama_file_integrity_exact(file_id())"
        "file->integrity_exact_at_open()")
    contract_require_token("${LLAMA_MMAP_CPP}${LLAMA_MODEL_CPP}" "${REQUIRED}"
        "ZC3a immutability-before-consumption contract")
endforeach()
foreach(REQUIRED IN ITEMS
        "!params_base.mmproj.path.empty()"
        "a multimodal execution fingerprint is honestly unavailable")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC3a actual-mmproj-loader fail-closed contract")
endforeach()
foreach(REQUIRED IN ITEMS
        "integrity_exact = false;"
        "GGML_BACKEND_DEVICE_IDENTITY_V1_PROC"
        "GGML_BACKEND_DEVICE_LINK_V1_PROC"
        "cpu_backend_identity_v1("
        "driver_version, hardware_exact"
        "runtime_version, hardware_exact")
    contract_require_token("${LLAMA_MMAP_CPP}${FINGERPRINT_CPP}${CONTEXT_CPP}"
        "${REQUIRED}" "ZC3a resolved hardware/mmproj identity contract")
endforeach()
contract_require_token("${FINGERPRINT_H}" "mmproj  = 3"
    "ZC3a canonical artifact-role table")
contract_forbid_token("${FINGERPRINT_CPP}"
    "driver.exact = params.devices.empty();"
    "ZC3a auto GPU selection was mislabeled exact CPU")

contract_find_forbidden("${FINGERPRINT_H}${FINGERPRINT_CPP}" FORBIDDEN
    "cache_plan_authority"
    "common_cache_plan_calib_find"
    "authorize("
    "std::filesystem"
    "fopen("
    "open("
    "path_model"
    "basename")
if (FORBIDDEN)
    message(FATAL_ERROR
        "ZC3a identity acquired authority, fitted data, or path reopening: ${FORBIDDEN}")
endif()

foreach(REQUIRED IN ITEMS
        "if (llama_model_artifact_capture_enabled())"
        "model->capture_artifact_descriptors(ml)"
        "llama_model_dup_artifact_descriptors_bounded("
        "duplicate_artifact_descriptors_bounded("
        "llama_model_artifact_descriptors_close_bounded(")
    contract_require_token("${LLAMA_CPP}${LLAMA_MODEL_CPP}${LLAMA_EXT_H}" "${REQUIRED}"
        "ZC3a loader-owned descriptor capture")
endforeach()
foreach(REQUIRED IN ITEMS
        "params_base.cache_optimizer.observer_store_enabled"
        "llama_model_artifact_capture_set("
        "cache_fingerprint_pending = result"
        "cache_calibration->resolve_load("
        "cache_optimizer_observations->set_execution_fingerprint("
        "cache_fingerprint_worker->set_scheduler_demand(true)"
        "cache_fingerprint_worker->configure("
        "cache_fingerprint_worker->add_descriptor("
        "cache_fingerprint_worker->add_fixed_artifact("
        "cache_fingerprint_worker->launch()"
        "cache_fingerprint_scheduler_busy = true"
        "cache_fingerprint_worker.reset();")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC3a observer-only production wiring")
endforeach()

contract_extract_region("${FINGERPRINT_CPP}"
    "bool fingerprint_config_root_from_params("
    "void close_descriptor("
    BOUNDED_CONFIG_REGION BOUNDED_CONFIG_REGION_FOUND)
set(FINGERPRINT_CPP_WITH_SENTINEL
    "${FINGERPRINT_CPP}\n// ZC3A_FINGERPRINT_CPP_EOF")
contract_extract_region("${FINGERPRINT_CPP_WITH_SENTINEL}"
    "void server_cache_fingerprint_worker::run() noexcept"
    "// ZC3A_FINGERPRINT_CPP_EOF"
    WORKER_RUN_REGION WORKER_RUN_REGION_FOUND)
contract_extract_region("${CONTEXT_CPP}"
    "void cache_fingerprint_start() noexcept"
    "void cache_fingerprint_lifecycle_point() noexcept"
    PRODUCTION_START_REGION PRODUCTION_START_REGION_FOUND)
if (NOT BOUNDED_CONFIG_REGION_FOUND OR NOT WORKER_RUN_REGION_FOUND OR
    NOT PRODUCTION_START_REGION_FOUND)
    message(FATAL_ERROR "ZC3a bounded fingerprint regions are incomplete")
endif()
contract_find_forbidden(
    "${BOUNDED_CONFIG_REGION}${WORKER_RUN_REGION}${PRODUCTION_START_REGION}"
    FINGERPRINT_HEAP_ESCAPE
    "std::vector<server_cache_fingerprint_field>"
    "std::vector<server_cache_fingerprint_descriptor>"
    "std::vector<server_cache_fingerprint_artifact>"
    "std::vector<llama_model_artifact_descriptor>"
    "server_cache_execution_fingerprint_v1("
    "server_cache_fingerprint_fields_v1(")
if (FINGERPRINT_HEAP_ESCAPE)
    message(FATAL_ERROR
        "ZC3a production fingerprint escaped bounded arena ownership: ${FINGERPRINT_HEAP_ESCAPE}")
endif()

foreach(REQUIRED IN ITEMS
        "key.profile_execution_digest = execution_fingerprint_.execution_root;"
        "key.identity_complete = false;")
    contract_require_token("${OBSERVER_CPP}" "${REQUIRED}"
        "ZC3a profile/operation identity separation")
endforeach()
contract_forbid_token("${OBSERVER_CPP}"
    "key.participant_execution_digest = execution_fingerprint_.execution_root;"
    "ZC3a global profile root impersonated participant identity")
contract_forbid_token("${OBSERVER_CPP}"
    "key.representation_digest = execution_fingerprint_.config_root;"
    "ZC3a global config root impersonated representation identity")

foreach(GOLDEN IN ITEMS
        "3c6440ad78d136e44565da591e0171606d66fe1d561be4663c65dc605bed5ab6"
        "6ca7e20b5bedd77c62565a8853b959e6d85d709dd25655293fa203fad7e12aff"
        "bad581506275f13c4118cf01d56ba31bb1f0141dd4371250daf8adc4f4b15084"
        "b3f15fa073cad9076b22cd15fae92ce16e48b7604e85bd84844d5910342dcdf4")
    contract_require_token("${TEST_CPP}" "${GOLDEN}"
        "ZC3a production-codec golden")
endforeach()

# House-standard negative controls exercise the same predicate as production.
function(zc3_validate_capture_guard TEXT OUT)
    count_literal("${TEXT}"
        "if (llama_model_artifact_capture_enabled())" GUARD_COUNT)
    count_literal("${TEXT}"
        "model->capture_artifact_descriptors(ml)" CAPTURE_COUNT)
    if (GUARD_COUNT EQUAL 1 AND CAPTURE_COUNT EQUAL 1)
        set(${OUT} TRUE PARENT_SCOPE)
    else()
        set(${OUT} FALSE PARENT_SCOPE)
    endif()
endfunction()
zc3_validate_capture_guard("${LLAMA_CPP}" CAPTURE_GUARD_VALID)
if (NOT CAPTURE_GUARD_VALID)
    message(FATAL_ERROR "ZC3a production capture guard is incomplete")
endif()

set(MUTATED_HARDWARE "${FINGERPRINT_CPP}")
string(REPLACE
    "GGML_BACKEND_DEVICE_IDENTITY_V1_PROC"
    "ggml_backend_device_identity_unversioned"
    MUTATED_HARDWARE "${MUTATED_HARDWARE}")
string(FIND "${MUTATED_HARDWARE}"
    "ggml_backend_device_identity_unversioned" MUTATED_AUTO_EXACT)
if (MUTATED_AUTO_EXACT EQUAL -1)
    message(FATAL_ERROR "ZC3a auto-device exactness control did not mutate")
endif()
string(FIND "${MUTATED_HARDWARE}"
    "GGML_BACKEND_DEVICE_IDENTITY_V1_PROC" MUTATED_RESOLVED_EXACT)
if (NOT MUTATED_RESOLVED_EXACT EQUAL -1)
    message(FATAL_ERROR "ZC3a auto-device negative control did not trip")
endif()
set(MUTATED_LLAMA "${LLAMA_CPP}")
string(REPLACE
    "if (llama_model_artifact_capture_enabled()) {"
    "if (true) {"
    MUTATED_LLAMA "${MUTATED_LLAMA}")
zc3_validate_capture_guard("${MUTATED_LLAMA}" MUTATED_CAPTURE_VALID)
if (MUTATED_CAPTURE_VALID)
    message(FATAL_ERROR "ZC3a capture-gate negative control did not trip")
endif()

set(MUTATED_FINGERPRINT "${FINGERPRINT_CPP}\nstatic constexpr char EXTRA[] = \"buun-zc-exec-v1\";")
count_literal("${MUTATED_FINGERPRINT}" "buun-zc-exec-v1" MUTATED_DOMAIN_COUNT)
if (MUTATED_DOMAIN_COUNT EQUAL 1)
    message(FATAL_ERROR "ZC3a domain-owner negative control did not trip")
endif()

set(MUTATED_BOUNDED_CONFIG
    "${BOUNDED_CONFIG_REGION}\nstd::vector<server_cache_fingerprint_field> escaped;")
contract_find_forbidden("${MUTATED_BOUNDED_CONFIG}"
    MUTATED_HEAP_ESCAPE "std::vector<server_cache_fingerprint_field>")
if (NOT MUTATED_HEAP_ESCAPE)
    message(FATAL_ERROR
        "ZC3a bounded-fingerprint heap-escape negative control did not trip")
endif()

message(STATUS "ZC3a fingerprint contract checks passed")
