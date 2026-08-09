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
file(READ "${SOURCE_ROOT}/tools/mtmd/clip.cpp" MTMD_CLIP_CPP)
file(READ "${SOURCE_ROOT}/tools/mtmd/mtmd.cpp" MTMD_CPP)
file(READ "${SOURCE_ROOT}/tests/test-server-cache-fingerprint.cpp" TEST_CPP)

foreach(NAME IN ITEMS
        "server_cache_execution_fingerprint"
        "server_cache_fingerprint_worker")
    contract_forbid_token("${PUBLIC_LLAMA_H}" "${NAME}"
        "ZC3a internal fingerprint leaked into installed llama.h")
endforeach()

foreach(DOMAIN IN ITEMS
        "buun-zc-cost-structures-v1"
        "buun-zc-config-v1"
        "buun-zc-exec-v2"
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
        "FINGERPRINT_ARENA_BYTES = 1024 * 1024")
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
        "llama.cpp model execution-cost structure"
        "writer.string(&hparams, sizeof(hparams))"
        "writer.u32(uint32_t(tensor->type))"
        "writer.u64(uint64_t(tensor->ne[i]))"
        "writer.u64(uint64_t(tensor->nb[i]))"
        "ggml_backend_buft_name(buft)"
        "intentionally excludes tensor payload bytes")
    contract_require_token("${LLAMA_MODEL_CPP}${LLAMA_EXT_H}" "${REQUIRED}"
        "ZC6 loader-verified cost-structure identity")
endforeach()
foreach(REQUIRED IN ITEMS
        "mtmd_cost_structure_digest("
        "server_cache_fingerprint_artifact_role::mmproj"
        "0, tensor_bytes, structure, true"
        "params_base.mmproj_gpu_swap")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC3a actual-mmproj-loader structural identity contract")
endforeach()
function(zc3_validate_mmproj_structure TEXT OUT)
    set(VALID TRUE)
    foreach(REQUIRED IN ITEMS
            "mtmd GGUF execution-cost structure"
            "std::strncmp(key, \"clip.\", 5) != 0"
            "gguf_get_tensor_type("
            "gguf_get_tensor_ne("
            "gguf_get_tensor_size("
            "mtmd loaded execution-cost structure"
            "mtmd context execution-cost structure"
            "total = std::max(total, bytes);")
        string(FIND "${TEXT}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(VALID FALSE)
        endif()
    endforeach()
    set(${OUT} ${VALID} PARENT_SCOPE)
endfunction()
set(MMPROJ_STRUCTURE "${MTMD_CLIP_CPP}${MTMD_CPP}")
zc3_validate_mmproj_structure("${MMPROJ_STRUCTURE}" MMPROJ_STRUCTURE_VALID)
if (NOT MMPROJ_STRUCTURE_VALID)
    message(FATAL_ERROR
        "ZC3a mmproj identity is not canonical, path-free structural identity")
endif()
set(MUTATED_MMPROJ_STRUCTURE "${MMPROJ_STRUCTURE}")
string(REPLACE
    "std::strncmp(key, \"clip.\", 5) != 0"
    "false"
    MUTATED_MMPROJ_STRUCTURE "${MUTATED_MMPROJ_STRUCTURE}")
zc3_validate_mmproj_structure(
    "${MUTATED_MMPROJ_STRUCTURE}" MUTATED_MMPROJ_STRUCTURE_VALID)
if (MUTATED_MMPROJ_STRUCTURE_VALID)
    message(FATAL_ERROR
        "ZC3a mmproj metadata-scope negative control did not trip")
endif()
foreach(REQUIRED IN ITEMS
        "GGML_BACKEND_DEVICE_IDENTITY_V1_PROC"
        "GGML_BACKEND_DEVICE_LINK_V1_PROC"
        "cpu_backend_identity_v1("
        "driver_version, hardware_exact"
        "runtime_version, hardware_exact")
    contract_require_token("${FINGERPRINT_CPP}${CONTEXT_CPP}"
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
        "if (llama_model_cost_structure_capture_enabled())"
        "model->capture_cost_structure_digest()"
        "llama_model_cost_structure_digest("
        "model->cost_structure_digest(digest, bytes)")
    contract_require_token("${LLAMA_CPP}${LLAMA_MODEL_CPP}${LLAMA_EXT_H}" "${REQUIRED}"
        "ZC6 loader-owned cost-structure capture")
endforeach()
foreach(REQUIRED IN ITEMS
        "params_base.cache_optimizer.observer_store_enabled"
        "llama_model_cost_structure_capture_set("
        "cache_fingerprint_pending = result"
        "cache_calibration->resolve_load("
        "cache_optimizer_observations->set_execution_fingerprint("
        "cache_fingerprint_worker->configure("
        "llama_model_cost_structure_digest("
        "cache_fingerprint_worker->add_fixed_artifact("
        "cache_fingerprint_worker->launch()"
        "cache_fingerprint_worker.reset();")
    contract_require_token("${CONTEXT_CPP}" "${REQUIRED}"
        "ZC3a observer-only production wiring")
endforeach()

contract_extract_region("${FINGERPRINT_CPP}"
    "bool fingerprint_config_root_from_params("
    "} // namespace"
    BOUNDED_CONFIG_REGION BOUNDED_CONFIG_REGION_FOUND)
set(FINGERPRINT_CPP_WITH_SENTINEL
    "${FINGERPRINT_CPP}\n// ZC3A_FINGERPRINT_CPP_EOF")
contract_extract_region("${FINGERPRINT_CPP_WITH_SENTINEL}"
    "bool server_cache_fingerprint_worker::launch() noexcept"
    "bool server_cache_fingerprint_worker::poll("
    WORKER_LAUNCH_REGION WORKER_LAUNCH_REGION_FOUND)
contract_extract_region("${FINGERPRINT_CPP_WITH_SENTINEL}"
    "void server_cache_fingerprint_worker::combine() noexcept"
    "// ZC3A_FINGERPRINT_CPP_EOF"
    WORKER_COMBINE_REGION WORKER_COMBINE_REGION_FOUND)
contract_extract_region("${CONTEXT_CPP}"
    "void cache_fingerprint_start() noexcept"
    "void cache_fingerprint_lifecycle_point() noexcept"
    PRODUCTION_START_REGION PRODUCTION_START_REGION_FOUND)
if (NOT BOUNDED_CONFIG_REGION_FOUND OR NOT WORKER_LAUNCH_REGION_FOUND OR
    NOT WORKER_COMBINE_REGION_FOUND OR
    NOT PRODUCTION_START_REGION_FOUND)
    message(FATAL_ERROR "ZC3a bounded fingerprint regions are incomplete")
endif()
function(zc3_validate_synchronous_structure TEXT OUT)
    set(VALID TRUE)
    foreach(REQUIRED IN ITEMS
            "cache_fingerprint_worker->add_fixed_artifact("
            "cache_fingerprint_worker->launch()"
            "combine();"
            "void server_cache_fingerprint_worker::combine() noexcept")
        string(FIND "${TEXT}" "${REQUIRED}" FOUND)
        if (FOUND EQUAL -1)
            set(VALID FALSE)
        endif()
    endforeach()
    foreach(FORBIDDEN IN ITEMS
            "server_cache_fingerprint_descriptor"
            "add_descriptor("
            "set_scheduler_demand("
            "std::thread"
            "pread("
            "llama_model_artifact_descriptor"
            "llama_model_dup_artifact_descriptors"
            "llama_file_integrity_exact"
            "integrity_exact_at_open"
            "llama_model_artifact_capture_")
        string(FIND "${TEXT}" "${FORBIDDEN}" FOUND)
        if (NOT FOUND EQUAL -1)
            set(VALID FALSE)
        endif()
    endforeach()
    set(${OUT} ${VALID} PARENT_SCOPE)
endfunction()
set(SYNCHRONOUS_STRUCTURE
    "${FINGERPRINT_H}${FINGERPRINT_CPP}${PRODUCTION_START_REGION}${LLAMA_CPP}${LLAMA_MODEL_CPP}${LLAMA_MMAP_CPP}${LLAMA_EXT_H}")
zc3_validate_synchronous_structure(
    "${SYNCHRONOUS_STRUCTURE}" SYNCHRONOUS_STRUCTURE_VALID)
if (NOT SYNCHRONOUS_STRUCTURE_VALID)
    message(FATAL_ERROR
        "ZC6 structural fingerprint retained descriptor, payload, or thread machinery")
endif()
set(MUTATED_SYNCHRONOUS_THREAD
    "${SYNCHRONOUS_STRUCTURE}\nstd::thread verifier;")
zc3_validate_synchronous_structure(
    "${MUTATED_SYNCHRONOUS_THREAD}" MUTATED_SYNCHRONOUS_THREAD_VALID)
if (MUTATED_SYNCHRONOUS_THREAD_VALID)
    message(FATAL_ERROR
        "ZC6 structural-fingerprint thread negative control did not trip")
endif()
set(MUTATED_SYNCHRONOUS_DESCRIPTOR
    "${SYNCHRONOUS_STRUCTURE}\nvoid add_descriptor(int);")
zc3_validate_synchronous_structure(
    "${MUTATED_SYNCHRONOUS_DESCRIPTOR}" MUTATED_SYNCHRONOUS_DESCRIPTOR_VALID)
if (MUTATED_SYNCHRONOUS_DESCRIPTOR_VALID)
    message(FATAL_ERROR
        "ZC6 structural-fingerprint descriptor negative control did not trip")
endif()
contract_find_forbidden(
    "${BOUNDED_CONFIG_REGION}${WORKER_COMBINE_REGION}${PRODUCTION_START_REGION}"
    FINGERPRINT_HEAP_ESCAPE
    "std::vector<server_cache_fingerprint_field>"
    "std::vector<server_cache_fingerprint_artifact>"
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
        "24edd1bcdebef3935c728040619352fc62926b5faa31103c9afcb9ceb0ee6a88"
        "6ca7e20b5bedd77c62565a8853b959e6d85d709dd25655293fa203fad7e12aff"
        "62d54604460a097936629ffbe1d6cc224e85c0b9866b895351b10362a833859b"
        "b3f15fa073cad9076b22cd15fae92ce16e48b7604e85bd84844d5910342dcdf4")
    contract_require_token("${TEST_CPP}" "${GOLDEN}"
        "ZC3a production-codec golden")
endforeach()
foreach(REQUIRED IN ITEMS
        "std::memset(structural_tensor->data, 0xa5"
        "CHECK(content_changed == structure_before)"
        "structural_tensor->flags = GGML_TENSOR_FLAG_INPUT"
        "CHECK(descriptor_changed != structure_before)")
    contract_require_token("${TEST_CPP}" "${REQUIRED}"
        "ZC6 cost-structure versus payload identity oracle")
endforeach()

# House-standard negative controls exercise the same predicate as production.
function(zc3_validate_capture_guard TEXT OUT)
    count_literal("${TEXT}"
        "if (llama_model_cost_structure_capture_enabled())" GUARD_COUNT)
    count_literal("${TEXT}"
        "model->capture_cost_structure_digest()" CAPTURE_COUNT)
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
    "if (llama_model_cost_structure_capture_enabled()) {"
    "if (true) {"
    MUTATED_LLAMA "${MUTATED_LLAMA}")
zc3_validate_capture_guard("${MUTATED_LLAMA}" MUTATED_CAPTURE_VALID)
if (MUTATED_CAPTURE_VALID)
    message(FATAL_ERROR "ZC3a capture-gate negative control did not trip")
endif()

set(MUTATED_FINGERPRINT "${FINGERPRINT_CPP}\nstatic constexpr char EXTRA[] = \"buun-zc-exec-v2\";")
count_literal("${MUTATED_FINGERPRINT}" "buun-zc-exec-v2" MUTATED_DOMAIN_COUNT)
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
