# F4.0 controller-identity contract. Durable lineage and process-local controller
# routing are deliberately different trust domains; this scan makes their one-way
# boundary and the retired pool-UUID vocabulary mechanically reviewable.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(GLOB_RECURSE controller_scan_files
    "${SOURCE_ROOT}/src/*.cpp" "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/server/*.cpp" "${SOURCE_ROOT}/tools/server/*.h"
    "${SOURCE_ROOT}/tests/*.cpp" "${SOURCE_ROOT}/tests/*.h")

set(retired_tokens
    vbr_pool_uuid
    pool_uuid_
    vbr_pool_id
    owner_pool_
    taken_by_
    vbr_operation_pool_key
    vbr_operation_instance_key
    vbr_binding_add_pool_target)
foreach(path IN LISTS controller_scan_files)
    file(READ "${path}" text)
    foreach(token IN LISTS retired_tokens)
        string(FIND "${text}" "${token}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR "retired VBR pool identity '${token}' remains in ${path}")
        endif()
    endforeach()
endforeach()

# The bare pool_uuid spelling is frozen wire/reason compatibility vocabulary only.
# No new identity-bearing field or helper may reuse it.
set(pool_uuid_reason_files
    "${SOURCE_ROOT}/src/llama-vbr-generation.h"
    "${SOURCE_ROOT}/src/llama-vbr-generation.cpp"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint-types.h"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint.cpp"
    "${SOURCE_ROOT}/common/common-checkpoint-shadow.h"
    "${SOURCE_ROOT}/common/common-checkpoint-shadow.cpp")
foreach(path IN LISTS controller_scan_files)
    file(READ "${path}" text)
    if (path MATCHES "/tests/test-vbr-representation-epoch\\.cpp$")
        count_literal("${text}"
            "vbr_checkpoint_eligibility_reason::pool_uuid" reason_test_uses)
        if (NOT reason_test_uses EQUAL 1)
            message(FATAL_ERROR
                "F4.0 lineage-rejection test must cite the frozen reason exactly once")
        endif()
        string(REPLACE "vbr_checkpoint_eligibility_reason::pool_uuid" "" text "${text}")
    endif()
    string(FIND "${text}" "pool_uuid" found)
    if (NOT found EQUAL -1)
        list(FIND pool_uuid_reason_files "${path}" allowed_index)
        if (allowed_index EQUAL -1)
            message(FATAL_ERROR "pool_uuid escaped the frozen reason allowlist: ${path}")
        endif()
    endif()
endforeach()

file(READ "${SOURCE_ROOT}/src/llama-vbr-controller-id.h" id_header)
file(READ "${SOURCE_ROOT}/src/llama-vbr-controller-id.cpp" id_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-generation.cpp" generation_source)

foreach(type IN ITEMS vbr_lineage_uuid vbr_controller_instance_id)
    count_literal("${id_header}" "struct ${type}" type_definitions)
    if (NOT type_definitions EQUAL 1)
        message(FATAL_ERROR "F4.0 identity ${type} must have exactly one definition")
    endif()
endforeach()
count_literal("${id_source}"
    "vbr_lineage_uuid vbr_lineage_uuid_allocate() noexcept" lineage_allocator_definitions)
count_literal("${id_source}"
    "vbr_controller_instance_id vbr_controller_instance_id_allocate() noexcept"
    instance_allocator_definitions)
if (NOT lineage_allocator_definitions EQUAL 1 OR
    NOT instance_allocator_definitions EQUAL 1)
    message(FATAL_ERROR "F4.0 requires exactly one allocator definition per identity class")
endif()
count_literal("${generation_source}" "vbr_lineage_uuid_allocate()" lineage_mint_sites)
count_literal("${generation_source}" "vbr_controller_instance_id_allocate()" instance_mint_sites)
count_literal("${generation_source}"
    "vbr_controller_instance_check_and_claim(instance_id_, this)" instance_claim_sites)
count_literal("${generation_source}"
    "vbr_controller_instance_release(instance_id_, this)" instance_release_sites)
if (NOT lineage_mint_sites EQUAL 1 OR NOT instance_mint_sites EQUAL 1 OR
    NOT instance_claim_sites EQUAL 1 OR NOT instance_release_sites EQUAL 1)
    message(FATAL_ERROR
        "tracker identity allocation/claim/release doors are not single-site "
        "(lineage=${lineage_mint_sites} instance=${instance_mint_sites} "
        "claim=${instance_claim_sites} release=${instance_release_sites})")
endif()

# Runtime instance IDs are never durable. These are the complete value-record,
# artifact codec, and canonical lineage-digest surfaces at F4.0.
set(durable_identity_files
    "${SOURCE_ROOT}/src/llama-vbr-generation-types.h"
    "${SOURCE_ROOT}/src/llama-vbr-checkpoint-compose.inc"
    "${SOURCE_ROOT}/src/llama-vbr-artifact.h"
    "${SOURCE_ROOT}/src/llama-vbr-artifact.cpp"
    "${SOURCE_ROOT}/src/llama-vbr-identity-digest.h")
foreach(path IN LISTS durable_identity_files)
    file(READ "${path}" text)
    foreach(token IN ITEMS
            vbr_controller_instance_id runtime_instance instance_id)
        string(FIND "${text}" "${token}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR
                "process-local controller identity leaked into durable surface: ${path} (${token})")
        endif()
    endforeach()
endforeach()

# Opaque lineage halves may be consumed only by the canonical artifact serializer/hash
# and identity digest. Everywhere else must use equality/nonzero on the strong type.
foreach(path IN LISTS controller_scan_files)
    if (path MATCHES "/tests/")
        continue()
    endif()
    file(READ "${path}" text)
    string(REGEX MATCH "lineage[^\n;]*(\\.hi|\\.lo)" raw_lineage_half "${text}")
    if (NOT raw_lineage_half STREQUAL "")
        if (NOT path MATCHES
            "/src/llama-vbr-(artifact\\.cpp|identity-digest\\.h|controller-id\\.cpp)$")
            message(FATAL_ERROR "opaque lineage halves interpreted outside canonical doors: ${path}")
        endif()
    endif()
endforeach()

file(READ "${SOURCE_ROOT}/include/llama.h" public_header)
foreach(token IN ITEMS vbr_lineage_uuid vbr_controller_instance_id)
    string(FIND "${public_header}" "${token}" public_leak)
    if (NOT public_leak EQUAL -1)
        message(FATAL_ERROR "F4.0 internal identity leaked into public llama.h: ${token}")
    endif()
endforeach()

# F4.1a is the reviewed artifact-only v2 move. F4.0 still pins the C/cache-plan
# schemas and confirms the runtime identity remains absent from the wire.
file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact.h" artifact_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" accounting_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" cache_plan_header)
count_literal("${artifact_header}" "VBR_UNIT_ARTIFACT_FORMAT_VERSION = 2" artifact_version)
count_literal("${accounting_header}"
    "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2" accounting_version)
count_literal("${cache_plan_header}"
    "COMMON_CACHE_PLAN_SCHEMA_VERSION = 4" cache_plan_version)
if (NOT artifact_version EQUAL 1 OR NOT accounting_version EQUAL 1 OR
    NOT cache_plan_version EQUAL 1)
    message(FATAL_ERROR
        "F4.0/F4.1a changed an unreviewed wire/schema version "
        "(artifact=${artifact_version} C=${accounting_version} plan=${cache_plan_version})")
endif()

# Negative controls prove both principal token detectors remain live.
set(retired_negative "struct bad { vbr_pool_uuid id; };")
string(FIND "${retired_negative}" "vbr_pool_uuid" retired_negative_found)
set(serialization_negative "struct bad_wire { vbr_controller_instance_id instance_id; };")
string(FIND "${serialization_negative}" "vbr_controller_instance_id" serialization_negative_found)
if (retired_negative_found EQUAL -1 OR serialization_negative_found EQUAL -1)
    message(FATAL_ERROR "F4.0 controller-identity negative control did not trip")
endif()

message(STATUS "F4.0 controller identity contracts passed")
