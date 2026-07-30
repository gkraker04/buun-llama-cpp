# F2.1 artifact-format contract: the wire vocabulary is internal, its six SHA-256
# identity purposes remain non-interchangeable, and process-local coordination/accounting
# identities never enter the codec. Negative controls mutate source text in memory.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact.h" artifact_header)
file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact.cpp" artifact_source)
file(READ "${SOURCE_ROOT}/include/llama.h" public_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" accounting_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" cache_plan_header)

foreach(identity IN ITEMS
        vbr_unit_version_id
        vbr_payload_digest
        vbr_stash_payload_id
        vbr_manifest_digest
        vbr_capture_generation_id
        vbr_transition_lineage_id)
    count_literal("${artifact_header}" "using ${identity}" identity_definitions)
    if (NOT identity_definitions EQUAL 1)
        message(FATAL_ERROR
            "F2.1 identity ${identity} must have exactly one typed definition")
    endif()
endforeach()

foreach(shared_type IN ITEMS
        llama_cache_acct_digest
        llama_cache_acct_shard_topology)
    string(FIND "${artifact_header}" "${shared_type}" shared_type_found)
    if (shared_type_found EQUAL -1)
        message(FATAL_ERROR
            "F2.1 must reuse the canonical accounting ${shared_type}")
    endif()
endforeach()

foreach(forbidden_topology_copy IN ITEMS
        "hash_topology"
        "buun.cache-acct.topology")
    string(FIND "${artifact_header}${artifact_source}"
        "${forbidden_topology_copy}" topology_copy_found)
    if (NOT topology_copy_found EQUAL -1)
        message(FATAL_ERROR
            "F2.1 topology digest must use the canonical accounting door")
    endif()
endforeach()
string(FIND "${artifact_source}"
    "llama_cache_acct_compute_topology_digest" topology_door_found)
if (topology_door_found EQUAL -1)
    message(FATAL_ERROR
        "F2.1 does not call the canonical accounting topology digest door")
endif()

foreach(forbidden IN ITEMS
        "llama_cache_acct_op_id"
        "llama_cache_acct_alloc_id"
        "llama_cache_acct_topology_id"
        "publish_seq"
        "mutation_serial")
    string(FIND "${artifact_header}${artifact_source}" "${forbidden}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "process-local identity '${forbidden}' leaked into the F2.1 format")
    endif()
endforeach()

set(process_local_negative
    "${artifact_header}\nstruct bad_wire { llama_cache_acct_op_id op; };")
string(FIND "${process_local_negative}" "llama_cache_acct_op_id" negative_found)
if (negative_found EQUAL -1)
    message(FATAL_ERROR
        "F2.1 process-local identity negative control did not trip")
endif()

string(FIND "${public_header}" "vbr_artifact_" public_leak)
if (NOT public_leak EQUAL -1)
    message(FATAL_ERROR "F2.1 internal artifact vocabulary leaked into public llama.h")
endif()

count_literal("${accounting_header}"
    "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2" accounting_schema)
count_literal("${cache_plan_header}"
    "COMMON_CACHE_PLAN_SCHEMA_VERSION = 4" cache_plan_schema)
if (NOT accounting_schema EQUAL 1 OR NOT cache_plan_schema EQUAL 1)
    message(FATAL_ERROR
        "F2.1 changed a frozen schema (C=${accounting_schema}, plan=${cache_plan_schema})")
endif()

foreach(enum_name IN ITEMS
        vbr_artifact_status
        vbr_artifact_layout
        vbr_artifact_side
        vbr_artifact_representation_kind
        vbr_artifact_recoverability
        vbr_artifact_clean_stash_state
        vbr_artifact_consistency_kind
        vbr_artifact_section_kind
        vbr_artifact_companion_kind
        vbr_artifact_accounting_role)
    string(REGEX MATCH
        "enum class ${enum_name}[^\\{]*\\{[^}]*\\}"
        enum_match "${artifact_header}")
    if (enum_match STREQUAL "")
        message(FATAL_ERROR "F2.1 closed enum ${enum_name} is missing")
    endif()
    string(FIND "${enum_match}" "_count" count_sentinel)
    if (count_sentinel EQUAL -1)
        message(FATAL_ERROR "F2.1 closed enum ${enum_name} lacks _count")
    endif()
endforeach()

message(STATUS "F2.1 VBR artifact format contracts passed")
