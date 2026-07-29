# B0/C0 source-contract scans [P2]: one closed reason enum in one header, no enum or
# name-table replicas, pinned band starts, policy-free C0 leaf header, and the wired-once
# observer surfaces. Mechanical greps in the spirit of check-vbr-generation-isolation.cmake;
# each scan is negative-controlled in the gate by mutating a file COPY and expecting failure.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(GLOB_RECURSE contract_files
    "${SOURCE_ROOT}/src/*.cpp"    "${SOURCE_ROOT}/src/*.h"
    "${SOURCE_ROOT}/common/*.cpp" "${SOURCE_ROOT}/common/*.h"
    "${SOURCE_ROOT}/tools/*.cpp"  "${SOURCE_ROOT}/tools/*.h"
    "${SOURCE_ROOT}/tests/*.cpp")

set(all_source "")
foreach(path IN LISTS contract_files)
    file(READ "${path}" text)
    string(APPEND all_source "${text}")
endforeach()

# --- one closed reason enum, defined exactly once, in common/common-cache-plan.h ---
count_literal("${all_source}" "enum common_cache_plan_reason : uint16_t" reason_defs)
if (NOT reason_defs EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one common_cache_plan_reason definition, found ${reason_defs}")
endif()

# (numeric band-start values are pinned at COMPILE TIME by exact-value static_asserts in
# common-cache-plan.h — stronger than any source-text scan, so no grep pin here)

# --- the other B0/C0 closed enums are each defined exactly once ---
foreach(def
        "enum class common_cache_plan_disposition : uint8_t"
        "enum class common_cache_plan_provider : uint8_t"
        "enum class common_cache_plan_outcome : uint8_t"
        "enum class common_cache_plan_selection : uint8_t"
        "enum class common_cache_plan_inventory_state : uint8_t"
        "enum class common_cache_plan_planner_status : uint8_t"
        "enum class llama_cache_acct_category : uint8_t"
        "enum class llama_cache_acct_residency : uint8_t"
        "enum class llama_cache_acct_domain_kind : uint8_t"
        "enum class llama_cache_acct_producer : uint8_t"
        "enum class llama_cache_acct_measure : uint8_t"
        "enum class llama_cache_acct_known : uint8_t"
        "enum class llama_cache_acct_unit : uint8_t"
        "enum class llama_cache_acct_cost_kind : uint8_t"
        "enum class llama_cache_acct_txn_state : uint8_t"
        "enum class common_retention_source_state : uint8_t"
        "enum class common_retention_pool : uint8_t"
        "enum class common_retention_artifact_kind : uint8_t"
        "enum class common_retention_score_state : uint8_t"
        "enum class server_cache_lease_scope_kind : uint8_t"
        "enum class server_cache_lease_class : uint8_t"
        "enum class server_cache_lease_eval_state : uint8_t"
        "enum class server_cache_lease_eligibility : uint8_t"
        "enum class server_cache_lease_fallback_state : uint8_t"
        "enum class server_cache_lease_event_kind : uint8_t")
    count_literal("${all_source}" "${def}" def_count)
    if (NOT def_count EQUAL 1)
        message(FATAL_ERROR "expected exactly one definition of '${def}', found ${def_count}")
    endif()
endforeach()

# D-S3 has an independent fail-closed binary envelope. It is not the cache-plan wire
# schema and must remain singular/versioned rather than acquiring a server-side mirror.
foreach(sidecar_pin
        "constexpr uint32_t COMMON_RETENTION_SIDECAR_VERSION = 1"
        "constexpr uint32_t COMMON_RETENTION_TURN_TABLE_VERSION = 1"
        "constexpr uint32_t SIDECAR_MAGIC = 0x44533352")
    count_literal("${all_source}" "${sidecar_pin}" sidecar_pin_count)
    if (NOT sidecar_pin_count EQUAL 1)
        message(FATAL_ERROR
            "D-S3 retention-sidecar format authority drifted: '${sidecar_pin}'")
    endif()
endforeach()
set(sidecar_pin_negative
    "${all_source}\nconstexpr uint32_t COMMON_RETENTION_SIDECAR_VERSION = 1;")
count_literal("${sidecar_pin_negative}"
    "constexpr uint32_t COMMON_RETENTION_SIDECAR_VERSION = 1"
    sidecar_negative_count)
if (sidecar_negative_count EQUAL 1)
    message(FATAL_ERROR "D-S3 one-definition negative control did not trip")
endif()

# D-S3 containment: durable turn vectors and score records belong only to the
# observer-owned catalog. The three shipped artifact structs must remain free of
# common_retention_* members so cloning, sizing, and pressure decisions cannot
# accidentally absorb shadow metadata.
function(retention_extract_struct source struct_name output)
    string(REPLACE ";" "\\;" escaped_source "${source}")
    string(REPLACE "\n" ";" lines "${escaped_source}")
    set(active FALSE)
    set(found FALSE)
    set(depth 0)
    set(body "")
    foreach(raw_line IN LISTS lines)
        if (NOT active AND
            raw_line MATCHES "(^|[^a-zA-Z0-9_])struct[ \t]+${struct_name}[ \t]*\\{")
            set(active TRUE)
            set(found TRUE)
        endif()
        if (active)
            string(APPEND body "${raw_line}\n")
            string(REGEX MATCHALL "\\{" opens "${raw_line}")
            string(REGEX MATCHALL "\\}" closes "${raw_line}")
            list(LENGTH opens n_open)
            list(LENGTH closes n_close)
            math(EXPR depth "${depth} + ${n_open} - ${n_close}")
            if (depth LESS_EQUAL 0)
                set(active FALSE)
                break()
            endif()
        endif()
    endforeach()
    if (NOT found)
        message(FATAL_ERROR "D-S3 containment scan could not find struct ${struct_name}")
    endif()
    set(${output} "${body}" PARENT_SCOPE)
endfunction()

file(READ "${SOURCE_ROOT}/tools/server/server-task.h" retention_server_header)
file(READ "${SOURCE_ROOT}/common/common.h" retention_common_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" cache_plan_wire_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.h" lease_header)
foreach(struct_name server_prompt server_prompt_cache_state)
    retention_extract_struct(
        "${retention_server_header}" "${struct_name}" retention_struct_body)
    string(FIND "${retention_struct_body}" "common_retention_" retention_member)
    string(FIND "${retention_struct_body}" "server_cache_lease_" lease_member)
    if (NOT retention_member EQUAL -1 OR NOT lease_member EQUAL -1)
        message(FATAL_ERROR
            "D-S3/D-S5 containment violation: ${struct_name} embeds retention/lease metadata")
    endif()
endforeach()
retention_extract_struct(
    "${retention_common_header}" "common_prompt_checkpoint" retention_struct_body)
string(FIND "${retention_struct_body}" "common_retention_" retention_member)
string(FIND "${retention_struct_body}" "server_cache_lease_" lease_member)
if (NOT retention_member EQUAL -1 OR NOT lease_member EQUAL -1)
    message(FATAL_ERROR
        "D-S3/D-S5 containment violation: common_prompt_checkpoint embeds retention/lease metadata")
endif()
retention_extract_struct(
    "${cache_plan_wire_header}" "common_cache_plan_record" lease_wire_body)
string(FIND "${lease_wire_body}" "server_cache_lease_" lease_wire_member)
if (NOT lease_wire_member EQUAL -1)
    message(FATAL_ERROR
        "D-S5 containment violation: common_cache_plan_record embeds lease state")
endif()

string(REPLACE
    "struct server_prompt {"
    "struct server_prompt {\\n    common_retention_stamp forbidden_inline;"
    retention_struct_negative
    "${retention_server_header}")
retention_extract_struct(
    "${retention_struct_negative}" "server_prompt" retention_negative_body)
string(FIND "${retention_negative_body}" "common_retention_" retention_negative_hit)
if (retention_negative_hit EQUAL -1)
    message(FATAL_ERROR "D-S3 struct-purity negative control did not trip")
endif()
string(REPLACE
    "struct server_prompt_cache_state {"
    "struct server_prompt_cache_state {\\n    server_cache_lease_id forbidden_lease;"
    lease_struct_negative
    "${retention_server_header}")
retention_extract_struct(
    "${lease_struct_negative}" "server_prompt_cache_state" lease_negative_body)
string(FIND "${lease_negative_body}" "server_cache_lease_" lease_negative_hit)
if (lease_negative_hit EQUAL -1)
    message(FATAL_ERROR "D-S5 struct-purity negative control did not trip")
endif()
string(REPLACE
    "struct common_cache_plan_record {"
    "struct common_cache_plan_record {\\n    server_cache_lease_id forbidden_lease;"
    lease_wire_negative
    "${cache_plan_wire_header}")
retention_extract_struct(
    "${lease_wire_negative}" "common_cache_plan_record" lease_wire_negative_body)
string(FIND "${lease_wire_negative_body}" "server_cache_lease_" lease_wire_negative_hit)
if (lease_wire_negative_hit EQUAL -1)
    message(FATAL_ERROR "D-S5 wire-purity negative control did not trip")
endif()

# D-S5 identity is an observer-only mirror of WS-4's canonical three opaque
# computation-frontier keys. Keep the lease library independent of common.h,
# but fail source CI if either side silently changes shape.
retention_extract_struct(
    "${retention_common_header}" "common_computation_frontier" frontier_identity_body)
retention_extract_struct(
    "${lease_header}" "server_cache_lease_identity" lease_identity_body)
foreach(field
        execution_identity
        adapter_config_identity
        media_content_identity)
    foreach(body frontier_identity_body lease_identity_body)
        string(FIND "${${body}}" "std::string ${field};" identity_field)
        if (identity_field EQUAL -1)
            message(FATAL_ERROR
                "D-S5/WS-4 identity mirror missing std::string ${field} in ${body}")
        endif()
    endforeach()
endforeach()
string(REGEX MATCHALL "std::string[ \t]+[a-zA-Z0-9_]+" lease_identity_strings
    "${lease_identity_body}")
list(LENGTH lease_identity_strings lease_identity_string_count)
if (NOT lease_identity_string_count EQUAL 3)
    message(FATAL_ERROR
        "D-S5 identity mirror must contain exactly the three canonical WS-4 keys")
endif()

foreach(alias
        "using llama_cache_acct_device_digest ="
        "using llama_cache_acct_topology_digest =")
    count_literal("${all_source}" "${alias}" alias_count)
    if (NOT alias_count EQUAL 1)
        message(FATAL_ERROR "expected exactly one tagged digest alias '${alias}', found ${alias_count}")
    endif()
endforeach()

# --- C schema-v2 resource identity and explicit v1 adapter are one-definition contracts ---
foreach(def
        "struct llama_cache_acct_resource_domain"
        "struct llama_cache_acct_shard_topology"
        "struct llama_cache_acct_topology_row"
        "struct llama_cache_acct_completeness_row"
        "struct llama_cache_acct_snapshot_v1")
    count_literal("${all_source}" "${def}" def_count)
    if (NOT def_count EQUAL 1)
        message(FATAL_ERROR "expected exactly one definition of '${def}', found ${def_count}")
    endif()
endforeach()
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" acct_v2_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.cpp" acct_v2_source)
count_literal("${acct_v2_header}" "bool llama_cache_acct_snapshot_to_v1(" v1_adapter_decls)
count_literal("${acct_v2_source}" "bool llama_cache_acct_snapshot_to_v1(" v1_adapter_defs)
if (NOT v1_adapter_decls EQUAL 1 OR NOT v1_adapter_defs EQUAL 1)
    message(FATAL_ERROR
        "expected one declaration + one definition of explicit C-v1 adapter, found "
        "${v1_adapter_decls}/${v1_adapter_defs}")
endif()
count_literal("${acct_v2_header}" "bool llama_cache_acct_build_shard_topology(" topology_builder_decls)
count_literal("${acct_v2_source}" "bool llama_cache_acct_build_shard_topology(" topology_builder_defs)
if (NOT topology_builder_decls EQUAL 1 OR NOT topology_builder_defs EQUAL 1)
    message(FATAL_ERROR
        "canonical topology builder drifted (header/source definitions "
        "${topology_builder_decls}/${topology_builder_defs})")
endif()
count_literal("${acct_v2_source}" "llama_sha256_writer" topology_writer_uses)
if (NOT topology_writer_uses EQUAL 2)
    message(FATAL_ERROR
        "accounting identity digests must use the canonical writer (found "
        "${topology_writer_uses} uses)")
endif()
foreach(retired_shape "llama_sha256 hash" "staged_now" "find_staged")
    string(FIND "${acct_v2_source}" "${retired_shape}" retired_found)
    string(FIND "${acct_v2_header}" "${retired_shape}" retired_header_found)
    if (NOT retired_found EQUAL -1 OR NOT retired_header_found EQUAL -1)
        message(FATAL_ERROR
            "retired accounting identity/staging shape returned: '${retired_shape}'")
    endif()
endforeach()

# Record schema 3 embeds accounting schema 2. The compile-time table is the authority; these
# source pins ensure the JSON schema could not move while either side's version stayed put.
foreach(schema_pin
        "constexpr uint32_t COMMON_CACHE_PLAN_SCHEMA_VERSION = 3"
        "common_cache_plan_accounting_schema(COMMON_CACHE_PLAN_SCHEMA_VERSION)"
        "LLAMA_CACHE_ACCT_SCHEMA_VERSION          = 2")
    count_literal("${all_source}" "${schema_pin}" schema_pin_count)
    if (NOT schema_pin_count EQUAL 1)
        message(FATAL_ERROR "record/accounting schema coupling drifted: '${schema_pin}'")
    endif()
endforeach()
file(READ "${SOURCE_ROOT}/tools/server/bench/cache_plan_common.py" cache_plan_reader)
string(FIND "${cache_plan_reader}" "SUPPORTED_SCHEMAS = (1, 2, 3)" schema_reader_pin)
if (schema_reader_pin EQUAL -1)
    message(FATAL_ERROR "cache-plan reader does not explicitly accept v1/v2/v3")
endif()

# --- name spellings are SINGULAR: every reason name is extracted mechanically from the
# X-macro list (its one authoritative spelling) and any second quoted occurrence anywhere in
# the tree is a shadow replica. "none" is excluded — it is a legitimate name in other closed
# vocabularies (e.g. the A2 tombstone table). ---
file(READ "${SOURCE_ROOT}/common/common-cache-plan.h" plan_header)
file(READ "${SOURCE_ROOT}/common/common-cache-plan.cpp" plan_source)
count_literal("${plan_header}"
    "nlohmann::ordered_json common_cache_plan_value_json(" value_json_decls)
count_literal("${plan_source}"
    "json common_cache_plan_value_json(" value_json_defs)
if (NOT value_json_decls EQUAL 1 OR NOT value_json_defs EQUAL 1)
    message(FATAL_ERROR
        "canonical accounting-value JSON helper drifted "
        "(decls=${value_json_decls}, defs=${value_json_defs})")
endif()
string(FIND "${plan_source}" "static json cache_plan_value_json(" old_value_json)
if (NOT old_value_json EQUAL -1)
    message(FATAL_ERROR
        "private accounting-value JSON replica returned")
endif()
set(value_json_negative
    "${plan_header}\nnlohmann::ordered_json common_cache_plan_value_json();")
count_literal("${value_json_negative}"
    "nlohmann::ordered_json common_cache_plan_value_json(" value_json_negative_count)
if (value_json_negative_count EQUAL 1)
    message(FATAL_ERROR
        "accounting-value JSON one-definition negative control did not trip")
endif()

string(REGEX MATCHALL "X\\([A-Z_0-9]+, +\"[a-z_0-9]+\"" reason_entries "${plan_header}")
list(LENGTH reason_entries reason_entry_count)
if (reason_entry_count LESS 30)
    message(FATAL_ERROR
        "expected at least 30 X-macro reason entries, found ${reason_entry_count}")
endif()
foreach(entry IN LISTS reason_entries)
    string(REGEX REPLACE ".*\"([a-z_0-9]+)\"" "\\1" name "${entry}")
    if (name STREQUAL "none")
        continue()
    endif()
    count_literal("${all_source}" "\"${name}\"" name_count)
    if (NOT name_count EQUAL 1)
        message(FATAL_ERROR "name-table replica: \"${name}\" spelled ${name_count} times")
    endif()
endforeach()

# Operation/allocation ids are process-local ledger handles. They may live in the in-memory
# server cache owner, but must not enter a public/state envelope or the JSON bridge. Glob the
# whole serialization/API surface: a renamed/new file must enter the scan automatically.
file(GLOB_RECURSE acct_serialization_files LIST_DIRECTORIES false
    "${SOURCE_ROOT}/include/*"
    "${SOURCE_ROOT}/common/*.h" "${SOURCE_ROOT}/common/*.cpp"
    "${SOURCE_ROOT}/src/llama-io*"
    "${SOURCE_ROOT}/tools/server/server-common*")
foreach(path IN LISTS acct_serialization_files)
    file(READ "${path}" text)
    foreach(process_local_symbol IN ITEMS
            llama_cache_acct_op_id
            llama_cache_acct_alloc_id)
        string(FIND "${text}" "${process_local_symbol}" found)
        if (NOT found EQUAL -1)
            message(FATAL_ERROR
                "process-local accounting id reached serialization/API surface: "
                "${path} (${process_local_symbol})")
        endif()
    endforeach()
endforeach()

# C/F freeze requirement 9: the ledger snapshot is the sole aggregate byte-cell shape.
# Production consumers may carry the snapshot, but cannot grow a parallel container of its
# cell/allocation rows. This is reader-agnostic and globbed; the negative control proves the
# scanner, rather than a hand-listed current file.
function(acct_find_private_aggregates corpus output)
    string(REGEX MATCHALL
        "(array|vector|unordered_map)[ \t]*<[^;\\n]*(llama_cache_acct_cell|llama_cache_acct_cell_row|llama_cache_acct_allocation_row)"
        private_hits "${corpus}")
    set(${output} "${private_hits}" PARENT_SCOPE)
endfunction()
set(acct_consumer_source "")
foreach(path IN LISTS contract_files)
    if (path STREQUAL "${SOURCE_ROOT}/src/llama-cache-accounting.h" OR
        path STREQUAL "${SOURCE_ROOT}/src/llama-cache-accounting.cpp" OR
        path MATCHES "/tests/")
        continue()
    endif()
    file(READ "${path}" text)
    string(APPEND acct_consumer_source "${text}\n")
endforeach()
acct_find_private_aggregates("${acct_consumer_source}" private_acct_aggregates)
if (private_acct_aggregates)
    message(FATAL_ERROR
        "C/F freeze requirement 9: private accounting aggregate container found: "
        "${private_acct_aggregates}")
endif()
set(acct_private_negative
    "${acct_consumer_source}\nstd::vector<llama_cache_acct_cell_row> private_cache_bytes;")
acct_find_private_aggregates("${acct_private_negative}" private_acct_negative_hits)
if (NOT private_acct_negative_hits)
    message(FATAL_ERROR "C/F freeze requirement 9 negative control did not trip")
endif()

# non-reason closed names keep the representative replica ban
foreach(name "\"valid_not_chosen_cost\"" "\"restore_failed_fell_back_cold\"" "\"rolling_window_tape\"")
    count_literal("${all_source}" "${name}" name_count)
    if (NOT name_count EQUAL 1)
        message(FATAL_ERROR "name-table replica: ${name} spelled ${name_count} times")
    endif()
endforeach()

# --- C0 leaf header stays policy-free: no name strings, no JSON, no server includes ---
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.h" acct_header)
file(READ "${SOURCE_ROOT}/src/llama-cache-accounting.cpp" acct_source)
foreach(banned "nlohmann" "#include \"server" "const char *")
    string(FIND "${acct_header}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "llama-cache-accounting.h must stay policy/presentation-free (found '${banned}')")
    endif()
    string(FIND "${acct_source}" "${banned}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "llama-cache-accounting.cpp must stay policy/presentation-free (found '${banned}')")
    endif()
endforeach()

# (once-only finalization is a RUNTIME invariant: cache_plan_finalize early-returns and
# fault-counts on an already-finalized record — outcome != unknown is the finalized state)

# D-S1: device domains must be interned before the ONE production completeness-manifest
# configuration. A second configure call cannot work after cells exist, and a gauge on a
# domain omitted from the first call fails closed. Keep this ordering mechanically pinned.
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" server_context_source)
function(cache_plan_live_manifest_shape source output)
    count_literal("${source}" "configure_required_producers(" configure_count)
    string(FIND "${source}" "make_device_domain(" domain_pos)
    string(FIND "${source}" "configure_required_producers(" configure_pos)
    string(FIND "${source}"
        "binding.domain, llama_cache_acct_producer::live_memory" live_requirement_pos)
    set(ok TRUE)
    if (NOT configure_count EQUAL 1 OR domain_pos EQUAL -1 OR configure_pos EQUAL -1 OR
        live_requirement_pos EQUAL -1 OR NOT domain_pos LESS configure_pos OR
        NOT live_requirement_pos LESS configure_pos)
        set(ok FALSE)
    endif()
    set(${output} "${ok}" PARENT_SCOPE)
endfunction()
cache_plan_live_manifest_shape("${server_context_source}" live_manifest_ok)
if (NOT live_manifest_ok)
    message(FATAL_ERROR
        "D-S1 live-memory domains/requirements must precede the single producer manifest")
endif()
set(live_manifest_negative
    "configure_required_producers(required, n); make_device_domain(topology, ordinal, domain); binding.domain, llama_cache_acct_producer::live_memory")
cache_plan_live_manifest_shape("${live_manifest_negative}" live_manifest_negative_ok)
if (live_manifest_negative_ok)
    message(FATAL_ERROR "D-S1 manifest-order negative control did not trip")
endif()

# VBR name census completeness (D-pins r6): scan quoted VBR_* literals independently of
# the reader spelling. This catches direct getenv, wrapper reads, programmatic producers,
# diagnostics, and scripts; every one must be classified in COMMON_CACHE_PLAN_VBR_ENV_LIST.
# The registry header is deliberately excluded from observations so it cannot prove its own
# coverage.
set(vbr_census_path "${SOURCE_ROOT}/common/common-cache-plan-estimate.h")
file(READ "${vbr_census_path}" census_src)
function(vbr_find_uncensused names_var output)
    set(missing "")
    foreach(name IN LISTS ${names_var})
        string(FIND "${census_src}" "X(\"${name}\"" found)
        if (found EQUAL -1)
            list(APPEND missing "${name}")
        endif()
    endforeach()
    set(${output} "${missing}" PARENT_SCOPE)
endfunction()

# ONE spelling of the extraction pattern: the negative control below must exercise the
# SAME regex as production, or it can keep passing against a retired pattern.
set(vbr_name_re "\"VBR_[A-Z_0-9]+\"")
set(vbr_env_names "")
foreach(dir src common tools ggml/src)
    file(GLOB_RECURSE dir_files LIST_DIRECTORIES false
         "${SOURCE_ROOT}/${dir}/*.c"   "${SOURCE_ROOT}/${dir}/*.cpp"
         "${SOURCE_ROOT}/${dir}/*.cu"  "${SOURCE_ROOT}/${dir}/*.cuh"
         "${SOURCE_ROOT}/${dir}/*.h"   "${SOURCE_ROOT}/${dir}/*.hpp"
         "${SOURCE_ROOT}/${dir}/*.inc" "${SOURCE_ROOT}/${dir}/*.py"
         "${SOURCE_ROOT}/${dir}/*.sh"  "${SOURCE_ROOT}/${dir}/*.cmake")
    foreach(f ${dir_files})
        if ("${f}" STREQUAL "${vbr_census_path}")
            continue()
        endif()
        file(READ "${f}" body)
        string(REGEX MATCHALL "${vbr_name_re}" hits "${body}")
        foreach(hit ${hits})
            string(REGEX REPLACE "^\"" "" name "${hit}")
            string(REGEX REPLACE "\"$" "" name "${name}")
            list(APPEND vbr_env_names "${name}")
        endforeach()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES vbr_env_names)
vbr_find_uncensused(vbr_env_names uncensused_vbr)
if (uncensused_vbr)
    message(FATAL_ERROR "VBR names used in the tree but missing from "
                        "COMMON_CACHE_PLAN_VBR_ENV_LIST: ${uncensused_vbr}")
endif()
list(LENGTH vbr_env_names n_vbr_env)
message(STATUS "vbr literal census covers ${n_vbr_env} classified names")

# Negative control for the exact historical hole: VBR_LAYER_STRICT is read through the real
# turbo_vbr_env_enabled wrapper (and set programmatically), so it must occur in the REAL scan
# output. A regression to getenv("VBR_*") extraction loses it and fails here.
list(FIND vbr_env_names "VBR_LAYER_STRICT" wrapper_name_index)
if (wrapper_name_index EQUAL -1)
    message(FATAL_ERROR "VBR reader-agnostic census missed real wrapper-only VBR_LAYER_STRICT")
endif()

message(STATUS "cache-plan/accounting contract scans passed")
