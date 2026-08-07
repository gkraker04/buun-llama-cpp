if(NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.cpp" control)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.h" header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.cpp" leases)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-lease.h" lease_header)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)

file(GLOB server_sources "${SOURCE_ROOT}/tools/server/*.cpp")
file(GLOB server_headers "${SOURCE_ROOT}/tools/server/*.h")
set(all_server_source "")
foreach(source ${server_sources})
    file(READ "${source}" source_text)
    string(APPEND all_server_source "\n${source_text}")
endforeach()
set(all_server_headers "")
foreach(source ${server_headers})
    file(READ "${source}" source_text)
    string(APPEND all_server_headers "\n${source_text}")
endforeach()

function(require_token text token label)
    string(FIND "${text}" "${token}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "${label}: missing '${token}'")
    endif()
endfunction()

function(forbid_token text token label)
    string(FIND "${text}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "${label}: forbidden '${token}'")
    endif()
endfunction()

# The only production execution door is the scheduler switch. E1.2 constructs
# the contiguous typed task through one path-derived scheduler enqueue; no
# route may invoke the authority itself or spell a second per-operation door.
string(REGEX MATCHALL "cache_control_authority->execute\\(" calls "${context}")
list(LENGTH calls n_calls)
if(NOT n_calls EQUAL 1)
    message(FATAL_ERROR "E1 control kernel must have one scheduler call site")
endif()
require_token("${context}" "case SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE:" "scheduler handler")
require_token("${control}" "cache control authority must run on scheduler thread" "scheduler assertion")
require_token("${context}" "cache_control_authority.reset();" "shutdown proof drain")
require_token("${context}" "server_cache_control_task_precheck(" "lifecycle task gate")
require_token("${context}" "params_base.cache_optimizer.cache_lifecycle," "lifecycle task gate input")
require_token("${control}" "if (!lifecycle_available || !substrate_available)" "debug-only refusal")
foreach(task_name
        CACHE_HOLDER_CREATE CACHE_HOLDER_CLOSE CACHE_HOLDER_REATTACH
        CACHE_FAMILY_REGISTER CACHE_FAMILY_BIND
        CACHE_LEASE_ACQUIRE CACHE_LEASE_INSPECT CACHE_LEASE_RENEW
        CACHE_LEASE_RELEASE CACHE_CONTROL_EVENTS)
    require_token("${context}" "case SERVER_TASK_TYPE_${task_name}:" "typed task census")
    string(REGEX MATCH
        "server_task[^;\n]*\\([^\\)]*SERVER_TASK_TYPE_${task_name}"
        task_constructor "${all_server_source}")
    if(NOT task_constructor STREQUAL "")
        message(FATAL_ERROR "scheduler task ${task_name} became constructible in production")
    endif()
    string(REGEX MATCH
        "\\.type[ \t\r\n]*=[ \t\r\n]*SERVER_TASK_TYPE_${task_name}"
        task_assignment "${all_server_source}")
    if(NOT task_assignment STREQUAL "")
        message(FATAL_ERROR "scheduler task ${task_name} became assignable in production")
    endif()
endforeach()
require_token("${context}"
    "server_task::cache_control_task(operation)"
    "E1.2 contiguous task construction")
require_token("${context}" "res->rd.post_task(std::move(task));"
    "E1.2 scheduler enqueue")

# One lease table remains authoritative. The control object receives a pointer
# and becomes its one proof provider; it may not own a parallel table/evaluator.
string(REGEX MATCHALL "server_cache_lease_table leases"
    table_members "${all_server_headers}")
list(LENGTH table_members n_table_members)
if(NOT n_table_members EQUAL 1)
    message(FATAL_ERROR "single lease-table member census found ${n_table_members}")
endif()
require_token("${control}" "bind_fallback_provider(this)" "single provider bind")
require_token("${control}" "grant_hard_owned(" "hard lease one-table door")
set(all_server_production "${all_server_headers}\n${all_server_source}")
string(REGEX MATCHALL "server_cache_lease_evaluate_request\\(" evaluators
    "${all_server_production}")
list(LENGTH evaluators n_evaluators)
if(NOT n_evaluators EQUAL 2)
    message(FATAL_ERROR "legacy lease evaluator census found ${n_evaluators}")
endif()
string(REGEX MATCHALL "grant_hard_owned\\(" owned_hard_calls "${all_server_source}")
list(LENGTH owned_hard_calls n_owned_hard_calls)
if(NOT n_owned_hard_calls EQUAL 2)
    message(FATAL_ERROR "owned hard evaluator census found ${n_owned_hard_calls}")
endif()

# Explicit-only hard lifetime and proven-frontier typing are structural.
require_token("${leases}" "!leases[i].explicit_hard" "hard timer skip")
require_token("${control}" "orphan_owner(holder.id)" "holder orphan transition")
require_token("${control}" "server_cache_control_status::partially_stale" "frontier typing")
require_token("${control}" "server_cache_control_status::subject_lost" "subject-lost typing")
require_token("${context}" "const bool append_continuity =" "append-stable subject identity")
require_token("${context}" "retained_prefix == slot.prompt.tokens.size()" "full-prefix continuity gate")
require_token("${context}" "identity_known && prior_artifact.v != 0" "replacement frontier gate")
require_token("${leases}" "server_cache_lease_table::artifact_replaced(" "append lease migration")
require_token("${context}" "server_cache_context_scope_id implicit_soft_lease_scope;"
    "one implicit soft scope per slot")
require_token("${context}" "server_cache_lease_scope::from(implicit_soft_lease_scope)"
    "implicit soft renewal scope")
require_token("${control}" "server_cache_control_subject_kind::live_checkpoint" "checkpoint reject")
require_token("${control}" "return server_cache_control_status::not_supported;" "checkpoint fallback closed")
require_token("${context}" "cache_control_authority->lifecycle_point();" "scheduler lifecycle wiring")
require_token("${task_source}" "server_prompt_cache_host_fallback_proof(" "host pin door")
require_token("${task_source}" "victim_catalog.recovery_pinned" "thin-lane pin guard")
forbid_token("${context}" "test_fail_note_after =" "test note fault production inertness")
forbid_token("${context}" "test_fail_remember =" "test remember fault production inertness")

# Only the scheduler-owned resolver may invoke production proof adapters.
foreach(source ${server_sources})
    if(source MATCHES "server-cache-retention-proof.cpp$" OR
       source MATCHES "server-cache-vbr-proof.cpp$")
        continue()
    endif()
    file(READ "${source}" source_text)
    if(NOT source MATCHES "server-task.cpp$")
        forbid_token("${source_text}" "server_cache_retention_fallback_proof(" "retention proof caller census")
    endif()
    if(NOT source MATCHES "server-cache-control.cpp$")
        forbid_token("${source_text}" "server_cache_vbr_fallback_proof(" "F proof caller census")
    endif()
endforeach()
string(REGEX MATCHALL "server_cache_retention_fallback_proof\\(" retention_adapters
    "${task_source}")
list(LENGTH retention_adapters n_retention_adapters)
if(NOT n_retention_adapters EQUAL 1)
    message(FATAL_ERROR "host fallback adapter census found ${n_retention_adapters}")
endif()
string(REGEX MATCHALL "resolve_control_reference\\(" control_resolvers
    "${control}")
list(LENGTH control_resolvers n_control_resolvers)
if(NOT n_control_resolvers EQUAL 1)
    message(FATAL_ERROR "F resolver call census drifted")
endif()

function(scheduler_contract_valid text output)
    string(FIND "${text}" "case SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE:" found)
    if(found EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(explicit_lifetime_valid text output)
    string(FIND "${text}" "!leases[i].explicit_hard" found)
    if(found EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(two_copy_contract_valid task output)
    string(FIND "${task}" "server_prompt_cache_host_fallback_proof(" host_pin)
    string(FIND "${task}" "victim_catalog.recovery_pinned" thin_pin)
    if(host_pin EQUAL -1 OR thin_pin EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(subject_lost_contract_valid lease control_text output)
    string(FIND "${lease}" "mark_subject_lost(" table_transition)
    string(FIND "${control_text}" "server_cache_control_status::subject_lost" typed_transition)
    if(table_transition EQUAL -1 OR typed_transition EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(append_continuity_contract_valid context_text output)
    string(FIND "${context_text}" "const bool append_continuity =" append_gate)
    string(FIND "${context_text}"
        "retained_prefix == slot.prompt.tokens.size()" full_prefix)
    string(FIND "${context_text}"
        "identity_known && prior_artifact.v != 0" replacement_gate)
    if(append_gate EQUAL -1 OR full_prefix EQUAL -1 OR
       replacement_gate EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(append_replacement_contract_valid lease_text output)
    string(FIND "${lease_text}"
        "server_cache_lease_table::artifact_replaced(" replacement)
    string(FIND "${lease_text}"
        "lease.proven_frontier.sequence_epoch" epoch_guard)
    if(replacement EQUAL -1 OR epoch_guard EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()
function(implicit_soft_scope_contract_valid context_text output)
    string(FIND "${context_text}"
        "server_cache_context_scope_id implicit_soft_lease_scope;" scope_member)
    string(FIND "${context_text}"
        "server_cache_lease_scope::from(implicit_soft_lease_scope)" scope_use)
    if(scope_member EQUAL -1 OR scope_use EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

# Negative controls rerun the same predicates after mutation.
string(REPLACE "case SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE:" "case SERVER_TASK_TYPE_COMPLETION:"
    bad_context "${context}")
scheduler_contract_valid("${bad_context}" bad_scheduler_valid)
if(bad_scheduler_valid)
    message(FATAL_ERROR "scheduler negative control did not mutate")
endif()
set(bad_task_construction
    "${all_server_source}\nserver_task copied = server_task(SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE);")
string(REGEX MATCH
    "server_task[^;\n]*\\([^\\)]*SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE"
    bad_task_match "${bad_task_construction}")
if(bad_task_match STREQUAL "")
    message(FATAL_ERROR "copy-init task-construction negative control did not trip")
endif()
set(bad_task_assignment
    "${all_server_source}\nserver_task assigned; assigned.type = SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE;")
string(REGEX MATCH
    "\\.type[ \t\r\n]*=[ \t\r\n]*SERVER_TASK_TYPE_CACHE_LEASE_ACQUIRE"
    bad_task_assignment_match "${bad_task_assignment}")
if(bad_task_assignment_match STREQUAL "")
    message(FATAL_ERROR "task-assignment negative control did not trip")
endif()
string(REPLACE "!leases[i].explicit_hard" "true"
    bad_leases "${leases}")
explicit_lifetime_valid("${bad_leases}" bad_lifetime_valid)
if(bad_lifetime_valid)
    message(FATAL_ERROR "explicit-only negative control did not mutate")
endif()
string(REPLACE "victim_catalog.recovery_pinned"
    "false /* recovery pin removed */" bad_task_source "${task_source}")
two_copy_contract_valid("${bad_task_source}" bad_two_copy_valid)
if(bad_two_copy_valid)
    message(FATAL_ERROR "two-copy negative control did not trip")
endif()
string(REPLACE "mark_subject_lost(" "mark_subject_missing_removed("
    bad_lease "${leases}")
subject_lost_contract_valid("${bad_lease}" "${control}" bad_lost_valid)
if(bad_lost_valid)
    message(FATAL_ERROR "subject-lost negative control did not trip")
endif()
string(REPLACE "retained_prefix == slot.prompt.tokens.size()"
    "retained_prefix != slot.prompt.tokens.size()"
    bad_append_context "${context}")
append_continuity_contract_valid("${bad_append_context}" bad_append_valid)
if(bad_append_valid)
    message(FATAL_ERROR "append-continuity negative control did not trip")
endif()
string(REPLACE "server_cache_lease_table::artifact_replaced("
    "server_cache_lease_table::artifact_replaced_removed("
    bad_replacement_leases "${leases}")
append_replacement_contract_valid(
    "${bad_replacement_leases}" bad_replacement_valid)
if(bad_replacement_valid)
    message(FATAL_ERROR "append-replacement negative control did not trip")
endif()
string(REPLACE "server_cache_lease_scope::from(implicit_soft_lease_scope)"
    "server_cache_lease_scope::from(lease_obs->new_context_scope())"
    bad_implicit_scope_context "${context}")
implicit_soft_scope_contract_valid(
    "${bad_implicit_scope_context}" bad_implicit_scope_valid)
if(bad_implicit_scope_valid)
    message(FATAL_ERROR "implicit-soft-scope negative control did not trip")
endif()
set(bad_headers "${all_server_headers}\nserver_cache_lease_table leases;")
string(REGEX MATCHALL "server_cache_lease_table leases"
    bad_table_members "${bad_headers}")
list(LENGTH bad_table_members n_bad_table_members)
if(NOT n_bad_table_members GREATER 1)
    message(FATAL_ERROR "single-table negative control did not trip")
endif()
set(bad_evaluator "${all_server_production}\nserver_cache_lease_evaluate_request()")
string(REGEX MATCHALL "server_cache_lease_evaluate_request\\(" bad_evaluators
    "${bad_evaluator}")
list(LENGTH bad_evaluators n_bad_evaluators)
if(NOT n_bad_evaluators GREATER 2)
    message(FATAL_ERROR "second-evaluator negative control did not trip")
endif()

message(STATUS "E1 control authority contract checks passed")
