if(NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/common/common-cache-family.h" family_header)
file(READ "${SOURCE_ROOT}/common/common-checkpoint-shadow.cpp" checkpoint_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" task_header)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/server-context.h" context_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.h" control_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.cpp" control_source)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)

function(require_token source token label)
    string(FIND "${source}" "${token}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "E1.1b ${label}: missing '${token}'")
    endif()
endfunction()

function(forbid_token source token label)
    string(FIND "${source}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "E1.1b ${label}: forbidden '${token}'")
    endif()
endfunction()

function(extract_region source begin_token end_token output)
    string(FIND "${source}" "${begin_token}" begin)
    string(FIND "${source}" "${end_token}" end)
    if(begin EQUAL -1 OR end EQUAL -1 OR NOT end GREATER begin)
        message(FATAL_ERROR
            "E1.1b region missing: '${begin_token}' .. '${end_token}'")
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} region)
    set(${output} "${region}" PARENT_SCOPE)
endfunction()

function(checkpoint_carrier_valid header source output)
    string(FIND "${header}"
        "common_cache_family_binding cache_family;" field)
    extract_region("${source}"
        "common_prompt_checkpoint::common_prompt_checkpoint(const common_prompt_checkpoint & other)"
        "common_prompt_checkpoint & common_prompt_checkpoint::operator=" copy_ctor)
    extract_region("${source}"
        "void common_prompt_checkpoint::clear()"
        "common_checkpoint_shadow_reason common_checkpoint_shadow_capture_scoped" clear_body)
    string(FIND "${copy_ctor}" "cache_family(other.cache_family)" copied)
    string(FIND "${clear_body}" "cache_family = {};" cleared)
    if(field EQUAL -1 OR copied EQUAL -1 OR cleared EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(propagation_valid task control context output)
    foreach(token
            "server_cache_control_token cache_family_binding_token;"
            "server_cache_control_operation::family_register"
            "server_cache_control_operation::family_bind"
            "resolve_family_binding("
            "server_cache_family_resolve_for_launch("
            "server_prompt_cache_apply_family("
            "copy.cache_family_binding_token = cache_family_binding_token;"
            "common_cache_family_follow_lineage("
            "other.cache_family = cache_family;"
            "next.cache_family = slot.cache_family;"
            "common_cache_family_binding restored_family = cache_family;"
            "*restored_family = delivery.cache_family;"
            "state->slot->cache_family = {};"
            "if (prompt.tokens.empty())")
        set(joined "${task}\n${control}\n${context}")
        string(FIND "${joined}" "${token}" found)
        if(found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

function(single_carrier_valid task context output)
    string(FIND "${task}"
        "common_cache_family_allows_additional_weight("
        callback_guard)
    string(FIND "${context}"
        "server_prompt_cache_apply_family("
        save_write)
    if(callback_guard EQUAL -1 OR save_write EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(lineage_boundary_valid source output)
    string(FIND "${source}"
        "retained_prefix == current_tokens" full_prefix)
    if(full_prefix EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
    else()
        set(${output} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(task_strong_binding_absent task output)
    string(FIND "${task}"
        "common_cache_family_binding cache_family;" found)
    if(found EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(test_door_call_absent source output)
    string(FIND "${source}"
        "server_cache_family_slot_round_trip_for_test(" found)
    if(found EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

checkpoint_carrier_valid(
    "${common_header}" "${checkpoint_source}" carrier_ok)
if(NOT carrier_ok)
    message(FATAL_ERROR "E1.1b checkpoint copy/clear carrier contract failed")
endif()

propagation_valid(
    "${task_header}\n${task_source}" "${control_header}\n${control_source}"
    "${context_source}" propagation_ok)
if(NOT propagation_ok)
    message(FATAL_ERROR "E1.1b propagation contract failed")
endif()

single_carrier_valid("${task_source}" "${context_source}" carrier_price_ok)
if(NOT carrier_price_ok)
    message(FATAL_ERROR "E1.1b single pricing carrier contract failed")
endif()

lineage_boundary_valid("${family_header}" lineage_boundary_ok)
if(NOT lineage_boundary_ok)
    message(FATAL_ERROR "E1.1b append-stable lineage boundary missing")
endif()

require_token("${control_source}" "holder.family_bindings"
    "holder-owned binding resolution")
extract_region("${task_header}" "struct server_task {"
    "struct server_task_result" task_struct)
task_strong_binding_absent("${task_struct}" task_strong_absent)
if(NOT task_strong_absent)
    message(FATAL_ERROR "E1.1b task carries injectable strong binding")
endif()
string(REGEX MATCHALL "cache_family = [{][}]" family_clears
    "${context_source}")
list(LENGTH family_clears family_clear_count)
if(NOT family_clear_count EQUAL 4)
    message(FATAL_ERROR
        "E1.1b slot family-clear census changed: ${family_clear_count} != 4")
endif()
forbid_token("${context_source}" "POST /cache/families"
    "no E1.2 family route")

require_token("${context_header}"
    "server_cache_family_slot_round_trip_for_test("
    "actual-slot regression door")
file(GLOB server_production "${SOURCE_ROOT}/tools/server/*.cpp")
set(slot_test_production "")
foreach(path IN LISTS server_production)
    if(path STREQUAL "${SOURCE_ROOT}/tools/server/server-context.cpp")
        continue()
    endif()
    file(READ "${path}" production)
    string(APPEND slot_test_production "\n${production}")
endforeach()
test_door_call_absent("${slot_test_production}" test_door_absent)
if(NOT test_door_absent)
    message(FATAL_ERROR
        "E1.1b test-only slot door has production callers")
endif()

# House-standard negative controls rerun the actual predicates.
string(REPLACE "cache_family(other.cache_family)"
    "cache_family()" bad_copy "${checkpoint_source}")
checkpoint_carrier_valid("${common_header}" "${bad_copy}" bad_copy_ok)
if(bad_copy_ok)
    message(FATAL_ERROR "checkpoint-copy negative control did not trip")
endif()

string(REPLACE "cache_family = {};"
    "/* family clear removed */" bad_clear "${checkpoint_source}")
checkpoint_carrier_valid("${common_header}" "${bad_clear}" bad_clear_ok)
if(bad_clear_ok)
    message(FATAL_ERROR "checkpoint-clear negative control did not trip")
endif()

string(REPLACE "next.cache_family = slot.cache_family;"
    "/* checkpoint propagation removed */" bad_context "${context_source}")
propagation_valid(
    "${task_header}\n${task_source}" "${control_header}\n${control_source}"
    "${bad_context}" bad_propagation_ok)
if(bad_propagation_ok)
    message(FATAL_ERROR "propagation negative control did not trip")
endif()

string(REPLACE "copy.cache_family_binding_token = cache_family_binding_token;"
    "/* child token propagation removed */" bad_task_header "${task_header}")
propagation_valid(
    "${bad_task_header}\n${task_source}"
    "${control_header}\n${control_source}"
    "${context_source}" bad_child_ok)
if(bad_child_ok)
    message(FATAL_ERROR "child-token negative control did not trip")
endif()

string(REPLACE "other.cache_family = cache_family;"
    "/* child slot provenance removed */" bad_child_context "${context_source}")
propagation_valid(
    "${task_header}\n${task_source}"
    "${control_header}\n${control_source}"
    "${bad_child_context}" bad_child_slot_ok)
if(bad_child_slot_ok)
    message(FATAL_ERROR "child-slot negative control did not trip")
endif()

string(REPLACE "state->slot->cache_family = {};"
    "/* import provenance reset removed */" bad_import "${context_source}")
propagation_valid(
    "${task_header}\n${task_source}"
    "${control_header}\n${control_source}"
    "${bad_import}" bad_import_ok)
if(bad_import_ok)
    message(FATAL_ERROR "import-reset negative control did not trip")
endif()

string(REPLACE "if (prompt.tokens.empty())"
    "if (false)" bad_empty_terminal "${context_source}")
propagation_valid(
    "${task_header}\n${task_source}"
    "${control_header}\n${control_source}"
    "${bad_empty_terminal}" bad_empty_ok)
if(bad_empty_ok)
    message(FATAL_ERROR "empty-terminal negative control did not trip")
endif()

string(REPLACE "*restored_family = delivery.cache_family;"
    "/* restored family provenance removed */" bad_restore "${task_source}")
propagation_valid(
    "${task_header}\n${bad_restore}" "${control_header}\n${control_source}"
    "${context_source}" bad_restore_ok)
if(bad_restore_ok)
    message(FATAL_ERROR "restore-provenance negative control did not trip")
endif()

string(REPLACE "common_cache_family_allows_additional_weight("
    "common_cache_family_additional_weight_removed("
    bad_task "${task_source}")
single_carrier_valid("${bad_task}" "${context_source}" bad_price_ok)
if(bad_price_ok)
    message(FATAL_ERROR "single-carrier negative control did not trip")
endif()

set(bad_task_struct
    "${task_struct}\ncommon_cache_family_binding cache_family;")
task_strong_binding_absent("${bad_task_struct}" bad_task_absent)
if(bad_task_absent)
    message(FATAL_ERROR "task-strong-binding negative control did not trip")
endif()

string(REPLACE "retained_prefix == current_tokens"
    "retained_prefix != 0" bad_family_header "${family_header}")
lineage_boundary_valid("${bad_family_header}" bad_lineage_boundary_ok)
if(bad_lineage_boundary_ok)
    message(FATAL_ERROR "lineage-boundary negative control did not trip")
endif()

set(bad_production
    "${slot_test_production}\nserver_cache_family_slot_round_trip_for_test(authority, token);")
test_door_call_absent("${bad_production}" bad_test_door_absent)
if(bad_test_door_absent)
    message(FATAL_ERROR "test-door caller negative control did not trip")
endif()

message(STATUS "E1.1b declared-family propagation contracts passed")
