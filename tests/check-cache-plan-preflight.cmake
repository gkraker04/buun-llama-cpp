if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-plan-preflight.cpp" preflight_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/server-http.cpp" http_source)
file(READ "${SOURCE_ROOT}/tools/server/server-http.h" http_header)
file(READ "${SOURCE_ROOT}/tools/server/server.cpp" server_source)
file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/common/arg.cpp" arg_source)

function(extract_between source begin_token end_token output ok)
    string(FIND "${source}" "${begin_token}" begin)
    string(FIND "${source}" "${end_token}" end)
    if (begin EQUAL -1 OR end LESS_EQUAL begin)
        set(${output} "" PARENT_SCOPE)
        set(${ok} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} body)
    set(${output} "${body}" PARENT_SCOPE)
    set(${ok} TRUE PARENT_SCOPE)
endfunction()

function(preflight_contract_valid source output)
    string(FIND "${source}"
        "server_cache_plan_preflight_view cache_plan_preflight(" begin)
    string(FIND "${source}" "bool build_capture_request(" end)
    if (begin EQUAL -1 OR end LESS_EQUAL begin)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} body)
    foreach(required
            "assert_scheduler_thread("
            "cache_plan_preflight_scheduler_thread,"
            "cache_plan_stage1_mode_for(plan_authority, true)"
            "server_cache_plan_preflight_build_view(")
        string(FIND "${body}" "${required}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    foreach(forbidden
            "destruction_quote_sequence"
            "quote_samples"
            "cache_plan_obs"
            "server_cache_plan_disarm_unlaunched("
            "finalize_execution("
            "authorize("
            "server_cache_prepare_release_set("
            "server_cache_prepared_release_capability"
            "server_cache_recovery_pin::acquire("
            "prompt_save("
            "prompt_load("
            "prompt_clear")
        string(FIND "${body}" "${forbidden}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

preflight_contract_valid("${context_source}" contract_valid)
if (NOT contract_valid)
    message(FATAL_ERROR "E0.1 read-only preflight contract failed")
endif()

string(REPLACE
    "server_cache_plan_preflight_view view;"
    "server_cache_plan_preflight_view view; cache_plan_authority->finalize_execution(*(common_cache_plan_record *) nullptr);"
    mutation_negative "${context_source}")
preflight_contract_valid("${mutation_negative}" mutation_negative_valid)
if (mutation_negative_valid)
    message(FATAL_ERROR "E0.1 mutation negative control did not trip")
endif()

function(stage1_mode_wiring_valid source output)
    extract_between("${source}"
        "void cache_plan_inventory_and_plan_before_mutation("
        "server_slot * get_available_slot("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(required
            "common_cache_plan_destruction_counters throwaway_destruction_counters;"
            "auto * destruction_counters = mode.preflight\n            ? &throwaway_destruction_counters"
            "!mode.preflight && cache_authority\n                ? &cache_authority->destruction_quote_sequence"
            "const bool quote_lifecycle_available = mode.preflight\n            ? preview_lifecycle_available\n            : true;"
            "if (destruction_counters &&"
            "mode.preflight,"
            "*destruction_counters, &source_registry")
        string(FIND "${body}" "${required}" found_pin)
        if (found_pin EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

stage1_mode_wiring_valid("${context_source}" wiring_valid)
if (NOT wiring_valid)
    message(FATAL_ERROR "E0.1 throwaway/sequence mode wiring drifted")
endif()
string(REPLACE
    "? preview_lifecycle_available\n            : true;"
    "? preview_lifecycle_available\n            : preview_lifecycle_available;"
    lifecycle_negative "${context_source}")
stage1_mode_wiring_valid("${lifecycle_negative}" lifecycle_negative_valid)
if (lifecycle_negative_valid)
    message(FATAL_ERROR "E0.1 real-path lifecycle literal negative control did not trip")
endif()
string(REPLACE
    "? &throwaway_destruction_counters"
    "? &cache_authority->destruction_counters"
    counter_negative "${context_source}")
stage1_mode_wiring_valid("${counter_negative}" counter_negative_valid)
if (counter_negative_valid)
    message(FATAL_ERROR "E0.1 production-counter use-site negative control did not trip")
endif()

function(shared_kernel_read_only source output)
    extract_between("${source}"
        "bool cache_plan_inventory_live_rows("
        "server_slot * get_available_slot("
        body found)
    if (NOT found)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(forbidden
            "cache_plan_begin_inventory("
            "prompt_cache->update("
            "prompt_cache->states.erase"
            "prompt_cache->states.emplace"
            "prompt_save("
            "prompt_load("
            "prompt_clear("
            "server_prompt_cache_destroy_entry"
            "cache_plan_source_id =")
        string(FIND "${body}" "${forbidden}" found_forbidden)
        if (NOT found_forbidden EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

shared_kernel_read_only("${context_source}" kernel_read_only)
if (NOT kernel_read_only)
    message(FATAL_ERROR "E0.1 shared stage-1 kernel gained a prompt-cache mutation")
endif()
string(REPLACE
    "GGML_ASSERT(mode.plan_authority);"
    "GGML_ASSERT(mode.plan_authority); prompt_cache->cache_plan_begin_inventory();"
    prompt_mutation_negative "${context_source}")
shared_kernel_read_only("${prompt_mutation_negative}" prompt_mutation_valid)
if (prompt_mutation_valid)
    message(FATAL_ERROR "E0.1 prompt-cache mutation negative control did not trip")
endif()

foreach(registry_pin
        "source_registry.get("
        "find_host_source("
        "server_cache_plan_local_source_registry local_;")
    string(FIND "${context_source}" "${registry_pin}" registry_pin_pos)
    if (registry_pin_pos EQUAL -1)
        message(FATAL_ERROR
            "E0.1 source registry missed a consumer: '${registry_pin}'")
    endif()
endforeach()

foreach(unminted_pin
        "bool preview_unminted = false;"
        "(options.admission_sequence == 0 && !options.preview_unminted)"
        "quote.receipt.admission_sequence == 0 ||")
    string(FIND "${context_source}${preflight_source}" "${unminted_pin}"
        unminted_pos)
    if (unminted_pos EQUAL -1)
        file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.h"
            quote_header)
        file(READ "${SOURCE_ROOT}/tools/server/server-cache-destruction-quote.cpp"
            quote_source)
        string(FIND "${quote_header}${quote_source}" "${unminted_pin}"
            unminted_pos)
    endif()
    if (unminted_pos EQUAL -1)
        message(FATAL_ERROR "E0.1 unminted receipt contract lost '${unminted_pin}'")
    endif()
endforeach()

foreach(forbidden_tu
        "server_cache_prepare_release_set"
        "server_cache_prepared_release_capability"
        "server_cache_recovery_pin"
        "prompt_save"
        "prompt_load"
        "prompt_clear"
        "finalize_execution"
        "destruction_quote_sequence")
    string(FIND "${preflight_source}" "${forbidden_tu}" forbidden_pos)
    if (NOT forbidden_pos EQUAL -1)
        message(FATAL_ERROR
            "E0.1 pure projection linked mutation/capability symbol '${forbidden_tu}'")
    endif()
endforeach()

string(FIND "${task_source}"
    "json server_task_result_cache_plan_preflight::to_json()" result_json)
string(FIND "${task_source}"
    "return server_cache_plan_preflight_json(view);" public_json)
if (result_json EQUAL -1 OR public_json LESS_EQUAL result_json)
    message(FATAL_ERROR "E0.2 task result lost the redacted production serializer")
endif()

extract_between("${context_source}"
    "this->post_cache_plan ="
    "this->get_props ="
    route_body route_found)
if (NOT route_found)
    message(FATAL_ERROR "E0.2 POST /cache/plan handler missing")
endif()
foreach(required
        "res->headers[\"Cache-Control\"] = \"no-store\";"
        "if (!params.cache_plan_preflight)"
        "server_cache_plan_preflight_exposure_allowed("
        "SERVER_TASK_TYPE_CACHE_PLAN_PREFLIGHT"
        "res->rd.post_task(std::move(task));"
        "server_task_result_cache_plan_preflight"
        "server_cache_plan_preflight_request_field_allowed("
        "!json_is_array_and_contains_numbers(prompt)"
        "cache-plan preflight message_delimiters must be an array"
        "inputs.size() != 1")
    string(FIND "${route_body}" "${required}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "E0.2 route contract lost '${required}'")
    endif()
endforeach()
foreach(forbidden
        "cache_plan_preflight(task)"
        "path_prompts_log_dir"
        "prompt_save("
        "prompt_load("
        "finalize_execution("
        "server_cache_prepare_release_set("
        "\"ticket\""
        "\"claim\""
        "\"preview_id\""
        "\"nonce\""
        "\"manifest_digest\""
        "\"artifact_id\""
        "n_predict"
        "sampling")
    string(FIND "${route_body}" "${forbidden}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR "E0.2 route admitted forbidden door/field '${forbidden}'")
    endif()
endforeach()
string(FIND "${route_body}" "res->headers[\"Cache-Control\"]" no_store_pos)
string(FIND "${route_body}" "if (!params.cache_plan_preflight)" gate_pos)
string(FIND "${route_body}" "json::parse(req.body)" parse_pos)
string(FIND "${route_body}" "res->rd.post_task(std::move(task));" queue_pos)
if (no_store_pos EQUAL -1 OR gate_pos LESS no_store_pos OR
    parse_pos LESS gate_pos OR queue_pos LESS parse_pos)
    message(FATAL_ERROR
        "E0.2 route order must be no-store -> flag gate -> parse -> task queue")
endif()

function(route_reserved_fields_absent source output)
    # Deliberate security-critical subset. The exhaustive canonical runtime
    # oracle is assert_redacted_keys() in test-cache-plan-preflight.cpp.
    foreach(reserved
            "\"ticket\"" "\"claim\"" "\"preview_id\""
            "\"nonce\"" "\"manifest_digest\"" "\"artifact_id\"")
        string(FIND "${source}" "${reserved}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
foreach(accepted_field
        "\"prompt\"" "\"id_slot\"" "\"cache_prompt\""
        "\"lora\"" "\"message_delimiters\"")
    string(FIND "${preflight_source}" "${accepted_field}" field_found)
    if (field_found EQUAL -1)
        message(FATAL_ERROR
            "E0.2 request allowlist lost ${accepted_field}")
    endif()
endforeach()
# Unknown/reserved fields are rejected by a closed allowlist. The negative
# control deliberately admits E1's ticket and must make this check fail.
string(REPLACE
    "\"message_delimiters\","
    "\"message_delimiters\", \"ticket\","
    route_allowlist_negative "${preflight_source}")
route_reserved_fields_absent(
    "${route_allowlist_negative}" route_allowlist_negative_ok)
if (route_allowlist_negative_ok)
    message(FATAL_ERROR "E0.2 request-allowlist negative control did not mutate")
endif()

function(route_prompt_log_free source output)
    string(FIND "${source}" "path_prompts_log_dir" prompt_log)
    if (prompt_log EQUAL -1)
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()
route_prompt_log_free("${route_body}" prompt_log_free)
if (NOT prompt_log_free)
    message(FATAL_ERROR "E0.2 route reaches the raw prompt log")
endif()
string(REPLACE
    "auto inputs = tokenize_input_prompts("
    "if (!params.path_prompts_log_dir.empty()) {} auto inputs = tokenize_input_prompts("
    prompt_log_negative "${route_body}")
route_prompt_log_free("${prompt_log_negative}" prompt_log_negative_ok)
if (prompt_log_negative_ok)
    message(FATAL_ERROR "E0.2 prompt-log bypass negative control did not trip")
endif()

extract_between("${preflight_source}"
    "json server_cache_plan_preflight_json("
    "bool server_cache_plan_preflight_exposure_allowed("
    serializer_body serializer_found)
if (NOT serializer_found)
    message(FATAL_ERROR "E0.2 redacted serializer missing")
endif()
foreach(required
        "{ \"authoritative\", false }"
        "{ \"reservation\", \"none\" }"
        "{ \"valid_until\", nullptr }"
        "{ \"estimate_scope\", \"cache_path_only\" }"
        "point_in_time"
        "no_reservation"
        "queue_and_contention_not_modeled"
        "post_generation_maintenance_not_modeled")
    string(FIND "${serializer_body}" "${required}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "E0.2 wire constant lost '${required}'")
    endif()
endforeach()
function(serializer_private_keys_absent source output)
    # Deliberate wire-key subset. The exhaustive canonical runtime oracle is
    # assert_redacted_keys() in test-cache-plan-preflight.cpp.
    foreach(private_key
            "\"target_slot_id\"" "\"source_id\""
            "\"candidate_id\"" "\"artifact_id\""
            "\"recovery_source\"" "\"manifest_digest\""
            "\"accounting_serial\"" "\"admission_sequence\""
            "\"topology_id\"" "\"domains\"" "\"journal_id\""
            "\"ticket\"" "\"claim\"" "\"nonce\""
            "\"preview_id\"" "\"valid_until_token\"")
        string(FIND "${source}" "${private_key}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
string(REPLACE
    "{ \"object\", \"cache_plan_preflight\" },"
    "{ \"object\", \"cache_plan_preflight\" }, { \"artifact_id\", 7 },"
    serializer_negative "${serializer_body}")
serializer_private_keys_absent(
    "${serializer_negative}" serializer_negative_redacted)
if (serializer_negative_redacted)
    message(FATAL_ERROR "E0.2 serializer redaction negative control did not trip")
endif()

foreach(exposure_pin
        "bool cache_plan_preflight = false;"
        "--cache-plan-preflight"
        "LLAMA_ARG_CACHE_PLAN_PREFLIGHT"
        "ctx_http.post(\"/cache/plan\""
        "server_cache_plan_preflight_exposure_allowed(")
    string(FIND
        "${common_header}${arg_source}${server_source}${preflight_source}"
        "${exposure_pin}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "E0.2 exposure pin lost '${exposure_pin}'")
    endif()
endforeach()

function(body_redaction_valid source output)
    foreach(required
            "\"/cache/plan\""
            "server_http_redacts_request_bodies(req.path)"
            "request:  [body redacted]"
            "response: [body redacted]")
        string(FIND "${source}" "${required}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
body_redaction_valid("${http_source}" redaction_ok)
if (NOT redaction_ok)
    message(FATAL_ERROR "E0.2 per-route request/response body redaction missing")
endif()
string(REPLACE
    "server_http_redacts_request_bodies(req.path)"
    "false"
    redaction_negative "${http_source}")
body_redaction_valid("${redaction_negative}" redaction_negative_ok)
if (redaction_negative_ok)
    message(FATAL_ERROR "E0.2 logger-redaction negative control did not trip")
endif()

function(route_no_store_valid source output)
    foreach(required
            "if (params.cache_receipt || params.cache_plan_preflight ||"
            "const bool cache_oracle_route ="
            "(cache_plan_preflight || cache_control_api) &&"
            "server_http_redacts_request_bodies(req.path)"
            "if (cache_oracle_route ||"
            "res.set_header(\"Cache-Control\", \"no-store\")")
        string(FIND "${source}" "${required}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
route_no_store_valid("${http_source}" route_no_store_ok)
if (NOT route_no_store_ok)
    message(FATAL_ERROR
        "E0.2 middleware-error responses lost their route-local no-store backstop")
endif()
string(REPLACE
    "(cache_plan_preflight || cache_control_api) &&"
    "(cache_plan_preflight && cache_control_api) &&"
    route_no_store_negative "${http_source}")
route_no_store_valid("${route_no_store_negative}" route_no_store_negative_ok)
if (route_no_store_negative_ok)
    message(FATAL_ERROR "E0.2 no-store negative control did not trip")
endif()

function(gcp_exclusion_valid header source output)
    foreach(required
            "return path != \"/cache/plan\" &&"
            "if (server_http_gcp_predict_dispatch_allowed(path))"
            "server_http_gcp_predict_dispatch_allowed(format) &&")
        string(FIND "${header}${source}" "${required}" found)
        if (found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
gcp_exclusion_valid("${http_header}" "${http_source}" gcp_exclusion_ok)
if (NOT gcp_exclusion_ok)
    message(FATAL_ERROR "E0.2 /predict dispatch exclusion missing")
endif()
string(REPLACE
    "return path != \"/cache/plan\" &&"
    "return true &&"
    gcp_header_negative "${http_header}")
gcp_exclusion_valid(
    "${gcp_header_negative}" "${http_source}" gcp_exclusion_negative_ok)
if (gcp_exclusion_negative_ok)
    message(FATAL_ERROR "E0.2 GCP-exclusion negative control did not trip")
endif()

string(FIND "${context_source}"
    "cache_plan_obs || cache_plan_authority || cache_authority ||\n            params_base.cache_plan_preflight"
    profile_gate)
string(FIND "${context_source}"
    "cache_plan_calibration_profile = common_cache_plan_calib_profile("
    profile_store)
foreach(profile_consumer
        "cache_plan_obs->calibration_profile =\n                        cache_plan_calibration_profile;"
        "cache_plan_authority->calibration_profile =\n                        cache_plan_calibration_profile;"
        "cache_authority->calibration_profile =\n                        cache_plan_calibration_profile;"
        "local_authority->calibration_profile =\n                cache_plan_calibration_profile;")
    string(FIND "${context_source}" "${profile_consumer}" consumer_found)
    if (consumer_found EQUAL -1)
        message(FATAL_ERROR
            "E0.2 profile consumer drifted from the shared source: ${profile_consumer}")
    endif()
endforeach()
if (profile_gate EQUAL -1 OR profile_store EQUAL -1)
    message(FATAL_ERROR "E0.2 authority-off init-only profile gate missing")
endif()

message(STATUS "E0.2 cache-plan preflight contracts passed")
