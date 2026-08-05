if(NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/common/common.h" common_header)
file(READ "${SOURCE_ROOT}/common/arg.cpp" arg_source)
file(READ "${SOURCE_ROOT}/tools/server/server.cpp" server_source)
file(READ "${SOURCE_ROOT}/tools/server/server-http.cpp" http_source)
file(READ "${SOURCE_ROOT}/tools/server/server-http.h" http_header)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.h" control_header)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control.cpp" control_source)
file(READ "${SOURCE_ROOT}/tools/server/server-cache-control-wire.cpp" wire_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/bench/fixtures/cache-control-v1.json" golden)

function(require_token source token label)
    string(FIND "${source}" "${token}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "E1.2 ${label}: missing '${token}'")
    endif()
endfunction()

function(forbid_token source token label)
    string(FIND "${source}" "${token}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "E1.2 ${label}: forbidden '${token}'")
    endif()
endfunction()

function(extract_region source begin_token end_token output)
    string(FIND "${source}" "${begin_token}" begin)
    string(FIND "${source}" "${end_token}" end)
    if(begin EQUAL -1 OR end LESS_EQUAL begin)
        message(FATAL_ERROR "E1.2 region missing: ${begin_token} .. ${end_token}")
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${source}" ${begin} ${length} region)
    set(${output} "${region}" PARENT_SCOPE)
endfunction()

# The header table is the only route list. The CMake checker derives its route
# census from that table, then verifies every mechanism consumes the table.
string(REGEX MATCHALL
    "[{] \"/cache/[^\"]+\", server_cache_control_operation::[a-z_]+ [}]"
    route_rows "${control_header}")
list(LENGTH route_rows route_count)
if(NOT route_count EQUAL 10)
    message(FATAL_ERROR "E1.2 route table census ${route_count} != 10")
endif()
set(routes "")
foreach(row IN LISTS route_rows)
    string(REGEX REPLACE
        "[{] \"([^\"]+)\",.*" "\\1" route "${row}")
    list(APPEND routes "${route}")
endforeach()

function(route_closure_valid header server http http_h context output)
    string(FIND "${header}" "SERVER_CACHE_CONTROL_ROUTES" table)
    string(FIND "${header}"
        "for (const auto & route : SERVER_CACHE_CONTROL_ROUTES)" operation)
    string(FIND "${server}"
        "for (const auto & route : SERVER_CACHE_CONTROL_ROUTES)" registration)
    string(FIND "${server}" "ctx_http.post(std::string(route.path)" post)
    string(FIND "${http}" "server_cache_control_is_route(path)" redaction)
    string(FIND "${http_h}" "!server_cache_control_is_route(path)" gcp)
    string(FIND "${context}"
        "server_cache_control_operation_for_path(req.path, operation)" dispatch)
    if(table EQUAL -1 OR registration EQUAL -1 OR post EQUAL -1 OR
            redaction EQUAL -1 OR gcp EQUAL -1 OR dispatch EQUAL -1 OR
            operation EQUAL -1)
        set(${output} FALSE PARENT_SCOPE)
        return()
    endif()
    set(${output} TRUE PARENT_SCOPE)
endfunction()

route_closure_valid(
    "${control_header}" "${server_source}" "${http_source}"
    "${http_header}" "${context_source}" closure_ok)
if(NOT closure_ok)
    message(FATAL_ERROR "E1.2 route-table closure rejected production source")
endif()

require_token("${common_header}" "bool cache_control_api = false;" "default-off flag")
require_token("${arg_source}" "{\"--cache-control-api\"}" "flag parser")
require_token("${server_source}" "if (params.cache_control_api)" "registration gate")
require_token("${server_source}" "!params.cache_lifecycle" "lifecycle startup gate")
require_token("${http_source}" "server_http_is_cache_control_route(req.path)"
    "disabled typed-refusal classifier")
require_token("${http_source}" "\"not_supported_error\"" "disabled typed refusal")

extract_region("${context_source}"
    "this->post_cache_control =" "this->get_props =" route_body)
require_token("${route_body}" "server_cache_control_prepare_request("
    "common parser before semantic resolution")
require_token("${route_body}" "server_cache_control_selector_field_allowed("
    "nested allowlist")
require_token("${route_body}" "res->headers[\"Cache-Control\"] = \"no-store\";"
    "route no-store")
require_token("${route_body}" "res->rd.post_task(std::move(task));" "scheduler queue")
require_token("${route_body}" "server_task::cache_control_task(operation)" "typed task factory")
require_token("${route_body}" "server_cache_capture_tenant_key(req)" "F tenant authorization")
require_token("${route_body}" "result->is_error()" "error-before-cast guard")
require_token("${route_body}" "res->status = 400;" "parse HTTP 400 convention")
require_token("${route_body}" "std::numeric_limits<int32_t>::max()" "slot range check")
require_token("${context_source}" "params_base.vbr_enabled()) {"
    "pre-E1.1c hard-live VBR refusal")

string(FIND "${route_body}"
    "const auto prepared = server_cache_control_prepare_request(" common_parse)
string(FIND "${route_body}" "valid = parse_selector(" selector_use)
string(FIND "${route_body}"
    "server_cache_control_selector_field_allowed(" nested_allowlist)
string(FIND "${route_body}" "tokenize_input_prompts(" tokenization)
if(common_parse EQUAL -1 OR selector_use EQUAL -1 OR nested_allowlist EQUAL -1 OR
        tokenization EQUAL -1 OR NOT common_parse LESS selector_use OR
        NOT nested_allowlist LESS tokenization)
    message(FATAL_ERROR "E1.2 allowlist no longer precedes selector tokenization")
endif()
foreach(forbidden
        "cache_control_authority->execute("
        "resolve_family_binding("
        "grant_hard_owned("
        "grant_soft("
        "prompt_save("
        "server_vbr_artifact_store::resolve_control_reference(")
    forbid_token("${route_body}" "${forbidden}" "HTTP-worker authority isolation")
endforeach()

set(serializer "${wire_source}")
foreach(forbidden artifact_id op_id manifest_digest accounting_serial retention_key tenant_key content)
    forbid_token("${serializer}" "\"${forbidden}\"" "serializer leak ${forbidden}")
endforeach()
require_token("${wire_source}" "{ \"object\", \"cache_control\" }" "wire object")
require_token("${wire_source}" "{ \"schema_version\", 1 }" "wire schema")
require_token("${wire_source}" "result.status != server_cache_control_status::ok"
    "refusal body early terminal")
require_token("${wire_source}" "return nullptr;" "absent fallback is null")
require_token("${wire_source}" "\"orphaned_leases\"" "reattach lease summary")
require_token("${wire_source}" "\"timestamp_ms\"" "event timestamp")
require_token("${wire_source}" "\"protected_bytes\"" "byte evidence")
require_token("${wire_source}" "server_cache_control_prepare_request("
    "testable pure parser")
forbid_token("${control_source}" "nlohmann" "authority JSON-free altitude")
forbid_token("${control_header}" "content," "content-handle enum removed")
forbid_token("${golden}" "\"content\"" "content-handle golden removed")
require_token("${golden}" "\"inspect\"" "inspect golden")

require_token("${control_source}" "scrub_replays(holder)" "expiry replay scrub")
require_token("${control_source}" "scrub_replays(*holder)" "close replay scrub")
require_token("${control_source}" "next_event_ordinal++" "per-holder event ordinals")
require_token("${control_source}" "family.label = request.family_label;" "family label storage")
require_token("${control_source}"
    "fallback == server_cache_control_status::not_found"
    "renew fallback-unavailable mapping")
require_token("${context_source}"
    "task.cache_control_fallback);\n                                if (selector_status ==\n                                        server_cache_control_status::not_found)"
    "scheduler renew fallback-unavailable mapping")

# Prompt-log bypass is structural: the route region has no prompt-log write.
forbid_token("${route_body}" "path_prompts_log_dir" "prompt-log bypass")

# House-standard negative controls rerun the closure and ordering predicates.
string(REPLACE
    "for (const auto & route : SERVER_CACHE_CONTROL_ROUTES)"
    "for (const auto & route : local_routes)"
    bad_server "${server_source}")
route_closure_valid(
    "${control_header}" "${bad_server}" "${http_source}"
    "${http_header}" "${context_source}" bad_closure)
if(bad_closure)
    message(FATAL_ERROR "E1.2 route closure negative control did not trip")
endif()

string(REPLACE "server_cache_control_prepare_request("
    "server_cache_control_prepare_unchecked("
    bad_context "${context_source}")
extract_region("${bad_context}"
    "this->post_cache_control =" "this->get_props =" bad_route)
string(FIND "${bad_route}" "server_cache_control_prepare_request(" bad_allowlist)
if(NOT bad_allowlist EQUAL -1)
    message(FATAL_ERROR "E1.2 allowlist negative control did not trip")
endif()

string(REPLACE "server_http_is_cache_control_route(req.path)" "false"
    bad_http "${http_source}")
string(FIND "${bad_http}" "server_http_is_cache_control_route(req.path)" bad_disabled)
if(NOT bad_disabled EQUAL -1)
    message(FATAL_ERROR "E1.2 disabled classifier negative control did not trip")
endif()

string(REPLACE "res->rd.post_task(std::move(task));"
    "cache_control_authority->execute(server_cache_control_operation::holder_close, *request); res->rd.post_task(std::move(task));"
    bad_direct "${context_source}")
extract_region("${bad_direct}"
    "this->post_cache_control =" "this->get_props =" bad_direct_route)
string(FIND "${bad_direct_route}" "cache_control_authority->execute(" direct_call)
if(direct_call EQUAL -1)
    message(FATAL_ERROR "E1.2 direct-authority negative control did not trip")
endif()

require_token("${task_source}" "server_cache_control_json(operation, result)"
    "production serializer")
message(STATUS "E1.2 HTTP contract passed (${route_count} table-driven routes)")
