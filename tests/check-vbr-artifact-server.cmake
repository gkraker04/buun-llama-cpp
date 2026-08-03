# F3.3 server-route contract. The route stays a typed, explicit scheduler
# operation; the server cannot substitute its own generation identity or fall
# back to legacy state-file export. Negative controls mutate source text only.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/contract-scan-utils.cmake")

file(READ "${SOURCE_ROOT}/tools/server/server-vbr-artifact-store.h" store_header)
file(READ "${SOURCE_ROOT}/tools/server/server-vbr-artifact-store.cpp" store_source)
file(READ "${SOURCE_ROOT}/tools/server/server-task.h" task_header)
file(READ "${SOURCE_ROOT}/tools/server/server-task.cpp" task_source)
file(READ "${SOURCE_ROOT}/tools/server/server-context.cpp" context_source)
file(READ "${SOURCE_ROOT}/src/llama-vbr-identity-digest.h" identity_header)
file(READ "${SOURCE_ROOT}/src/llama-vbr-checkpoint-compose.inc" checkpoint_compose)
file(READ "${SOURCE_ROOT}/src/llama-vbr-explicit-capture.cpp" capture_source)

string(REGEX MATCH
    "enum class server_vbr_artifact_capture_status[^\\{]*\\{[^}]*\\}"
    status_enum "${store_header}")
if (status_enum STREQUAL "")
    message(FATAL_ERROR "F3.3 closed server capture status enum is missing")
endif()
string(FIND "${status_enum}" "_count" status_count)
if (status_count EQUAL -1)
    message(FATAL_ERROR "F3.3 server capture status lacks _count")
endif()

foreach(required IN ITEMS
        "SERVER_TASK_TYPE_CACHE_CAPTURE"
        "case SERVER_TASK_TYPE_CACHE_CAPTURE:"
        "action == \"capture\""
        "if (!vbr_artifact_store)"
        "VBR_ARTIFACT_CAPTURE begin"
        "VBR_ARTIFACT_CAPTURE end"
        "library_status=%s"
        "phase=%s"
        "inner_status=%s"
        "generation_failure=%s"
        "size_failure=%s"
        "reservation_group=%s"
        "admission_status=%s"
        "ring_failure=%s"
        "constructed_ring=%")
    string(FIND "${task_header}${context_source}" "${required}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "F3.3 route contract is missing '${required}'")
    endif()
endforeach()

string(FIND "${capture_source}"
    "source.backend != lane.binding.backend" exact_backend_binding)
if (NOT exact_backend_binding EQUAL -1)
    message(FATAL_ERROR
        "F3.3 ring must bind lazy VBR side streams by physical device, not backend handle")
endif()

string(FIND "${context_source}"
    "case SERVER_TASK_TYPE_CACHE_CAPTURE:" capture_case_begin)
string(FIND "${context_source}"
    "case SERVER_TASK_TYPE_SLOT_RESTORE:" capture_case_end)
if (capture_case_begin EQUAL -1 OR capture_case_end EQUAL -1 OR
    NOT capture_case_begin LESS capture_case_end)
    message(FATAL_ERROR "F3.3 capture task is not isolated beside slot save")
endif()
math(EXPR capture_case_length "${capture_case_end} - ${capture_case_begin}")
string(SUBSTRING "${context_source}" ${capture_case_begin}
    ${capture_case_length} capture_case)
string(FIND "${context_source}"
    "bool build_capture_request(" capture_builder_begin)
string(FIND "${context_source}"
    "void process_single_task(" capture_builder_end)
if (capture_builder_begin EQUAL -1 OR capture_builder_end EQUAL -1 OR
    NOT capture_builder_begin LESS capture_builder_end)
    message(FATAL_ERROR "F3.3 capture request builder is missing")
endif()
math(EXPR capture_builder_length
    "${capture_builder_end} - ${capture_builder_begin}")
string(SUBSTRING "${context_source}" ${capture_builder_begin}
    ${capture_builder_length} capture_builder)
foreach(legacy_export IN ITEMS
        "llama_state_seq_save_file"
        "llama_state_seq_load_file"
        "slot_action.filepath")
    string(FIND "${capture_case}${capture_builder}"
        "${legacy_export}" legacy_found)
    if (NOT legacy_found EQUAL -1)
        message(FATAL_ERROR
            "F3.3 capture task reached legacy state-file token '${legacy_export}'")
    endif()
endforeach()

string(FIND "${context_source}"
    "request.identity_policy_order_digest =" server_identity_override)
if (NOT server_identity_override EQUAL -1)
    message(FATAL_ERROR
        "F3.3 server must not invent the canonical child-policy digest")
endif()
count_literal(
    "${identity_header}${checkpoint_compose}${capture_source}"
    "vbr checkpoint identity/policy/order digest v1"
    identity_domain_count)
if (NOT identity_domain_count EQUAL 1)
    message(FATAL_ERROR
        "checkpoint and artifact capture must share one identity digest recipe")
endif()

foreach(server_codec_token IN ITEMS
        "GGML_TYPE_"
        "TURBO_ROTATION_R"
        "ggml_turbo_meansub_table"
        "resolve_reference")
    string(FIND "${store_header}${store_source}"
        "${server_codec_token}" codec_token_found)
    if (NOT codec_token_found EQUAL -1)
        message(FATAL_ERROR
            "F3.3 server store owns forbidden codec/auth token '${server_codec_token}'")
    endif()
endforeach()

string(FIND "${task_source}"
    "json server_task_result_cache_capture::to_json()" capture_json_begin)
string(FIND "${task_source}"
    "json server_task_result_get_lora::to_json()" capture_json_end)
if (capture_json_begin EQUAL -1 OR capture_json_end EQUAL -1 OR
    NOT capture_json_begin LESS capture_json_end)
    message(FATAL_ERROR "F3.3 typed result serializer is missing")
endif()
math(EXPR capture_json_length "${capture_json_end} - ${capture_json_begin}")
string(SUBSTRING "${task_source}" ${capture_json_begin}
    ${capture_json_length} capture_json)
foreach(raw_field IN ITEMS "\"filename\"" "\"filepath\"" "\"data\"")
    string(FIND "${capture_json}" "${raw_field}" raw_found)
    if (NOT raw_found EQUAL -1)
        message(FATAL_ERROR
            "F3.3 typed result leaked raw storage field ${raw_field}")
    endif()
endforeach()

set(legacy_negative
    "${capture_case}${capture_builder}\nllama_state_seq_save_file(ctx, path, 0, nullptr, 0);")
string(FIND "${legacy_negative}" "llama_state_seq_save_file"
    legacy_negative_found)
if (legacy_negative_found EQUAL -1)
    message(FATAL_ERROR "F3.3 legacy-export negative control did not trip")
endif()

message(STATUS "F3.3 VBR artifact server contracts passed")
