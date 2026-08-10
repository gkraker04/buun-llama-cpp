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
file(READ "${SOURCE_ROOT}/src/llama-vbr-explicit-capture.cpp" capture_source)
file(READ "${SOURCE_ROOT}/src/llama-kv-cache.cpp" kv_cache_source)
file(READ "${SOURCE_ROOT}/src/llama-memory-recurrent.cpp" recurrent_source)
file(READ "${SOURCE_ROOT}/src/llama-graph.cpp" graph_source)

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
        "SERVER_TASK_TYPE_CACHE_IMPORT"
        "case SERVER_TASK_TYPE_CACHE_CAPTURE:"
        "case SERVER_TASK_TYPE_CACHE_IMPORT:"
        "action == \"capture\""
        "action == \"import\""
        "if (!vbr_artifact_store)"
        "VBR_ARTIFACT_CAPTURE begin"
        "VBR_ARTIFACT_CAPTURE end"
        "VBR_ARTIFACT_IMPORT begin"
        "VBR_ARTIFACT_IMPORT end"
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
    "${identity_header}${capture_source}"
    "vbr checkpoint identity/policy/order digest v1"
    identity_domain_count)
if (NOT identity_domain_count EQUAL 1)
    message(FATAL_ERROR
        "checkpoint and artifact capture must share one identity digest recipe")
endif()

foreach(server_codec_token IN ITEMS
        "GGML_TYPE_"
        "TURBO_ROTATION_R"
        "ggml_turbo_meansub_table")
    string(FIND "${store_header}${store_source}"
        "${server_codec_token}" codec_token_found)
    if (NOT codec_token_found EQUAL -1)
        message(FATAL_ERROR
            "F3.3 server store owns forbidden codec/auth token '${server_codec_token}'")
    endif()
endforeach()

# F4.3 deliberately restores catalog resolution in this server-only owner.
# The opaque handle must first match the capture-time tenant binding; a wrong
# tenant and a missing token share the same closed not_found terminal.
foreach(import_auth IN ITEMS
        "found->second.tenant_key != tenant_key"
        "impl_->references.authorize("
        "server_vbr_artifact_import_status::not_found"
        "impl_->catalog.resolve_reference("
        "vbr_validate_unit_manifest("
        "vbr_stage_validated_manifest("
        "vbr_adopt_empty_manifest(")
    string(FIND "${store_source}" "${import_auth}" import_auth_found)
    if (import_auth_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 authenticated import pipeline is missing '${import_auth}'")
    endif()
endforeach()

string(FIND "${context_source}"
    "case SERVER_TASK_TYPE_CACHE_IMPORT:" import_case_begin)
string(FIND "${context_source}"
    "case SERVER_TASK_TYPE_SLOT_RESTORE:" import_case_end)
if (import_case_begin EQUAL -1 OR import_case_end EQUAL -1 OR
    NOT import_case_begin LESS import_case_end)
    message(FATAL_ERROR "F4.3 import task boundary is missing")
endif()
math(EXPR import_case_length "${import_case_end} - ${import_case_begin}")
string(SUBSTRING "${context_source}" ${import_case_begin}
    ${import_case_length} import_case)
string(FIND "${import_case}"
    "server_vbr_artifact_import_route_precheck(" import_precheck)
if (import_precheck EQUAL -1)
    message(FATAL_ERROR "F4.3 typed empty-slot precheck is missing")
endif()
foreach(forbidden_import_mutation IN ITEMS
        "prompt_clear("
        "llama_memory_seq_rm("
        "common_context_seq_rm(")
    string(FIND "${import_case}" "${forbidden_import_mutation}"
        forbidden_import_found)
    if (NOT forbidden_import_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 import route silently destroys target state via '${forbidden_import_mutation}'")
    endif()
endforeach()
set(import_mutation_negative "${import_case}\nprompt_clear();")
string(FIND "${import_mutation_negative}" "prompt_clear("
    import_mutation_negative_found)
if (import_mutation_negative_found EQUAL -1)
    message(FATAL_ERROR "F4.3 no-silent-erase negative control did not trip")
endif()

# Import publication must preserve the slot-owned prompt configuration exactly
# as SLOT_RESTORE does. Replacing the complete server_prompt with a default
# object loses decode-side configuration (notably the multimodal token-ledger
# mode) before the first continuation decode.
foreach(import_publish_pin IN ITEMS
        "state->prompt.tokens.has_mtmd ="
        "state->slot->prompt.tokens.has_mtmd"
        "swap(state->slot->prompt.tokens,"
        "state->slot->prompt.sequence_epoch =")
    string(FIND "${import_case}" "${import_publish_pin}"
        import_publish_found)
    if (import_publish_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 import prompt publication is missing '${import_publish_pin}'")
    endif()
endforeach()
string(FIND "${import_case}" "swap(state->slot->prompt,"
    whole_prompt_swap)
if (NOT whole_prompt_swap EQUAL -1)
    message(FATAL_ERROR
        "F4.3 import must not replace the slot-owned server_prompt")
endif()
set(import_publish_negative "${import_case}\nswap(state->slot->prompt, state->prompt);")
string(FIND "${import_publish_negative}" "swap(state->slot->prompt,"
    import_publish_negative_found)
if (import_publish_negative_found EQUAL -1)
    message(FATAL_ERROR
        "F4.3 prompt-publication negative control did not trip")
endif()

# The conservative live-image receipt is held only while imported cells remain
# live. Ordinary slot erase reaches llama_kv_cache::seq_rm rather than clear(),
# so the empty transition must retire the receipt there or a second import into
# the construction-empty cache is falsely refused at live_image_prepare.
foreach(receipt_pin IN ITEMS
        "void llama_kv_cache::vbr_import_receipts_release_if_empty() noexcept"
        "cells.get_used() == 0"
        "vbr_import_receipts_release_if_empty();")
    string(FIND "${kv_cache_source}" "${receipt_pin}" receipt_found)
    if (receipt_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 import receipt lifecycle is missing '${receipt_pin}'")
    endif()
endforeach()
set(receipt_negative "${kv_cache_source}")
string(REPLACE "vbr_import_receipts_release_if_empty();"
    "/* receipt release removed */" receipt_negative "${receipt_negative}")
string(FIND "${receipt_negative}" "vbr_import_receipts_release_if_empty();"
    receipt_negative_found)
if (NOT receipt_negative_found EQUAL -1)
    message(FATAL_ERROR
        "F4.3 receipt-lifecycle negative control did not trip")
endif()

# Artifact adoption swaps the recurrent companion's concrete tensor and buffer
# owners. A live server may still have a shape-compatible graph cached against
# the previous owners, so every recurrent graph-reuse door must also cite the
# binding epoch published by the companion swap.
foreach(binding_pin IN ITEMS
        "target->bump_tensor_binding_epoch();"
        "inp->tensor_binding_epoch = mctx_cur->get_tensor_binding_epoch();"
        "tensor_binding_epoch == mctx->get_tensor_binding_epoch()"
        "inp_rs->tensor_binding_epoch ==")
    string(FIND "${recurrent_source}${graph_source}" "${binding_pin}"
        binding_pin_found)
    if (binding_pin_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 recurrent graph-binding fence is missing '${binding_pin}'")
    endif()
endforeach()
count_literal("${graph_source}" "get_tensor_binding_epoch()"
    binding_epoch_consumer_count)
if (NOT binding_epoch_consumer_count EQUAL 5)
    message(FATAL_ERROR
        "F4.3 recurrent binding epoch must fence standalone + 3 hybrid reuse doors and initialize the graph")
endif()
set(binding_negative "${recurrent_source}")
string(REPLACE "target->bump_tensor_binding_epoch();"
    "/* recurrent binding epoch publication removed */"
    binding_negative "${binding_negative}")
string(FIND "${binding_negative}" "target->bump_tensor_binding_epoch();"
    binding_negative_found)
if (NOT binding_negative_found EQUAL -1)
    message(FATAL_ERROR
        "F4.3 recurrent graph-binding negative control did not trip")
endif()

# ORDERING pin, not just presence: authorization must precede the first catalog
# resolution or the route becomes an existence oracle. Both tokens are unique
# call sites in the store source.
string(FIND "${store_source}" "impl_->references.authorize(" authorize_pos)
string(FIND "${store_source}" "impl_->catalog.resolve_reference(" resolve_pos)
if (NOT authorize_pos LESS resolve_pos)
    message(FATAL_ERROR
        "F4.3 authorization must precede catalog resolution (existence oracle)")
endif()

set(auth_negative "${store_source}")
string(REPLACE
    "found->second.tenant_key != tenant_key"
    "false"
    auth_negative "${auth_negative}")
string(FIND "${auth_negative}"
    "found->second.tenant_key != tenant_key"
    auth_negative_found)
if (NOT auth_negative_found EQUAL -1)
    message(FATAL_ERROR "F4.3 tenant-auth negative control did not trip")
endif()

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
foreach(import_result_field IN ITEMS
        "\"validation_status\""
        "\"stage_status\""
        "\"downward_reserve_status\""
        "\"adopt_status\""
        "\"phase\""
        "\"downward_subphase\""
        "\"downward_edge\""
        "\"decision\""
        "\"consistency\"")
    string(FIND "${capture_json}" "${import_result_field}"
        import_result_found)
    if (import_result_found EQUAL -1)
        message(FATAL_ERROR
            "F4.3 typed import result is missing ${import_result_field}")
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
