# F4.1b is an evidence classifier, not an import implementation. Keep the
# validator translation unit unable to reach a target-write/adoption door;
# the negative control below proves this scan fails on a representative H2D
# mutation token.

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact-validate.h" validator_header)
file(READ "${SOURCE_ROOT}/src/llama-vbr-artifact-validate.cpp" validator_source)

foreach(required IN ITEMS
        "enum class vbr_import_decision"
        "enum class vbr_manifest_validation_status"
        "class vbr_validated_manifest"
        "std::unique_ptr<vbr_validated_manifest> proof"
        "vbr_validate_unit_manifest("
        "vbr_validate_unit_manifest_snapshot("
        "llama_memory_tree_collect(&target, tree)"
        "package.validate()"
        "package.retain(retained)")
    string(FIND "${validator_header}${validator_source}" "${required}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "F4.1b validator contract is missing '${required}'")
    endif()
endforeach()

foreach(forbidden IN ITEMS
        "ggml_backend_tensor_set"
        "ggml_backend_tensor_copy"
        "vbr_adopt_empty_manifest"
        "vbr_generation_begin"
        "vbr_operation_begin"
        "publish_unit("
        "materialize_and_commit("
        "reserve_if_serial("
        "gauge_set("
        "seq_rm("
        "seq_cp("
        "seq_add("
        "state_read(")
    string(FIND "${validator_source}" "${forbidden}" found)
    if (NOT found EQUAL -1)
        message(FATAL_ERROR
            "F4.1b validator reached forbidden target-write token '${forbidden}'")
    endif()
endforeach()

# The broad generation-isolation scan exempts this TU so it may copy captured
# generation tuples into the proof. Restore the other half of that fence here:
# live generation comparisons remain owned by checkpoint_vbr_eligibility().
# The validator may consume only the inspector's closed compatibility bits.
function(find_live_generation_comparisons text output)
    set(hits "")
    foreach(symbol IN ITEMS
            "->active()"
            "live.controllers"
            "current.tracker"
            "controller_generation()"
            "page_generation("
            "page_destructive_generation("
            "page_import_generation("
            "dependency_generation("
            "membership_generation("
            "unit_generation(")
        string(FIND "${text}" "${symbol}" found)
        if (NOT found EQUAL -1)
            list(APPEND hits "${symbol}")
        endif()
    endforeach()
    string(REGEX MATCH
        "(live|current)[^\n;]*global_generation" raw_global "${text}")
    if (raw_global)
        list(APPEND hits "raw-live-global_generation")
    endif()
    set(${output} "${hits}" PARENT_SCOPE)
endfunction()

find_live_generation_comparisons(
    "${validator_header}${validator_source}" live_comparison_hits)
if (live_comparison_hits)
    message(FATAL_ERROR
        "F4.1b validator grew forbidden live-generation comparisons: ${live_comparison_hits}")
endif()
foreach(snapshot_bit IN ITEMS
        "generation_compatible"
        "ownership_compatible"
        "stash_compatible")
    string(FIND "${validator_source}" "${snapshot_bit}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR
            "F4.1b validator stopped consuming opaque '${snapshot_bit}' evidence")
    endif()
endforeach()

string(FIND "${validator_header}"
    "vbr_validated_manifest() = default;" private_ctor)
string(FIND "${validator_header}" "private:" private_section)
if (private_ctor EQUAL -1 OR private_section EQUAL -1 OR
    private_ctor LESS private_section)
    message(FATAL_ERROR "F4.1b proof constructor is not private")
endif()

set(negative_source
    "${validator_source}\nggml_backend_tensor_set(target, bytes, 0, n);")
string(FIND "${negative_source}" "ggml_backend_tensor_set" negative_found)
if (negative_found EQUAL -1)
    message(FATAL_ERROR "F4.1b target-write negative control did not trip")
endif()

set(live_negative
    "${validator_source}\nif (current.tracker->active()) { auto g = current.tracker->controller_generation(); }")
find_live_generation_comparisons("${live_negative}" live_negative_hits)
if (NOT live_negative_hits)
    message(FATAL_ERROR
        "F4.1b live-generation-comparison negative control did not trip")
endif()

message(STATUS "F4.1b VBR artifact validator contracts passed")
