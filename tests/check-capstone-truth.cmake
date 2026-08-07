file(READ "${SOURCE_ROOT}/tools/server/bench/trace_common.py" common_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/capture-proxy.py" capture_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/trace-replay.py" replay_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/trace-compare.py" compare_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/zc0-light-gate.py" gate_src)

function(capstone_truth_contract common capture replay compare output)
    set(ok TRUE)
    foreach(token IN ITEMS
            "class SSEPayloadAccumulator"
            "\"truncated\", \"stop_type\""
            "merge_server_metrics")
        string(FIND "${common}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    foreach(token IN ITEMS
            "sse_reader = SSEPayloadAccumulator() if streaming else None"
            "for payload in sse_reader.feed(chunk)"
            "merge_server_metrics(server_metrics, payload)")
        string(FIND "${capture}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    foreach(token IN ITEMS
            "--claim-grade"
            "capture_pin_issues(records"
            "context_truncation_unreported"
            "observe_server_parallel(args)"
            "--serialize-overlap"
            "if isinstance(metrics.get(\"truncated\"), bool) else None")
        string(FIND "${replay}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    foreach(token IN ITEMS
            "--parity"
            "--null-concurrent"
            "--serialized-baseline"
            "--serialized-null"
            "--planner-evidence"
            "authority.get(\"state\") == \"authoritative\"")
        string(FIND "${compare}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    set(${output} "${ok}" PARENT_SCOPE)
endfunction()

function(capstone_gate_arms_ok gate output)
    set(ok TRUE)
    foreach(token IN ITEMS
            "branch-no-features"
            "baseline-concurrent-null"
            "baseline-serialized-null"
            "branch-active-unfitted"
            "branch-serialized-absent"
            "branch-serialized-explicit-off"
            "assert_serialized_identity(config_absent, config_off)"
            "[\"--cache-optimizer\", \"off\", \"--cache-debug\""
            "active-unfitted negative control unexpectedly claimed optimization")
        string(FIND "${gate}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    set(${output} "${ok}" PARENT_SCOPE)
endfunction()

capstone_gate_arms_ok("${gate_src}" gate_ok)
if (NOT gate_ok)
    message(FATAL_ERROR "ZC0 light gate lost a mandatory arm")
endif()
string(REPLACE "baseline-serialized-null" "baseline-repeat"
               bad_gate "${gate_src}")
capstone_gate_arms_ok("${bad_gate}" bad_gate_ok)
if (bad_gate_ok)
    message(FATAL_ERROR "ZC0 mandatory-arm negative control did not trip")
endif()

capstone_truth_contract("${common_src}" "${capture_src}" "${replay_src}" "${compare_src}" original_ok)
if (NOT original_ok)
    message(FATAL_ERROR "capstone claim-truth contract is incomplete")
endif()

# Negative controls prove the checker, not string(REPLACE), by rerunning the full contract.
string(REPLACE "\"truncated\", \"stop_type\"" "\"stop_type\"" bad_common "${common_src}")
capstone_truth_contract("${bad_common}" "${capture_src}" "${replay_src}" "${compare_src}" bad_common_ok)
if (bad_common_ok)
    message(FATAL_ERROR "capstone truncation negative control did not trip")
endif()

string(REPLACE "authority.get(\"state\") == \"authoritative\""
               "authority.get(\"state\") == \"shadow\""
               bad_compare "${compare_src}")
capstone_truth_contract("${common_src}" "${capture_src}" "${replay_src}" "${bad_compare}" bad_compare_ok)
if (bad_compare_ok)
    message(FATAL_ERROR "capstone authority-evidence negative control did not trip")
endif()

string(REPLACE "for payload in sse_reader.feed(chunk)"
               "for payload in []"
               bad_capture "${capture_src}")
capstone_truth_contract("${common_src}" "${bad_capture}" "${replay_src}" "${compare_src}" bad_capture_ok)
if (bad_capture_ok)
    message(FATAL_ERROR "capstone streaming-wiring negative control did not trip")
endif()
