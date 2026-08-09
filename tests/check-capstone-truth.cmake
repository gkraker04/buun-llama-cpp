file(READ "${SOURCE_ROOT}/tools/server/bench/trace_common.py" common_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/capture-proxy.py" capture_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/trace-replay.py" replay_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/trace-compare.py" compare_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/zc0-light-gate.py" gate_src)
file(READ "${SOURCE_ROOT}/tools/server/bench/zc6-capstone-report.py" zc6_report_src)
file(READ "${SOURCE_ROOT}/common/fit.cpp" fit_src)
file(READ "${SOURCE_ROOT}/tools/server/server.cpp" server_src)

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
			"--calibration-replay"
			"apply_calibration_replay_preset(args)"
			"real_server_admission_clocks"
			"recorded_inter_group_gaps_dropped"
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
            "\"--cache-optimizer\", \"off\", \"--cache-debug\""
            "active-unfitted negative control unexpectedly claimed optimization"
            "run_capstone_session(args)"
            "(\"baseline\", \"learn\", \"auto\")"
            "capstone arms must use one executable identity"
            "command = [\"/usr/bin/taskset\", \"-c\", \"0-11\", *command]"
            "\"cpu_affinity\": \"0-11\""
            "capstone arms must use identical non-mode server arguments"
            "--primary-order must be one permutation of baseline,learn,auto"
            "use only --capstone-extra-server-arg for a ZC6 capstone session"
            "class ResourceSampler"
            "class NvmlProcessInfoV3"
            "nvmlDeviceGetComputeRunningProcesses_v3"
            "self.stop_event.wait(0.020)"
            "self._sample_rss()"
            "resource sampler is unsupported (requires Linux /proc and NVML v3)"
            "failed readiness/replay must not strand a non-daemon thread"
            "resources = sampler.finish()"
            "raise primary_error.with_traceback(primary_error.__traceback__)"
            "def self_test_cleanup():"
            "peak_rss_bytes"
            "peak_vram_bytes"
            "vram_process_samples"
            "nvml_process_bytes_20ms"
            "LLAMA_DEVICE_MEMORY_WITNESS"
            "def device_memory_witness(log_path):"
            "def capstone_kv_cache_args(extra):"
            "return [] if has_k else [\"-ctk\", \"f16\", \"-ctv\", \"f16\"]"
            "kv_cache_args = (capstone_kv_cache_args(extra) if capstone_mode else"
            "LLAMA_DEVICE_MEMORY_BEGIN"
            "LLAMA_DEVICE_MEMORY_END"
            "llama_owned_buffers_v1"
            "max_state_dir_bytes")
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
string(REPLACE "capstone arms must use one executable identity"
               "capstone executable unchecked" bad_executable_gate "${gate_src}")
capstone_gate_arms_ok("${bad_executable_gate}" bad_executable_gate_ok)
if (bad_executable_gate_ok)
    message(FATAL_ERROR "ZC6 executable-identity negative control did not trip")
endif()
string(REPLACE "resources = sampler.finish()" "resources = None"
               bad_cleanup_gate "${gate_src}")
capstone_gate_arms_ok("${bad_cleanup_gate}" bad_cleanup_gate_ok)
if (bad_cleanup_gate_ok)
    message(FATAL_ERROR "ZC6 failed-arm cleanup negative control did not trip")
endif()
string(REPLACE "self.stop_event.wait(0.020)"
               "self.stop_event.wait(1.000)"
               bad_nvml_cadence_gate "${gate_src}")
capstone_gate_arms_ok("${bad_nvml_cadence_gate}" bad_nvml_cadence_ok)
if (bad_nvml_cadence_ok)
    message(FATAL_ERROR "ZC6 continuous-NVML cadence negative control did not trip")
endif()

capstone_truth_contract("${common_src}" "${capture_src}" "${replay_src}" "${compare_src}" original_ok)
if (NOT original_ok)
    message(FATAL_ERROR "capstone claim-truth contract is incomplete")
endif()

function(zc6_report_contract report output)
    set(ok TRUE)
    foreach(token IN ITEMS
            "CELL_LABELS = frozenset("
            "def validate_cells(cells):"
            "capstone cells do not share one executable identity"
            "metadata.get(\"cpu_affinity\") != \"0-11\""
            "def validate_scorecard(scorecard, label, arm):"
            "claim.get(\"requested\")"
            "scorecard.get(\"truncated_requests\") != 0"
            "row.get(\"generated_recorded\") != generated"
            "total < ttft"
            "def hierarchical_sample(sessions, rng):"
            "session = sessions[rng.randrange(len(sessions))]"
            "requires two complete balanced launch-order cycles"
            "\"learning_tax\": (0, 1)"
            "\"planner_tax\": (1, 2)"
            "\"total_tax\": (0, 2)"
            "\"learning_tax_passed\": tax_passes[\"learning_tax\"]"
            "\"planner_tax_passed\": tax_passes[\"planner_tax\"]"
            "\"fresh_overhead_upper_us\": fresh_overhead_upper_us"
            "\"fresh_overhead_by_cell_us\": fresh_overhead_by_cell_us"
            "def overhead_upper_us(cell):"
            "resource.get(\"vram_sampler\") != \"nvml_process_bytes_20ms\""
            "resource.get(\"vram_poll_interval_ms\") != 20"
            "resource[\"peak_rss_bytes\"] - base[\"peak_rss_bytes\"]"
            "resource[\"peak_vram_bytes\"] - base[\"peak_vram_bytes\"]"
            "NVML_DRIVER_ENVELOPE_BYTES = 2 * MiB"
            "abs(vram_delta) <= NVML_DRIVER_ENVELOPE_BYTES"
            "llama_memory_equal"
            "llama_device_memory_witness"
            "resource[\"max_state_dir_bytes\"] <= 32 * MiB"
            "\"evaluation\": \"fresh_three_arm_overhead_only\""
            "\"optimization_claim\": False"
            "assert not cancelled[\"passed\"]"
            "clustered = {(slow_label, \"auto\", 5): 20.0}"
            "del no_resource[0][1][0][\"capstone_arm\"][\"resources\"]"
            "coarse_vram[0][1][0][\"capstone_arm\"][\"resources\"][\"vram_sampler\"] = \"nvidia-smi\""
            "slow_vram[0][1][0][\"capstone_arm\"][\"resources\"][\"vram_poll_interval_ms\"] = 1000")
        string(FIND "${report}" "${token}" found)
        if (found EQUAL -1)
            set(ok FALSE)
        endif()
    endforeach()
    set(${output} "${ok}" PARENT_SCOPE)
endfunction()

zc6_report_contract("${zc6_report_src}" zc6_report_ok)
if (NOT zc6_report_ok)
    message(FATAL_ERROR "ZC6 three-arm/resource report contract is incomplete")
endif()

foreach(token IN ITEMS
        "LLAMA_DEVICE_MEMORY_WITNESS"
        "LLAMA_DEVICE_MEMORY_BEGIN count=%zu"
        "LLAMA_DEVICE_MEMORY_END count=%zu"
        "LLAMA_DEVICE_MEMORY device=%zu model=%zu context=%zu compute=%zu total=%zu")
    string(FIND "${fit_src}" "${token}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "ZC6 llama-owned device-memory witness is incomplete")
    endif()
endforeach()
string(FIND "${server_src}" "common_memory_breakdown_print(ll_ctx, true)"
    final_device_witness)
if (final_device_witness EQUAL -1)
    message(FATAL_ERROR "ZC6 final device-memory witness owner is missing")
endif()

set(bad_static_fixture "${gate_src}")
string(REPLACE
    "return [] if has_k else [\"-ctk\", \"f16\", \"-ctv\", \"f16\"]"
    "return []"
    bad_static_fixture "${bad_static_fixture}")
capstone_gate_arms_ok("${bad_static_fixture}" bad_static_fixture_ok)
if (bad_static_fixture_ok)
    message(FATAL_ERROR "ZC6 explicit-static fixture negative control did not trip")
endif()

file(GLOB optimizer_device_owners
    "${SOURCE_ROOT}/tools/server/server-cache-calibration-*.cpp"
    "${SOURCE_ROOT}/tools/server/server-cache-observer.cpp"
    "${SOURCE_ROOT}/tools/server/server-cache-plan-authority.cpp"
    "${SOURCE_ROOT}/tools/server/server-cache-fingerprint.cpp"
    "${SOURCE_ROOT}/tools/server/server-context.cpp"
    "${SOURCE_ROOT}/tools/server/server-task.cpp")
set(optimizer_device_text "")
foreach(owner IN LISTS optimizer_device_owners)
    file(READ "${owner}" owner_text)
    string(APPEND optimizer_device_text "${owner_text}")
endforeach()
function(zc6_optimizer_device_owners_ok text output)
    foreach(forbidden IN ITEMS
            "cudaMalloc("
            "ggml_backend_buft_alloc_buffer("
            "ggml_backend_alloc_ctx_tensors("
            "llama_vram_hold_alloc_ctx_tensors(")
        string(FIND "${text}" "${forbidden}" found)
        if (NOT found EQUAL -1)
            set(${output} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${output} TRUE PARENT_SCOPE)
endfunction()
zc6_optimizer_device_owners_ok("${optimizer_device_text}" optimizer_device_ok)
if (NOT optimizer_device_ok)
    message(FATAL_ERROR "ZC6 optimizer owner acquired a device-allocation door")
endif()
zc6_optimizer_device_owners_ok(
    "${optimizer_device_text}\ncudaMalloc(1);" mutated_optimizer_device_ok)
if (mutated_optimizer_device_ok)
    message(FATAL_ERROR "ZC6 optimizer device-allocation negative control did not trip")
endif()
string(REPLACE "\"optimization_claim\": False"
               "\"optimization_claim\": passed"
               bad_zc6_report "${zc6_report_src}")
zc6_report_contract("${bad_zc6_report}" bad_zc6_claim_ok)
if (bad_zc6_claim_ok)
    message(FATAL_ERROR "ZC6 optimization-claim negative control did not trip")
endif()

string(REPLACE "\"planner_tax\": (1, 2)" "\"planner_tax\": (0, 2)"
               bad_zc6_tax "${zc6_report_src}")
zc6_report_contract("${bad_zc6_tax}" bad_zc6_tax_ok)
if (bad_zc6_tax_ok)
    message(FATAL_ERROR "ZC6 independent-tax negative control did not trip")
endif()

string(REPLACE "session = sessions[rng.randrange(len(sessions))]"
               "session = sessions[0]" bad_zc6_cluster "${zc6_report_src}")
zc6_report_contract("${bad_zc6_cluster}" bad_zc6_cluster_ok)
if (bad_zc6_cluster_ok)
    message(FATAL_ERROR "ZC6 clustered-bootstrap negative control did not trip")
endif()

string(REPLACE "resource[\"max_state_dir_bytes\"] <= 32 * MiB"
               "True" bad_zc6_resource "${zc6_report_src}")
zc6_report_contract("${bad_zc6_resource}" bad_zc6_resource_ok)
if (bad_zc6_resource_ok)
    message(FATAL_ERROR "ZC6 resource-cap negative control did not trip")
endif()
string(REPLACE
    "resource.get(\"vram_sampler\") != \"nvml_process_bytes_20ms\""
    "False"
    bad_zc6_nvml "${zc6_report_src}")
zc6_report_contract("${bad_zc6_nvml}" bad_zc6_nvml_ok)
if (bad_zc6_nvml_ok)
    message(FATAL_ERROR "ZC6 continuous-NVML report negative control did not trip")
endif()

string(REPLACE "capstone cells do not share one executable identity"
               "unchecked executable" bad_zc6_executable "${zc6_report_src}")
zc6_report_contract("${bad_zc6_executable}" bad_zc6_executable_ok)
if (bad_zc6_executable_ok)
    message(FATAL_ERROR "ZC6 report executable-census negative control did not trip")
endif()

string(REPLACE "total < ttft" "False" bad_zc6_row "${zc6_report_src}")
zc6_report_contract("${bad_zc6_row}" bad_zc6_row_ok)
if (bad_zc6_row_ok)
    message(FATAL_ERROR "ZC6 claim-grade row negative control did not trip")
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
