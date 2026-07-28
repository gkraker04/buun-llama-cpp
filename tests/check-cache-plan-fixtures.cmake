# B-6 golden-fixture gate [P2]: the replay tool must parse the checked-in v1+v2 CACHE_PLAN
# samples exactly and reproduce the pinned report values. A schema drift that breaks either
# the emitted record shape or the tool's reading of it fails here, not in the field.
#
# Usage: cmake -DSOURCE_ROOT=<repo root> -P tests/check-cache-plan-fixtures.cmake

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "pass -DSOURCE_ROOT=<repo root>")
endif()

find_program(PYTHON3 python3)
if (NOT PYTHON3)
    message(FATAL_ERROR "python3 is required for the cache-plan fixture gate")
endif()

execute_process(
    COMMAND "${PYTHON3}" "${SOURCE_ROOT}/tools/server/bench/cache-plan-replay.py"
            "${SOURCE_ROOT}/tools/server/bench/fixtures/cache-plan-golden.jsonl" --json
    OUTPUT_VARIABLE report
    RESULT_VARIABLE rc
)
if (NOT rc EQUAL 0)
    message(FATAL_ERROR "cache-plan-replay.py failed on the golden fixture (rc=${rc})")
endif()

# pinned values: 4 records (1x v1 + 3x v2), two agreements (incl. the composed
# host→checkpoint chain as the complete shipped-plan ordinal) + one exact disagreement
foreach(expect
        "\"records\": 4"
        "\"shadow_evaluated\": 3"
        "\"shadow_unavailable\": 1"
        "\"agreement_rate\": 0.666"
        "\"disagreements\": 1"
        "\"predicted_saving_us\": 49600"
        "\"shipped_provider\": \"host_cache_entry\""
        "\"shadow_provider\": \"live_slot\""
        "live_context_checkpoint:truncated_by_shipped_short_circuit")
    string(FIND "${report}" "${expect}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "golden-fixture report missing pinned value: ${expect}\n${report}")
    endif()
endforeach()

# fail-closed negative: an unsupported schema_version must ERROR, never count as a record
execute_process(
    COMMAND "${PYTHON3}" "${SOURCE_ROOT}/tools/server/bench/cache-plan-replay.py"
            "${SOURCE_ROOT}/tools/server/bench/fixtures/cache-plan-unknown-schema.jsonl"
    OUTPUT_QUIET ERROR_QUIET
    RESULT_VARIABLE neg_rc
)
if (neg_rc EQUAL 0)
    message(FATAL_ERROR "unknown-schema fixture was ACCEPTED (rc 0) — schema gate is open")
endif()

# calibration fit arithmetic (D-pins r2 finding 6): the unidentifiable-slope branch
execute_process(
    COMMAND "${PYTHON3}" "${SOURCE_ROOT}/tools/server/bench/cache-plan-calibrate.py" --self-test
    OUTPUT_QUIET ERROR_VARIABLE selftest_err RESULT_VARIABLE selftest_rc
)
if (NOT selftest_rc EQUAL 0)
    message(FATAL_ERROR "cache-plan-calibrate self-test failed:\n${selftest_err}")
endif()

message(STATUS "cache-plan golden-fixture gate passed")
