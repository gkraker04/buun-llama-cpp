#!/usr/bin/env python3

import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/server/bench/zc4-shadow-report.py"
SPEC = importlib.util.spec_from_file_location("zc4_shadow_report", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(actual, point, radius, state="provisional", assignment=2):
    return {
        "calibration_instance_slot": 3,
        "operation": "replay",
        "provider": "live_slot",
        "terminal": "accepted",
        "calibration_assignment": assignment,
        "calibration_profile_state": state,
        "calibration_n_fit": 20,
        "calibration_n_validation": 4,
        "calibration_prediction_available": True,
        "owned_service_us": actual,
        "calibration_prediction_us": point,
        "calibration_radius_us": radius,
        "tail_exceeded": False,
    }


report = MODULE.summarize([
    row(100, 90, 20),
    row(200, 150, 10, "active"),
    row(999, 0, 1, assignment=1),
])
assert report["observation_rows"] == 3
assert report["predictions"] == 2
assert report["coverage"] == 0.5
instance = report["instances"][0]
assert instance["mae_us"] == 30
assert instance["max_error_us"] == 50
assert instance["states"] == {"active": 1, "provisional": 2}
json.dumps(report)
print("PASS")
