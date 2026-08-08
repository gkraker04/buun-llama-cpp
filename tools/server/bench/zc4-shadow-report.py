#!/usr/bin/env python3
"""Summarize ZC4's debug-only calibration receipts without touching authority.

The input is a llama-server log.  This tool deliberately consumes the same
CACHE_OPTIMIZER_OBSERVATION JSON that operators see; it never opens or edits
the calibration store and cannot manufacture fitting evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFIX = "CACHE_OPTIMIZER_OBSERVATION "


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            try:
                value = json.loads(line[marker + len(PREFIX):])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("calibration_boot_claim_ordinal"),
            row.get("calibration_profile_generation"),
            row.get("calibration_instance_slot"),
            row.get("calibration_fit_generation"),
            row.get("operation"),
            row.get("provider"),
            row.get("calibration_model_kind"),
            row.get("size_family"),
            row.get("contention_bucket"),
            row.get("start_bucket"),
            row.get("batch_bucket"),
            row.get("ubatch_bucket"),
        )
        groups[key].append(row)

    instances: list[dict[str, Any]] = []
    total_predictions = 0
    total_covered = 0
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        values = groups[key]
        predicted: list[tuple[float, float, float]] = []
        for row in values:
            if (row.get("terminal") != "accepted" or
                    row.get("calibration_assignment") != 2 or
                    not row.get("calibration_prediction_available")):
                continue
            actual = row.get("owned_service_us")
            point = row.get("calibration_prediction_us")
            radius = row.get("calibration_radius_us")
            if not all(isinstance(item, (int, float)) and math.isfinite(item)
                       for item in (actual, point, radius)):
                continue
            predicted.append((float(actual), float(point), float(radius)))
        errors = [abs(actual - point) for actual, point, _ in predicted]
        covered = sum(error <= radius for error, (_, _, radius) in zip(errors, predicted))
        total_predictions += len(predicted)
        total_covered += covered
        states: dict[str, int] = defaultdict(int)
        assignments: dict[str, int] = defaultdict(int)
        for row in values:
            states[str(row.get("calibration_profile_state", "unavailable"))] += 1
            assignments[str(row.get("calibration_assignment", 0))] += 1
        instances.append({
            "boot_claim_ordinal": key[0],
            "profile_generation": key[1],
            "slot": key[2],
            "fit_generation": key[3],
            "operation": key[4],
            "provider": key[5],
            "model_kind": key[6],
            "size_family": key[7],
            "contention_bucket": key[8],
            "start_bucket": key[9],
            "batch_bucket": key[10],
            "ubatch_bucket": key[11],
            "rows": len(values),
            "terminal_rows": {
                terminal: sum(row.get("terminal") == terminal for row in values)
                for terminal in ("accepted", "diagnostic", "operation_unavailable")
            },
            "assignments": dict(sorted(assignments.items())),
            "states": dict(sorted(states.items())),
            "last_n_fit": max((int(row.get("calibration_n_fit", 0)) for row in values), default=0),
            "last_n_validation": max((int(row.get("calibration_n_validation", 0)) for row in values), default=0),
            "predictions": len(predicted),
            "coverage": covered / len(predicted) if predicted else None,
            "mae_us": sum(errors) / len(errors) if errors else None,
            "max_error_us": max(errors) if errors else None,
            "tail_rows": sum(bool(row.get("tail_exceeded")) for row in values),
        })
    return {
        "schema": "zc4-shadow-report/v1",
        "observation_rows": len(rows),
        "instances": instances,
        "predictions": total_predictions,
        "coverage": total_covered / total_predictions if total_predictions else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-observations", action="store_true")
    args = parser.parse_args()
    report = summarize(load_rows(args.log))
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.require_observations and report["observation_rows"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
