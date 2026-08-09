#!/usr/bin/env python3
"""Evaluate the frozen, three-arm ZC6 fresh-overhead capstone.

The independent experimental unit is a server restart.  Four-request blocks
are sampled only inside a sampled restart, so a long/slow restart cannot be
diluted into request-level pseudoreplication.  This remains an overhead gate,
never an optimization claim; persisted-policy efficacy is evaluated elsewhere.
"""

import argparse
import collections
import json
import math
import pathlib
import random


CELL_LABELS = frozenset(
    f"{size}-{kv}-{shape}"
    for size in ("small", "large")
    for kv in ("static", "vbr")
    for shape in ("concurrent", "serialized")
)
ARM_NAMES = ("baseline", "learn", "auto")
PRIMARY_ORDERS = frozenset(
    (first, second, third)
    for first in ARM_NAMES for second in ARM_NAMES for third in ARM_NAMES
    if len({first, second, third}) == 3
)
OUTPUT_FIELDS = (
    "seq", "status", "generated_tokens", "generated_recorded", "truncated",
    "stop_type", "output_sha",
)
MiB = 1024 * 1024
NVML_DRIVER_ENVELOPE_BYTES = 2 * MiB
RESTARTS_PER_ORDER = 2


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[rank]


def finite_number(value, field):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"invalid {field}")
    return value


def validate_resource(resource):
    if not isinstance(resource, dict):
        raise ValueError("missing resource evidence")
    required = (
        "peak_rss_bytes", "peak_vram_bytes", "final_state_dir_bytes",
        "max_state_dir_bytes", "samples",
    )
    for field in required:
        value = resource.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid resource field {field}")
    if resource["samples"] == 0:
        raise ValueError("resource evidence has no live samples")
    if not resource.get("rss_supported") or not resource.get("vram_supported"):
        raise ValueError("resource sampler unsupported")
    if (resource.get("vram_sampler") != "nvml_process_bytes_20ms" or
            resource.get("vram_poll_interval_ms") != 20 or
            not isinstance(resource.get("vram_process_samples"), int) or
            isinstance(resource.get("vram_process_samples"), bool) or
            resource["vram_process_samples"] <= 0):
        raise ValueError("resource evidence lacks continuous NVML process sampling")
    if resource["final_state_dir_bytes"] > resource["max_state_dir_bytes"]:
        raise ValueError("final state bytes exceed sampled maximum")
    if resource.get("llama_device_memory_witness") != "llama_owned_buffers_v1":
        raise ValueError("missing llama-owned device-memory witness")
    device_rows = resource.get("llama_device_memory")
    if not isinstance(device_rows, list) or not device_rows:
        raise ValueError("missing llama-owned device-memory rows")
    for index, row in enumerate(device_rows):
        if not isinstance(row, dict) or row.get("device") != index:
            raise ValueError("invalid llama-owned device-memory ordinal")
        values = []
        for field in ("model_bytes", "context_bytes", "compute_bytes", "total_bytes"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid llama-owned device-memory {field}")
            values.append(value)
        if values[3] != sum(values[:3]):
            raise ValueError("inconsistent llama-owned device-memory total")


def validate_scorecard(scorecard, label, arm):
    metadata = scorecard.get("capstone_arm")
    if not isinstance(metadata, dict) or metadata.get("mode") != arm:
        raise ValueError(f"{label}/{arm}: missing or wrong arm metadata")
    session_id = metadata.get("session_id")
    launch_order = metadata.get("launch_order")
    executable = metadata.get("executable")
    if metadata.get("cpu_affinity") != "0-11":
        raise ValueError(f"{label}/{arm}: wrong CPU affinity")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"{label}/{arm}: missing session id")
    if (not isinstance(launch_order, list) or tuple(launch_order) not in PRIMARY_ORDERS):
        raise ValueError(f"{label}/{arm}: invalid launch order")
    if (not isinstance(executable, dict) or
            not isinstance(executable.get("sha256"), str) or
            len(executable["sha256"]) != 64 or
            not isinstance(executable.get("size_bytes"), int) or
            executable["size_bytes"] <= 0):
        raise ValueError(f"{label}/{arm}: missing executable provenance")
    validate_resource(metadata.get("resources"))

    claim = scorecard.get("claim_validation") or {}
    sampling = scorecard.get("sampling") or {}
    timeline = scorecard.get("timeline") or {}
    execution = scorecard.get("execution") or {}
    if not claim.get("requested") or not claim.get("valid") or claim.get("reasons"):
        raise ValueError(f"{label}/{arm}: scorecard is not claim-grade")
    if not sampling.get("greedy") or not sampling.get("pin_generation_complete"):
        raise ValueError(f"{label}/{arm}: sampling/generation pin is incomplete")
    if not timeline.get("faithful") or timeline.get("gaps_capped") != 0:
        raise ValueError(f"{label}/{arm}: replay timeline is not faithful")
    serialized = label.endswith("-serialized")
    if execution.get("serialize_overlap") is not serialized:
        raise ValueError(f"{label}/{arm}: serialized execution metadata disagrees")
    observed_parallel = execution.get("server_parallel_observed")
    if (not isinstance(observed_parallel, int) or
            (serialized and observed_parallel != 1) or
            (not serialized and observed_parallel < 2)):
        raise ValueError(f"{label}/{arm}: invalid observed server parallelism")

    value = scorecard.get("requests")
    if not isinstance(value, list) or len(value) < 80 or len(value) % 4:
        raise ValueError(f"{label}/{arm}: needs >=80 requests in four-row blocks")
    if (scorecard.get("requests_total") != len(value) or
            scorecard.get("requests_scored") != len(value) or
            scorecard.get("requests_ok") != len(value) or
            scorecard.get("status_other") not in ([], ()) or
            scorecard.get("truncated_requests") != 0 or
            scorecard.get("generated_matched_capture") != len(value)):
        raise ValueError(f"{label}/{arm}: scorecard summary is unsuccessful")
    seen = set()
    for row in value:
        seq = row.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq in seen:
            raise ValueError(f"{label}/{arm}: invalid or duplicate sequence")
        seen.add(seq)
        if row.get("status") != 200 or row.get("truncated") is not False:
            raise ValueError(f"{label}/{arm}: failed or truncated row {seq}")
        generated = row.get("generated_tokens")
        if (not isinstance(generated, int) or generated <= 0 or
                row.get("generated_recorded") != generated):
            raise ValueError(f"{label}/{arm}: generation mismatch at {seq}")
        ttft = finite_number(row.get("ttft_ms"), "ttft_ms")
        total = finite_number(row.get("total_ms"), "total_ms")
        if ttft < 0 or total < ttft:
            raise ValueError(f"{label}/{arm}: invalid timeline at {seq}")
        fresh = row.get("fresh_prefill_tokens")
        prompt = row.get("prompt_tokens")
        if (not isinstance(fresh, int) or fresh < 0 or
                not isinstance(prompt, int) or prompt < fresh):
            raise ValueError(f"{label}/{arm}: invalid prompt accounting at {seq}")
        if not isinstance(row.get("stop_type"), str) or not row.get("output_sha"):
            raise ValueError(f"{label}/{arm}: incomplete output evidence at {seq}")
    ordered = sorted(value, key=lambda row: row["seq"])
    if [row["seq"] for row in ordered] != list(range(len(ordered))):
        raise ValueError(f"{label}/{arm}: request sequence is incomplete")
    return {
        "session_id": session_id,
        "launch_order": tuple(launch_order),
        "executable": executable,
        "resources": metadata["resources"],
        "blocks": [ordered[index:index + 4] for index in range(0, len(ordered), 4)],
    }


def align_sessions(label, arms):
    indexed = []
    for arm_name, scorecards in zip(ARM_NAMES, arms):
        by_session = {}
        for scorecard in scorecards:
            session = validate_scorecard(scorecard, label, arm_name)
            if session["session_id"] in by_session:
                raise ValueError(f"{label}/{arm_name}: duplicate session id")
            by_session[session["session_id"]] = session
        indexed.append(by_session)
    ids = set(indexed[0])
    if not ids or any(set(values) != ids for values in indexed[1:]):
        raise ValueError(f"{label}: arm session sets differ")
    sessions = []
    for session_id in sorted(ids):
        rows = [values[session_id] for values in indexed]
        executable_ids = {
            (row["executable"]["sha256"], row["executable"]["size_bytes"])
            for row in rows
        }
        if len(executable_ids) != 1:
            raise ValueError(f"{label}/{session_id}: arms use different executables")
        if len({row["launch_order"] for row in rows}) != 1:
            raise ValueError(f"{label}/{session_id}: arms disagree on launch order")
        block_counts = {len(row["blocks"]) for row in rows}
        if len(block_counts) != 1:
            raise ValueError(f"{label}/{session_id}: arm block counts differ")
        fields = OUTPUT_FIELDS if label.endswith("-serialized") else tuple(
            key for key in OUTPUT_FIELDS if key != "output_sha")
        for blocks in zip(*(row["blocks"] for row in rows)):
            for triplet in zip(*blocks):
                reference = tuple(triplet[0].get(key) for key in fields)
                if any(tuple(row.get(key) for key in fields) != reference for row in triplet[1:]):
                    raise ValueError(
                        f"{label}/{session_id}: output mismatch at seq {triplet[0]['seq']}")
        sessions.append({
            "session_id": session_id,
            "launch_order": rows[0]["launch_order"],
            "arms": rows,
            "blocks": [tuple(blocks) for blocks in zip(*(row["blocks"] for row in rows))],
        })
    order_counts = collections.Counter(
        session["launch_order"] for session in sessions)
    if (len(sessions) != len(PRIMARY_ORDERS) * RESTARTS_PER_ORDER or
            order_counts != collections.Counter({
                order: RESTARTS_PER_ORDER for order in PRIMARY_ORDERS
            })):
        raise ValueError(f"{label}: requires two complete balanced launch-order cycles")
    return sessions


def hierarchical_sample(sessions, rng):
    sample = []
    for _ in sessions:
        session = sessions[rng.randrange(len(sessions))]
        blocks = session["blocks"]
        sample.extend(blocks[rng.randrange(len(blocks))] for _ in blocks)
    return sample


def metric(blocks, side, q):
    return percentile([row["ttft_ms"] for block in blocks for row in block[side]], q)


def fresh_total(blocks, side):
    return sum(row["fresh_prefill_tokens"] for block in blocks for row in block[side])


def evaluate_cell(label, baseline, learn, auto, iterations, seed):
    sessions = align_sessions(label, (baseline, learn, auto))
    blocks = [block for session in sessions for block in session["blocks"]]
    rng = random.Random(seed)
    comparisons = {
        "learning_tax": (0, 1),
        "planner_tax": (1, 2),
        "total_tax": (0, 2),
    }
    taxes = {}
    for tax, (reference, measured) in comparisons.items():
        taxes[tax] = {}
        for endpoint, q, relative, absolute in (
                ("p50", 0.50, 0.02, 1.0),
                ("p95", 0.95, 0.03, 2.0)):
            base = metric(blocks, reference, q)
            active = metric(blocks, measured, q)
            taxes[tax][endpoint] = {
                "reference_ms": base,
                "measured_ms": active,
                "delta_ms": active - base,
                "margin_ms": max(relative * base, absolute),
                "samples": [],
            }
    fresh = {
        "baseline_tokens": fresh_total(blocks, 0),
        "learn_tokens": fresh_total(blocks, 1),
        "auto_tokens": fresh_total(blocks, 2),
    }
    fresh["margin_tokens"] = (0 if label.endswith("static-serialized") else
        max(8 * len(blocks) / 20, 0.0025 * fresh["baseline_tokens"]))
    fresh["schedule_exact"] = all(
        block[0][row]["fresh_prefill_tokens"] == block[arm][row]["fresh_prefill_tokens"] and
        block[0][row]["prompt_tokens"] == block[arm][row]["prompt_tokens"]
        for block in blocks for row in range(4) for arm in (1, 2))
    fresh["samples"] = {"learn": [], "auto": []}

    for _ in range(iterations):
        sample = hierarchical_sample(sessions, rng)
        for tax, (reference, measured) in comparisons.items():
            for endpoint, q in (("p50", 0.50), ("p95", 0.95)):
                taxes[tax][endpoint]["samples"].append(
                    metric(sample, measured, q) - metric(sample, reference, q))
        for name, side in (("learn", 1), ("auto", 2)):
            fresh["samples"][name].append(
                fresh_total(sample, side) - fresh_total(sample, 0))

    resources = {"sessions": [], "pass": True}
    for session in sessions:
        arm_resources = [arm["resources"] for arm in session["arms"]]
        base = arm_resources[0]
        resources["pass"] = resources["pass"] and (
            base["max_state_dir_bytes"] <= 32 * MiB and
            base["final_state_dir_bytes"] <= 32 * MiB)
        comparisons_out = {}
        for name, resource in zip(ARM_NAMES[1:], arm_resources[1:]):
            rss_delta = resource["peak_rss_bytes"] - base["peak_rss_bytes"]
            vram_delta = resource["peak_vram_bytes"] - base["peak_vram_bytes"]
            llama_memory_equal = (
                resource["llama_device_memory"] == base["llama_device_memory"])
            disk_ok = (resource["max_state_dir_bytes"] <= 32 * MiB and
                       resource["final_state_dir_bytes"] <= 32 * MiB)
            arm_pass = (rss_delta <= 96 * MiB and
                        abs(vram_delta) <= NVML_DRIVER_ENVELOPE_BYTES and
                        llama_memory_equal and disk_ok)
            comparisons_out[name] = {
                "peak_rss_delta_bytes": rss_delta,
                "peak_vram_delta_bytes": vram_delta,
                "nvml_driver_envelope_bytes": NVML_DRIVER_ENVELOPE_BYTES,
                "llama_device_memory_equal": llama_memory_equal,
                "max_state_dir_bytes": resource["max_state_dir_bytes"],
                "final_state_dir_bytes": resource["final_state_dir_bytes"],
                "pass": arm_pass,
            }
            resources["pass"] = resources["pass"] and arm_pass
        resources["sessions"].append({
            "session_id": session["session_id"],
            "baseline": base,
            "comparisons": comparisons_out,
        })
    return {
        "label": label,
        "restarts": len(sessions),
        "blocks": len(blocks),
        "taxes": taxes,
        "fresh": fresh,
        "resources": resources,
    }


def apply_holm(evaluated, tax):
    tests = []
    for cell in evaluated:
        for endpoint in ("p50", "p95"):
            point = cell["taxes"][tax][endpoint]
            values = point["samples"]
            p_value = (1 + sum(value >= point["margin_ms"] for value in values)) / (len(values) + 1)
            tests.append((p_value, cell, endpoint))
    ordered = sorted(tests, key=lambda item: item[0])
    family_pass = True
    for index, (p_value, cell, endpoint) in enumerate(ordered):
        threshold = 0.05 / (len(ordered) - index)
        point = cell["taxes"][tax][endpoint]
        upper = percentile(point["samples"], 1.0 - threshold)
        passed = family_pass and p_value <= threshold and upper <= point["margin_ms"]
        family_pass = passed
        point.update({
            "p_value": p_value,
            "holm_threshold": threshold,
            "upper_delta_ms": upper,
            "holm_pass": passed,
        })
    return family_pass


def validate_cells(cells):
    labels = [cell[0] for cell in cells]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate capstone cell label")
    unknown = set(labels) - CELL_LABELS
    missing = CELL_LABELS - set(labels)
    if unknown or missing:
        raise ValueError(
            f"capstone labels are not closed: unknown={sorted(unknown)} missing={sorted(missing)}")


def report(cells, iterations=10000, seed=0x5A436):
    validate_cells(cells)
    evaluated = [
        evaluate_cell(label, baseline, learn, auto, iterations, seed + index)
        for index, (label, baseline, learn, auto) in enumerate(sorted(cells))
    ]
    executable_ids = {
        (card["capstone_arm"]["executable"]["sha256"],
         card["capstone_arm"]["executable"]["size_bytes"])
        for _, baseline, learn, auto in cells
        for cards in (baseline, learn, auto) for card in cards
    }
    if len(executable_ids) != 1:
        raise ValueError("capstone cells do not share one executable identity")
    tax_passes = {
        tax: apply_holm(evaluated, tax)
        for tax in ("learning_tax", "planner_tax", "total_tax")
    }
    schedule_pass = True
    resource_pass = True
    for cell in evaluated:
        fresh = cell["fresh"]
        for arm in ("learn", "auto"):
            fresh[f"{arm}_upper_delta_tokens"] = percentile(fresh["samples"][arm], 0.95)
        fresh["pass"] = (fresh["schedule_exact"] if cell["label"].endswith("static-serialized") else
            all(fresh[f"{arm}_tokens"] - fresh["baseline_tokens"] <= fresh["margin_tokens"]
                for arm in ("learn", "auto")))
        schedule_pass = schedule_pass and fresh["pass"]
        resource_pass = resource_pass and cell["resources"]["pass"]
        del fresh["samples"]
        for tax in cell["taxes"].values():
            for endpoint in tax.values():
                del endpoint["samples"]

    def overhead_upper_us(cell):
        maximum_total = max(
            cell["taxes"]["total_tax"][endpoint]["upper_delta_ms"]
            for endpoint in ("p50", "p95"))
        maximum_learning = max(
            cell["taxes"]["learning_tax"][endpoint]["upper_delta_ms"]
            for endpoint in ("p50", "p95"))
        maximum_planner = max(
            cell["taxes"]["planner_tax"][endpoint]["upper_delta_ms"]
            for endpoint in ("p50", "p95"))
        return int(math.ceil(1000.0 * max(
            0.0, maximum_total,
            max(0.0, maximum_learning) + max(0.0, maximum_planner))))

    fresh_overhead_by_cell_us = {
        cell["label"]: overhead_upper_us(cell) for cell in evaluated
    }
    # Retain the global maximum for conservative summaries.  Authority efficacy
    # must consume the preregistered matching cell rather than this cross-shape
    # maximum, whose restart and concurrency costs are different currencies.
    fresh_overhead_upper_us = max(fresh_overhead_by_cell_us.values())
    passed = all(tax_passes.values()) and schedule_pass and resource_pass
    return {
        "schema": "zc6-capstone-report/v1",
        "evaluation": "fresh_three_arm_overhead_only",
        "bootstrap": "hierarchical_restart_then_four_row_block",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "holm_family_alpha": 0.05,
        "learning_tax_passed": tax_passes["learning_tax"],
        "planner_tax_passed": tax_passes["planner_tax"],
        "total_tax_passed": tax_passes["total_tax"],
        "schedule_passed": schedule_pass,
        "resource_passed": resource_pass,
        "fresh_overhead_upper_us": fresh_overhead_upper_us,
        "fresh_overhead_by_cell_us": fresh_overhead_by_cell_us,
        "noninferiority_passed": passed,
        "optimization_claim": False,
        "authority_efficacy_required": True,
        "passed": passed,
        "cells": evaluated,
    }


def fixture_card(label, arm, session, launch_order, extra=0.0, resource=True):
    serialized = label.endswith("-serialized")
    rows = [{
        "seq": index, "status": 200, "generated_tokens": 1,
        "generated_recorded": 1, "truncated": False,
        "stop_type": "limit", "output_sha": str(index),
        "ttft_ms": 100.0 + extra, "total_ms": 110.0 + extra,
        "fresh_prefill_tokens": 10, "prompt_tokens": 20,
    } for index in range(80)]
    card = {
        "requests": rows, "requests_total": 80, "requests_scored": 80, "requests_ok": 80,
        "status_other": [],
        "truncated_requests": 0, "generated_matched_capture": 80,
        "claim_validation": {"requested": True, "valid": True, "reasons": []},
        "sampling": {"greedy": True, "pin_generation_complete": True},
        "timeline": {"faithful": True, "gaps_capped": 0},
        "execution": {"serialize_overlap": serialized,
                      "server_parallel_observed": 1 if serialized else 2},
        "capstone_arm": {
            "mode": arm, "session_id": session,
            "launch_order": list(launch_order),
            "executable": {"sha256": "a" * 64, "size_bytes": 1234},
            "cpu_affinity": "0-11",
        },
    }
    if resource:
        card["capstone_arm"]["resources"] = {
            "peak_rss_bytes": 100 * MiB, "peak_vram_bytes": 1000 * MiB,
            "final_state_dir_bytes": MiB, "max_state_dir_bytes": MiB,
            "samples": 10, "rss_supported": True, "vram_supported": True,
            "vram_process_samples": 10,
            "vram_sampler": "nvml_process_bytes_20ms",
            "vram_poll_interval_ms": 20,
            "llama_device_memory_witness": "llama_owned_buffers_v1",
            "llama_device_memory": [{
                "device": 0,
                "model_bytes": 700 * MiB,
                "context_bytes": 200 * MiB,
                "compute_bytes": 100 * MiB,
                "total_bytes": 1000 * MiB,
            }],
        }
    return card


def fixture_cells(extra=None):
    extra = extra or {}
    cells = []
    orders = sorted(PRIMARY_ORDERS)
    for label in sorted(CELL_LABELS):
        arms = []
        session_count = len(orders) * RESTARTS_PER_ORDER
        for arm in ARM_NAMES:
            arms.append([
                fixture_card(label, arm, f"session-{session}",
                             orders[session % len(orders)],
                             extra.get((label, arm, session), 0.0))
                for session in range(session_count)
            ])
        cells.append((label, *arms))
    return cells


def expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test():
    passing = report(fixture_cells(), 1024)
    assert passing["passed"] and passing["learning_tax_passed"]
    assert passing["planner_tax_passed"] and passing["total_tax_passed"]
    assert passing["fresh_overhead_upper_us"] == 0
    assert set(passing["fresh_overhead_by_cell_us"]) == CELL_LABELS
    assert all(value == 0 for value in passing["fresh_overhead_by_cell_us"].values())
    assert not passing["optimization_claim"] and passing["authority_efficacy_required"]

    # A combined baseline->auto delta of zero must not conceal learning tax.
    cancellation = {
        (label, "learn", session): 20.0
        for label in CELL_LABELS
        for session in range(len(PRIMARY_ORDERS) * RESTARTS_PER_ORDER)
    }
    cancelled = report(fixture_cells(cancellation), 1024)
    assert not cancelled["passed"] and not cancelled["learning_tax_passed"]
    assert cancelled["total_tax_passed"]

    # One slow restart must remain a cluster rather than eighty diluted rows.
    slow_label = "small-static-concurrent"
    clustered = {(slow_label, "auto", 5): 20.0}
    assert not report(fixture_cells(clustered), 1024)["passed"]

    missing = fixture_cells()[:-1]
    expect_value_error(lambda: report(missing, 16))
    duplicate = fixture_cells()
    duplicate.append(duplicate[0])
    expect_value_error(lambda: report(duplicate, 16))
    unknown = fixture_cells()
    unknown[0] = ("medium-static-concurrent", *unknown[0][1:])
    expect_value_error(lambda: report(unknown, 16))

    bad_row = fixture_cells()
    bad_row[0][1][0]["requests"][0]["truncated"] = True
    expect_value_error(lambda: report(bad_row, 16))
    no_resource = fixture_cells()
    del no_resource[0][1][0]["capstone_arm"]["resources"]
    expect_value_error(lambda: report(no_resource, 16))
    coarse_vram = fixture_cells()
    coarse_vram[0][1][0]["capstone_arm"]["resources"]["vram_sampler"] = "nvidia-smi"
    expect_value_error(lambda: report(coarse_vram, 16))
    slow_vram = fixture_cells()
    slow_vram[0][1][0]["capstone_arm"]["resources"]["vram_poll_interval_ms"] = 1000
    expect_value_error(lambda: report(slow_vram, 16))

    wrong_executable = fixture_cells()
    wrong_executable[0][3][0]["capstone_arm"]["executable"]["sha256"] = "b" * 64
    expect_value_error(lambda: report(wrong_executable, 16))
    oversized = fixture_cells()
    oversized[0][2][0]["capstone_arm"]["resources"]["peak_rss_bytes"] += 96 * MiB + 1
    oversized_result = report(oversized, 1024)
    assert not oversized_result["resource_passed"] and not oversized_result["passed"]
    driver_quantum = fixture_cells()
    driver_quantum[0][3][0]["capstone_arm"]["resources"]["peak_vram_bytes"] += 2 * MiB
    assert report(driver_quantum, 1024)["resource_passed"]
    driver_overage = fixture_cells()
    driver_overage[0][3][0]["capstone_arm"]["resources"]["peak_vram_bytes"] += 2 * MiB + 1
    assert not report(driver_overage, 1024)["resource_passed"]
    owned_overage = fixture_cells()
    owned_memory = owned_overage[0][3][0]["capstone_arm"]["resources"]["llama_device_memory"][0]
    owned_memory["compute_bytes"] += 1
    owned_memory["total_bytes"] += 1
    assert not report(owned_overage, 1024)["resource_passed"]


def load_scorecards(spec):
    paths = [pathlib.Path(value) for value in spec.split(",") if value]
    if not paths:
        raise ValueError("empty scorecard list")
    return [json.loads(path.read_text()) for path in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell", nargs=4, action="append",
        metavar=("LABEL", "BASELINE", "LEARN", "AUTO"))
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.cell or not args.out:
        parser.error("--cell and --out are required")
    cells = [
        (label, load_scorecards(baseline), load_scorecards(learn), load_scorecards(auto))
        for label, baseline, learn, auto in args.cell
    ]
    result = report(cells)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("ZC6 capstone three-arm non-inferiority gate failed")


if __name__ == "__main__":
    main()
