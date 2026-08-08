#!/usr/bin/env python3
"""Compare fresh historical and intentional ZC1 live-gate arms."""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics


def load_transcript(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{path}: transcript has no records")
    return payload


def load_cache_plans(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "CACHE_PLAN "
        pos = line.find(marker)
        if pos < 0:
            continue
        start = line.find("{", pos + len(marker))
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if value.get("schema_version") == 7:
            records.append(value)
    return records


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise RuntimeError("empty metric population")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize(records: list[dict], expected_mode: str) -> dict:
    if not records:
        raise RuntimeError(f"{expected_mode}: no schema-7 CACHE_PLAN records")
    modes = collections.Counter(
        row.get("optimizer", {}).get("mode") for row in records)
    if set(modes) != {expected_mode}:
        raise RuntimeError(f"{expected_mode}: optimizer modes were {dict(modes)}")
    ttft = [row.get("ttft_us") for row in records]
    if not all(isinstance(value, int) and value >= 0 for value in ttft):
        raise RuntimeError(f"{expected_mode}: incomplete measured TTFT")
    reuse = [row.get("n_reused_tokens") for row in records]
    replay = [row.get("n_replayed_tokens") for row in records]
    if not all(isinstance(value, int) and value >= 0 for value in reuse + replay):
        raise RuntimeError(f"{expected_mode}: incomplete reuse/replay evidence")

    outcomes: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    retention_events = 0
    released_bytes = 0
    released_tokens = 0
    for row in records:
        summary = row.get("optimizer", {}).get("retention_summary", {})
        for key, value in summary.get("outcome_counts", {}).items():
            outcomes[key] += int(value)
            retention_events += int(value)
        for key, value in summary.get("reason_counts", {}).items():
            reasons[key] += int(value)
        released_bytes += int(summary.get("released_bytes", 0))
        released_tokens += int(summary.get("released_tokens", 0))

    if reasons["internal_fault"]:
        raise RuntimeError(
            f"{expected_mode}: retention internal_fault={reasons['internal_fault']}")
    return {
        "requests": len(records),
        "ttft_us": {
            "mean": round(statistics.fmean(ttft), 3),
            "p50": nearest_rank(ttft, 0.50),
            "p95": nearest_rank(ttft, 0.95),
            "p99": nearest_rank(ttft, 0.99),
            "max": max(ttft),
        },
        "reused_tokens": sum(reuse),
        "replayed_tokens": sum(replay),
        "request_outcomes": dict(collections.Counter(
            row.get("outcome") for row in records)),
        "retention_events": retention_events,
        "retention_outcomes": dict(outcomes),
        "retention_reasons": {key: value for key, value in reasons.items() if value},
        "released_bytes": released_bytes,
        "released_tokens": released_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-transcript", type=pathlib.Path, required=True)
    parser.add_argument("--historical-log", type=pathlib.Path, required=True)
    parser.add_argument("--intentional-transcript", type=pathlib.Path, required=True)
    parser.add_argument("--intentional-log", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    historical_transcript = load_transcript(args.historical_transcript)
    intentional_transcript = load_transcript(args.intentional_transcript)
    historical_rows = historical_transcript["records"]
    intentional_rows = intentional_transcript["records"]
    historical_outputs = {
        (row["round"], row["chain"]): row["reply_sha256"]
        for row in historical_rows
    }
    intentional_outputs = {
        (row["round"], row["chain"]): row["reply_sha256"]
        for row in intentional_rows
    }
    if historical_outputs != intentional_outputs:
        mismatches = sorted(
            key for key in set(historical_outputs) | set(intentional_outputs)
            if historical_outputs.get(key) != intentional_outputs.get(key))
        raise RuntimeError(f"output mismatch at {mismatches[:8]}")

    historical = summarize(load_cache_plans(args.historical_log), "off")
    intentional = summarize(load_cache_plans(args.intentional_log), "baseline")
    expected_requests = len(historical_rows)
    if historical["requests"] != expected_requests or \
            intentional["requests"] != expected_requests:
        raise RuntimeError(
            "CACHE_PLAN/transcript count mismatch: "
            f"transcript={expected_requests} historical={historical['requests']} "
            f"intentional={intentional['requests']}")
    if intentional["retention_events"] == 0:
        raise RuntimeError("intentional arm exercised no retention terminal")

    report = {
        "schema": "zc1_retention_live_report/v1",
        "label": args.label,
        "outputs_byte_identical": True,
        "requests_compared": expected_requests,
        "historical": historical,
        "intentional": intentional,
    }
    args.out.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "ZC1_RETENTION_LIVE PASS "
        f"label={args.label} requests={expected_requests} "
        f"historical_p95_us={historical['ttft_us']['p95']} "
        f"intentional_p95_us={intentional['ttft_us']['p95']} "
        f"historical_replay={historical['replayed_tokens']} "
        f"intentional_replay={intentional['replayed_tokens']} "
        f"retention_events={intentional['retention_events']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
