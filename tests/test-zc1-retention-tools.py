#!/usr/bin/env python3
"""Model-free codec/grid tests for the ZC1 training tools."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def frontier(round_id: int) -> int:
    if round_id <= 11:
        return (round_id + 1) * 1000
    if round_id <= 15:
        return (round_id - 7) * 1000
    if round_id <= 21:
        return (round_id - 14) * 1000
    return (round_id - 17) * 1000


def parent(round_id: int) -> int | None:
    if round_id in (0, 16):
        return None
    if round_id == 12:
        return 3
    if round_id == 22:
        return 18
    return round_id - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    extractor = root / "tools/server/bench/zc1-retention-extract.py"
    reporter = root / "tools/server/bench/zc1-retention-report.py"
    fixture = root / "tools/server/bench/fixtures/zc1-retention-events.json"

    with tempfile.TemporaryDirectory(prefix="zc1-retention-tools-") as temp_dir:
        temp = pathlib.Path(temp_dir)
        records = []
        trace_rows = []
        base_seed = 61001
        for chain in range(3):
            for round_id in range(24):
                seed = base_seed + chain * 100 + round_id
                records.append({
                    "round": round_id,
                    "chain": chain,
                    "parent_round": parent(round_id),
                    "transition": "deep_rewind" if round_id in (12, 22) else
                                  "compact" if round_id == 16 else "append",
                    "request_seed": seed,
                })
                trace_rows.append({
                    "record": "request",
                    "path": "/v1/chat/completions",
                    "status": 200,
                    "body": {"seed": seed},
                    "server": {"timings": {
                        "prompt_n": frontier(round_id) - 100,
                        "cache_n": 100,
                    }},
                })
        transcript = {
            "schema": "zc1_retention_pressure_transcript/v1",
            "chains": 3,
            "rounds": 24,
            "records": records,
        }
        transcript_path = temp / "transcript.json"
        trace_path = temp / "trace.jsonl"
        event_path = temp / "events.json"
        transcript_path.write_text(
            json.dumps(transcript, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        trace_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n"
                    for row in reversed(trace_rows)),
            encoding="utf-8",
        )
        subprocess.run([
            sys.executable,
            str(extractor),
            "--trace", str(trace_path),
            "--transcript", str(transcript_path),
            "--out", str(event_path),
            "--capacity", "4",
            "--checkpoint-min-step", "100",
        ], check=True)
        events = json.loads(event_path.read_text(encoding="utf-8"))
        assert len(events["chains"]) == 3
        assert all(len(chain["events"]) == 24 for chain in events["chains"])
        assert events["chains"][0]["events"][12]["parent_node"] == 3
        assert events["chains"][0]["events"][22]["parent_node"] == 18

        grid_out = temp / "grid.json"
        subprocess.run([
            args.grid,
            "--events", str(fixture),
            "--out", str(grid_out),
        ], check=True)
        grid = json.loads(grid_out.read_text(encoding="utf-8"))
        assert grid["schema"] == "zc1_retention_grid_result/v2"
        assert grid["candidate_count"] == 432
        assert grid["winner"]["valid"] is True
        assert grid["winner"]["v7_eligible"] is True
        assert grid["winner"]["total_replay_tokens"] <= \
            grid["v7_contract"]["total_replay_limit"]
        assert grid["winner"]["p95_replay_tokens"] <= \
            grid["v7_contract"]["p95_replay_limit"]
        assert grid["winner"]["p99_replay_tokens"] < grid["fifo"]["p99_replay_tokens"]
        assert grid["winner"]["max_replay_tokens"] < grid["fifo"]["max_replay_tokens"]
        assert grid["winner"]["zero_coverage_deep_rewinds"] == 0
        assert grid["winner"]["checkpoint_mutations"] <= grid["fifo"]["checkpoint_mutations"]
        assert grid["winner"]["publication_skips"] == 0

        # A duplicate trace seed must fail closed rather than silently picking
        # one concurrency-order outcome.
        bad_trace = temp / "bad-trace.jsonl"
        bad_trace.write_text(
            trace_path.read_text(encoding="utf-8") +
            json.dumps(trace_rows[0], separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        failed = subprocess.run([
            sys.executable,
            str(extractor),
            "--trace", str(bad_trace),
            "--transcript", str(transcript_path),
            "--out", str(temp / "bad-events.json"),
            "--capacity", "4",
            "--checkpoint-min-step", "100",
        ], capture_output=True, text=True)
        assert failed.returncode != 0
        assert "duplicate trace request seed" in failed.stderr

        # The live reporter joins deterministic outputs to measured schema-7
        # records and rejects a silent/no-terminal intentional arm.
        live_records = [{
            "round": 0, "chain": 0, "reply_sha256": "a" * 64,
        }]
        historical_transcript = temp / "historical.json"
        intentional_transcript = temp / "intentional.json"
        for path, label in (
                (historical_transcript, "historical"),
                (intentional_transcript, "intentional")):
            path.write_text(json.dumps({
                "schema": "zc1_retention_shape_transcript/v1",
                "label": label,
                "records": live_records,
            }) + "\n", encoding="utf-8")

        def plan(mode: str, events: int) -> dict:
            return {
                "schema_version": 7,
                "outcome": "replayed",
                "n_reused_tokens": 10,
                "n_replayed_tokens": 20,
                "ttft_us": 30,
                "optimizer": {
                    "mode": mode,
                    "retention_summary": {
                        "outcome_counts": {
                            "executed": events,
                            "deferred": 0,
                            "publication_skipped": 0,
                            "blocked": 0,
                        },
                        "reason_counts": {"internal_fault": 0},
                        "released_bytes": events * 40,
                        "released_tokens": events * 20,
                    },
                },
            }

        historical_log = temp / "historical.log"
        intentional_log = temp / "intentional.log"
        historical_log.write_text(
            "CACHE_PLAN " + json.dumps(plan("off", 0)) + "\n",
            encoding="utf-8")
        intentional_log.write_text(
            "CACHE_PLAN " + json.dumps(plan("baseline", 1)) + "\n",
            encoding="utf-8")
        live_report = temp / "live-report.json"
        subprocess.run([
            sys.executable, str(reporter),
            "--historical-transcript", str(historical_transcript),
            "--historical-log", str(historical_log),
            "--intentional-transcript", str(intentional_transcript),
            "--intentional-log", str(intentional_log),
            "--label", "synthetic", "--out", str(live_report),
        ], check=True)
        report = json.loads(live_report.read_text(encoding="utf-8"))
        assert report["outputs_byte_identical"] is True
        assert report["intentional"]["retention_events"] == 1

    print("ZC1_RETENTION_TOOLS_TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
