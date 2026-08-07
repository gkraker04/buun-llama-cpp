#!/usr/bin/env python3
"""Lower one ZC1 pressure capture into the frozen counterfactual event codec."""

from __future__ import annotations

import argparse
import hashlib
import json


def digest(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--checkpoint-min-step", type=int, required=True)
    args = parser.parse_args()
    if args.capacity <= 0 or args.checkpoint_min_step <= 0:
        raise SystemExit("capacity and checkpoint-min-step must be positive")

    with open(args.transcript, encoding="utf-8") as source:
        transcript = json.load(source)
    if (
        transcript.get("schema") != "zc1_retention_pressure_transcript/v1"
        or transcript.get("chains") != 3
        or transcript.get("rounds") != 24
        or len(transcript.get("records", [])) != 72
    ):
        raise SystemExit("invalid pressure transcript")

    expected = {}
    for record in transcript["records"]:
        key = record.get("request_seed")
        if not isinstance(key, int) or key in expected:
            raise SystemExit("invalid or duplicate transcript request seed")
        expected[key] = record

    observed = {}
    with open(args.trace, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if (
                row.get("record") != "request"
                or row.get("path") != "/v1/chat/completions"
                or row.get("status") != 200
            ):
                continue
            body = row.get("body") or {}
            seed = body.get("seed")
            timings = ((row.get("server") or {}).get("timings") or {})
            prompt_n = timings.get("prompt_n")
            cache_n = timings.get("cache_n")
            if seed not in expected or seed in observed:
                raise SystemExit("unexpected or duplicate trace request seed")
            if not isinstance(prompt_n, int) or not isinstance(cache_n, int):
                raise SystemExit("trace request lacks exact cache timings")
            frontier = prompt_n + cache_n
            if frontier <= 0:
                raise SystemExit("trace request has invalid prompt frontier")
            observed[seed] = frontier
    if observed.keys() != expected.keys():
        raise SystemExit("trace/transcript request set mismatch")

    chains = []
    for chain in range(3):
        events = []
        chain_records = sorted(
            (record for record in transcript["records"] if record["chain"] == chain),
            key=lambda record: record["round"],
        )
        if [record["round"] for record in chain_records] != list(range(24)):
            raise SystemExit("incomplete chain rounds")
        for record in chain_records:
            parent = record["parent_round"]
            frontier = observed[record["request_seed"]]
            if parent is not None:
                parent_frontier = events[parent]["frontier_tokens"]
                if frontier < parent_frontier:
                    raise SystemExit("child frontier precedes its parent")
            events.append({
                "node": record["round"],
                "parent_node": parent,
                "frontier_tokens": frontier,
                "deep_rewind": record["transition"] == "deep_rewind",
            })
        chains.append({"chain": chain, "events": events})

    result = {
        "schema": "zc1_retention_events/v1",
        "source_trace_sha256": digest(args.trace),
        "source_transcript_sha256": digest(args.transcript),
        "capacity": args.capacity,
        "checkpoint_min_step": args.checkpoint_min_step,
        "chains": chains,
    }
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(f"ZC1_RETENTION_EXTRACT PASS events=72 out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

