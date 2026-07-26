#!/usr/bin/env python3
"""Aggregate Gate-5 edit/divergence samples from llama-server logs."""

import argparse
import json
import re
import sys
from collections import Counter


LINE_RE = re.compile(
    r"edit/divergence sample "
    r"\(cached/incoming/lcp/reusable/rewind/append/cache_prompt\) = "
    r"\((\d+)/(\d+)/(\d+)/(\d+)/(\d+)/(\d+)/([01])\)"
)


def percentile(values: Counter, q: float) -> int | None:
    total = sum(values.values())
    if total == 0:
        return None
    target = max(1, int(total * q + 0.999999999))
    seen = 0
    for value in sorted(values):
        seen += values[value]
        if seen >= target:
            return value
    raise AssertionError("unreachable percentile")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate logical prompt-edit depths emitted by llama-server"
    )
    parser.add_argument("logs", nargs="*", help="server logs (default: stdin)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    sources = [open(path, "r", encoding="utf-8", errors="replace") for path in args.logs]
    if not sources:
        sources = [sys.stdin]

    samples: Counter[tuple[int, int, int, int, int, int, int]] = Counter()
    try:
        for source in sources:
            for line in source:
                match = LINE_RE.search(line)
                if match:
                    samples[tuple(map(int, match.groups()))] += 1
    finally:
        for source in sources:
            if source is not sys.stdin:
                source.close()

    if not samples:
        print("no edit/divergence samples found", file=sys.stderr)
        return 1

    warm_rewinds: Counter[int] = Counter()
    reusable_rewinds: Counter[int] = Counter()
    n_cold = 0
    n_extension = 0
    n_cache_disabled = 0
    n_events = sum(samples.values())

    rows = []
    for sample, count in sorted(samples.items()):
        cached, incoming, lcp, reusable, rewind, append, cache_prompt = sample
        rows.append(
            {
                "cached": cached,
                "incoming": incoming,
                "lcp": lcp,
                "reusable": reusable,
                "rewind": rewind,
                "append": append,
                "cache_prompt": bool(cache_prompt),
                "events": count,
            }
        )
        if not cache_prompt:
            n_cache_disabled += count
        if lcp == 0:
            n_cold += count
        elif rewind == 0:
            n_extension += count
        else:
            # Logical user branch/regenerate depth.
            warm_rewinds[rewind] += count
            # Work required by the current reuse policy. This can exceed the
            # content divergence when adapter/alora constraints cap reuse.
            reusable_rewinds[max(0, cached - reusable)] += count

    def stats(values: Counter) -> dict:
        total = sum(values.values())
        return {
            "events": total,
            "mean": (
                sum(value * count for value, count in values.items()) / total
                if total
                else None
            ),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
            "hist": [
                {"depth": value, "events": count}
                for value, count in sorted(values.items())
            ],
        }

    result = {
        "events": n_events,
        "cold_lcp0": n_cold,
        "pure_extensions": n_extension,
        "cache_disabled": n_cache_disabled,
        "warm_edit_rewind": stats(warm_rewinds),
        "policy_rewind": stats(reusable_rewinds),
        "samples": rows,
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        warm = result["warm_edit_rewind"]
        policy = result["policy_rewind"]
        print(
            f"events={n_events} cold_lcp0={n_cold} "
            f"pure_extensions={n_extension} cache_disabled={n_cache_disabled}"
        )
        print(
            "warm_edit_rewind: "
            f"events={warm['events']} mean={warm['mean']} p50={warm['p50']} "
            f"p90={warm['p90']} p95={warm['p95']} p99={warm['p99']} "
            f"max={warm['max']}"
        )
        print(
            "policy_rewind: "
            f"events={policy['events']} mean={policy['mean']} p50={policy['p50']} "
            f"p90={policy['p90']} p95={policy['p95']} p99={policy['p99']} "
            f"max={policy['max']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
