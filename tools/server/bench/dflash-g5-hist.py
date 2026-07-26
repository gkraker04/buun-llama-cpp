#!/usr/bin/env python3
"""Aggregate DFlash Gate-5 verify/rollback histograms from llama-server logs."""

import argparse
import json
import re
import sys
from collections import Counter


LINE_RE = re.compile(
    r"verify/rollback histogram \(draft/rejected:cycles\) = \(([^)]*)\)"
)
ITEM_RE = re.compile(r"(\d+)/(\d+):(\d+)")


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
        description="Aggregate draft-length/rollback-depth samples emitted by llama-server"
    )
    parser.add_argument("logs", nargs="*", help="server logs (default: stdin)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    sources = [open(path, "r", encoding="utf-8", errors="replace") for path in args.logs]
    if not sources:
        sources = [sys.stdin]

    joint: Counter[tuple[int, int]] = Counter()
    try:
        for source in sources:
            for line in source:
                match = LINE_RE.search(line)
                if not match:
                    continue
                for n_draft, rollback, count in ITEM_RE.findall(match.group(1)):
                    joint[(int(n_draft), int(rollback))] += int(count)
    finally:
        for source in sources:
            if source is not sys.stdin:
                source.close()

    if not joint:
        print("no verify/rollback histogram samples found", file=sys.stderr)
        return 1

    rollback_counts: Counter[int] = Counter()
    cycles = 0
    draft_rows = 0
    rollback_rows = 0
    for (n_draft, rollback), count in joint.items():
        cycles += count
        draft_rows += n_draft * count
        rollback_rows += rollback * count
        rollback_counts[rollback] += count

    result = {
        "cycles": cycles,
        "mean_draft": draft_rows / cycles,
        "mean_rollback": rollback_rows / cycles,
        "rollback_p50": percentile(rollback_counts, 0.50),
        "rollback_p90": percentile(rollback_counts, 0.90),
        "rollback_p95": percentile(rollback_counts, 0.95),
        "rollback_p99": percentile(rollback_counts, 0.99),
        "rollback_max": max(rollback_counts),
        "joint": [
            {"draft": n_draft, "rollback": rollback, "cycles": count}
            for (n_draft, rollback), count in sorted(joint.items())
        ],
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "cycles={cycles} mean_draft={mean_draft:.6f} "
            "mean_rollback={mean_rollback:.6f} p50={rollback_p50} "
            "p90={rollback_p90} p95={rollback_p95} p99={rollback_p99} "
            "max={rollback_max}".format(**result)
        )
        print(
            "hist="
            + ",".join(
                f"{item['draft']}/{item['rollback']}:{item['cycles']}"
                for item in result["joint"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
