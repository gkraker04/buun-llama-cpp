#!/usr/bin/env python3
"""Paired correctness-first comparison for dflash-g6-margin.py outputs."""

import argparse
import json
import math
import statistics


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def paired_summary(name, a, b, subset=None, label="all"):
    allowed = set(a) & set(b) if subset is None else set(subset)
    ids = sorted(
        case_id for case_id in allowed
        if a[case_id].get(name) is not None and
           b[case_id].get(name) is not None)
    delta = [a[case_id][name] - b[case_id][name] for case_id in ids]
    if not delta:
        print(f"{name}[{label}]: no jointly scorable cases")
        return
    mean = statistics.fmean(delta)
    if len(delta) > 1:
        sd = statistics.stdev(delta)
        t_stat = mean / (sd / math.sqrt(len(delta))) if sd else math.inf
    else:
        t_stat = math.nan
    worst_a = min(ids, key=lambda case_id: a[case_id][name])
    worst_b = min(ids, key=lambda case_id: b[case_id][name])
    print(
        f"{name}[{label}]: n={len(delta)} mean_delta_A-B={mean:.6f} "
        f"t={t_stat:.3f} A>B={sum(x > 0 for x in delta)} "
        f"B>A={sum(x < 0 for x in delta)} ties={sum(x == 0 for x in delta)} "
        f"minA={a[worst_a][name]:.6f}@{worst_a} "
        f"minB={b[worst_b][name]:.6f}@{worst_b}")


def case_min_output(row):
    margins = row.get("all_output_margins") or []
    return min(margins) if margins else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    a = load(args.a)
    b = load(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common case IDs")

    bad_restore_a = [
        case_id for case_id in common
        if (a[case_id].get("restore") or {}).get("result") != "installed"]
    bad_restore_b = [
        case_id for case_id in common
        if (b[case_id].get("restore") or {}).get("result") != "installed"]
    if bad_restore_a or bad_restore_b:
        raise SystemExit(
            f"restore evidence invalid: A={bad_restore_a[:5]} "
            f"B={bad_restore_b[:5]}")

    exact_a = sum(bool(a[case_id].get("exact")) for case_id in common)
    exact_b = sum(bool(b[case_id].get("exact")) for case_id in common)
    a_only = [
        case_id for case_id in common
        if a[case_id].get("exact") and not b[case_id].get("exact")]
    b_only = [
        case_id for case_id in common
        if b[case_id].get("exact") and not a[case_id].get("exact")]
    print(
        f"common={len(common)} exact_A={exact_a} exact_B={exact_b} "
        f"A_only={len(a_only)} B_only={len(b_only)}")
    print(f"A_only_ids={a_only}")
    print(f"B_only_ids={b_only}")

    # Correctness is reported before margins. Target-span margins are primary;
    # all-output minimum is a secondary fragility lens.
    paired_summary("target_min_margin", a, b)
    groups = {
        "first_half": common[:len(common) // 2],
        "last_half": common[len(common) // 2:],
        "even": common[::2],
        "odd": common[1::2],
    }
    for label, ids in groups.items():
        paired_summary(
            "target_min_margin", a, b, subset=ids, label=label)
    a_output = {
        case_id: {"all_output_min_margin": case_min_output(row)}
        for case_id, row in a.items()
    }
    b_output = {
        case_id: {"all_output_min_margin": case_min_output(row)}
        for case_id, row in b.items()
    }
    paired_summary("all_output_min_margin", a_output, b_output)

    unscorable_a = sum(
        a[case_id].get("target_min_margin") is None for case_id in common)
    unscorable_b = sum(
        b[case_id].get("target_min_margin") is None for case_id in common)
    missing_runner_a = sum(
        a[case_id].get("all_output_unscorable", 0) for case_id in common)
    missing_runner_b = sum(
        b[case_id].get("all_output_unscorable", 0) for case_id in common)
    print(
        f"target_unscorable_A={unscorable_a} "
        f"target_unscorable_B={unscorable_b} "
        f"missing_runner_up_tokens_A={missing_runner_a} "
        f"missing_runner_up_tokens_B={missing_runner_b}")


if __name__ == "__main__":
    main()
