#!/usr/bin/env python3
"""Diff two trace-replay scorecards into headline numbers and a regression list.

	./trace-compare.py --baseline master.scorecard.json --arm ours.scorecard.json

Prints the claim-grade summary (both arms scored by the same instrument on the
same request stream) and the per-request rows that moved most, which is the
input to policy tuning.
"""

import argparse
import json


def ratio(baseline, arm):
	if baseline in (None, 0) or arm in (None, 0):
		return None
	return round(baseline / arm, 3)


def percent_drop(baseline, arm):
	if baseline in (None, 0) or arm is None:
		return None
	return round(100.0 * (baseline - arm) / baseline, 1)


def index_requests(scorecard):
	return {row.get("seq"): row for row in scorecard.get("requests") or []}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--baseline", required=True)
	parser.add_argument("--arm", required=True)
	parser.add_argument("--out", default=None)
	parser.add_argument("--top", type=int, default=15)
	args = parser.parse_args()

	with open(args.baseline) as handle:
		base = json.load(handle)
	with open(args.arm) as handle:
		arm = json.load(handle)

	base_line = base.get("timeline") or {}
	arm_line = arm.get("timeline") or {}
	headline = {
		"baseline_label": base.get("label"),
		"arm_label": arm.get("label"),
		"trace": arm.get("trace"),
		"requests_scored": arm.get("requests_scored"),
		"project_speedup_x": ratio(base.get("wall_clock_ms"), arm.get("wall_clock_ms")),
		"server_time_speedup_x": ratio(base_line.get("server_ms"), arm_line.get("server_ms")),
		"tool_and_idle_share_of_project": base_line.get("tool_gap_share"),
		"server_time_fraction": {
			"baseline": base_line.get("server_time_fraction"),
			"arm": arm_line.get("server_time_fraction"),
		},
		"timeline_faithful": {
			"baseline": base_line.get("faithful"),
			"arm": arm_line.get("faithful"),
		},
		"ttft_p50_speedup_x": ratio(base.get("ttft_p50_ms"), arm.get("ttft_p50_ms")),
		"ttft_p95_speedup_x": ratio(base.get("ttft_p95_ms"), arm.get("ttft_p95_ms")),
		"ttft_weighted_p95_speedup_x": ratio(
			base.get("ttft_weighted_p95_ms"), arm.get("ttft_weighted_p95_ms")),
		"fresh_prefill_reduction_pct": percent_drop(
			base.get("fresh_prefill_tokens"), arm.get("fresh_prefill_tokens")),
		"latency_weighted_fresh_prefill_reduction_pct": percent_drop(
			base.get("latency_weighted_fresh_prefill"),
			arm.get("latency_weighted_fresh_prefill")),
		"prefix_reuse_ratio": {
			"baseline": base.get("prefix_reuse_ratio"),
			"arm": arm.get("prefix_reuse_ratio"),
		},
	}

	by_class = {}
	for name, arm_stats in (arm.get("by_class") or {}).items():
		base_stats = (base.get("by_class") or {}).get(name) or {}
		by_class[name] = {
			"requests": arm_stats.get("requests"),
			"ttft_p50_speedup_x": ratio(
				base_stats.get("ttft_p50_ms"), arm_stats.get("ttft_p50_ms")),
			"ttft_p95_speedup_x": ratio(
				base_stats.get("ttft_p95_ms"), arm_stats.get("ttft_p95_ms")),
			"fresh_prefill_reduction_pct": percent_drop(
				base_stats.get("fresh_prefill_tokens"),
				arm_stats.get("fresh_prefill_tokens")),
		}

	base_rows = index_requests(base)
	arm_rows = index_requests(arm)
	deltas = []
	for seq, arm_row in arm_rows.items():
		base_row = base_rows.get(seq)
		if not base_row:
			continue
		if arm_row.get("status") != 200 or base_row.get("status") != 200:
			continue
		entry = {
			"seq": seq,
			"class": arm_row.get("class"),
			"ttft_base_ms": base_row.get("ttft_ms"),
			"ttft_arm_ms": arm_row.get("ttft_ms"),
			"fresh_base": base_row.get("fresh_prefill_tokens"),
			"fresh_arm": arm_row.get("fresh_prefill_tokens"),
		}
		if entry["ttft_base_ms"] is not None and entry["ttft_arm_ms"] is not None:
			entry["ttft_delta_ms"] = round(
				entry["ttft_arm_ms"] - entry["ttft_base_ms"], 3)
		if entry["fresh_base"] is not None and entry["fresh_arm"] is not None:
			entry["fresh_delta"] = entry["fresh_arm"] - entry["fresh_base"]
		deltas.append(entry)

	regressions = sorted(
		[row for row in deltas if (row.get("ttft_delta_ms") or 0) > 0],
		key=lambda row: -(row.get("ttft_delta_ms") or 0))[:args.top]
	wins = sorted(
		[row for row in deltas if (row.get("ttft_delta_ms") or 0) < 0],
		key=lambda row: (row.get("ttft_delta_ms") or 0))[:args.top]
	fresh_regressions = sorted(
		[row for row in deltas if (row.get("fresh_delta") or 0) > 0],
		key=lambda row: -(row.get("fresh_delta") or 0))[:args.top]

	report = {
		"headline": headline,
		"by_class": by_class,
		"top_ttft_regressions": regressions,
		"top_ttft_wins": wins,
		"top_fresh_prefill_regressions": fresh_regressions,
	}

	if args.out:
		with open(args.out, "w") as handle:
			json.dump(report, handle, indent=2)
			handle.write("\n")

	print(f"=== {base.get('label')} -> {arm.get('label')} "
		f"({arm.get('requests_scored')} scored requests) ===")
	print(f"project wall clock  {base.get('wall_clock_ms'):>12.0f} -> "
		f"{arm.get('wall_clock_ms'):>12.0f} ms  "
		f"({headline['project_speedup_x']}x)")
	print(f"server time only    {(base_line.get('server_ms') or 0):>12.0f} -> "
		f"{(arm_line.get('server_ms') or 0):>12.0f} ms  "
		f"({headline['server_time_speedup_x']}x)")
	print(f"  harness+tool gaps {(base_line.get('harness_gap_ms_slept') or 0):>12.0f} ms"
		f"  (tool share of gaps: {base_line.get('tool_gap_share')};"
		f" server = {base_line.get('server_time_fraction')} of project)")
	if not (base_line.get("faithful") and arm_line.get("faithful")):
		print("  NOTE: timeline not faithful (think time scaled/capped/dropped) —"
			" project wall clock is NOT a claim-grade number for these runs")
	print(f"TTFT p50            {str(base.get('ttft_p50_ms')):>12} -> "
		f"{str(arm.get('ttft_p50_ms')):>12} ms  "
		f"({headline['ttft_p50_speedup_x']}x)")
	print(f"TTFT p95            {str(base.get('ttft_p95_ms')):>12} -> "
		f"{str(arm.get('ttft_p95_ms')):>12} ms  "
		f"({headline['ttft_p95_speedup_x']}x)")
	print(f"fresh prefill tok   {str(base.get('fresh_prefill_tokens')):>12} -> "
		f"{str(arm.get('fresh_prefill_tokens')):>12}     "
		f"(-{headline['fresh_prefill_reduction_pct']}%)")
	print(f"prefix reuse        {str(base.get('prefix_reuse_ratio')):>12} -> "
		f"{str(arm.get('prefix_reuse_ratio')):>12}")
	for name, stats in by_class.items():
		print(f"  [{name}] n={stats['requests']} "
			f"ttft_p95 {stats['ttft_p95_speedup_x']}x "
			f"fresh -{stats['fresh_prefill_reduction_pct']}%")
	if regressions:
		print(f"top TTFT regressions (seq, class, +ms): " + ", ".join(
			f"{row['seq']}/{row['class']}/+{row['ttft_delta_ms']:.0f}"
			for row in regressions[:5]))


if __name__ == "__main__":
	main()
