#!/usr/bin/env python3
"""Diff capstone scorecards into headline numbers and a regression list.

	./trace-compare.py --baseline master.scorecard.json --arm ours.scorecard.json

Ordinary two-file use is exploratory. ``--claim-grade`` is deliberately a
closed six-scorecard proof: intentional baseline, branch-with-features-disabled
parity, active arm, a concurrent baseline null, and a serialized baseline null
pair. It also consumes a separate debug probe proving the active planner was
fitted and actually authoritative.
"""

import argparse
import json

from cache_plan_common import iter_cache_plan_records


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


CONCURRENT_EXACT_FIELDS = (
	"status", "generated_tokens", "generated_recorded", "truncated",
)
SERIAL_EXACT_FIELDS = CONCURRENT_EXACT_FIELDS + (
	"fresh_prefill_tokens", "prompt_tokens",
)


def load_scorecard(path):
	with open(path) as handle:
		return json.load(handle)


def scorecard_claim_reasons(name, scorecard):
	claim = scorecard.get("claim_validation") or {}
	if not claim.get("requested"):
		return [f"{name}:replay:claim_grade_not_requested"]
	if not claim.get("valid"):
		return [f"{name}:replay:" + reason
			for reason in claim.get("reasons") or ["claim_validation_missing"]]
	return []


def parity_reasons(name, left, right, require_output=False, fields=SERIAL_EXACT_FIELDS):
	"""Compare behavior evidence, not concurrent floating-point output hashes."""
	reasons = []
	left_rows = index_requests(left)
	right_rows = index_requests(right)
	if set(left_rows) != set(right_rows):
		reasons.append(f"{name}:request_set_mismatch")
	for seq in sorted(set(left_rows) & set(right_rows)):
		for field in fields:
			if left_rows[seq].get(field) != right_rows[seq].get(field):
				reasons.append(f"{name}:seq={seq}:{field}")
		if require_output and left_rows[seq].get("output_sha") != right_rows[seq].get("output_sha"):
			reasons.append(f"{name}:seq={seq}:output_sha")
	return reasons


def aggregate(rows, field):
	values = [row.get(field) for row in rows.values() if row.get(field) is not None]
	return len(values), sum(values)


def concurrent_parity_reasons(baseline, parity, null_concurrent):
	"""Branch-off drift must not exceed the cache noise measured by the null."""
	reasons = parity_reasons(
		"branch_no_features", baseline, parity, fields=CONCURRENT_EXACT_FIELDS)
	reasons.extend(parity_reasons(
		"null_concurrent", baseline, null_concurrent,
		fields=CONCURRENT_EXACT_FIELDS))
	base_rows = index_requests(baseline)
	parity_rows = index_requests(parity)
	null_rows = index_requests(null_concurrent)
	for field in ("fresh_prefill_tokens", "prompt_tokens"):
		base_known, base_total = aggregate(base_rows, field)
		parity_known, parity_total = aggregate(parity_rows, field)
		null_known, null_total = aggregate(null_rows, field)
		if parity_known != base_known or null_known != base_known:
			reasons.append(f"concurrent:{field}:coverage_mismatch")
			continue
		if abs(parity_total - base_total) > abs(null_total - base_total):
			reasons.append(f"branch_no_features:{field}:outside_null_envelope")
	return reasons


def workload_reasons(name, baseline, other):
	reasons = []
	if not baseline.get("trace_tag") or baseline.get("trace_tag") != other.get("trace_tag"):
		reasons.append(f"{name}:trace_tag_mismatch")
	base_rows = index_requests(baseline)
	other_rows = index_requests(other)
	if set(base_rows) != set(other_rows):
		reasons.append(f"{name}:request_set_mismatch")
	for seq in sorted(set(base_rows) & set(other_rows)):
		if base_rows[seq].get("generated_recorded") != other_rows[seq].get("generated_recorded"):
			reasons.append(f"{name}:seq={seq}:captured_generation_mismatch")
	return reasons


def planner_evidence(paths):
	statuses = {}
	authoritative = []
	total = 0
	for rec in iter_cache_plan_records(paths):
		total += 1
		status = rec.get("planner_status", "missing")
		statuses[status] = statuses.get(status, 0) + 1
		authority = rec.get("authority") or {}
		if (status == "ok" and authority.get("state") == "authoritative"
				and isinstance(authority.get("executed_plan_candidate"), int)
				and authority.get("decision_tier") not in (None, "none")):
			authoritative.append({
				"id_task": rec.get("id_task"),
				"decision_tier": authority.get("decision_tier"),
				"executed_plan_candidate": authority.get("executed_plan_candidate"),
			})
	return {
		"records": total,
		"planner_status": statuses,
		"authoritative_count": len(authoritative),
		"authoritative_examples": authoritative[:8],
	}


def validate_claim_arms(base, parity, arm, null_concurrent,
		serialized_base, serialized_null, evidence):
	reasons = []
	for name, scorecard in (
			("baseline", base), ("branch_no_features", parity), ("active", arm),
			("null_concurrent", null_concurrent),
			("null_serialized_baseline", serialized_base),
			("null_serialized_repeat", serialized_null)):
		reasons.extend(scorecard_claim_reasons(name, scorecard))
		if not (scorecard.get("timeline") or {}).get("faithful"):
			reasons.append(f"{name}:timeline_not_faithful")
		if name != "baseline":
			reasons.extend(workload_reasons(name, base, scorecard))
	for name, scorecard in (
			("baseline", base), ("branch_no_features", parity), ("active", arm),
			("null_concurrent", null_concurrent)):
		execution = scorecard.get("execution") or {}
		if execution.get("serialize_overlap"):
			reasons.append(f"{name}:unexpected_serialized_execution")
		if (execution.get("server_parallel_observed") or 0) < 2:
			reasons.append(f"{name}:concurrent_server_has_fewer_than_two_slots")
	for name, scorecard in (
			("null_serialized_baseline", serialized_base),
			("null_serialized_repeat", serialized_null)):
		execution = scorecard.get("execution") or {}
		if not execution.get("serialize_overlap"):
			reasons.append(f"{name}:overlap_not_serialized")
		if execution.get("server_parallel_observed") != 1:
			reasons.append(f"{name}:server_not_np1")
	reasons.extend(concurrent_parity_reasons(base, parity, null_concurrent))
	reasons.extend(parity_reasons(
		"null_serialized", serialized_base, serialized_null, require_output=True))
	if evidence.get("planner_status", {}).get("ok", 0) == 0:
		reasons.append("active_evidence:planner_status_not_ok")
	if evidence.get("authoritative_count", 0) == 0:
		reasons.append("active_evidence:no_authoritative_execution")
	return reasons


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--baseline", required=True)
	parser.add_argument("--arm", required=True)
	parser.add_argument("--claim-grade", action="store_true",
		help="require branch-no-feature parity, concurrent and serialized nulls, "
			"and separate fitted authoritative planner evidence")
	parser.add_argument("--parity", help="branch scorecard with cache features disabled")
	parser.add_argument("--null-concurrent", help="repeat baseline under scoring concurrency")
	parser.add_argument("--serialized-baseline", help="baseline replay at -np 1")
	parser.add_argument("--serialized-null", help="repeat baseline replay at -np 1")
	parser.add_argument("--planner-evidence", action="append", default=[],
		help="CACHE_PLAN debug log from a separate active-arm probe (repeatable)")
	parser.add_argument("--out", default=None)
	parser.add_argument("--top", type=int, default=15)
	args = parser.parse_args()

	base = load_scorecard(args.baseline)
	arm = load_scorecard(args.arm)
	claim = {
		"requested": bool(args.claim_grade),
		"valid": False,
		"reasons": [],
	}
	if args.claim_grade:
		missing = [name for name, value in (
			("--parity", args.parity),
			("--null-concurrent", args.null_concurrent),
			("--serialized-baseline", args.serialized_baseline),
			("--serialized-null", args.serialized_null),
			("--planner-evidence", args.planner_evidence),
		) if not value]
		if missing:
			raise SystemExit("claim-grade comparison requires " + ", ".join(missing))
		parity = load_scorecard(args.parity)
		null_concurrent = load_scorecard(args.null_concurrent)
		serialized_base = load_scorecard(args.serialized_baseline)
		serialized_null = load_scorecard(args.serialized_null)
		evidence = planner_evidence(args.planner_evidence)
		claim["reasons"] = validate_claim_arms(
			base, parity, arm, null_concurrent, serialized_base, serialized_null,
			evidence)
		claim["valid"] = not claim["reasons"]
		claim["planner_evidence"] = evidence

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
		"output_identity": None,
		"context_overflow": {
			"baseline_truncated": base.get("truncated_requests"),
			"arm_truncated": arm.get("truncated_requests"),
		},
		"decode_comparability": {
			"baseline_generated_tokens": base.get("generated_tokens_total"),
			"arm_generated_tokens": arm.get("generated_tokens_total"),
			"generated_delta_pct": percent_drop(
				base.get("generated_tokens_total"), arm.get("generated_tokens_total")),
			"baseline_sampling": base.get("sampling"),
			"arm_sampling": arm.get("sampling"),
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
	shared = [seq for seq in arm_rows if seq in base_rows]
	hashed = [seq for seq in shared
		if base_rows[seq].get("output_sha") and arm_rows[seq].get("output_sha")]
	matched = [seq for seq in hashed
		if base_rows[seq]["output_sha"] == arm_rows[seq]["output_sha"]]
	headline["output_identity"] = {
		"compared": len(hashed),
		"identical": len(matched),
		"first_divergence_seq": next(
			(seq for seq in sorted(hashed) if seq not in matched), None),
	}
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
		"claim_validation": claim,
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
	overflow = headline["context_overflow"]
	if (overflow.get("baseline_truncated") or overflow.get("arm_truncated")):
		print(f"context truncation  {overflow['baseline_truncated']:>12} -> "
			f"{overflow['arm_truncated']:>12} requests")
		if overflow["baseline_truncated"] != overflow["arm_truncated"]:
			print("  NOTE: arms truncated different numbers of prompts — this is a "
				"capability difference, not just a speed one. Report it beside "
				"the timing table.")
	identity = headline["output_identity"]
	if identity and identity["compared"]:
		print(f"output identity     {identity['identical']:>12}"
			f" / {identity['compared']} requests byte-identical")
		if identity["identical"] < identity["compared"]:
			print("  NOTE: concurrent output bytes are diagnostic only: batch "
				"composition can flip near-ties even in a static-KV null replay, "
				"and VBR adds pressure-driven tier differences. The mandatory "
				"serialized null is the determinism oracle — first divergence "
				f"here was seq {identity['first_divergence_seq']}.")
	decode = headline["decode_comparability"]
	base_gen = decode["baseline_generated_tokens"] or 0
	arm_gen = decode["arm_generated_tokens"] or 0
	skew = abs(base_gen - arm_gen) / base_gen if base_gen else 0.0
	print(f"decode work         {base_gen:>12} -> {arm_gen:>12} tokens"
		f"  (skew {skew * 100:.1f}%)")
	if skew > 0.02 and not (base.get("sampling") or {}).get("pinned_generation"):
		print("  WARNING: arms did unequal decode work — wall clock and total "
			"latency are polluted by generation-length divergence. Re-run with "
			"--greedy --pin-generation for a claim-grade wall clock.")
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
	if args.claim_grade:
		print("claim validation    " + ("PASS" if claim["valid"] else "FAIL"))
		for reason in claim["reasons"]:
			print(f"  {reason}")
		if not claim["valid"]:
			raise SystemExit(2)


if __name__ == "__main__":
	main()
