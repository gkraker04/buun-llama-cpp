#!/usr/bin/env python3
"""Describe a captured trace: gap taxonomy, request classes, compaction events.

Answers the questions a capture session raises before any replay happens: where
did the wall clock actually go, which requests are foreground work, and where
did the harness compact the conversation.

	./trace-analyze.py --trace hermes-capstone-01.jsonl --out analysis.json

Gap taxonomy matters for the headline. A gap between requests is one of:
  tool      - the next request carries new tool results (a script/build ran)
  human     - the next request carries a new user message (someone was typing,
              or eating dinner); arm-independent AND not machine work
  harness   - neither; the harness itself was thinking
Only server time is arm-dependent. Tool time is real machine work that belongs
in an end-to-end claim; human idle is not, and including hours of it would
dilute any ratio toward 1.0 while telling the reader nothing.
"""

import argparse
import json
from collections import Counter


def load(path):
	header = None
	rows = []
	with open(path, errors="replace") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				row = json.loads(line)
			except ValueError:
				continue
			if row.get("record") == "capture_header":
				header = row
			elif row.get("record") == "request":
				rows.append(row)
	rows.sort(key=lambda r: r.get("t_arrival_ms") or 0.0)
	return header, rows


def messages(row):
	body = row.get("body") or {}
	out = []
	for message in body.get("messages") or []:
		if not isinstance(message, dict):
			continue
		content = message.get("content")
		if isinstance(content, list):
			content = json.dumps(content, sort_keys=True)[:400]
		out.append((message.get("role"), (content or "")[:400]))
	return out


def role_counts(msgs):
	return Counter(role for role, _ in msgs)


def shared_prefix(a, b):
	n = 0
	for x, y in zip(a, b):
		if x != y:
			break
		n += 1
	return n


def overlap_groups(rows):
	groups = []
	current = []
	end = None
	for row in rows:
		arrival = row.get("t_arrival_ms") or 0.0
		finish = arrival + (row.get("total_ms") or 0.0)
		if current and arrival < (end or 0.0):
			current.append(row)
			end = max(end, finish)
			continue
		if current:
			groups.append({"rows": current, "end_ms": end})
		current = [row]
		end = finish
	if current:
		groups.append({"rows": current, "end_ms": end})
	return groups


COMPACTION_MARKERS = ("summarization agent", "context checkpoint")


def is_compaction(row):
	"""Hermes compacts by asking a summarizer to rewrite the conversation."""
	body = row.get("body") or {}
	for message in body.get("messages") or []:
		if not isinstance(message, dict):
			continue
		content = message.get("content")
		if isinstance(content, str):
			lowered = content[:600].lower()
			if any(marker in lowered for marker in COMPACTION_MARKERS):
				return True
	return False


def classify_gap(index, groups, main_family):
	"""Attribute a wall-clock gap to what ended it.

	Role counts are only comparable WITHIN a conversation family: a compaction
	or background call is its own tiny conversation, so comparing it against the
	main thread makes every gap look like harness overhead. Look ahead to the
	next main-family request instead and ask what it gained relative to the
	previous main-family request — a new user message means someone was away, new
	tool results mean a tool ran."""
	previous_main = None
	for earlier in range(index - 1, -1, -1):
		candidates = [r for r in groups[earlier]["rows"]
			if (r.get("fingerprint") or {}).get("system_prefix_sha") == main_family]
		if candidates:
			previous_main = candidates[-1]
			break
	next_main = None
	compaction_first = False
	for later in range(index, len(groups)):
		rows = groups[later]["rows"]
		if any(is_compaction(r) for r in rows) and next_main is None:
			compaction_first = True
		candidates = [r for r in rows
			if (r.get("fingerprint") or {}).get("system_prefix_sha") == main_family]
		if candidates:
			next_main = candidates[0]
			break
	# A harness reconnect (someone came back to the machine) announces itself by
	# re-probing the metadata endpoints. Check this BEFORE role counts: a
	# compaction rewrites the conversation, so the turn after it can carry FEWER
	# user messages than the turn before, which makes role counting blind here.
	for later in range(index, min(index + 6, len(groups))):
		for row in groups[later]["rows"]:
			if row.get("path") in ("/api/tags", "/api/v1/models", "/v1/props"):
				return "human"
		if any((r.get("fingerprint") or {}).get("system_prefix_sha") == main_family
				for r in groups[later]["rows"]):
			break
	if previous_main is None or next_main is None:
		return "harness"
	before = role_counts(messages(previous_main))
	after = role_counts(messages(next_main))
	if after.get("user", 0) > before.get("user", 0):
		return "human"
	if after.get("tool", 0) > before.get("tool", 0):
		return "tool"
	if compaction_first:
		return "compaction_wait"
	return "harness"


def detect_compactions(rows):
	"""A compaction replaces a long conversation with a summary: the message
	list collapses and the shared prefix with the previous turn breaks."""
	events = []
	by_family = {}
	for row in rows:
		if row.get("path") != "/v1/chat/completions":
			continue
		key = (row.get("fingerprint") or {}).get("system_prefix_sha")
		by_family.setdefault(key, []).append(row)
	for key, family in by_family.items():
		for previous, current in zip(family, family[1:]):
			before = messages(previous)
			after = messages(current)
			if len(before) < 8 or len(after) >= len(before):
				continue
			kept = shared_prefix(before, after)
			# A real compaction drops most turns AND breaks the prefix; a plain
			# new conversation on the same system prompt keeps only the system
			# message too, so require that the drop be large.
			if len(after) <= len(before) / 2 and kept <= max(2, len(after) // 2):
				events.append({
					"family": key,
					"at_seq": current.get("seq"),
					"t_arrival_ms": current.get("t_arrival_ms"),
					"messages_before": len(before),
					"messages_after": len(after),
					"shared_prefix_messages": kept,
				})
	events.sort(key=lambda e: e["t_arrival_ms"] or 0)
	return events


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--trace", required=True)
	parser.add_argument("--out", default=None)
	parser.add_argument("--tooled-system-chars", type=int, default=1500)
	args = parser.parse_args()

	header, rows = load(args.trace)
	chats = [r for r in rows if r.get("path") == "/v1/chat/completions"]
	if not chats:
		raise SystemExit("no chat completions in trace")

	span_ms = max((r["t_arrival_ms"] + (r.get("total_ms") or 0.0)) for r in rows)
	groups = overlap_groups(rows)

	main_family = Counter(
		(r.get("fingerprint") or {}).get("system_prefix_sha") for r in chats
	).most_common(1)[0][0]
	gaps = {"tool": 0.0, "human": 0.0, "harness": 0.0, "compaction_wait": 0.0}
	gap_counts = Counter()
	longest = []
	previous = None
	for index, group in enumerate(groups):
		if previous is not None:
			first = min(r["t_arrival_ms"] for r in group["rows"])
			delta = max(0.0, first - previous["end_ms"])
			kind = classify_gap(index, groups, main_family)
			gaps[kind] += delta
			gap_counts[kind] += 1
			longest.append((delta, kind, group["rows"][0].get("seq")))
		previous = group
	longest.sort(reverse=True)

	busy_ms = sum((r.get("total_ms") or 0.0) for r in chats)
	server_ms = span_ms - sum(gaps.values())

	families = Counter((r.get("fingerprint") or {}).get("system_prefix_sha") for r in chats)
	dominant = main_family

	compaction_calls = [r for r in chats if is_compaction(r)]
	compaction_server_ms = sum((r.get("total_ms") or 0.0) for r in compaction_calls)
	compaction_fresh = sum(
		(((r.get("server") or {}).get("timings") or {}).get("prompt_n") or 0)
		for r in compaction_calls)

	def klass(row):
		fingerprint = row.get("fingerprint") or {}
		if fingerprint.get("system_prefix_sha") == dominant:
			return "foreground_main"
		if (fingerprint.get("system_chars") or 0) >= args.tooled_system_chars:
			return "foreground_subagent"
		return "background"

	classes = Counter(klass(r) for r in chats)
	fresh_by_class = Counter()
	for row in chats:
		fresh = ((row.get("server") or {}).get("timings") or {}).get("prompt_n")
		if isinstance(fresh, int):
			fresh_by_class[klass(row)] += fresh

	compactions = detect_compactions(rows)

	report = {
		"trace": args.trace,
		"tag": (header or {}).get("tag"),
		"requests": {
			"total": len(rows),
			"chat_completions": len(chats),
			"non_chat": len(rows) - len(chats),
			"statuses": dict(Counter(r.get("status") for r in rows)),
		},
		"timeline_hours": round(span_ms / 3600000.0, 3),
		"where_the_time_went_ms": {
			"server_busy_or_overlap": round(server_ms, 1),
			"tool_execution": round(gaps["tool"], 1),
			"human_idle": round(gaps["human"], 1),
			"harness_overhead": round(gaps["harness"], 1),
			"compaction_wait": round(gaps["compaction_wait"], 1),
		},
		"where_the_time_went_pct": {
			k: round(100.0 * v / span_ms, 1) for k, v in {
				"server_busy_or_overlap": server_ms,
				"tool_execution": gaps["tool"],
				"human_idle": gaps["human"],
				"harness_overhead": gaps["harness"],
				"compaction_wait": gaps["compaction_wait"],
			}.items()
		},
		"gap_counts": dict(gap_counts),
		"longest_gaps": [
			{"seconds": round(d / 1000.0, 1), "kind": k, "before_seq": s}
			for d, k, s in longest[:10]
		],
		"request_seconds_sum": round(busy_ms / 1000.0, 1),
		"concurrency": {
			"groups": len(groups),
			"max_parallel": max(len(g["rows"]) for g in groups),
			"groups_with_parallel": sum(1 for g in groups if len(g["rows"]) > 1),
		},
		"compaction_calls": {
			"count": len(compaction_calls),
			"server_seconds": round(compaction_server_ms / 1000.0, 1),
			"share_of_session_pct": round(100.0 * compaction_server_ms / span_ms, 1),
			"fresh_prefill_tokens": compaction_fresh,
			"ttft_seconds": [round((r.get("ttft_ms") or 0) / 1000.0, 1)
				for r in compaction_calls],
		},
		"classes": dict(classes),
		"fresh_prefill_by_class": dict(fresh_by_class),
		"families": {str(k): v for k, v in families.most_common()},
		"compactions": compactions,
	}

	if args.out:
		with open(args.out, "w") as handle:
			json.dump(report, handle, indent=2)
			handle.write("\n")

	pct = report["where_the_time_went_pct"]
	print(f"=== {report['tag']} — {report['timeline_hours']}h, "
		f"{report['requests']['chat_completions']} completions ===")
	print(f"server/busy      {pct['server_busy_or_overlap']:>5}%")
	print(f"tool execution   {pct['tool_execution']:>5}%   ({gap_counts['tool']} gaps)")
	print(f"human idle       {pct['human_idle']:>5}%   ({gap_counts['human']} gaps)")
	print(f"harness overhead {pct['harness_overhead']:>5}%   ({gap_counts['harness']} gaps)")
	if "compaction_wait" in pct:
		print(f"compaction wait  {pct['compaction_wait']:>5}%   "
			f"({gap_counts.get('compaction_wait', 0)} gaps)")
	comp = report["compaction_calls"]
	print(f"COMPACTION: {comp['count']} calls, {comp['server_seconds']}s server "
		f"({comp['share_of_session_pct']}% of session), "
		f"{comp['fresh_prefill_tokens']:,} fresh prefill tokens, "
		f"TTFT {comp['ttft_seconds']}")
	print(f"concurrency: {report['concurrency']['max_parallel']} max parallel, "
		f"{report['concurrency']['groups_with_parallel']}/{report['concurrency']['groups']} "
		f"groups had overlap")
	print(f"classes: {report['classes']}")
	print(f"fresh prefill by class: {report['fresh_prefill_by_class']}")
	print(f"compactions detected: {len(compactions)}")
	for event in compactions:
		print(f"  seq {event['at_seq']}: {event['messages_before']} -> "
			f"{event['messages_after']} messages "
			f"(kept {event['shared_prefix_messages']})")
	print("longest gaps:")
	for gap in report["longest_gaps"][:5]:
		print(f"  {gap['seconds']:>8.1f}s {gap['kind']} (before seq {gap['before_seq']})")


if __name__ == "__main__":
	main()
