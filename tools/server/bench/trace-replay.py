#!/usr/bin/env python3
"""Replay a captured harness trace against one server and score it.

The trace is an open-loop request stream: the recorded prompts are sent
verbatim, at their recorded arrival offsets, regardless of what this server
answers. Two arms therefore see byte-identical prompt sequences with identical
concurrency, which is what makes their scorecards directly comparable.

	./trace-replay.py --trace hermes-website.jsonl \\
		--target 127.0.0.1:8080 --label ours --out ours.scorecard.json

Score the production config (no --cache-debug); run a second replay with debug
enabled when you want the planner's reasoning for a specific request.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


BLOCK = 4096


def now_ms():
	return time.monotonic() * 1000.0


def percentile(values, fraction):
	if not values:
		return None
	ordered = sorted(values)
	index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
	return round(ordered[index], 3)


def load_trace(path):
	header = None
	records = []
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
			elif row.get("record") == "request" and "body" in row:
				records.append(row)
	records.sort(key=lambda row: row.get("t_arrival_ms") or 0.0)
	return header, records


def dominant_system_prefix(records):
	"""The main family's system preamble: the prefix hash carried by the most
	request volume. Sub-agents equipped with the same tooling share it; tight
	background calls do not."""
	weight = {}
	for row in records:
		fingerprint = row.get("fingerprint") or {}
		key = fingerprint.get("system_prefix_sha")
		if not key:
			continue
		weight[key] = weight.get(key, 0) + 1
	if not weight:
		return None
	return max(weight.items(), key=lambda item: item[1])[0]


def classify(row, dominant, args):
	"""Foreground vs background, per the system-prompt fingerprint ensemble.

	A productive sub-agent is equipped like the main agent (long shared tooling
	preamble); a backgrounded call ("summarize this for the UI", "write a
	memory") carries a tight targeted prompt. Streaming and n_predict are weak
	corroborators, never the decision on their own."""
	fingerprint = row.get("fingerprint") or {}
	system_chars = fingerprint.get("system_chars") or 0
	prefix = fingerprint.get("system_prefix_sha")
	tools = fingerprint.get("tools") or 0
	predict = fingerprint.get("n_predict")
	if prefix and dominant and prefix == dominant:
		return "foreground_main"
	if system_chars >= args.tooled_system_chars or tools >= args.tooled_tools:
		return "foreground_subagent"
	if isinstance(predict, int) and 0 < predict <= args.background_predict:
		return "background"
	if system_chars < args.tooled_system_chars and not row.get("stream"):
		return "background"
	return "foreground_subagent"


def tool_message_count(body):
	"""Tool results the harness has fed back into the conversation.

	A gap that ends with new tool results is tool-execution time (a script ran,
	a build happened); a gap with none is harness overhead or human idle. Both
	are arm-independent, but the split is what makes the headline honest: a 2x
	server speedup on a project that is 85% tool time is not a 2x project."""
	if not isinstance(body, dict):
		return 0
	count = 0
	for message in body.get("messages") or []:
		if not isinstance(message, dict):
			continue
		if message.get("role") == "tool":
			count += 1
			continue
		content = message.get("content")
		if isinstance(content, list):
			for block in content:
				if isinstance(block, dict) and block.get("type") in (
						"tool_result", "tool_use"):
					count += 1
	return count


def overlap_groups(records):
	"""Split the trace into dependency-chained groups.

	An agentic harness is serial: it cannot issue turn N+1 until turn N has
	answered, so a faster server compounds into a shorter project. Requests
	that genuinely overlapped in the capture (parallel sub-agents) stay
	together in one group; groups are chained, and the recorded harness think
	time between a group's last completion and the next arrival is replayed as
	a real gap. Wall clock is then a claim-grade end-to-end number: server time
	is the only arm-dependent term."""
	groups = []
	current = []
	current_end = None
	for row in records:
		arrival = row.get("t_arrival_ms") or 0.0
		end = arrival + (row.get("total_ms") or 0.0)
		if current and arrival < (current_end or 0.0):
			current.append(row)
			current_end = max(current_end, end)
			continue
		if current:
			groups.append({"rows": current, "end_ms": current_end})
		current = [row]
		current_end = end
	if current:
		groups.append({"rows": current, "end_ms": current_end})
	previous_end = None
	previous_tools = None
	for group in groups:
		first_arrival = min(row.get("t_arrival_ms") or 0.0 for row in group["rows"])
		group["think_ms"] = (max(0.0, first_arrival - previous_end)
			if previous_end is not None else 0.0)
		group["spread_ms"] = [
			(row.get("t_arrival_ms") or 0.0) - first_arrival for row in group["rows"]]
		tools = max(tool_message_count(row.get("body")) for row in group["rows"])
		group["tool_gap"] = (previous_tools is not None and tools > previous_tools)
		group["new_tool_results"] = (max(0, tools - previous_tools)
			if previous_tools is not None else 0)
		previous_end = group["end_ms"]
		previous_tools = tools
	return groups


def sse_first_content_offset(text):
	for line in text.splitlines():
		line = line.strip()
		if not line.startswith("data:"):
			continue
		data = line[5:].strip()
		if not data or data == "[DONE]":
			continue
		try:
			payload = json.loads(data)
		except ValueError:
			continue
		if payload.get("content"):
			return True
		choices = payload.get("choices")
		if isinstance(choices, list):
			for choice in choices:
				if not isinstance(choice, dict):
					continue
				delta = choice.get("delta")
				if isinstance(delta, dict) and delta.get("content"):
					return True
				if choice.get("text"):
					return True
	return False


def server_metrics(text, streaming):
	found = {}
	blobs = []
	if streaming:
		for line in text.splitlines():
			line = line.strip()
			if line.startswith("data:"):
				data = line[5:].strip()
				if data and data != "[DONE]":
					blobs.append(data)
	else:
		blobs.append(text)
	for blob in blobs:
		try:
			payload = json.loads(blob)
		except ValueError:
			continue
		if not isinstance(payload, dict):
			continue
		timings = payload.get("timings")
		if isinstance(timings, dict):
			found["timings"] = timings
		usage = payload.get("usage")
		if isinstance(usage, dict):
			found["usage"] = usage
		for key in ("tokens_evaluated", "tokens_cached", "id_slot"):
			if key in payload:
				found[key] = payload[key]
	return found


def fresh_prefill_tokens(metrics):
	"""Prompt tokens this server actually had to evaluate. Every reuse path
	(prefix hit, checkpoint restore, host entry) shows up here as a smaller
	number, which is why it is the honest cross-arm reuse metric."""
	timings = metrics.get("timings") or {}
	for key in ("prompt_n", "n_prompt_tokens_processed", "prompt_tokens_processed"):
		value = timings.get(key)
		if isinstance(value, (int, float)):
			return int(value)
	value = metrics.get("tokens_evaluated")
	if isinstance(value, (int, float)):
		return int(value)
	return None


def prompt_tokens_total(metrics):
	usage = metrics.get("usage") or {}
	value = usage.get("prompt_tokens")
	if isinstance(value, (int, float)):
		return int(value)
	cached = metrics.get("tokens_cached")
	fresh = fresh_prefill_tokens(metrics)
	if isinstance(cached, (int, float)) and fresh is not None:
		return int(cached) + fresh
	return None


def fire(row, args, results, lock):
	body = json.dumps(row["body"]).encode()
	url = f"http://{args.target}{row['path']}"
	request = urllib.request.Request(url, data=body, method=row.get("method", "POST"))
	request.add_header("Content-Type", "application/json")
	if args.api_key:
		request.add_header("Authorization", f"Bearer {args.api_key}")

	streaming = bool(row.get("stream"))
	start = now_ms()
	t_first_token = None
	text = ""
	status = 0
	error = None
	try:
		with urllib.request.urlopen(request, timeout=args.timeout) as response:
			status = response.status
			while True:
				chunk = response.read(BLOCK)
				if not chunk:
					break
				text += chunk.decode("utf-8", "replace")
				if streaming and t_first_token is None:
					if sse_first_content_offset(text):
						t_first_token = now_ms()
			if not streaming:
				t_first_token = now_ms()
	except urllib.error.HTTPError as http_error:
		status = http_error.code
		error = f"http {http_error.code}"
		try:
			text = http_error.read().decode("utf-8", "replace")
		except Exception:
			text = ""
	except Exception as generic:
		status = 599
		error = str(generic)

	end = now_ms()
	metrics = server_metrics(text, streaming)
	result = {
		"seq": row.get("seq"),
		"path": row.get("path"),
		"class": row.get("_class"),
		"stream": streaming,
		"status": status,
		"ttft_ms": round(t_first_token - start, 3) if t_first_token else None,
		"total_ms": round(end - start, 3),
		"fresh_prefill_tokens": fresh_prefill_tokens(metrics),
		"prompt_tokens": prompt_tokens_total(metrics),
	}
	if error:
		result["error"] = error
	with lock:
		results.append(result)


def summarize(results, args, wall_ms, timeline):
	scored = [row for row in results if (row.get("seq") or 0) >= args.warmup]
	ok = [row for row in scored if row.get("status") == 200]
	weights = {
		"foreground_main": 1.0,
		"foreground_subagent": args.subagent_weight,
		"background": args.background_weight,
	}

	def ttfts(rows):
		return [row["ttft_ms"] for row in rows if row.get("ttft_ms") is not None]

	by_class = {}
	for name in ("foreground_main", "foreground_subagent", "background"):
		rows = [row for row in ok if row.get("class") == name]
		values = ttfts(rows)
		by_class[name] = {
			"requests": len(rows),
			"ttft_p50_ms": percentile(values, 0.50),
			"ttft_p95_ms": percentile(values, 0.95),
			"fresh_prefill_tokens": sum(
				row["fresh_prefill_tokens"] for row in rows
				if row.get("fresh_prefill_tokens") is not None),
		}

	fresh_known = [row for row in ok if row.get("fresh_prefill_tokens") is not None]
	total_fresh = sum(row["fresh_prefill_tokens"] for row in fresh_known)
	total_prompt = sum(row["prompt_tokens"] for row in ok
		if row.get("prompt_tokens") is not None)
	weighted_fresh = sum(
		row["fresh_prefill_tokens"] * weights.get(row.get("class"), 1.0)
		for row in fresh_known)
	weighted_ttft = [
		row["ttft_ms"] * weights.get(row.get("class"), 1.0)
		for row in ok if row.get("ttft_ms") is not None]

	server_ms = max(0.0, wall_ms - timeline["think_ms_slept"])
	recorded_total = timeline["think_ms_recorded"]
	return {
		"label": args.label,
		"target": args.target,
		"timeline": {
			"groups": timeline["groups"],
			"wall_clock_ms": round(wall_ms, 3),
			"harness_gap_ms_slept": round(timeline["think_ms_slept"], 3),
			"harness_gap_ms_recorded": round(recorded_total, 3),
			"tool_gap_ms_recorded": round(timeline["think_ms_tool"], 3),
			"other_gap_ms_recorded": round(timeline["think_ms_other"], 3),
			"tool_gap_share": (round(timeline["think_ms_tool"] / recorded_total, 4)
				if recorded_total else None),
			"server_ms": round(server_ms, 3),
			"server_time_fraction": (round(server_ms / wall_ms, 4)
				if wall_ms else None),
			"gaps_capped": timeline["gaps_capped"],
			"think_ms_dropped_by_cap": round(timeline["think_ms_dropped_by_cap"], 3),
			"think_speed": args.think_speed,
			"faithful": (not args.no_think_time and args.think_speed == 1.0
				and timeline["gaps_capped"] == 0),
		},
		"requests_total": len(results),
		"requests_scored": len(scored),
		"requests_ok": len(ok),
		"status_other": sorted({row["status"] for row in scored
			if row.get("status") != 200}),
		"wall_clock_ms": round(wall_ms, 3),
		"ttft_p50_ms": percentile(ttfts(ok), 0.50),
		"ttft_p95_ms": percentile(ttfts(ok), 0.95),
		"ttft_weighted_p50_ms": percentile(weighted_ttft, 0.50),
		"ttft_weighted_p95_ms": percentile(weighted_ttft, 0.95),
		"latency_ms_total": round(sum(row["total_ms"] for row in ok), 3),
		"fresh_prefill_tokens": total_fresh,
		"fresh_prefill_reported_for": len(fresh_known),
		"prompt_tokens_total": total_prompt or None,
		"prefix_reuse_ratio": (round(1.0 - (total_fresh / total_prompt), 4)
			if total_prompt else None),
		"latency_weighted_fresh_prefill": round(weighted_fresh, 1),
		"weights": weights,
		"by_class": by_class,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--trace", required=True)
	parser.add_argument("--target", required=True, help="host:port")
	parser.add_argument("--label", required=True)
	parser.add_argument("--out", required=True)
	parser.add_argument("--mode", choices=("chained", "paced"), default="chained",
		help="chained: replay the harness dependency chain (wall clock is a "
			"claim-grade end-to-end number). paced: hold recorded arrival "
			"offsets (concurrency-faithful; wall clock is NOT a speedup claim)")
	parser.add_argument("--speed", type=float, default=1.0,
		help="paced mode only: arrival-pacing multiplier")
	parser.add_argument("--think-speed", type=float, default=1.0,
		help="chained mode: divide recorded harness think time by this")
	parser.add_argument("--no-think-time", action="store_true",
		help="chained mode: drop harness/tool time, measure server time only")
	parser.add_argument("--max-think-ms", type=float, default=0.0,
		help="chained mode: cap any single gap (0 = faithful). Capping is "
			"disclosed in the scorecard; it suppresses idle-driven cache work "
			"(idle reclaim, TTL expiry) so use only for absurd outliers")
	parser.add_argument("--warmup", type=int, default=0,
		help="skip the first N recorded requests when scoring")
	parser.add_argument("--timeout", type=float, default=900.0)
	parser.add_argument("--api-key", default=None)
	parser.add_argument("--tooled-system-chars", type=int, default=1500)
	parser.add_argument("--tooled-tools", type=int, default=3)
	parser.add_argument("--background-predict", type=int, default=256)
	parser.add_argument("--subagent-weight", type=float, default=1.0)
	parser.add_argument("--background-weight", type=float, default=0.1)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	header, records = load_trace(args.trace)
	if args.limit:
		records = records[:args.limit]
	if not records:
		raise SystemExit("trace contained no replayable requests")

	dominant = dominant_system_prefix(records)
	for row in records:
		row["_class"] = classify(row, dominant, args)

	results = []
	lock = threading.Lock()
	base = now_ms()

	timeline = {
		"groups": 0,
		"think_ms_recorded": 0.0,
		"think_ms_slept": 0.0,
		"think_ms_tool": 0.0,
		"think_ms_other": 0.0,
		"gaps_capped": 0,
		"think_ms_dropped_by_cap": 0.0,
	}

	if args.mode == "chained":
		groups = overlap_groups(records)
		timeline["groups"] = len(groups)
		for group in groups:
			recorded = group["think_ms"]
			timeline["think_ms_recorded"] += recorded
			if group["tool_gap"]:
				timeline["think_ms_tool"] += recorded
			else:
				timeline["think_ms_other"] += recorded
			slept = 0.0
			if recorded > 0 and not args.no_think_time:
				# Tool execution and human idle are arm-independent, but they are
				# NOT inert: idle-slot reclaim, TTL expiry and displacement saves
				# all happen in these gaps, so they are replayed faithfully by
				# default. The cap exists only for absurd outliers (a 30-minute
				# training job) and is disclosed in the scorecard.
				capped = recorded
				if args.max_think_ms and capped > args.max_think_ms:
					timeline["gaps_capped"] += 1
					timeline["think_ms_dropped_by_cap"] += capped - args.max_think_ms
					capped = args.max_think_ms
				slept = capped / args.think_speed
				time.sleep(slept / 1000.0)
			timeline["think_ms_slept"] += slept
			group_start = now_ms()
			threads = []
			for row, spread in zip(group["rows"], group["spread_ms"]):
				delay = (group_start + spread) - now_ms()
				if delay > 0:
					time.sleep(delay / 1000.0)
				thread = threading.Thread(
					target=fire, args=(row, args, results, lock), daemon=True)
				thread.start()
				threads.append(thread)
			for thread in threads:
				thread.join()
	else:
		threads = []
		first_offset = records[0].get("t_arrival_ms") or 0.0
		for row in records:
			if args.speed > 0:
				offset = ((row.get("t_arrival_ms") or 0.0) - first_offset) / args.speed
				delay = (base + offset) - now_ms()
				if delay > 0:
					time.sleep(delay / 1000.0)
			thread = threading.Thread(
				target=fire, args=(row, args, results, lock), daemon=True)
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join()
	wall_ms = now_ms() - base

	results.sort(key=lambda row: row.get("seq") or 0)
	scorecard = summarize(results, args, wall_ms, timeline)
	scorecard["mode"] = args.mode
	scorecard["think_time"] = (not args.no_think_time) and args.mode == "chained"
	scorecard["trace"] = args.trace
	scorecard["trace_tag"] = (header or {}).get("tag")
	scorecard["dominant_system_prefix"] = dominant
	scorecard["requests"] = results

	with open(args.out, "w") as handle:
		json.dump(scorecard, handle, indent=2)
		handle.write("\n")

	line = scorecard["timeline"]
	print(f"TRACE_REPLAY {args.label} timeline wall={line['wall_clock_ms']:.0f}ms"
		f" server={line['server_ms']:.0f}ms ({line['server_time_fraction']})"
		f" gaps={line['harness_gap_ms_slept']:.0f}ms"
		f" tool_share={line['tool_gap_share']}"
		f" faithful={line['faithful']}")
	print(f"TRACE_REPLAY {args.label} requests={scorecard['requests_ok']}"
		f"/{scorecard['requests_scored']} wall={scorecard['wall_clock_ms']:.0f}ms"
		f" ttft_p50={scorecard['ttft_p50_ms']} ttft_p95={scorecard['ttft_p95_ms']}"
		f" fresh_pp={scorecard['fresh_prefill_tokens']}"
		f" reuse={scorecard['prefix_reuse_ratio']}")
	if scorecard["status_other"]:
		print(f"TRACE_REPLAY {args.label} NON-200 statuses:"
			f" {scorecard['status_other']}")


if __name__ == "__main__":
	main()
