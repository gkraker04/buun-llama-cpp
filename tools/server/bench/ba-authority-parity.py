#!/usr/bin/env python3
"""B-A authority ratchet gate: level=off vs authority-level parity + liveness.

Boots the same server twice (identical config except --cache-plan-authority)
and drives an identical forced-slot request sequence through both. Requires:
byte-identical content and logprobs across arms, forced-slot identity in every
response, live authority counters in the on arm (authoritative executions at
the flipped tier, verdicts recorded), and zero authoritative executions in the
off arm. A deliberately unfitted arm proves the typed-refusal path leaves
outputs untouched. No server backdoor: public routes + /slots debug JSON only.
"""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request


def request(base, path, body=None, timeout=600):
	headers = {}
	data = None
	if body is not None:
		data = json.dumps(body).encode()
		headers["Content-Type"] = "application/json"
	req = urllib.request.Request(base + path, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
	try:
		with urllib.request.urlopen(req, timeout=timeout) as response:
			return response.status, json.loads(response.read())
	except urllib.error.HTTPError as error:
		payload = error.read()
		try:
			return error.code, json.loads(payload)
		except Exception:
			return error.code, {"raw": payload.decode(errors="replace")}


class ServerArm:
	def __init__(self, args, level, port, log_path):
		self.level = level
		self.base = f"http://127.0.0.1:{port}"
		cmd = [args.server_bin, "-m", args.model, "--port", str(port),
			"-ngl", "99", "-c", str(args.ctx), "-b", str(args.batch),
			"-np", str(args.parallel), "-fa", "on",
			"--cache-ram", str(args.cache_ram), "--slots", "--cache-debug",
			"--cache-plan-authority", level, "--seed", str(args.seed)]
		if args.slot_prompt_similarity:
			cmd += ["--slot-prompt-similarity",
				str(args.slot_prompt_similarity)]
		if args.ctk:
			cmd += ["-ctk", args.ctk, "-ctv", args.ctv or args.ctk]
		self.log = open(log_path, "w")
		self.proc = subprocess.Popen(cmd, stdout=self.log,
			stderr=subprocess.STDOUT)

	def wait_healthy(self, deadline=360):
		start = time.time()
		while time.time() - start < deadline:
			try:
				status, _ = request(self.base, "/health", timeout=5)
				if status == 200:
					return
			except Exception:
				pass
			if self.proc.poll() is not None:
				raise RuntimeError(f"arm {self.level}: server exited early")
			time.sleep(2)
		raise RuntimeError(f"arm {self.level}: never healthy")

	def stop(self):
		self.proc.terminate()
		try:
			self.proc.wait(timeout=30)
		except subprocess.TimeoutExpired:
			self.proc.kill()
		self.log.close()


def sequence_steps(args):
	"""Forced-slot sequence covering live / host / cold candidate shapes,
	or (sequence=similarity) a non-forced battery whose reuse shapes select
	via the similarity tier: cold fills, short-suffix returns (live replay),
	a displacing newcomer, and post-displacement returns (host restore)."""
	filler = " ".join(
		f"authority parity row {i} keeps the ledger deterministic."
		for i in range(args.prompt_rows))
	convo_a = "Conversation A. " + filler
	convo_b = "Conversation B considers different state. " + filler
	if args.sequence == "forced":
		return [
			(0, convo_a),
			(0, convo_a),
			(0, convo_a + " Continue the ledger with one more row."),
			(0, convo_b),
			(0, convo_a),
			(1, convo_a),
		]
	convo_c = "Conversation C opens a third thread. " + filler
	if args.sequence == "route_home":
		# Main-agent/sub-agent displacement (plan §4.3): a long main context
		# homes on one slot; short sub-agent tasks churn the other; the main
		# agent returns with a continuation long enough to push similarity
		# below threshold while the home prefix stays nonzero -> route_home.
		# The final step is a genuinely new stream (LRU/cold control).
		# The continuation must outweigh the retained prefix so similarity
		# (lcp/prompt) falls below the threshold while the home lcp stays
		# large -- that is what routes the return through route_home.
		long_cont = " The main agent now resumes and reviews every ledger " \
			"row in detail, considering " + " and ".join(
			f"aspect {i} of the retained plan"
			for i in range(int(args.prompt_rows * 2.5))) + "."
		sub = "Sub-agent task: reply tersely. "
		return [
			(None, convo_a),
			(None, sub + "List one prime number."),
			(None, sub + "Name one color of the rainbow."),
			(None, convo_a + "<0>" + long_cont),
			(None, sub + "State one day of the week."),
			(None, convo_a + "<0>" + long_cont + "<3> Conclude briefly."),
			(None, "Entirely new stream D with no shared prefix beyond "
				"structural tokens. " + filler),
		]
	# Chat-echo battery: follow-ups accumulate the model's own reply (the
	# "<i>" placeholder splices step i's completion), so live replay is a
	# pure append (f_keep == 1.0). A trimming replay would rewind the
	# recurrent frontier on hybrid models and correctly demote — that arm
	# is structurally non-authoritative and not a liveness vehicle.
	return [
		(None, convo_a),
		(None, convo_b),
		(None, convo_a + "<0> Add one short clarifying row."),
		(None, convo_b + "<1> Add one short clarifying row."),
		(None, convo_c),
		(None, convo_a + "<0> Add one short clarifying row.<2> Summarize."),
		(None, convo_b + "<1> Add one short clarifying row.<3> Continue "
			"with a medium extension before stopping."),
	]


def drive_sequence(base, args):
	transcript = []
	for slot, prompt in sequence_steps(args):
		for i, row in enumerate(transcript):
			prompt = prompt.replace(f"<{i}>", row["content"] or "")
		body = {
			"prompt": prompt, "n_predict": args.n_predict,
			"temperature": 0, "seed": args.seed, "cache_prompt": True,
			"n_probs": args.n_probs,
		}
		if slot is not None:
			body["id_slot"] = slot
		status, payload = request(base, "/completion", body)
		if status != 200:
			raise RuntimeError(f"completion failed: {status} {payload}")
		probs = []
		for tok in payload.get("completion_probabilities") or []:
			probs.append((tok.get("id"), tok.get("logprob"), [
				(p.get("id"), p.get("logprob"))
				for p in tok.get("top_logprobs") or []]))
		transcript.append({
			"slot_requested": slot,
			"slot_served": payload.get("id_slot"),
			"content": payload.get("content"),
			"probs": probs,
			"ttft_ms": (payload.get("timings") or {}).get("prompt_ms"),
		})
	return transcript


def start_contention(base, args):
	"""Long background generation occupying one slot for the whole battery."""
	import threading

	def spin():
		request(base, "/completion", {
			"prompt": "Background contention stream. " * 30,
			"n_predict": args.contention_tokens, "temperature": 0,
			"seed": 11, "cache_prompt": False, "ignore_eos": True,
		}, timeout=1800)
	thread = threading.Thread(target=spin, daemon=True)
	thread.start()
	time.sleep(2)
	return thread


def authority_totals(base):
	status, slots = request(base, "/slots")
	if status != 200:
		raise RuntimeError(f"/slots failed: {status}")
	totals = {}
	for entry in slots:
		observer = (entry.get("cache_plan") or {}).get("observer") or {}
		for tier, counters in (observer.get("authority") or {}).items():
			bucket = totals.setdefault(tier, dict.fromkeys(counters, 0))
			for key, value in counters.items():
				bucket[key] = max(bucket[key], value)
	return totals


def run_arm(args, level, port, workdir):
	arm = ServerArm(args, level, port, f"{workdir}/server-{level}.log")
	try:
		arm.wait_healthy()
		if args.contention:
			start_contention(arm.base, args)
		transcript = drive_sequence(arm.base, args)
		totals = authority_totals(arm.base)
	finally:
		arm.stop()
	return transcript, totals


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-bin", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--workdir", required=True)
	parser.add_argument("--base-port", type=int, default=8440)
	parser.add_argument("--level", default="by_id",
		help="authority level for the on arm")
	parser.add_argument("--tier", default="by_id",
		help="tier whose counters must show authoritative executions")
	parser.add_argument("--ctx", type=int, default=4096)
	parser.add_argument("--batch", type=int, default=512)
	parser.add_argument("--parallel", type=int, default=2)
	parser.add_argument("--cache-ram", type=int, default=2048)
	parser.add_argument("--ctk", default="")
	parser.add_argument("--ctv", default="")
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--n-predict", type=int, default=16)
	parser.add_argument("--n-probs", type=int, default=8)
	parser.add_argument("--prompt-rows", type=int, default=40)
	parser.add_argument("--expect-fallback", default="",
		help="expected sole fallback_reason; arm must show zero "
		"authoritative executions (unfitted-profile negative cell)")
	parser.add_argument("--expect-shadow", action="store_true",
		help="the workload's selections fall outside the implemented "
		"authority domain (e.g. dynamic-VBR servers route reuse through "
		"route_home): assert parity + zero executions, not liveness")
	parser.add_argument("--sequence", default="forced",
		choices=["forced", "similarity", "route_home"],
		help="forced = id_slot battery (by_id tier); similarity = "
		"non-forced reuse battery (similarity tier); route_home = "
		"main/sub-agent displacement battery (route_home tier)")
	parser.add_argument("--slot-prompt-similarity", type=float, default=0.0,
		help="pass through to the server (0 = server default); the "
		"route_home battery needs a threshold the long continuation "
		"actually crosses")
	parser.add_argument("--liveness-optional", action="store_true",
		help="do not require authoritative executions (nondeterministic "
		"drift arms: parity + typed-fallback discipline still assert)")
	parser.add_argument("--allow-internal-fault", action="store_true",
		help="skip the record-level assertion that no receipt carries "
		"fallback_reason=internal_fault (deterministic gate cells must "
		"not book designed policy resets as faults)")
	parser.add_argument("--contention", action="store_true",
		help="run the battery under a long background generation; "
		"content parity is skipped (scheduling nondeterminism), "
		"liveness and TTFT sanity still assert")
	parser.add_argument("--contention-tokens", type=int, default=384)
	parser.add_argument("--ttft-max-regress", type=float, default=0.05,
		help="on-arm mean TTFT may not exceed off-arm mean by more "
		"than this fraction")
	args = parser.parse_args()

	off_transcript, off_totals = run_arm(
		args, "off", args.base_port, args.workdir)
	on_transcript, on_totals = run_arm(
		args, args.level, args.base_port + 1, args.workdir)

	internal_faults = 0
	if not args.allow_internal_fault:
		import re
		log_path = f"{args.workdir}/server-{args.level}.log"
		with open(log_path, errors="replace") as handle:
			for line in handle:
				match = re.search(r"CACHE_PLAN (\{.*)", line)
				if not match:
					continue
				try:
					rec = json.loads(match.group(1))
				except ValueError:
					continue
				authority = rec.get("authority") or {}
				if authority.get("fallback_reason") == "internal_fault":
					internal_faults += 1

	failures = []
	if internal_faults:
		failures.append(
			f"{internal_faults} receipt(s) booked internal_fault -- "
			"designed policy resets must demote typed, and genuine "
			"faults must not occur in a deterministic gate cell")
	for i, (off, on) in enumerate(zip(off_transcript, on_transcript)):
		if not args.contention:
			if off["content"] != on["content"]:
				failures.append(f"step {i}: content diverged")
			if off["probs"] != on["probs"]:
				failures.append(f"step {i}: logprobs diverged")
		for arm_name, row in (("off", off), ("on", on)):
			if (row["slot_requested"] is not None and
					row["slot_served"] != row["slot_requested"]):
				failures.append(
					f"step {i} {arm_name}: served slot "
					f"{row['slot_served']} != forced {row['slot_requested']}")

	ttft_off = [r["ttft_ms"] for r in off_transcript if r["ttft_ms"]]
	ttft_on = [r["ttft_ms"] for r in on_transcript if r["ttft_ms"]]
	ttft_report = {}
	if ttft_off and ttft_on:
		mean_off = sum(ttft_off) / len(ttft_off)
		mean_on = sum(ttft_on) / len(ttft_on)
		deltas = [round(a - b, 3) for a, b in zip(ttft_off, ttft_on)]
		ttft_report = {
			"mean_off_ms": round(mean_off, 3),
			"mean_on_ms": round(mean_on, 3),
			"per_step_win_ms": deltas,
		}
		if any(t < 0 for t in ttft_on):
			failures.append("negative TTFT measured on the on arm")
		if mean_on > mean_off * (1.0 + args.ttft_max_regress):
			failures.append(
				f"on-arm mean TTFT {mean_on:.1f}ms exceeds off-arm "
				f"{mean_off:.1f}ms by more than "
				f"{args.ttft_max_regress:.0%}")

	def tier_sum(totals, key):
		return sum(t.get(key, 0) for t in totals.values())

	if tier_sum(off_totals, "executed") != 0:
		failures.append("off arm shows authoritative executions")
	tier_on = on_totals.get(args.tier, {})
	if args.expect_shadow:
		if tier_sum(on_totals, "executed") != 0:
			failures.append(
				"out-of-domain workload executed authoritatively")
		if tier_sum(on_totals, "observed") == 0:
			failures.append("no records observed at all")
	elif args.expect_fallback:
		if tier_sum(on_totals, "executed") != 0:
			failures.append(
				"unfitted arm executed authoritatively; expected "
				f"pure {args.expect_fallback} fallback")
		if tier_on.get("fallback_legacy", 0) == 0:
			failures.append(
				f"unfitted arm recorded no fallback_legacy[{args.tier}]")
	elif not args.liveness_optional:
		if tier_on.get("executed", 0) == 0:
			failures.append(
				f"on arm executed nothing at tier {args.tier}; "
				"gate requires live authoritative executions")
		if tier_on.get("agree", 0) + tier_on.get("disagree", 0) == 0:
			failures.append(
				f"on arm recorded no verdicts at tier {args.tier}")

	report = {
		"off_totals": off_totals,
		"on_totals": on_totals,
		"steps": len(on_transcript),
		"ttft": ttft_report,
		"failures": failures,
	}
	print(json.dumps(report, indent=1))
	if failures:
		print(f"BA_AUTHORITY_CELL={args.tier} FAIL")
		raise SystemExit(1)
	print(f"BA_AUTHORITY_CELL={args.tier} PASS")


if __name__ == "__main__":
	main()
