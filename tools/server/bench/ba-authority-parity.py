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


def drive_sequence(base, args):
	"""Forced-slot sequence covering live / host / cold candidate shapes:
	cold fill on slot 0, exact re-ask (live), extension (live replay),
	displacement by a second conversation (host save), return of the first
	(host restore), plus a cold fill on slot 1 proving non-forced isolation."""
	filler = " ".join(
		f"authority parity row {i} keeps the ledger deterministic."
		for i in range(args.prompt_rows))
	convo_a = "Conversation A. " + filler
	convo_b = "Conversation B considers different state. " + filler
	steps = [
		(0, convo_a),
		(0, convo_a),
		(0, convo_a + " Continue the ledger with one more row."),
		(0, convo_b),
		(0, convo_a),
		(1, convo_a),
	]
	transcript = []
	for slot, prompt in steps:
		status, payload = request(base, "/completion", {
			"prompt": prompt, "n_predict": args.n_predict,
			"temperature": 0, "seed": args.seed, "cache_prompt": True,
			"id_slot": slot, "n_probs": args.n_probs,
		})
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
		})
	return transcript


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
	args = parser.parse_args()

	off_transcript, off_totals = run_arm(
		args, "off", args.base_port, args.workdir)
	on_transcript, on_totals = run_arm(
		args, args.level, args.base_port + 1, args.workdir)

	failures = []
	for i, (off, on) in enumerate(zip(off_transcript, on_transcript)):
		if off["content"] != on["content"]:
			failures.append(f"step {i}: content diverged")
		if off["probs"] != on["probs"]:
			failures.append(f"step {i}: logprobs diverged")
		for arm_name, row in (("off", off), ("on", on)):
			if row["slot_served"] != row["slot_requested"]:
				failures.append(
					f"step {i} {arm_name}: served slot "
					f"{row['slot_served']} != forced {row['slot_requested']}")

	def tier_sum(totals, key):
		return sum(t.get(key, 0) for t in totals.values())

	if tier_sum(off_totals, "executed") != 0:
		failures.append("off arm shows authoritative executions")
	tier_on = on_totals.get(args.tier, {})
	if args.expect_fallback:
		if tier_sum(on_totals, "executed") != 0:
			failures.append(
				"unfitted arm executed authoritatively; expected "
				f"pure {args.expect_fallback} fallback")
		if tier_on.get("fallback_legacy", 0) == 0:
			failures.append(
				f"unfitted arm recorded no fallback_legacy[{args.tier}]")
	else:
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
		"failures": failures,
	}
	print(json.dumps(report, indent=1))
	if failures:
		print(f"BA_AUTHORITY_CELL={args.tier} FAIL")
		raise SystemExit(1)
	print(f"BA_AUTHORITY_CELL={args.tier} PASS")


if __name__ == "__main__":
	main()
