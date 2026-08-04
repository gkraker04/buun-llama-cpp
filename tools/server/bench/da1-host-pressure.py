#!/usr/bin/env python3
"""D-A1 repeated-resume host-pressure gate.

Runs the same resume-heavy workload through two arms -- lifecycle ON
(non-consuming restores) and lifecycle OFF (consuming legacy) -- and compares:
byte-identical content, host inventory growth (bounded: the retained entry must
be reused, not duplicated, across resumes), retained-restore evidence on the on
arm, and restore-path timing (the clone cost is O(checkpoint-ring bytes) and
must not blow up TTFT). Design gate: 'may not trade a miss-rate win for
unbounded host growth'.
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request


def request(base, path, body=None, timeout=900):
	data = json.dumps(body).encode() if body is not None else None
	req = urllib.request.Request(base + path, data=data,
		headers={"Content-Type": "application/json"},
		method="POST" if data is not None else "GET")
	with urllib.request.urlopen(req, timeout=timeout) as response:
		return json.loads(response.read())


def run_arm(args, lifecycle, port, log_path):
	cmd = [args.server_bin, "-m", args.model, "--port", str(port),
		"-ngl", "99", "-c", str(args.ctx), "-b", str(args.batch),
		"-np", "2", "-fa", "on", "-ctk", args.ctk, "-ctv", args.ctv,
		"--cache-ram", str(args.cache_ram), "--slots", "--cache-debug",
		"--cache-plan-authority", "lru", "--slot-prompt-similarity", "0.5",
		"--seed", "7"]
	if lifecycle:
		cmd.append("--cache-lifecycle")
	proc = subprocess.Popen(cmd, stdout=open(log_path, "w"),
		stderr=subprocess.STDOUT)
	base = f"http://127.0.0.1:{port}"
	transcript = []
	try:
		for _ in range(180):
			time.sleep(2)
			try:
				request(base, "/health", timeout=5)
				break
			except Exception:
				if proc.poll() is not None:
					raise RuntimeError("server exited early")
		filler = " ".join(
			f"host pressure ledger row {i} retains deterministic state."
			for i in range(args.prompt_rows))
		main = "Main resumable conversation. " + filler
		echo = ""
		for cycle in range(args.cycles):
			# displace the main conversation off its slot with two churn
			# tasks, then resume it -- lifecycle-on restores non-consuming
			for c in range(2):
				request(base, "/completion", {
					"prompt": f"Churn {cycle}-{c}. " + filler,
					"n_predict": 8, "temperature": 0, "seed": 7,
					"cache_prompt": True})
			# history must grow STRICTLY (marker included) so each re-save is
			# a token-prefix extension of the previous entry -- that is what
			# lets host_dedup prune the predecessor (the D-A2 seam)
			marker = f" Resume cycle {cycle}."
			r = request(base, "/completion", {
				"prompt": main + echo + marker,
				"n_predict": 12, "temperature": 0, "seed": 7,
				"cache_prompt": True})
			echo += marker + (r.get("content") or "")
			transcript.append({
				"content": r.get("content"),
				"prompt_ms": (r.get("timings") or {}).get("prompt_ms"),
			})
	finally:
		proc.terminate()
		try:
			proc.wait(timeout=30)
		except subprocess.TimeoutExpired:
			proc.kill()

	# host_entries from the lifecycle evidence is the honest growth metric
	# (the earlier inventory_states proxy always read zero -- vacuous)
	entries_peak = 0
	retained_restores = 0
	with open(log_path, errors="replace") as handle:
		for line in handle:
			match = re.search(r"CACHE_HOST_LIFECYCLE (\{.*)", line)
			if match:
				try:
					evt = json.loads(match.group(1))
				except ValueError:
					continue
				if evt.get("mode") == "non_consuming":
					retained_restores += 1
				entries = evt.get("host_entries")
				if isinstance(entries, int):
					entries_peak = max(entries_peak, entries)
	return transcript, entries_peak, retained_restores


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-bin", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--workdir", required=True)
	parser.add_argument("--base-port", type=int, default=8470)
	parser.add_argument("--ctx", type=int, default=4096)
	parser.add_argument("--batch", type=int, default=512)
	parser.add_argument("--ctk", default="f16")
	parser.add_argument("--ctv", default="f16")
	parser.add_argument("--cache-ram", type=int, default=2048)
	parser.add_argument("--prompt-rows", type=int, default=50)
	parser.add_argument("--cycles", type=int, default=4)
	parser.add_argument("--inventory-slack", type=int, default=2,
		help="on-arm host inventory peak may exceed the off arm by at "
		"most this many entries (retained-entry reuse, not duplication)")
	parser.add_argument("--timing-max-regress", type=float, default=0.25)
	parser.add_argument("--expect-da2-evidence", action="store_true",
		help="the on arm must show D-A2 destruction receipts on grown-save "
		"prunes: typed refusals (fail-closed conservatism) or certified "
		"evictions, and zero internal_fault")
	args = parser.parse_args()

	off_t, off_peak, _ = run_arm(args, False, args.base_port,
		f"{args.workdir}/server-off.log")
	on_t, on_peak, retained = run_arm(args, True, args.base_port + 1,
		f"{args.workdir}/server-on.log")

	da2_receipts, da2_faults = {}, 0
	if args.expect_da2_evidence:
		with open(f"{args.workdir}/server-on.log", errors="replace") as handle:
			for line in handle:
				match = re.search(r"CACHE_HOST_DESTRUCTION (\{.*)", line)
				if not match:
					continue
				try:
					destruction = json.loads(match.group(1))
				except ValueError:
					continue
				state = destruction.get("state")
				if state in ("refused", "certified", "executed", "quoted"):
					key = f'{state}:{destruction.get("reason")}'
					da2_receipts[key] = da2_receipts.get(key, 0) + 1
				if destruction.get("reason") == "internal_fault":
					da2_faults += 1

	failures = []
	for i, (off, on) in enumerate(zip(off_t, on_t)):
		if off["content"] != on["content"]:
			failures.append(f"cycle {i}: content diverged")
	if retained == 0:
		failures.append("on arm recorded no retained (non-consuming) restores")
	# off arm emits no lifecycle evidence; the growth bound is absolute:
	# churn saves (2/cycle, distinct content) + main + one transient
	# successor -- dedup pruning must keep the main line at ~1 live entry
	entries_bound = args.cycles * 2 + 2 + args.inventory_slack
	if on_peak > entries_bound:
		failures.append(
			f"host entries peaked at {on_peak} > bound {entries_bound} -- "
			"grown-save pruning is not holding the main line to one entry")
	off_ms = [t["prompt_ms"] for t in off_t if t["prompt_ms"]]
	on_ms = [t["prompt_ms"] for t in on_t if t["prompt_ms"]]
	timing = {}
	if off_ms and on_ms:
		mean_off = sum(off_ms) / len(off_ms)
		mean_on = sum(on_ms) / len(on_ms)
		timing = {"mean_off_ms": round(mean_off, 2),
			"mean_on_ms": round(mean_on, 2)}
		if mean_on > mean_off * (1.0 + args.timing_max_regress):
			failures.append(
				f"resume timing regressed: {mean_on:.1f}ms vs "
				f"{mean_off:.1f}ms (limit +{args.timing_max_regress:.0%})")

	if args.expect_da2_evidence:
		if not da2_receipts:
			failures.append(
				"no D-A2 destruction receipts on the on arm -- the "
				"certification seam never engaged on grown-save prunes")
		if da2_faults:
			failures.append(f"{da2_faults} internal_fault destruction receipt(s)")

	report = {
		"cycles": len(on_t),
		"off_inventory_peak": off_peak,
		"on_inventory_peak": on_peak,
		"retained_restores": retained,
		"da2_receipts": da2_receipts,
		"timing": timing,
		"failures": failures,
	}
	print(json.dumps(report, indent=1))
	if failures:
		print("DA1_HOST_PRESSURE FAIL")
		raise SystemExit(1)
	print("DA1_HOST_PRESSURE PASS")


if __name__ == "__main__":
	main()
