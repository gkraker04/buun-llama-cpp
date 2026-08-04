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
			r = request(base, "/completion", {
				"prompt": main + echo + f" Resume cycle {cycle}.",
				"n_predict": 12, "temperature": 0, "seed": 7,
				"cache_prompt": True})
			echo += r.get("content") or ""
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

	inventory_peak = 0
	retained_restores = 0
	with open(log_path, errors="replace") as handle:
		for line in handle:
			match = re.search(r"CACHE_PLAN (\{.*)", line)
			if match:
				try:
					rec = json.loads(match.group(1))
				except ValueError:
					continue
				states = rec.get("inventory_states")
				if isinstance(states, list):
					inventory_peak = max(inventory_peak, len(states))
			if "CACHE_HOST_LIFECYCLE" in line and '"non_consuming"' in line:
				retained_restores += 1
	return transcript, inventory_peak, retained_restores


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
	args = parser.parse_args()

	off_t, off_peak, _ = run_arm(args, False, args.base_port,
		f"{args.workdir}/server-off.log")
	on_t, on_peak, retained = run_arm(args, True, args.base_port + 1,
		f"{args.workdir}/server-on.log")

	failures = []
	for i, (off, on) in enumerate(zip(off_t, on_t)):
		if off["content"] != on["content"]:
			failures.append(f"cycle {i}: content diverged")
	if retained == 0:
		failures.append("on arm recorded no retained (non-consuming) restores")
	if on_peak > off_peak + args.inventory_slack:
		failures.append(
			f"host inventory grew: on-arm peak {on_peak} vs off-arm "
			f"{off_peak} (+slack {args.inventory_slack}) -- non-consuming "
			"restores must reuse the retained entry, not duplicate it")
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

	report = {
		"cycles": len(on_t),
		"off_inventory_peak": off_peak,
		"on_inventory_peak": on_peak,
		"retained_restores": retained,
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
