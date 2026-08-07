#!/usr/bin/env python3
"""Self-booting ZC0/ZC0b harness/config gate (no performance claim).

It runs the mandatory branch-off/concurrent-null/serialized-null arms, then
proves that an active but unfitted debug arm is rejected as "optimized" and
that absent optimizer mode is identical to explicit off on the real server.
"""

import argparse
import json
import pathlib
import subprocess
import time
import urllib.request


def wait_ready(base, process, timeout):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		if process.poll() is not None:
			raise RuntimeError(f"server exited early with {process.returncode}")
		try:
			with urllib.request.urlopen(base + "/health", timeout=2) as response:
				if response.status == 200:
					return
		except Exception:
			pass
		time.sleep(0.5)
	raise RuntimeError("server readiness timeout")


def stop_server(process, log):
	if process.poll() is None:
		process.terminate()
		try:
			process.wait(timeout=30)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=10)
	log.close()


def make_trace(path):
	rows = [{
		"record": "capture_header", "schema": "capture_trace/v1",
		"tag": "zc0-light-gate",
	}]
	base = "ZC0 cache harness truth fixture. "
	for seq in range(4):
		rows.append({
			"record": "request", "seq": seq, "t_arrival_ms": seq * 0.2,
			"total_ms": 100.0, "status": 200, "method": "POST",
			"path": "/completion", "stream": True,
			"body": {
				"prompt": base + ("alpha beta gamma delta " * (seq + 1)),
				"n_predict": 6, "stream": True, "cache_prompt": True,
			},
			"fingerprint": {
				"system_prefix_sha": "zc0", "system_chars": 32,
				"prompt_chars": 64 * (seq + 1), "n_messages": 0,
				"n_predict": 6, "tools": 0,
			},
			"server": {"timings": {"predicted_n": 6}},
		})
	path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_arm(args, name, server, parallel, extra, serialized=False):
	log_path = args.workdir / f"{name}.server.log"
	scorecard = args.workdir / f"{name}.json"
	log = log_path.open("w")
	command = [
		str(server), "-m", str(args.model), "--host", "127.0.0.1",
		"--port", str(args.port), "-ngl", "99", "-c", "2048",
		"-np", str(parallel), "-fa", "on", "-ctk", "f16", "-ctv", "f16",
		"--slots", *extra,
	]
	process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
	try:
		wait_ready(f"http://127.0.0.1:{args.port}", process, args.boot_timeout)
		replay = [
			"python3", str(args.bench / "trace-replay.py"),
			"--trace", str(args.trace), "--target", f"127.0.0.1:{args.port}",
			"--label", name, "--out", str(scorecard), "--mode", "chained",
			"--greedy", "--pin-generation", "--claim-grade",
			"--timeout", str(args.request_timeout),
		]
		if serialized:
			replay.append("--serialize-overlap")
		subprocess.run(replay, check=True)
	finally:
		stop_server(process, log)
	return scorecard, log_path


def assert_serialized_identity(absent_path, explicit_path):
	absent = json.loads(absent_path.read_text())
	explicit = json.loads(explicit_path.read_text())
	fields = (
		"seq", "path", "status", "fresh_prefill_tokens", "prompt_tokens",
		"generated_tokens", "generated_recorded", "truncated", "stop_type",
		"output_sha",
	)
	absent_rows = [{key: row.get(key) for key in fields}
		for row in absent.get("requests", [])]
	explicit_rows = [{key: row.get(key) for key in fields}
		for row in explicit.get("requests", [])]
	if absent_rows != explicit_rows:
		raise SystemExit("absent optimizer mode diverged from explicit off: "
			f"absent={absent_rows} explicit={explicit_rows}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--master-server", type=pathlib.Path, required=True)
	parser.add_argument("--branch-server", type=pathlib.Path, required=True)
	parser.add_argument("--model", type=pathlib.Path, required=True)
	parser.add_argument("--workdir", type=pathlib.Path, required=True)
	parser.add_argument("--port", type=int, default=8081)
	parser.add_argument("--boot-timeout", type=float, default=180.0)
	parser.add_argument("--request-timeout", type=float, default=180.0)
	args = parser.parse_args()
	args.bench = pathlib.Path(__file__).resolve().parent
	args.workdir.mkdir(parents=True, exist_ok=True)
	args.trace = args.workdir / "zc0-truth.jsonl"
	make_trace(args.trace)

	base, _ = run_arm(args, "baseline-master", args.master_server, 2, [])
	parity, _ = run_arm(args, "branch-no-features", args.branch_server, 2, [])
	active, active_log = run_arm(args, "branch-active-unfitted", args.branch_server, 2,
		["--cache-optimizer", "off", "--cache-debug", "--cache-lifecycle",
		 "--cache-plan-authority", "lru"])
	null, _ = run_arm(args, "baseline-concurrent-null", args.master_server, 2, [])
	serial_base, _ = run_arm(
		args, "baseline-serialized", args.master_server, 1, [], serialized=True)
	serial_null, _ = run_arm(
		args, "baseline-serialized-null", args.master_server, 1, [], serialized=True)
	config_absent, _ = run_arm(
		args, "branch-serialized-absent", args.branch_server, 1, [], serialized=True)
	config_off, _ = run_arm(
		args, "branch-serialized-explicit-off", args.branch_server, 1,
		["--cache-optimizer", "off"], serialized=True)
	assert_serialized_identity(config_absent, config_off)

	report = args.workdir / "zc0-light.report.json"
	compare = [
		"python3", str(args.bench / "trace-compare.py"),
		"--baseline", str(base), "--parity", str(parity), "--arm", str(active),
		"--null-concurrent", str(null), "--serialized-baseline", str(serial_base),
		"--serialized-null", str(serial_null), "--planner-evidence", str(active_log),
		"--claim-grade", "--out", str(report),
	]
	completed = subprocess.run(compare)
	if completed.returncode == 0:
		raise SystemExit("active-unfitted negative control unexpectedly claimed optimization")
	result = json.loads(report.read_text())
	reasons = set((result.get("claim_validation") or {}).get("reasons") or [])
	expected = {
		"active_evidence:planner_status_not_ok",
		"active_evidence:no_authoritative_execution",
	}
	if reasons != expected:
		raise SystemExit(f"ZC0 light gate failed with unexpected reasons: {sorted(reasons)}")
	print("ZC0_LIGHT_GATE PASS parity=null serialized=null active_unfitted=rejected "
		"optimizer_absent=explicit_off")


if __name__ == "__main__":
	main()
