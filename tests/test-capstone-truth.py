#!/usr/bin/env python3
"""Model-free truth tests for the capstone capture/replay/compare pipeline."""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "tools" / "server" / "bench"
sys.path.insert(0, str(BENCH))


def load_script(name):
	path = BENCH / name
	spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


trace_common = load_script("trace_common.py")
trace_replay = load_script("trace-replay.py")
trace_compare = load_script("trace-compare.py")


def request(seq=0, *, fresh=10, output="same"):
	return {
		"seq": seq, "status": 200, "fresh_prefill_tokens": fresh,
		"prompt_tokens": 100, "generated_tokens": 7, "generated_recorded": 7,
		"truncated": False, "output_sha": output,
	}


def scorecard(rows=None, *, serialized=False):
	return {
		"claim_validation": {"requested": True, "valid": True, "reasons": []},
		"trace_tag": "capstone-test",
		"timeline": {"faithful": True},
		"execution": {
			"serialize_overlap": serialized,
			"server_parallel_observed": 1 if serialized else 4,
		},
		"requests": rows or [request()],
	}


class CapstoneTruthTest(unittest.TestCase):
	def test_incremental_sse_metrics_outlive_body_retention(self):
		reader = trace_common.SSEPayloadAccumulator()
		metrics = {}
		# This payload volume is larger than capture-proxy.TEXT_CAP. Only the
		# parser's unfinished line remains resident.
		chunk = b'data: {"content":"x"}\n\n'
		for _ in range(((2 << 20) // len(chunk)) + 32):
			for payload in reader.feed(chunk):
				trace_common.merge_server_metrics(metrics, payload)
		final = (b'data: {"timings":{"predicted_n":17},'
			b'"truncated":true,"stop_type":"limit"}\n\n')
		for payload in reader.feed(final) + reader.feed(b"", final=True):
			trace_common.merge_server_metrics(metrics, payload)
		self.assertEqual(metrics["timings"]["predicted_n"], 17)
		self.assertIs(metrics["truncated"], True)
		self.assertEqual(metrics["stop_type"], "limit")

	def test_replay_preserves_truncation_false_true_and_missing(self):
		false = trace_replay.server_metrics('data: {"truncated":false}\n', True)
		true = trace_replay.server_metrics('data: {"truncated":true}\n', True)
		missing = trace_replay.server_metrics('data: {"content":"x"}\n', True)
		self.assertIs(false["truncated"], False)
		self.assertIs(true["truncated"], True)
		self.assertNotIn("truncated", missing)

	def test_successful_capture_without_generation_is_unpinnable(self):
		rows = [
			{"seq": 1, "status": 200, "server": {"usage": {"completion_tokens": 4}}},
			{"seq": 2, "status": 200, "server": {"truncated": False}},
			{"seq": 3, "status": 500, "server": {}},
		]
		self.assertEqual(trace_replay.capture_pin_issues(rows), [2])

	def test_claim_replay_fails_before_network_when_capture_is_unpinnable(self):
		with tempfile.TemporaryDirectory() as directory:
			trace = pathlib.Path(directory) / "trace.jsonl"
			out = pathlib.Path(directory) / "scorecard.json"
			trace.write_text(json.dumps({
				"record": "request", "seq": 4, "status": 200,
				"t_arrival_ms": 0, "path": "/completion", "method": "POST",
				"stream": True, "body": {"prompt": "x", "n_predict": 3},
			}) + "\n")
			result = subprocess.run([
				sys.executable, str(BENCH / "trace-replay.py"),
				"--trace", str(trace), "--target", "127.0.0.1:1",
				"--label", "missing", "--out", str(out),
				"--greedy", "--pin-generation", "--claim-grade",
			], capture_output=True, text=True)
			self.assertNotEqual(result.returncode, 0)
			self.assertIn("lack generation counts", result.stderr)
			self.assertFalse(out.exists())

	def test_claim_arms_require_structural_parity_and_serial_identity(self):
		base = scorecard()
		parity = scorecard()
		active = scorecard()
		null = scorecard()
		serial_base = scorecard(serialized=True)
		serial_null = scorecard(serialized=True)
		evidence = {"planner_status": {"ok": 1}, "authoritative_count": 1}
		self.assertEqual(trace_compare.validate_claim_arms(
			base, parity, active, null, serial_base, serial_null, evidence), [])
		parity["requests"][0]["fresh_prefill_tokens"] = 11
		serial_null["requests"][0]["output_sha"] = "different"
		reasons = trace_compare.validate_claim_arms(
			base, parity, active, null, serial_base, serial_null, evidence)
		self.assertIn(
			"branch_no_features:fresh_prefill_tokens:outside_null_envelope", reasons)
		self.assertIn("null_serialized:seq=0:output_sha", reasons)
		null["requests"][0]["fresh_prefill_tokens"] = 12
		reasons = trace_compare.validate_claim_arms(
			base, parity, active, null, serial_base, scorecard(serialized=True), evidence)
		self.assertNotIn(
			"branch_no_features:fresh_prefill_tokens:outside_null_envelope", reasons)

	def test_active_unfitted_evidence_cannot_claim_optimization(self):
		with tempfile.NamedTemporaryFile("w", delete=False) as handle:
			path = handle.name
			handle.write(json.dumps({
				"schema_version": 6, "outcome": "cold",
				"planner_status": "profile_unfitted",
				"authority": {"state": "fallback_legacy"},
			}) + "\n")
		try:
			evidence = trace_compare.planner_evidence([path])
		finally:
			pathlib.Path(path).unlink()
		self.assertEqual(evidence["authoritative_count"], 0)
		reasons = trace_compare.validate_claim_arms(
			scorecard(), scorecard(), scorecard(), scorecard(),
			scorecard(serialized=True), scorecard(serialized=True), evidence)
		self.assertIn("active_evidence:planner_status_not_ok", reasons)
		self.assertIn("active_evidence:no_authoritative_execution", reasons)

	def test_authoritative_evidence_requires_executed_candidate_and_tier(self):
		with tempfile.NamedTemporaryFile("w", delete=False) as handle:
			path = handle.name
			handle.write("CACHE_PLAN " + json.dumps({
				"schema_version": 6, "outcome": "restored", "planner_status": "ok",
				"authority": {
					"state": "authoritative", "decision_tier": "lru",
					"executed_plan_candidate": 2,
				},
			}) + "\n")
		try:
			evidence = trace_compare.planner_evidence([path])
		finally:
			pathlib.Path(path).unlink()
		self.assertEqual(evidence["authoritative_count"], 1)


if __name__ == "__main__":
	unittest.main()
