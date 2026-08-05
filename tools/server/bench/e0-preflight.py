#!/usr/bin/env python3
"""E0 /cache/plan live acceptance gate.

Self-boots four sequential sibling-style server arms and proves: immediate-plan honesty,
destructive evidence, intervening-mutation non-authority, disabled-path parity,
lifecycle-only evidence inertness, and request/prompt-log redaction. The default
run is a gate, not a permissive probe: every cell must print a PASS marker before
the final E0_PREFLIGHT PASS marker.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request


# Deliberate live-gate subset. The exhaustive canonical private-key oracle is
# assert_redacted_keys() in tests/test-cache-plan-preflight.cpp.
FORBIDDEN_KEYS = {
	"ticket", "claim", "preview_id", "nonce", "manifest_digest",
	"artifact_id", "target_slot_id", "source_id", "accounting_serial",
	"admission_sequence", "topology_id", "domains", "journal_id",
}
CACHE_PLAN_RE = re.compile(r"CACHE_PLAN (\{.*\})")

# common_cache_plan_authority_fallback_name()'s execution-time
# currency/budget/certification family. A planner_if_still_current preview is
# conditional on these facts surviving until the real request reaches its
# mutation boundary. Planning refusals and internal_fault are deliberately not
# accepted here.
PLANNER_IF_CURRENT_FALLBACK_REASONS = frozenset({
	"stale_capability",
	"destruction_authority_required",
	"budget_or_lease_unavailable",
	"destruction_not_certified",
})


def request(base, path, body=None, timeout=600):
	headers = {}
	data = None
	if body is not None:
		data = json.dumps(body).encode("utf-8")
		headers["Content-Type"] = "application/json"
	req = urllib.request.Request(
		base + path, data=data, headers=headers,
		method="POST" if data is not None else "GET")
	try:
		with urllib.request.urlopen(req, timeout=timeout) as response:
			payload = response.read()
			return (response.status,
				json.loads(payload) if payload else None,
				dict(response.headers.items()))
	except urllib.error.HTTPError as error:
		payload = error.read()
		try:
			decoded = json.loads(payload) if payload else None
		except Exception:
			decoded = {"raw": payload.decode(errors="replace")}
		return error.code, decoded, dict(error.headers.items())


def header(headers, name):
	for key, value in headers.items():
		if key.lower() == name.lower():
			return value
	return None


def walk_keys(value):
	if isinstance(value, dict):
		for key, nested in value.items():
			yield key
			yield from walk_keys(nested)
	elif isinstance(value, list):
		for nested in value:
			yield from walk_keys(nested)


def assert_preview(payload, headers):
	if header(headers, "Cache-Control") != "no-store":
		raise RuntimeError("preflight response omitted Cache-Control: no-store")
	if not isinstance(payload, dict) or payload.get("object") != "cache_plan_preflight":
		raise RuntimeError(f"wrong preflight object: {payload}")
	if payload.get("authoritative") is not False:
		raise RuntimeError("preflight became authoritative")
	if payload.get("reservation") != "none" or payload.get("valid_until") is not None:
		raise RuntimeError("preflight exposed a replayable reservation")
	leaked = FORBIDDEN_KEYS.intersection(walk_keys(payload))
	if leaked:
		raise RuntimeError(f"private preflight keys leaked: {sorted(leaked)}")


def cache_plan_records(log_path):
	records = []
	with open(log_path, errors="replace") as handle:
		for line in handle:
			match = CACHE_PLAN_RE.search(line)
			if not match:
				continue
			try:
				records.append(json.loads(match.group(1)))
			except ValueError:
				pass
	return records


def wait_new_record(log_path, before, timeout=30):
	deadline = time.time() + timeout
	while time.time() < deadline:
		records = cache_plan_records(log_path)
		if len(records) > before:
			return records[-1]
		time.sleep(0.05)
	raise RuntimeError("completion emitted no CACHE_PLAN record")


def candidate_for(rec, candidate_id):
	if not isinstance(candidate_id, int):
		return None
	for candidate in rec.get("candidates") or []:
		if candidate.get("id") == candidate_id:
			return candidate
	return None


def planner_candidate(rec):
	authority = rec.get("authority") or {}
	return candidate_for(rec, authority.get("planner_plan_candidate"))


def cache_plan_slot_evidence(base):
	status, slots, _ = request(base, "/slots")
	if status != 200:
		raise RuntimeError(f"/slots failed: {status}")
	return [entry.get("cache_plan") for entry in slots]


class ServerArm:
	def __init__(self, args, name, port, *, preflight, debug, lifecycle,
			prompt_log_dir=None):
		self.name = name
		self.base = f"http://127.0.0.1:{port}"
		self.log_path = os.path.join(args.workdir, f"server-{name}.log")
		self.vbr_kv = any(
			value and value.lower() == "vbr" for value in (args.ctk, args.ctv))
		cmd = [
			args.server_bin, "-m", args.model, "--host", "127.0.0.1",
			"--port", str(port), "-ngl", str(args.ngl),
			"-c", str(args.ctx), "-b", str(args.batch),
			"-np", str(args.parallel), "-fa", "on",
			"--cache-ram", str(args.cache_ram), "--slots",
			"--cache-plan-authority", "lru",
			"--slot-prompt-similarity", str(args.slot_prompt_similarity),
			"--seed", str(args.seed),
		]
		if args.ctk:
			cmd += ["-ctk", args.ctk, "-ctv", args.ctv or args.ctk]
		if preflight:
			cmd.append("--cache-plan-preflight")
		if debug:
			cmd.append("--cache-debug")
		if lifecycle:
			cmd.append("--cache-lifecycle")
		if prompt_log_dir:
			cmd += ["--log-prompts-dir", prompt_log_dir]
		for extra in args.extra_server_arg:
			cmd += shlex.split(extra)
		self.log = open(self.log_path, "w")
		self.proc = subprocess.Popen(
			cmd, stdout=self.log, stderr=subprocess.STDOUT)

	def wait_healthy(self, deadline=360):
		started = time.time()
		while time.time() - started < deadline:
			try:
				status, _, _ = request(self.base, "/health", timeout=5)
				if status == 200:
					return
			except Exception:
				pass
			if self.proc.poll() is not None:
				raise RuntimeError(
					f"arm {self.name} exited early; see {self.log_path}")
			time.sleep(1)
		raise RuntimeError(f"arm {self.name} never became healthy")

	def stop(self):
		if self.proc.poll() is None:
			self.proc.terminate()
			try:
				self.proc.wait(timeout=30)
			except subprocess.TimeoutExpired:
				self.proc.kill()
				self.proc.wait(timeout=30)
		self.log.close()


def completion(base, prompt, slot, args):
	body = {
		"prompt": prompt,
		"cache_prompt": True,
		"n_predict": args.n_predict,
		"temperature": 0,
		"seed": args.seed,
		"n_probs": 8,
	}
	if slot is not None:
		body["id_slot"] = slot
	status, payload, _ = request(base, "/completion", body)
	if status != 200:
		raise RuntimeError(f"completion failed: {status} {payload}")
	probs = []
	for token in payload.get("completion_probabilities") or []:
		probs.append((token.get("id"), token.get("logprob")))
	return {
		"content": payload.get("content"),
		"slot": payload.get("id_slot"),
		"probs": probs,
	}


def parity_battery(base, args):
	filler = " ".join(
		f"E0 parity ledger row {i} is deterministic."
		for i in range(args.prompt_rows))
	steps = [
		(0, "E0 parity conversation A. " + filler),
		(1, "E0 parity conversation B. " + filler),
		(0, "E0 parity conversation A. " + filler + " Resume briefly."),
	]
	return [completion(base, prompt, slot, args) for slot, prompt in steps]


def cell_disabled_and_parity(args):
	off = ServerArm(
		args, "disabled", args.base_port,
		preflight=False, debug=True, lifecycle=True)
	try:
		off.wait_healthy()
		status, payload, headers = request(
			off.base, "/cache/plan", {"prompt": "E0 disabled arm"})
		if status != 501:
			raise RuntimeError(f"disabled route returned {status}, expected 501")
		error = payload.get("error") if isinstance(payload, dict) else None
		if not isinstance(error, dict) or error.get("type") != "not_supported_error":
			raise RuntimeError(f"disabled route returned wrong typed error: {payload}")
		if header(headers, "Cache-Control") != "no-store":
			raise RuntimeError("disabled typed 501 omitted no-store")
		off_transcript = parity_battery(off.base, args)
	finally:
		off.stop()

	on = ServerArm(
		args, "debug", args.base_port + 1,
		preflight=True, debug=True, lifecycle=True)
	try:
		on.wait_healthy()
		on_transcript = parity_battery(on.base, args)
		if off_transcript != on_transcript:
			raise RuntimeError(
				"flag-on/unused completion transcript differs from flag-off")
		print("E0_PREFLIGHT CELL disabled_parity PASS")
		return on
	except Exception:
		on.stop()
		raise


def preflight(arm, body):
	started = time.time()
	status, payload, headers = request(arm.base, "/cache/plan", body)
	latency_ms = (time.time() - started) * 1000.0
	if status != 200:
		raise RuntimeError(f"preflight failed: {status} {payload}")
	assert_preview(payload, headers)
	return payload, latency_ms


def assert_plan_agreement(preview, rec, *, require_authoritative=False):
	planner = preview["planner"]
	if planner["selection_tier"] != rec.get("selection"):
		raise RuntimeError("preview selection tier disagrees with actual record")
	planned = planner_candidate(rec)
	if planner.get("provider") is not None:
		if not planned or planned.get("provider") != planner["provider"]:
			raise RuntimeError("preview provider disagrees with actual planner candidate")
	expected = planner["expected_path"]
	authority = rec.get("authority") or {}
	if expected == "planner_if_still_current":
		# This preview is a conditional forecast, not a reservation. The real
		# request either executes the same plan or records the typed fact that
		# invalidated its currency/budget/certification before mutation.
		state = authority.get("state")
		if state == "authoritative":
			if authority.get("executed_plan_candidate") != authority.get("planner_plan_candidate"):
				raise RuntimeError("actual execution did not use the predicted plan")
		elif state == "fallback_legacy":
			if require_authoritative:
				raise RuntimeError(
					"clean-cache planner prediction did not execute authoritatively")
			reason = authority.get("fallback_reason")
			if reason not in PLANNER_IF_CURRENT_FALLBACK_REASONS:
				raise RuntimeError(
					f"planner forecast fell back for non-conditional reason: {reason}")
		else:
			raise RuntimeError(
				f"planner forecast produced unexpected authority state: {state}")
	elif expected == "conditional_on_destruction_certification":
		if not (rec.get("destruction") or {}).get("effects"):
			raise RuntimeError("conditional prediction lacked actual destruction evidence")
	elif expected != "legacy":
		raise RuntimeError(f"unknown expected_path: {expected}")


def cell_agreement(arm, args, latencies):
	prompt = "E0 agreement immediate request. " + " ".join(
		f"agreement row {i}" for i in range(args.prompt_rows))
	before_records = len(cache_plan_records(arm.log_path))
	before_slots = cache_plan_slot_evidence(arm.base)
	preview, latency = preflight(arm, {
		"prompt": prompt, "id_slot": 0, "cache_prompt": True})
	latencies.append(latency)
	after_slots = cache_plan_slot_evidence(arm.base)
	if before_slots != after_slots:
		raise RuntimeError("preflight changed /slots observer evidence")
	completion(arm.base, prompt, 0, args)
	rec = wait_new_record(arm.log_path, before_records)
	assert_plan_agreement(preview, rec, require_authoritative=True)
	print("E0_PREFLIGHT CELL agreement PASS")


def cell_clean_agreement(args, latencies):
	arm = ServerArm(
		args, "agreement", args.base_port + 1,
		preflight=True, debug=True, lifecycle=True)
	try:
		arm.wait_healthy()
		cell_agreement(arm, args, latencies)
	finally:
		arm.stop()


def effect_names(effects):
	return {
		effect.get("effect") for effect in effects or []
		if isinstance(effect, dict) and effect.get("effect")
	}


def cell_destructive(arm, args, latencies):
	filler = " ".join(
		f"destruction ledger row {i} preserves state."
		for i in range(args.prompt_rows))
	prompts = [
		"E0 destructive thread A. " + filler,
		"E0 destructive thread B. " + filler,
		"E0 destructive replacement C, unrelated prefix. " + filler,
		"E0 destructive replacement D, another prefix. " + filler,
	]
	seen = 0
	for prompt in prompts:
		before = len(cache_plan_records(arm.log_path))
		preview, latency = preflight(arm, {"prompt": prompt, "cache_prompt": True})
		latencies.append(latency)
		completion(arm.base, prompt, None, args)
		rec = wait_new_record(arm.log_path, before)
		# These requests run after the parity battery has warmed the cache. The
		# conditional forecast therefore accepts a typed currency/budget fallback
		# while selection-tier and planned-provider agreement remain mandatory.
		assert_plan_agreement(preview, rec)
		d_view = preview["destruction"]
		if arm.vbr_kv:
			# Dynamic VBR has no plain host prompt-cache inventory in this
			# battery: artifact persistence is the F4 capture/import path, KV
			# pressure is absorbed by deliberately planner-invisible legacy VBR
			# reclaim, and its ordinary similarity domain may be empty. Therefore
			# honest absence—not fabricated destruction evidence—is the gate.
			# This architectural reading is high-confidence but awaits buun's
			# confirmation; if he rules the host path should engage, this assertion
			# becomes the regression test for that implementation fix.
			if (d_view.get("required") is not False or
					d_view.get("assessment") != "not_required" or
					d_view.get("effects")):
				raise RuntimeError(
					f"VBR preview fabricated destruction evidence: {d_view}")
			# Live/checkpoint/replay candidates are normal on VBR arms; only
			# plain host prompt-cache rows are absent (F4 artifact flow instead).
			host_rows = [
				row for row in rec.get("candidates") or []
				if isinstance(row, dict) and row.get("provider") == "host_cache_entry"
			]
			if host_rows:
				raise RuntimeError(
					"VBR destructive battery unexpectedly produced host_cache_entry rows")
			d_actual = rec.get("destruction") or {}
			if (d_actual.get("state") != "not_required" or
					d_actual.get("effects")):
				raise RuntimeError(
					f"VBR completion emitted destruction evidence: {d_actual}")
			continue
		if not d_view.get("required"):
			continue
		seen += 1
		d_actual = rec.get("destruction") or {}
		preview_effects = effect_names(d_view.get("effects"))
		actual_effects = effect_names(d_actual.get("effects"))
		if preview_effects and preview_effects != actual_effects:
			raise RuntimeError(
				f"destruction effects drifted: {preview_effects} vs {actual_effects}")
		assessment = d_view.get("assessment")
		state = d_actual.get("state")
		if assessment == "eligible_at_snapshot" and state not in {
				"quoted", "certified", "executed", "refused"}:
			raise RuntimeError(
				f"eligible preview became inconsistent actual state {state}")
		if assessment == "blocked":
			if state != "refused" or d_view.get("reason") != d_actual.get("reason"):
				raise RuntimeError("blocked preview disagrees with actual refusal")
	if arm.vbr_kv:
		with open(arm.log_path, errors="replace") as handle:
			log = handle.read()
		if ("CACHE_HOST_LIFECYCLE {" in log or
				"CACHE_HOST_DESTRUCTION {" in log):
			raise RuntimeError(
				"VBR honest-absence arm emitted host lifecycle/destruction evidence")
		print("E0_PREFLIGHT CELL destructive_evidence PASS (vbr honest-absence)")
		return
	if seen == 0:
		raise RuntimeError("destructive battery produced no destructive preview")
	print("E0_PREFLIGHT CELL destructive_evidence PASS")


def cell_intervening_mutation(arm, args, latencies):
	prompt = "E0 stale preview target alpha beta gamma"
	preview, latency = preflight(arm, {
		"prompt": prompt, "id_slot": 0, "cache_prompt": True})
	latencies.append(latency)
	completion(
		arm.base, "E0 intervening mutation delta epsilon", 0, args)
	actual = completion(arm.base, prompt, 0, args)
	if not actual.get("content"):
		raise RuntimeError("actual request failed after intervening mutation")
	if preview.get("authoritative") is not False or preview.get("reservation") != "none":
		raise RuntimeError("stale preview exposed authority")
	print("E0_PREFLIGHT CELL intervening_mutation PASS")


def cell_log_absence_and_lifecycle_only(args, latencies):
	prompt_dir = os.path.join(args.workdir, "preflight-prompt-log")
	if os.path.exists(prompt_dir):
		raise RuntimeError(f"prompt-log path already exists: {prompt_dir}")
	arm = ServerArm(
		args, "lifecycle-only", args.base_port + 2,
		preflight=True, debug=False, lifecycle=True,
		prompt_log_dir=prompt_dir)
	sentinel = "E0_PRIVATE_BODY_SENTINEL_92f165"
	try:
		arm.wait_healthy()
		preview, latency = preflight(arm, {
			"prompt": sentinel, "cache_prompt": True})
		latencies.append(latency)
		if preview.get("status") != "ok":
			raise RuntimeError(f"lifecycle-only preflight status: {preview}")
		if any(evidence is not None
				for evidence in cache_plan_slot_evidence(arm.base)):
			raise RuntimeError(
				"lifecycle-only preflight populated /slots cache-plan evidence")
	finally:
		arm.stop()
	with open(arm.log_path, errors="replace") as handle:
		log = handle.read()
	if "CACHE_PLAN {" in log or "CACHE_HOST_DESTRUCTION {" in log:
		raise RuntimeError("lifecycle-only preflight emitted production evidence")
	if sentinel in log:
		raise RuntimeError("preflight request body appeared in the server log")
	if ('"object":"cache_plan_preflight"' in log or
			'"object": "cache_plan_preflight"' in log):
		raise RuntimeError("preflight response body appeared in the server log")
	if os.path.exists(prompt_dir):
		files = [
			os.path.join(root, name)
			for root, _, names in os.walk(prompt_dir) for name in names]
		if files:
			raise RuntimeError(f"preflight materialized prompt-log files: {files}")
	print("E0_PREFLIGHT CELL lifecycle_only PASS")
	print("E0_PREFLIGHT CELL log_absence PASS")


def stamp_workdir(args):
	os.makedirs(args.workdir, exist_ok=True)
	stamp = os.path.join(args.workdir, "BENCH_STAMP")
	with open(stamp, "w") as handle:
		handle.write(json.dumps({
			"gate": "e0-preflight",
			"created_unix": int(time.time()),
			"server_bin": os.path.abspath(args.server_bin),
			"model": os.path.abspath(args.model),
		}, sort_keys=True) + "\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-bin", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--workdir", required=True)
	parser.add_argument("--base-port", type=int, default=8480)
	parser.add_argument("--ctx", type=int, default=4096)
	parser.add_argument("--batch", type=int, default=512)
	parser.add_argument("--parallel", type=int, default=2)
	parser.add_argument("--cache-ram", type=int, default=512)
	parser.add_argument("--ngl", type=int, default=99)
	parser.add_argument("--ctk", default="f16")
	parser.add_argument("--ctv", default="f16")
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--n-predict", type=int, default=8)
	parser.add_argument("--prompt-rows", type=int, default=40)
	parser.add_argument("--slot-prompt-similarity", type=float, default=0.5)
	parser.add_argument("--extra-server-arg", action="append", default=[])
	args = parser.parse_args()

	stamp_workdir(args)
	latencies = []
	# The strict agreement request is the first cache-mutating request on its
	# arm. Stop it before booting the parity arms so a large model is never
	# resident in two sibling servers at once.
	cell_clean_agreement(args, latencies)
	debug_arm = cell_disabled_and_parity(args)
	try:
		cell_destructive(debug_arm, args, latencies)
		cell_intervening_mutation(debug_arm, args, latencies)
	finally:
		debug_arm.stop()
	cell_log_absence_and_lifecycle_only(args, latencies)

	ordered = sorted(latencies)
	p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
	print(json.dumps({
		"preflight_samples": len(latencies),
		"preflight_p95_ms_report_only": round(p95, 3),
	}, sort_keys=True))
	print("E0_PREFLIGHT PASS")


if __name__ == "__main__":
	main()
