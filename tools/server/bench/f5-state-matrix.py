#!/usr/bin/env python3
"""Milestone-F retained server round-trip gate.

This harness adds no server backdoor. It drives the public completion and
authenticated slot capture/erase/import routes, and reconstructs the live VBR
tier schedule from the shipped transition log. Every run pins both cache sides,
the budget, the freeze mode, and (when supplied) the degrade-order file.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request


DEGRADE_RE = re.compile(
	r"VBR degrade #\d+:\s+(cache_([kv])_\S*)\s+L(\d+)\s+->\s+(\S+)")

def request(base, path, body=None, tenant=None, timeout=600):
	headers = {}
	data = None
	if body is not None:
		data = json.dumps(body).encode()
		headers["Content-Type"] = "application/json"
	if tenant:
		headers["X-Api-Key"] = tenant
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


def make_prompt_tokens(base, wanted):
	text = " ".join(
		f"F5 ledger row {i} preserves deterministic state matrix evidence."
		for i in range(wanted))
	status, payload = request(base, "/tokenize", {
		"content": text, "add_special": True,
	})
	if status != 200 or not isinstance(payload.get("tokens"), list):
		raise RuntimeError(f"tokenize failed: {status} {payload}")
	tokens = payload["tokens"]
	if len(tokens) < wanted:
		raise RuntimeError(f"tokenize returned {len(tokens)} < {wanted}")
	return tokens[:wanted]


def completion(base, slot, tokens, n_predict, seed, n_probs=1):
	status, payload = request(base, "/completion", {
		"prompt": tokens,
		"id_slot": slot,
		"cache_prompt": True,
		"n_predict": n_predict,
		"temperature": 0.0,
		"seed": seed,
		"n_probs": n_probs,
		"ignore_eos": True,
		"return_tokens": True,
	})
	if status != 200:
		raise RuntimeError(f"completion failed: {status} {payload}")
	return payload


def prime_slot(base, slot, tokens, seed):
	payload = completion(base, slot, tokens, 1, seed)
	predicted = []
	for row in payload.get("completion_probabilities") or []:
		token = row.get("id")
		choices = row.get("top_logprobs") or []
		if token is None and choices:
			token = choices[0].get("id")
		if token is not None:
			predicted.append(token)
	if len(predicted) != 1:
		raise RuntimeError(
			f"prime must expose exactly one generated token id: {payload}")
	return tokens + predicted


def result_fingerprint(payload):
	top = []
	for token in payload.get("completion_probabilities") or []:
		token_id = token.get("id")
		logprob = token.get("logprob")
		if token_id is not None and logprob is not None:
			top.append((token_id, float(logprob)))
	return {
		"content_sha": hashlib.sha256(
			(payload.get("content") or "").encode()).hexdigest(),
		"content": payload.get("content"),
		"tokens": payload.get("tokens") or [value[0] for value in top],
		"scores": [value[1] for value in top],
		"tokens_predicted": payload.get("tokens_predicted"),
		"stop_type": payload.get("stop_type"),
		"stopping_word": payload.get("stopping_word"),
	}


def compare_results(reference, restored, tolerance):
	lhs = result_fingerprint(reference)
	rhs = result_fingerprint(restored)
	if lhs["content_sha"] != rhs["content_sha"] or lhs["tokens"] != rhs["tokens"]:
		return False, float("inf"), lhs, rhs
	if len(lhs["scores"]) != len(rhs["scores"]):
		return False, float("inf"), lhs, rhs
	max_abs = max((abs(a - b) for a, b in zip(
		lhs["scores"], rhs["scores"])), default=0.0)
	return max_abs <= tolerance, max_abs, lhs, rhs


def one_step_distribution(base, slot, tokens, seed, n_probs):
	payload = completion(base, slot, tokens, 1, seed, n_probs)
	rows = payload.get("completion_probabilities") or []
	if len(rows) != 1:
		raise AssertionError(
			f"one-step response has {len(rows)} probability rows: {payload}")
	row = rows[0]
	selected = row.get("id")
	distribution = {}
	if selected is not None and row.get("logprob") is not None:
		distribution[int(selected)] = float(row["logprob"])
	for choice in row.get("top_logprobs") or []:
		if choice.get("id") is not None and choice.get("logprob") is not None:
			distribution[int(choice["id"])] = float(choice["logprob"])
	if selected is None:
		tokens_out = payload.get("tokens") or []
		if len(tokens_out) == 1:
			selected = int(tokens_out[0])
	if selected is None or int(selected) not in distribution:
		raise AssertionError(f"one-step response lacks selected logprob: {payload}")
	return int(selected), distribution


def native_teacher_trace(base, slot, tokens, n_steps, seed, n_probs):
	prompt = list(tokens)
	trace = []
	for _ in range(n_steps):
		selected, distribution = one_step_distribution(
			base, slot, prompt, seed, n_probs)
		trace.append((selected, distribution))
		prompt.append(selected)
	return trace


def compare_teacher_trace(
		base, slot, tokens, reference, seed, n_probs, tolerance):
	prompt = list(tokens)
	max_abs = 0.0
	observations = []
	for step, (native_token, native_distribution) in enumerate(reference):
		restored_token, restored_distribution = one_step_distribution(
			base, slot, prompt, seed, n_probs)
		# Independently sampled continuations may fork at a near tie and then
		# compare different computations. P3 standard 2 instead teacher-forces
		# the native same-tier token sequence and compares the distributions at
		# the identical frontier. Both selected tokens must be visible on both
		# sides, and every shared top-N score is bounded by the declared gate.
		required = {native_token, restored_token}
		missing = [token for token in required
                   if token not in native_distribution or
                   token not in restored_distribution]
		if missing:
			raise AssertionError(
				f"downward step {step}: top-{n_probs} misses {missing}; "
				f"native={native_distribution} restored={restored_distribution}")
		# Documented cell-5 protocol: identical chosen tokens each step. A
		# genuine near-tie fork would still satisfy the distribution bound, so
		# argmax equality is the honest first assertion, not a relaxation.
		if restored_token != native_token:
			raise AssertionError(
				f"downward step {step}: chosen token diverged "
				f"(native={native_token} restored={restored_token})")
		shared = set(native_distribution) & set(restored_distribution)
		# Documented protocol: ALL top-N probabilities common each step.
		if len(shared) != n_probs:
			raise AssertionError(
				f"downward step {step}: only {len(shared)}/{n_probs} "
				f"common top-{n_probs} tokens")
		step_max = max(abs(native_distribution[token] -
                           restored_distribution[token]) for token in shared)
		max_abs = max(max_abs, step_max)
		if step_max > tolerance:
			raise AssertionError(
				f"downward step {step}: max_abs={step_max} > {tolerance}; "
				f"native={native_distribution} restored={restored_distribution}")
		observations.append({
			"step": step,
			"native_token": native_token,
			"restored_token": restored_token,
			"max_abs": step_max,
			"n_common": len(shared),
		})
		prompt.append(native_token)
	return max_abs, observations


def parse_tiers(log_path, start_offset=0):
	current = {}
	transitions = []
	with open(log_path, errors="replace") as log:
		log.seek(start_offset)
		for line in log:
			match = DEGRADE_RE.search(line)
			if not match:
				continue
			side = match.group(2)
			layer = int(match.group(3))
			tier = match.group(4).lower().replace("turbo", "t")
			tier = tier.split("_")[0]
			current[(layer, side)] = tier
			transitions.append((layer, side, tier))
	return current, transitions




def serializable_tiers(current):
	return {f"{layer}:{side}": tier
			for (layer, side), tier in sorted(current.items())}


class Server:
	def __init__(self, args):
		self.args = args
		self.base = f"http://127.0.0.1:{args.port}"
		self.log_path = os.path.join(args.workdir, f"f5-{args.cell}.log")
		self.process = None

	def start(self):
		env = os.environ.copy()
		env["VBR_FREEZE"] = "1"
		env["VBR_BUDGET_MIB"] = str(self.args.budget_mib)
		if self.args.cell == "downward":
			env["VBR_F5_PRESERVE_EMPTY_TIERS"] = "1"
		if self.args.degrade_order:
			env["VBR_DEGRADE_ORDER"] = os.path.abspath(
				self.args.degrade_order)
		command = [
			self.args.server_bin, "-m", self.args.model, "-ngl", "99",
			"-c", str(self.args.ctx), "-b", str(self.args.batch),
			"-ub", str(self.args.batch), "-np", "2",
			"-ctk", "vbr", "-ctv", "vbr", "--vbr-floor", "t1",
			"--vbr-vram", f"{self.args.budget_mib}M",
			"--vbr-reclaim-floor", "0", "--vbr-reset-keep-frac", "0",
			"--cache-lifecycle", "--slots", "-fa", "on",
			"--no-context-shift", "-lv", "4", "--port", str(self.args.port),
		]
		os.makedirs(self.args.workdir, exist_ok=True)
		log = open(self.log_path, "w")
		self.process = subprocess.Popen(command, env=env, stdout=log,
										stderr=subprocess.STDOUT)
		for _ in range(180):
			try:
				status, payload = request(self.base, "/health", timeout=5)
				if status == 200 and payload.get("status") == "ok":
					return
			except Exception:
				pass
			if self.process.poll() is not None:
				break
			time.sleep(1)
		raise RuntimeError("server failed health:\n" + subprocess.run(
			["tail", "-20", self.log_path], capture_output=True,
			text=True).stdout)

	def stop(self):
		if self.process is not None:
			self.process.terminate()
			try:
				self.process.wait(timeout=20)
			except subprocess.TimeoutExpired:
				self.process.kill()
				self.process.wait()


def slot_action(server, slot, action, tenant, body=None):
	return request(server.base, f"/slots/{slot}?action={action}",
                   {} if body is None else body, tenant)


def wait_slot_idle(server, slot, timeout=30):
	deadline = time.time() + timeout
	while time.time() < deadline:
		status, payload = request(server.base, "/slots")
		slots = payload.get("slots", payload) if isinstance(
			payload, dict) else payload
		current = next((row for row in (slots or [])
						if row.get("id", row.get("id_slot")) == slot), None)
		if status == 200 and current is not None and not current.get(
				"is_processing", False):
			# The erase result and idle bit may be visible in the same server
			# turn. Leave one update tick for llama_memory_breathe() to shrink
			# VBR watermarks (and, unless the F5 downward latch is armed, apply
			# the normal empty-boundary full reset).
			time.sleep(2.0)
			request(server.base, "/slots")
			return
		time.sleep(0.05)
	raise RuntimeError(f"slot {slot} did not become idle")


def assert_capture(captured, cell):
	if captured.get("status") != "ok" or not captured.get("reference"):
		raise AssertionError(f"{cell}: capture failed: {captured}")
	if captured.get("consistency") != "capture_exact":
		raise AssertionError(f"{cell}: dishonest capture label: {captured}")


def erase_and_wait(server, slot, check=True):
	status, payload = slot_action(
		server, slot, "erase", server.args.tenant)
	if check and status != 200:
		raise AssertionError(f"slot {slot} erase failed: {payload}")
	wait_slot_idle(server, slot)
	return status, payload


def import_reference(server, slot, reference):
	return slot_action(server, slot, "import", server.args.tenant,
		{"reference": reference})


def run_native_cell(server, args, tokens):
	# Null/reference arm first, then an empty-cache reset and a second prime.
	# VBR_FREEZE makes the second controller schedule a pure function of the
	# same token block and explicit budget, without keeping a peer sequence
	# live (the import validator correctly requires a dedicated target).
	reference_tokens = prime_slot(server.base, 0, tokens, args.seed)
	reference = completion(server.base, 0, reference_tokens,
                           args.n_predict, args.seed)
	erase_and_wait(server, 0)
	source_log_offset = os.path.getsize(server.log_path)
	tokens = prime_slot(server.base, 0, tokens, args.seed)
	if tokens != reference_tokens:
		raise AssertionError("VBR_FREEZE reference/source token blocks differ")
	_, captured = slot_action(server, 0, "capture", args.tenant)
	assert_capture(captured, args.cell)
	source_tiers, source_transitions = parse_tiers(
		server.log_path, source_log_offset)

	erase_and_wait(server, 0)
	status, imported = import_reference(server, 0, captured["reference"])
	if status != 200 or imported.get("status") != "ok":
		raise AssertionError(f"native import failed: {status} {imported}")
	expected = {
		"decision": "native_import",
		"consistency": "capture_exact",
		"downward_reserve_status": "not_attempted",
	}
	for key, value in expected.items():
		if imported.get(key) != value:
			raise AssertionError(f"native {key}: {imported}")
	restored = completion(server.base, 0, tokens, args.n_predict, args.seed)
	equal, max_abs, lhs, rhs = compare_results(reference, restored, 0.0)
	if not equal:
		raise AssertionError(
			f"native continuation differs max_abs={max_abs} {lhs} {rhs}")
	erase_and_wait(server, 0, check=False)
	status, second = import_reference(server, 0, captured["reference"])
	if status != 200 or second.get("status") != "ok" or \
			second.get("decision") != "native_import":
		raise AssertionError(f"native no-pin re-import failed: {second}")
	return (captured, imported, second, max_abs,
			serializable_tiers(source_tiers),
			source_transitions)


def run_downward_cell(server, args, source_tokens, pressure_tokens):
	source_tokens = prime_slot(
		server.base, 0, source_tokens, args.seed)
	_, captured = slot_action(server, 0, "capture", args.tenant)
	assert_capture(captured, args.cell)
	if captured.get("stash_bytes") != 0:
		raise AssertionError("downward source must be full-domain/stash-absent")

	# Slot 1 holds occupancy while slot 0 is erased twice. This is the public,
	# production path that prevents an empty-tree full reset between the native
	# same-tier oracle and the import target.
	prime_slot(server.base, 1, pressure_tokens, args.seed)
	current, transitions = parse_tiers(server.log_path)
	if len(transitions) < args.min_degrades:
		raise AssertionError(
			f"only {len(transitions)} degrade transitions, expected "
			f"{args.min_degrades}")

	# P3 standard 2 compares against the shipped live-degrade oracle: the
	# captured source remains live while the pressure slot lowers its KV
	# representation, then continues from that exact lower-tier state. This
	# preserves the recurrent companion frontier, unlike re-decoding the
	# whole hybrid prefix at a lower tier (which is a different computation).
	# Retire the pressure sequence first so both arms use the source artifact's
	# exact watermark; slot 0 keeps the tree non-empty and therefore retains
	# the controller-selected tier vector without the F5 empty-tier latch.
	erase_and_wait(server, 1, check=False)
	native = native_teacher_trace(
		server.base, 0, source_tokens, args.n_predict,
		args.seed, args.n_probs)
	erase_and_wait(server, 0, check=False)
	status, imported = import_reference(server, 0, captured["reference"])
	if status != 200 or imported.get("status") != "ok":
		raise AssertionError(f"downward import failed: {status} {imported}")
	expected = {
		"decision": "downward_rebase",
		"consistency": "live_rebased",
	}
	for key, value in expected.items():
		if imported.get(key) != value:
			raise AssertionError(f"downward {key}: {imported}")
	if imported.get("downward_reserve_status") not in (
			"reserved", "reserved_stashless"):
		raise AssertionError(f"downward reservation not usable: {imported}")
	max_abs, observations = compare_teacher_trace(
		server.base, 0, source_tokens, native, args.seed,
		args.n_probs, args.tolerance)
	erase_and_wait(server, 0, check=False)
	_, second = import_reference(server, 0, captured["reference"])
	# A whole_import resets representation provenance.  The original source
	# reference therefore cannot be replayed a second time into this retained
	# tier vector: unchanged units now carry whole_import rather than the
	# source's degrade transition.  The exact representation_mismatch proves
	# the catalog/reference resolved and failed at validation, not because the
	# first import pinned it, made it busy, or leaked accounting claims.
	if second.get("status") != "validation_failed" or \
			second.get("validation_status") != "representation_mismatch" or \
			second.get("downward_reserve_status") != "not_attempted":
		raise AssertionError(
			f"downward re-import terminal is not provenance-typed: {second}")
	return (captured, imported, second, max_abs,
			serializable_tiers(current), transitions,
			observations)


def run_auth_negative(server, args, tokens):
	tokens = prime_slot(server.base, 0, tokens, args.seed)
	_, captured = slot_action(server, 0, "capture", args.tenant)
	assert_capture(captured, args.cell)
	erase_and_wait(server, 0, check=False)
	status, missed = slot_action(server, 0, "import", "wrong-tenant",
                                 {"reference": captured["reference"]})
	if missed.get("status") != "not_found":
		raise AssertionError(
			f"wrong tenant exposed reference existence: {status} {missed}")
	status, nonexistent = slot_action(
		server, 0, "import", args.tenant,
		{"reference": "vbrref:00000000000000000000000000000000"})
	if nonexistent.get("status") != "not_found":
		raise AssertionError(f"missing reference response differs: {nonexistent}")
	return captured, missed


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-bin", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--cell", choices=(
		"native", "downward", "auth-negative"),
		required=True)
	parser.add_argument("--workdir", required=True)
	parser.add_argument("--budget-mib", type=int, required=True)
	parser.add_argument("--prime-tokens", type=int, default=256)
	parser.add_argument("--pressure-tokens", type=int, default=4096)
	parser.add_argument("--ctx", type=int, default=8192)
	parser.add_argument("--batch", type=int, default=512)
	parser.add_argument("--n-predict", type=int, default=16)
	parser.add_argument("--n-probs", type=int, default=16)
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--port", type=int, default=8265)
	parser.add_argument("--tenant", default="f5-gate-tenant")
	parser.add_argument("--degrade-order")
	parser.add_argument("--min-degrades", type=int, default=1)
	parser.add_argument("--tolerance", type=float, default=1.0e-4)
	args = parser.parse_args()
	if args.prime_tokens + args.n_predict >= args.ctx or \
			args.pressure_tokens + args.n_predict >= args.ctx:
		parser.error("token counts must leave continuation room in --ctx")

	server = Server(args)
	server.start()
	try:
		source_tokens = make_prompt_tokens(server.base, args.prime_tokens)
		if args.cell == "auth-negative":
			result = run_auth_negative(server, args, source_tokens)
		elif args.cell == "downward":
			pressure = make_prompt_tokens(server.base, args.pressure_tokens)
			result = run_downward_cell(server, args, source_tokens, pressure)
		else:
			result = run_native_cell(server, args, source_tokens)
			current, transitions = result[4], result[5]
			if transitions:
				raise AssertionError(
					f"native anchor unexpectedly degraded: {transitions}")
		output = {"cell": args.cell, "status": "PASS", "result": result}
		with open(os.path.join(args.workdir, f"f5-{args.cell}.json"), "w") as f:
			json.dump(output, f, indent=2, sort_keys=True, default=str)
		print(f"F5_CELL={args.cell} PASS")
	finally:
		server.stop()


if __name__ == "__main__":
	main()
