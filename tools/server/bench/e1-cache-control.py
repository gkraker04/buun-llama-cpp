#!/usr/bin/env python3
"""E1 trusted-local cache-control HTTP acceptance gate."""

import argparse
import json
import os
import re
import shlex
import time

from server_gate_common import ManagedServerArm, header, request, write_bench_stamp


DESTRUCTION_RE = re.compile(r"CACHE_HOST_DESTRUCTION (\{.*\})")
VBR_HARD_SEAL_RE = re.compile(r"CACHE_VBR_HARD_SEAL (\{.*\})")


def wait_for_log_since(arm, offset, token, timeout=10):
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		with open(arm.log_path, errors="replace") as handle:
			handle.seek(offset)
			if token in handle.read():
				return
		time.sleep(0.05)
	raise RuntimeError(f"timed out waiting for {token}")


def destruction_records_since(path, offset):
	records = []
	with open(path, errors="replace") as handle:
		handle.seek(offset)
		for line in handle:
			match = DESTRUCTION_RE.search(line)
			if match:
				records.append(json.loads(match.group(1)))
	return records


def vbr_hard_seal_records_since(path, offset):
	records = []
	with open(path, errors="replace") as handle:
		handle.seek(offset)
		for line in handle:
			match = VBR_HARD_SEAL_RE.search(line)
			if match:
				records.append(json.loads(match.group(1)))
	return records


def vbr_hard_seal_units(records):
	units = set()
	for row in records:
		ordinal = row.get("order_ordinal")
		layer = row.get("layer")
		side = row.get("side")
		if (row.get("evidence_event") != "sealed_step" or
			not isinstance(ordinal, int) or ordinal < 0 or
			not isinstance(layer, int) or layer < 0 or
			side not in ("k", "v")):
			raise RuntimeError(f"malformed VBR sealed-step evidence: {row}")
		units.add((ordinal, layer, side))
	return units


def vbr_trace_paths(prefix):
	return [path for path in (prefix, prefix + ".base", prefix + ".swa")
		if os.path.exists(path)]


def vbr_trace_records(path):
	records = []
	with open(path, errors="replace") as handle:
		for line in handle:
			if not line.strip() or line.startswith("#"):
				continue
			fields = line.rstrip("\n").split("\t")
			if len(fields) != 7:
				raise RuntimeError(f"malformed VBR schedule trace row: {line!r}")
			records.append({
				"phase": fields[0],
				"boundary": int(fields[1]),
				"cursor": int(fields[2]),
				"tier_fnv": fields[3],
				"watermark": int(fields[4]),
				"used": int(fields[5]),
				"mapped_bytes": int(fields[6]),
			})
	return records


def vbr_trace_snapshot(prefix):
	return {path: vbr_trace_records(path) for path in vbr_trace_paths(prefix)}


def vbr_trace_degrades_since(prefix, before):
	degrades = []
	for path in vbr_trace_paths(prefix):
		records = vbr_trace_records(path)
		prior = before.get(path, [])
		if len(records) < len(prior):
			raise RuntimeError(f"VBR schedule trace was truncated: {path}")
		window = records[max(0, len(prior) - 1):]
		for old, new in zip(window, window[1:]):
			if (old["tier_fnv"] == new["tier_fnv"] or
					new["cursor"] <= old["cursor"]):
				continue
			degrades.append({
				"path": path,
				"from_cursor": old["cursor"],
				"to_cursor": new["cursor"],
				"ordinals": tuple(range(old["cursor"], new["cursor"])),
			})
	return degrades


def vbr_trace_state_changes_since(prefix, before):
	changes = []
	for path in vbr_trace_paths(prefix):
		records = vbr_trace_records(path)
		prior = before.get(path, [])
		if len(records) < len(prior):
			raise RuntimeError(f"VBR schedule trace was truncated: {path}")
		window = records[max(0, len(prior) - 1):]
		for old, new in zip(window, window[1:]):
			# Occupancy/watermark growth alone is not a retier/reset. Only the
			# controller cursor or representation digest is epoch-change evidence.
			fields = ("cursor", "tier_fnv")
			if any(old[field] != new[field] for field in fields):
				changes.append({"path": path, "before": old, "after": new})
	return changes


def assert_vbr_unified_hold_frozen(degrades):
	if degrades:
		raise RuntimeError(
			"hard-leased unified-KV VBR transitioned below T4 during hold")


def assert_vbr_release_thawed(degrades, sealed_records):
	if not degrades:
		raise RuntimeError(
			"released VBR lease did not thaw a deferred below-T4 transition")
	if sealed_records:
		raise RuntimeError(
			"released VBR lease continued emitting sealed-step evidence")


def vbr_renew_disposition(before_epoch, after_epoch, status, trace_changes):
	if before_epoch == after_epoch:
		if status != "partially_stale":
			raise RuntimeError(
				"append-stable VBR frontier did not remain partially_stale")
		return "renew"
	if status != "subject_lost" or not trace_changes:
		raise RuntimeError(
			"VBR epoch change lacked typed subject_lost/WS-0 transition evidence")
	return "reacquire"


def make_prompt_tokens(base, tag, wanted):
	text = " ".join(
		f"E1 {tag} calibrated VBR pressure row {index}."
		for index in range(wanted))
	status, payload, _ = request(base, "/tokenize", {
		"content": text, "add_special": True,
	})
	if (status != 200 or not isinstance(payload, dict) or
			not isinstance(payload.get("tokens"), list)):
		raise RuntimeError(f"tokenize failed: {status} {payload}")
	tokens = payload["tokens"]
	if len(tokens) < wanted:
		raise RuntimeError(f"tokenize returned {len(tokens)} < {wanted}")
	return tokens[:wanted]


def completion_error_type(payload):
	if not isinstance(payload, dict):
		return None
	error = payload.get("error", payload)
	return error.get("type") if isinstance(error, dict) else None


def bounded_pressure_sizes(initial, step, cap):
	current = initial
	while current <= cap:
		yield current
		if current == cap:
			return
		current = min(cap, current + step)


def cache_plan_preview(arm, prompt, slot):
	status, payload, headers = request(arm.base, "/cache/plan", {
		"prompt": prompt, "cache_prompt": True, "id_slot": slot,
	})
	if status != 200:
		raise RuntimeError(f"cache-plan preview failed: {status} {payload}")
	if header(headers, "Cache-Control") != "no-store":
		raise RuntimeError("cache-plan preview omitted Cache-Control: no-store")
	if (not isinstance(payload, dict) or
			payload.get("object") != "cache_plan_preflight"):
		raise RuntimeError(f"cache-plan preview returned wrong envelope: {payload}")
	return payload


def assert_released_fallback_unprotected(preview):
	planner = preview.get("planner") or {}
	if planner.get("provider") != "host_cache_entry":
		raise RuntimeError(
			f"fallback preview did not select its host row: {planner}")
	destruction = preview.get("destruction") or {}
	protection = destruction.get("protection")
	reason = destruction.get("reason")
	# After release the preview may not require destruction at all
	# (assessment=not_required => no lease verdict => protection=unavailable),
	# which is itself release evidence. Reject only surviving hard evidence.
	acceptable = {"none", "weighted"}
	if destruction.get("assessment") == "not_required":
		acceptable.add("unavailable")
	if protection not in acceptable or reason == "hard_lease_blocked":
		raise RuntimeError(
			f"released fallback remains hard-protected: {destruction}")


def assert_recovery_pin_exclusion(records):
	excluded = {
		item.get("artifact_id")
		for row in records
		for item in [row.get("recovery_pin_excluded")]
		if (row.get("evidence_event") == "recovery_pin_excluded" and
				isinstance(item, dict) and
				isinstance(item.get("artifact_id"), int) and
				item.get("artifact_id") > 0)
	}
	if len(excluded) != 1:
		raise RuntimeError(
			f"held pressure did not name exactly one pinned fallback: {records}")
	for row in records:
		if row.get("victim_artifact_id") in excluded:
			raise RuntimeError(
				f"pinned fallback entered the quoted victim inventory: {row}")
		if row.get("floor_victim_artifact_id") in excluded:
			raise RuntimeError(
				f"pinned fallback reached the pressure floor: {row}")
		if (row.get("state") == "executed" and
				excluded.intersection(row.get("victim_ids") or [])):
			raise RuntimeError(
				f"pinned fallback appeared in an executed eviction: {row}")
	outcomes = {
		row.get("floor_outcome") for row in records
		if row.get("evidence_event") == "floor_outcome"
	}
	if not outcomes.intersection(
			{"priced_evicted", "legacy_evicted", "publication_skipped"}):
		raise RuntimeError(
			f"held pressure omitted its floor outcome: {records}")
	return next(iter(excluded))


def materialize_retained_pair(
		arm, args, prompt, displacer, slot=0, family_binding=None):
	# Displace the prompt into the host cache, then resume the same lineage.
	# Lifecycle mode retains the restored source, yielding an exact live subject
	# plus a distinct host fallback with the same epoch/frontier. The lifecycle
	# record is the deterministic proof that fallback materialization completed
	# before the hard acquire.
	first_content, first_slot, first_tokens = completion(
		arm.base, prompt, slot, args.n_predict, family_binding,
		return_tokens=True)
	completion(arm.base, displacer, slot, args.n_predict)
	lifecycle_offset = os.path.getsize(arm.log_path)
	resumed_content, _, resumed_tokens = completion(
		arm.base, prompt, slot, args.n_predict, family_binding,
		return_tokens=True)
	wait_for_log_since(
		arm, lifecycle_offset, 'CACHE_HOST_LIFECYCLE {"mode":"non_consuming"')
	if resumed_content != first_content or resumed_tokens != first_tokens:
		raise RuntimeError(
			"deterministic resume changed the retained frontier suffix")
	# Host selectors are exact-state selectors, not prefix selectors. The host
	# node saved on displacement contains the sampled suffix as well as the
	# request prompt. Passing only `prompt` used to fail semantic resolution as
	# fallback_unavailable before storage-disjointness was even evaluated.
	# A mixed prompt preserves the generated token IDs exactly and avoids any
	# tokenizer merge ambiguity at the prompt/completion text boundary.
	# The FINAL sampled token is returned to the client but never decoded into
	# the KV/host entry (generation stops before its forward pass), so the
	# retained entry holds one token fewer than the response reports. Probe
	# evidence 2026-08-05: selector want=579 vs entry size=578, lcp=578.
	return first_content, first_slot, first_tokens, [prompt] + first_tokens[:-1]


def retained_pair(arm, args, tag, slot=0):
	prompt = f"E1 {tag} retained " + " ".join(
		f"{tag}-row-{i}" for i in range(96))
	displacer = f"E1 {tag} displacement " + " ".join(
		f"other-{i}" for i in range(96))
	_, _, _, selector = materialize_retained_pair(
		arm, args, prompt, displacer, slot)
	return prompt, selector


def acquire_hard(arm, holder, prompt, slot, key):
	lease = control(arm.base, "/cache/leases/acquire", {
		"holder": holder, "class": "hard", "ttl_ms": 300000,
		"floor": "t4", "subject": {"kind": "live_prefix", "slot_id": slot},
		"fallback": {"kind": "host_snapshot", "prompt": prompt},
		"allow_soft_fallback": False, "idempotency_key": key,
	})
	if (lease.get("granted_class") != "hard" or
			(lease.get("fallback") or {}).get("state") != "resolved"):
		raise RuntimeError(f"hard lease did not pin its fallback: {lease}")
	if lease.get("fallback_pinned_bytes") in (None, 0):
		raise RuntimeError(f"hard lease omitted pinned-byte evidence: {lease}")
	return lease


def capture_vbr_reference(arm, slot):
	status, payload, _ = request(
		arm.base, f"/slots/{slot}?action=capture", {})
	if (status != 200 or not isinstance(payload, dict) or
			payload.get("status") != "ok" or not payload.get("reference")):
		raise RuntimeError(f"VBR fallback capture failed: {status} {payload}")
	return payload["reference"]


def acquire_hard_vbr(arm, holder, slot, reference, key):
	lease = control(arm.base, "/cache/leases/acquire", {
		"holder": holder, "class": "hard", "ttl_ms": 300000,
		"floor": "t4", "subject": {"kind": "live_prefix", "slot_id": slot},
		"fallback": {"kind": "vbr_reference", "reference": reference},
		"allow_soft_fallback": False, "idempotency_key": key,
	})
	if (lease.get("granted_class") != "hard" or
			(lease.get("fallback") or {}).get("state") != "resolved"):
		raise RuntimeError(f"VBR hard lease did not bind sealed fallback: {lease}")
	return lease


def prime_vbr_hard(arm, args, tag, slot=0, token_count=None):
	prompt = (make_prompt_tokens(arm.base, tag, token_count)
		if token_count is not None else
		f"E1 {tag} VBR hard prefix " + " ".join(
			f"{tag}-vbr-{index}" for index in range(160)))
	frontier, receipt = vbr_completion_frontier(
		arm, args, prompt, slot)
	return frontier, capture_vbr_reference(arm, slot), receipt


def vbr_completion_frontier(arm, args, prompt, slot):
	status, payload, _ = request(arm.base, "/completion", {
		"prompt": prompt, "cache_prompt": True,
		"n_predict": args.n_predict, "temperature": 0,
		"seed": 7, "id_slot": slot, "return_tokens": True,
	})
	if status != 200 or not isinstance(payload, dict):
		raise RuntimeError(f"VBR frontier completion failed: {status} {payload}")
	tokens = payload.get("tokens")
	receipt = payload.get("cache_receipt")
	if (not isinstance(tokens, list) or not tokens or
		not isinstance(receipt, dict) or
		not isinstance(receipt.get("sequence_epoch"), int) or
		not isinstance(receipt.get("token_count"), int)):
		raise RuntimeError(f"VBR frontier evidence missing: {payload}")
	# The final sampled token is returned but is not decoded into KV. Preserve
	# the exact mixed frontier so an append cannot accidentally become a branch.
	frontier = (list(prompt) if isinstance(prompt, list) else [prompt])
	frontier.extend(tokens[:-1])
	return frontier, receipt


def control_response(base, path, body):
	status, payload, headers = request(base, path, body)
	if status != 200:
		raise RuntimeError(f"{path} HTTP {status}: {payload}")
	if header(headers, "Cache-Control") != "no-store":
		raise RuntimeError(f"{path} omitted Cache-Control: no-store")
	if not isinstance(payload, dict) or payload.get("object") != "cache_control":
		raise RuntimeError(f"{path} returned wrong envelope: {payload}")
	if payload.get("schema_version") != 1:
		raise RuntimeError(f"{path} returned wrong schema: {payload}")
	return payload.get("status"), payload.get("result") or {}


def control(base, path, body, expected="ok"):
	status, result = control_response(base, path, body)
	if status != expected:
		raise RuntimeError(f"{path} returned {status}, expected {expected}")
	return result


def completion(base, prompt, slot, n_predict, family_binding=None,
		return_tokens=False):
	body = {
		"prompt": prompt, "cache_prompt": True, "n_predict": n_predict,
		"temperature": 0, "seed": 7,
	}
	if return_tokens:
		body["return_tokens"] = True
	if slot is not None:
		body["id_slot"] = slot
	if family_binding is not None:
		body["family_binding"] = family_binding
	status, payload, _ = request(base, "/completion", body)
	if status != 200:
		raise RuntimeError(f"completion failed: {status} {payload}")
	result = payload.get("content"), payload.get("id_slot")
	return result + (payload.get("tokens") or [],) if return_tokens else result


class Arm(ManagedServerArm):
	def __init__(self, args, name, port, enabled, prompt_log=None):
		base = f"http://127.0.0.1:{port}"
		log_path = os.path.join(args.workdir, f"server-{name}.log")
		self.trace_prefix = os.path.join(
			args.workdir, f"server-{name}.vbrtrace")
		for path in (self.trace_prefix, self.trace_prefix + ".base",
				self.trace_prefix + ".swa"):
			try:
				os.unlink(path)
			except FileNotFoundError:
				pass
		cmd = [
			args.server_bin, "-m", args.model, "--host", "127.0.0.1",
			"--port", str(port), "-ngl", str(args.ngl), "-c", str(args.ctx),
			"-b", str(args.batch), "-np", str(args.parallel), "--slots",
			"--cache-lifecycle", "--cache-debug", "--cache-plan-authority", "lru",
			# The E0 preflight route is the measurement instrument for the
			# hard-pressure cell's protection-state assertions.
			"--cache-plan-preflight",
			"--cache-ram", str(args.cache_ram), "-ctk", args.ctk, "-ctv", args.ctv,
		]
		if enabled:
			cmd.append("--cache-control-api")
		if prompt_log:
			cmd += ["--log-prompts-dir", prompt_log]
		for extra in args.extra_server_arg:
			cmd += shlex.split(extra)
		env = os.environ.copy()
		if args.ctk == "vbr" or args.ctv == "vbr":
			# Freeze the controller on the fitted F5-downward budget. The gate
			# also passes --vbr-vram so the process command remains self-describing.
			env["VBR_FREEZE"] = "1"
			env["VBR_BUDGET_MIB"] = str(args.vbr_budget_mib)
			env["VBR_TRACE"] = self.trace_prefix
			cmd += ["--cache-receipt", "--cache-receipt-key", "e1-vbr-epoch-gate"]
		super().__init__(name, base, log_path, cmd, env=env)

	def wait(self, timeout=360):
		self.wait_healthy(timeout)


def cell_holder_and_lease(arm, args):
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "holder-live-gate",
	})
	prompt = "E1 lease subject " + " ".join(f"row-{i}" for i in range(48))
	completion(arm.base, prompt, 0, args.n_predict)
	# Exercise the soft-class wire independently of the hard VBR and retained
	# fallback cells below. The fallback field is a hard-only proof input and
	# the wire serializer must report its absence as JSON null here.
	lease = control(arm.base, "/cache/leases/acquire", {
		"holder": holder["holder"], "class": "soft", "ttl_ms": 300000,
		"subject": {"kind": "live_prefix", "slot_id": 0},
		"fallback": {"kind": "host_snapshot", "prompt": prompt},
		"idempotency_key": "lease-live-gate",
	})
	if lease.get("fallback") is not None:
		raise RuntimeError("soft acquire misrepresented the supplied fallback as pinned")
	control(arm.base, "/cache/leases/inspect", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	control(arm.base, "/cache/leases/renew", {
		"holder": holder["holder"], "lease": lease["lease"], "ttl_ms": 300000,
	})
	control(arm.base, "/cache/leases/release", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	control(arm.base, "/cache/holders/close", {"holder": holder["holder"]})
	print("E1_CACHE_CONTROL CELL holder_lifecycle PASS")
	print("E1_CACHE_CONTROL CELL lease_round_trip PASS")


def cell_hard_lease_pressure(arm, args):
	if args.ctk == "vbr" or args.ctv == "vbr":
		holder = control(arm.base, "/cache/holders/create", {
			"ttl_ms": 300000, "idempotency_key": "vbr-hard-pressure-holder",
		})
		_, reference, _ = prime_vbr_hard(
			arm, args, "hard-pressure", 0, args.vbr_prime_tokens)
		lease = acquire_hard_vbr(
			arm, holder["holder"], 0, reference, "vbr-hard-pressure-lease")
		held_offset = os.path.getsize(arm.log_path)
		held_trace = vbr_trace_snapshot(arm.trace_prefix)
		pressure = make_prompt_tokens(
			arm.base, "hard-pressure-load", args.vbr_pressure_total_token_cap)
		# Unified KV is mandatory for dynamic VBR. Every layer/side unit therefore
		# contains the leased slot's range, so the range-qualified hard seal freezes
		# the entire below-T4 band while held; there cannot be an "unleased unit"
		# transcode in this live shape. T8->T4 remains outside the seal, and an
		# unmeetable all-hard wave has the separately unit-proven typed terminal.
		# Match the fitted F5-downward shape: a 1791-token protected prefix,
		# 767-token initial pressure, and 125 MiB. Grow the unleased slot until
		# the first structural sealed-step row appears. The total-token cap—not a
		# fixed request count—is the honest bound; live forcing of the all-hard
		# terminal would require exhausting hundreds of 27B transcodes and is
		# deliberately left to the model-free termination matrix.
		sealed_records = []
		pressure_tokens = args.vbr_pressure_tokens
		for pressure_tokens in bounded_pressure_sizes(
				args.vbr_pressure_tokens, args.vbr_pressure_step_tokens,
				args.vbr_pressure_total_token_cap):
			status, payload, _ = request(arm.base, "/completion", {
				"prompt": pressure[:pressure_tokens], "cache_prompt": True,
				"n_predict": args.n_predict, "temperature": 0,
				"seed": 7, "id_slot": 1,
			})
			if (status != 200 and
					completion_error_type(payload) != "hard_lease_blocked"):
				raise RuntimeError(
					f"VBR pressure returned unexpected error: {status} {payload}")
			sealed_records = vbr_hard_seal_records_since(
				arm.log_path, held_offset)
			if sealed_records:
				break
		if not sealed_records:
			raise RuntimeError(
				"VBR pressure exhausted its token cap before a sealed step")
		with open(arm.log_path, errors="replace") as handle:
			handle.seek(held_offset)
			held_log = handle.read()
		vbr_hard_seal_units(sealed_records)
		# Live "VBR degrade" lines are -v-gated (the fork CLAUDE.md fact), and
		# CACHE_BUDGET residency is arena-level/tier-invariant. Neither is gate
		# currency. VBR_FREEZE arms the WS-0 schedule trace used by the F5/L2
		# gates: the held window must remain tier-frozen, then the same trace must
		# show a cursor/digest transition immediately after explicit release.
		trace_degrades = vbr_trace_degrades_since(
			arm.trace_prefix, held_trace)
		assert_vbr_unified_hold_frozen(trace_degrades)
		if "hard lease seals the live prefix" not in held_log:
			raise RuntimeError(
				"VBR pressure did not prove reclaim guard and the unified-KV held freeze")
		current = control(arm.base, "/cache/leases/inspect", {
			"holder": holder["holder"], "lease": lease["lease"],
		})
		if current.get("protection_state") != "current":
			raise RuntimeError(f"VBR pressure lost hard protection: {current}")
		control(arm.base, "/cache/leases/release", {
			"holder": holder["holder"], "lease": lease["lease"],
		})
		released_offset = os.path.getsize(arm.log_path)
		released_trace = vbr_trace_snapshot(arm.trace_prefix)
		# Post-release the cache can settle just under the budget ceiling
		# (observed: ~124MiB/125M), so a single fixed round may never attempt
		# a crossing. Grow fresh pressure progressively until the trace shows
		# the thaw transition, mirroring the held-phase grower. Bounded so
		# failure stays honest.
		thawed = []
		for round_index in range(6):
			grow = (f"E1 vbr thaw pressure round {round_index} " +
				" ".join(f"thaw-{round_index}-{i}" for i in range(192)))
			completion(arm.base, grow, 1, args.n_predict)
			thawed = vbr_trace_degrades_since(arm.trace_prefix, released_trace)
			if thawed:
				break
		released_records = vbr_hard_seal_records_since(
			arm.log_path, released_offset)
		assert_vbr_release_thawed(thawed, released_records)
		control(arm.base, "/cache/holders/close", {
			"holder": holder["holder"],
		})
		print("E1_CACHE_CONTROL CELL hard_lease_pressure PASS (vbr enforced)")
		print("E1_CACHE_CONTROL CELL vbr_reclaim_guard PASS")
		return
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "hard-pressure-holder",
	})
	_, fallback_prompt = retained_pair(arm, args, "hard-pressure", 0)
	lease = acquire_hard(
		arm, holder["holder"], fallback_prompt, 0, "hard-pressure-lease")
	held_log_offset = os.path.getsize(arm.log_path)
	pressure = " ".join(f"pressure-row-{i}" for i in range(128))
	for index in range(18):
		completion(arm.base, f"E1 hard pressure {index} {pressure}",
			1, args.n_predict)
	time.sleep(0.1)
	held_destruction = destruction_records_since(
		arm.log_path, held_log_offset)
	if not held_destruction:
		raise RuntimeError(
			"host-pressure arm did not exercise an eviction floor while hard lease held")
	# A fallback proof is a recovery pin, not a hard lease on the host artifact.
	# The pinned node is therefore excluded before quoting; hard_lease_blocked is
	# emitted only when a hard-leased artifact itself enters the victim ladder.
	# Name the excluded artifact from the debug event, then prove it appears in
	# neither quoted nor executed victim evidence in this attributed window.
	assert_recovery_pin_exclusion(held_destruction)
	# Renew re-resolves the pinned retained source. Its success after pressure is
	# the end-to-end proof that the attributed floors skipped that source.
	renewed = control(arm.base, "/cache/leases/renew", {
		"holder": holder["holder"], "lease": lease["lease"],
		"ttl_ms": 300000,
		"fallback": {"kind": "host_snapshot", "prompt": fallback_prompt},
	})
	if renewed.get("protection_state") != "current":
		raise RuntimeError(f"hard lease did not remain current: {renewed}")
	control(arm.base, "/cache/leases/release", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	released_preview = cache_plan_preview(arm, fallback_prompt, 1)
	assert_released_fallback_unprotected(released_preview)
	control(arm.base, "/cache/holders/close", {"holder": holder["holder"]})
	print("E1_CACHE_CONTROL CELL hard_lease_pressure PASS")


def cell_reattach(arm, args):
	if args.ctk == "vbr" or args.ctv == "vbr":
		_, reference, _ = prime_vbr_hard(arm, args, "reattach", 0)
		holder = control(arm.base, "/cache/holders/create", {
			"ttl_ms": 250, "idempotency_key": "vbr-reattach-holder",
		})
		lease = acquire_hard_vbr(
			arm, holder["holder"], 0, reference, "vbr-reattach-lease")
		time.sleep(0.35)
		control(arm.base, "/cache/events/query", {
			"holder": holder["holder"], "after_ordinal": 0, "limit": 8,
		}, expected="not_found")
		reattached = control(arm.base, "/cache/holders/reattach", {
			"holder_recovery": holder["holder_recovery"], "ttl_ms": 300000,
		})
		if not any(item.get("lease") == lease["lease"]
				for item in reattached.get("orphaned_leases") or []):
			raise RuntimeError("VBR reattach omitted orphaned hard lease")
		control(arm.base, "/cache/leases/release", {
			"holder": reattached["holder"], "lease": lease["lease"],
		})
		print("E1_CACHE_CONTROL CELL reattach PASS (vbr hard)")
		return
	_, fallback_prompt = retained_pair(arm, args, "reattach", 0)
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 250, "idempotency_key": "reattach-holder",
	})
	lease = acquire_hard(
		arm, holder["holder"], fallback_prompt, 0, "reattach-lease")
	time.sleep(0.35)
	# Any scheduler operation is a lifecycle point. The stale session becomes
	# not_found, while the recovery token names the orphaned holder.
	control(arm.base, "/cache/events/query", {
		"holder": holder["holder"], "after_ordinal": 0, "limit": 8,
	}, expected="not_found")
	reattached = control(arm.base, "/cache/holders/reattach", {
		"holder_recovery": holder["holder_recovery"], "ttl_ms": 300000,
	})
	orphaned = reattached.get("orphaned_leases") or []
	if not any(item.get("lease") == lease["lease"] for item in orphaned):
		raise RuntimeError(f"reattach omitted orphaned lease summary: {reattached}")
	control(arm.base, "/cache/leases/inspect", {
		"holder": reattached["holder"], "lease": lease["lease"],
	}, expected="orphaned")
	control(arm.base, "/cache/leases/release", {
		"holder": reattached["holder"], "lease": lease["lease"],
	})
	print("E1_CACHE_CONTROL CELL reattach PASS")


def cell_renew_new_fallback(arm, args):
	if args.ctk == "vbr" or args.ctv == "vbr":
		holder = control(arm.base, "/cache/holders/create", {
			"ttl_ms": 300000, "idempotency_key": "vbr-renew-holder",
		})
		prompt, reference, before_receipt = prime_vbr_hard(
			arm, args, "renew", 0)
		lease = acquire_hard_vbr(
			arm, holder["holder"], 0, reference, "vbr-renew-lease")
		before = control(arm.base, "/cache/leases/inspect", {
			"holder": holder["holder"], "lease": lease["lease"],
		})
		before_epoch = before_receipt["sequence_epoch"]
		if (before.get("proven_frontier", {}).get("token_count") !=
				before_receipt["token_count"]):
			raise RuntimeError(
				f"VBR lease/receipt frontier mismatch: {before} {before_receipt}")
		growth_trace = vbr_trace_snapshot(arm.trace_prefix)
		# Extend the exact decoded frontier. Reusing the original user prompt here
		# omits the sampled suffix and correctly looks like an identity-changing
		# branch—the same driver bug the f16 cell previously caught.
		grown_prompt = list(prompt) + [prompt[-1]]
		_, after_receipt = vbr_completion_frontier(
			arm, args, grown_prompt, 0)
		after_status, _ = control_response(
			arm.base, "/cache/leases/inspect", {
				"holder": holder["holder"], "lease": lease["lease"],
			})
		after_epoch = after_receipt["sequence_epoch"]
		trace_changes = vbr_trace_state_changes_since(
			arm.trace_prefix, growth_trace)
		disposition = vbr_renew_disposition(
			before_epoch, after_epoch, after_status, trace_changes)
		print("E1_CACHE_CONTROL EPOCH renew "
			f"before={before_epoch} after={after_epoch} "
			f"status={after_status} trace_changes={len(trace_changes)}")
		grown_reference = capture_vbr_reference(arm, 0)
		if disposition == "reacquire":
			control(arm.base, "/cache/leases/release", {
				"holder": holder["holder"], "lease": lease["lease"],
			})
			reacquired = acquire_hard_vbr(
				arm, holder["holder"], 0, grown_reference,
				"vbr-renew-reacquire")
			current = control(arm.base, "/cache/leases/inspect", {
				"holder": holder["holder"], "lease": reacquired["lease"],
			})
			if current.get("protection_state") != "current":
				raise RuntimeError(
					f"VBR epoch-change reacquire was not current: {current}")
			control(arm.base, "/cache/leases/release", {
				"holder": holder["holder"], "lease": reacquired["lease"],
			})
			print("E1_CACHE_CONTROL CELL renew_new_fallback PASS "
				"(vbr subject_lost+reacquire)")
			return
		renewed = control(arm.base, "/cache/leases/renew", {
			"holder": holder["holder"], "lease": lease["lease"],
			"ttl_ms": 300000,
			"fallback": {
				"kind": "vbr_reference", "reference": grown_reference,
			},
		})
		if (renewed.get("proven_frontier", {}).get("token_count", 0) <=
				before.get("proven_frontier", {}).get("token_count", 0)):
			raise RuntimeError("VBR renew did not advance the proven frontier")
		control(arm.base, "/cache/leases/release", {
			"holder": holder["holder"], "lease": lease["lease"],
		})
		print("E1_CACHE_CONTROL CELL renew_new_fallback PASS "
			"(vbr partially_stale+renew)")
		return
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "renew-holder",
	})
	_, fallback_prompt = retained_pair(arm, args, "renew", 0)
	lease = acquire_hard(
		arm, holder["holder"], fallback_prompt, 0, "renew-lease")
	before = control(arm.base, "/cache/leases/inspect", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	# A live-prefix lease follows append growth, not a branch from the original
	# user text. `fallback_prompt` is the exact decoded frontier (the final
	# sampled-but-not-decoded token is already trimmed by retained_pair). Reusing
	# only the original text would omit the generated suffix and correctly drive
	# the identity-change/subject_lost path instead of partially_stale.
	grown = list(fallback_prompt) + [" renewed frontier extension"]
	grown_content, _, grown_tokens = completion(
		arm.base, grown, 0, args.n_predict, return_tokens=True)
	partial = control(arm.base, "/cache/leases/inspect", {
		"holder": holder["holder"], "lease": lease["lease"],
	}, expected="partially_stale")
	# Materialize the grown fallback from another slot, then displace that slot;
	# this does not destroy the hard-leased subject on slot 0.
	fallback_content, _, fallback_tokens = completion(
		arm.base, grown, 1, args.n_predict, return_tokens=True)
	if fallback_content != grown_content or fallback_tokens != grown_tokens:
		raise RuntimeError("renew fallback did not reproduce the grown frontier")
	completion(arm.base, "E1 renewed fallback displacer", 1, args.n_predict)
	renewed = control(arm.base, "/cache/leases/renew", {
		"holder": holder["holder"], "lease": lease["lease"],
		"ttl_ms": 300000,
		"fallback": {
			"kind": "host_snapshot", "prompt": grown + fallback_tokens[:-1],
		},
	})
	if (renewed.get("proven_frontier", {}).get("token_count", 0) <=
			before.get("proven_frontier", {}).get("token_count", 0)):
		raise RuntimeError(
			f"renew did not advance proven frontier: {before} -> {renewed}")
	if partial:
		# A typed partially_stale response intentionally has an empty result body.
		pass
	current = control(arm.base, "/cache/leases/inspect", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	if current.get("protection_state") != "current":
		raise RuntimeError(f"renew left lease partially stale: {current}")
	control(arm.base, "/cache/leases/release", {
		"holder": holder["holder"], "lease": lease["lease"],
	})
	print("E1_CACHE_CONTROL CELL renew_new_fallback PASS")


def cell_events(arm):
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "events-holder",
	})
	for index in range(70):
		family = control(arm.base, "/cache/families/register", {
			"holder": holder["holder"], "label": f"event-{index}",
			"idempotency_key": f"event-family-{index}",
		})
		if not family.get("family"):
			raise RuntimeError("event producer did not return family")
	events = control(arm.base, "/cache/events/query", {
		"holder": holder["holder"], "after_ordinal": 0, "limit": 64,
	})
	rows = events.get("events") or []
	if not events.get("overflowed") or not rows:
		raise RuntimeError(f"event overflow was not reported honestly: {events}")
	for row in rows:
		for required in ("ordinal", "timestamp_ms", "kind", "status",
				"subject_kind", "family_role", "lease"):
			if required not in row:
				raise RuntimeError(f"event omitted {required}: {row}")
	other = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "events-other",
	})
	foreign = control(arm.base, "/cache/events/query", {
		"holder": other["holder"], "after_ordinal": 0, "limit": 64,
	})
	if any(row.get("ordinal", 0) > 1 for row in foreign.get("events") or []):
		raise RuntimeError("event query crossed holder boundary")
	print("E1_CACHE_CONTROL CELL events PASS")


def cell_family(arm, args):
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": "family-holder",
	})
	family = control(arm.base, "/cache/families/register", {
		"holder": holder["holder"], "label": "main-agent",
		"idempotency_key": "family-register",
	})
	binding = control(arm.base, "/cache/families/bind", {
		"holder": holder["holder"], "family": family["family"], "role": "main",
		"idempotency_key": "family-bind",
	})
	branch_binding = control(arm.base, "/cache/families/bind", {
		"holder": holder["holder"], "family": family["family"], "role": "branch",
		"idempotency_key": "family-bind-branch",
	})
	if args.ctk == "vbr" or args.ctv == "vbr":
		# Dynamic VBR has no plain host prompt-cache inventory; E1.1c owns
		# controller enforcement. This arm proves propagation only.
		content, slot = completion(
			arm.base, "E1 declared main family retained conversation", 1,
			args.n_predict, binding["family_binding"])
		if not content or slot != 1:
			raise RuntimeError(
				"declared family completion failed or moved slots")
		print("E1_CACHE_CONTROL CELL family_binding PASS (vbr propagation)")
		return

	words = " ".join(f"family-row-{i}" for i in range(96))
	# Both roles must cross the host-save seam before pressure starts. Merely
	# decoding a declared branch leaves it live and absent from the D-A3 host
	# victim inventory, making a branch-first ranking assertion vacuous. The
	# non-consuming resume event proves each saved node is retained.
	content, slot, _, _ = materialize_retained_pair(
		arm, args,
		"E1 main retained " + words,
		"E1 main retained displacement " + words,
		1, binding["family_binding"])
	if not content or slot != 1:
		raise RuntimeError("declared main family materialization moved slots")
	materialize_retained_pair(
		arm, args,
		"E1 branch retained " + words,
		"E1 branch retained displacement " + words,
		0, branch_binding["family_binding"])
	# Continued equal-size churn forces the D-A3 ranker to compare those
	# retained entries under the configured host bound; debug evidence names
	# only the declared role, never family ID.
	ranking_offset = os.path.getsize(arm.log_path)
	for index in range(20):
		completion(arm.base,
			f"E1 family pressure {index} " + words, index % 2,
			args.n_predict)
	time.sleep(0.2)
	with open(arm.log_path, errors="replace") as handle:
		handle.seek(ranking_offset)
		ranked = []
		for line in handle:
			match = DESTRUCTION_RE.search(line)
			if not match:
				continue
			try:
				record = json.loads(match.group(1))
			except ValueError:
				continue
			if (record.get("declared_family_role") in ("main", "branch")
					and isinstance(record.get("price_us"), (int, float))):
				ranked.append(record)
	branches = [row for row in ranked
		if row.get("declared_family_role") == "branch"]
	mains = [row for row in ranked
		if row.get("declared_family_role") == "main"]
	if (not branches or not mains or
			ranked[0].get("declared_family_role") != "branch" or
			not any(row.get("retention_weight_milli") == 2000
				for row in branches) or
			not any(row.get("retention_weight_milli") == 4000
				for row in mains) or
			min(row["price_us"] for row in branches) >=
				min(row["price_us"] for row in mains)):
		raise RuntimeError(
			"declared branch was not the cheaper first priced family victim: "
			f"{ranked[:3]}")
	print("E1_CACHE_CONTROL CELL family_binding PASS (declared victim order)")


def cell_redaction(arm, prompt_log):
	sentinel = "E1_PRIVATE_CONTROL_BODY_5ed7d21"
	# This is a shared arm: earlier completion cells legitimately create prompt
	# files. Attribute this assertion only to the control-route window, then
	# scan all accumulated files for the private control values.
	before_files = {}
	if os.path.isdir(prompt_log):
		before_files = {
			os.path.join(root, name): os.path.getsize(os.path.join(root, name))
			for root, _, names in os.walk(prompt_log) for name in names}
	holder = control(arm.base, "/cache/holders/create", {
		"ttl_ms": 300000, "idempotency_key": sentinel,
	})
	private_values = {
		value for value in holder.values()
		if isinstance(value, str) and value}
	private_values.add(sentinel)
	arm.stop()
	with open(arm.log_path, errors="replace") as handle:
		log = handle.read()
	if any(value in log for value in private_values):
		raise RuntimeError("cache-control bearer/body appeared in server log")
	after_files = {}
	if os.path.isdir(prompt_log):
		after_files = {
			os.path.join(root, name): os.path.getsize(os.path.join(root, name))
			for root, _, names in os.walk(prompt_log) for name in names}
	new_files = set(after_files) - set(before_files)
	if new_files:
		raise RuntimeError(
			f"cache-control route created prompt-log files: {sorted(new_files)}")
	grown_files = {
		path: (before_files[path], size)
		for path, size in after_files.items()
		if path in before_files and size != before_files[path]}
	if grown_files:
		raise RuntimeError(
			f"cache-control route changed prompt-log files: {grown_files}")
	for path in after_files:
		with open(path, errors="replace") as handle:
			contents = handle.read()
		if any(value in contents for value in private_values):
			raise RuntimeError(
				f"cache-control bearer/body appeared in prompt log: {path}")
	print("E1_CACHE_CONTROL CELL redaction_no_store PASS")


def parity(args):
	transcripts = []
	for index, enabled in enumerate((False, True)):
		arm = Arm(args, f"parity-{index}", args.base_port + index, enabled)
		try:
			arm.wait()
			if not enabled:
				status, payload, headers = request(
					arm.base, "/cache/holders/create", {"ttl_ms": 1000})
				if (status != 501 or
						(payload or {}).get("error", {}).get("type") !=
							"not_supported_error" or
						header(headers, "Cache-Control") != "no-store"):
					raise RuntimeError(
						"disabled cache-control route did not return typed/no-store "
						f"refusal: {status} {payload}")
			transcripts.append(completion(
				arm.base, "E1 disabled parity deterministic prompt", 0,
				args.n_predict))
		finally:
			arm.stop()
	if transcripts[0] != transcripts[1]:
		raise RuntimeError("cache-control flag changed unused completion output/schedule")
	print("E1_CACHE_CONTROL CELL disabled_parity PASS")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-bin", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--workdir", required=True)
	parser.add_argument("--base-port", type=int, default=8580)
	parser.add_argument("--ctx", type=int, default=4096)
	parser.add_argument("--batch", type=int, default=512)
	parser.add_argument("--parallel", type=int, default=2)
	parser.add_argument("--cache-ram", type=int, default=512)
	parser.add_argument("--ngl", type=int, default=99)
	parser.add_argument("--ctk", default="f16")
	parser.add_argument("--ctv", default="f16")
	parser.add_argument("--n-predict", type=int, default=8)
	parser.add_argument("--vbr-budget-mib", type=int,
		default=int(os.environ.get("VBR_BUDGET_MIB", "125")))
	parser.add_argument("--vbr-prime-tokens", type=int, default=1791)
	parser.add_argument("--vbr-pressure-tokens", type=int, default=767)
	parser.add_argument("--vbr-pressure-step-tokens", type=int, default=256)
	parser.add_argument("--vbr-pressure-total-token-cap", type=int, default=3584)
	parser.add_argument("--extra-server-arg", action="append", default=[])
	args = parser.parse_args()
	if (args.vbr_pressure_tokens <= 0 or args.vbr_pressure_step_tokens <= 0 or
		args.vbr_pressure_tokens > args.vbr_pressure_total_token_cap or
		args.vbr_pressure_total_token_cap + args.n_predict >= args.ctx):
		parser.error("VBR pressure token bounds must fit inside --ctx")
	os.makedirs(args.workdir, exist_ok=True)
	write_bench_stamp(
		args.workdir, "e1-cache-control", args.model, args.server_bin)

	parity(args)
	prompt_log = os.path.join(args.workdir, "prompt-log")
	arm = Arm(args, "enabled", args.base_port + 2, True, prompt_log)
	try:
		arm.wait()
		cell_holder_and_lease(arm, args)
		# Shared-arm attribution rule: cell_family needs an unsaturated host
		# cache to surface priced family rankings; cell_hard_lease_pressure
		# deliberately saturates it with unique recovery-refused entries, so
		# family (and its evidence window) must run BEFORE the pressure cell.
		cell_family(arm, args)
		cell_hard_lease_pressure(arm, args)
		cell_reattach(arm, args)
		cell_renew_new_fallback(arm, args)
		cell_events(arm)
		cell_redaction(arm, prompt_log)
	finally:
		arm.stop()
	print("E1_CACHE_CONTROL PASS")


if __name__ == "__main__":
	main()
