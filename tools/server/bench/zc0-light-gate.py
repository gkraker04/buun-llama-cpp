#!/usr/bin/env python3
"""Self-booting ZC0/ZC0b harness/config gate (no performance claim).

It runs the mandatory branch-off/concurrent-null/serialized-null arms, then
proves that an active but unfitted debug arm is rejected as "optimized" and
that the ZC6 default preserves explicit-off output semantics on the real
server. The modes intentionally differ in calibration and selection state.
"""

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request


DEVICE_MEMORY_RE = re.compile(
	r"LLAMA_DEVICE_MEMORY device=(\d+) model=(\d+) context=(\d+) "
	r"compute=(\d+) total=(\d+)")
DEVICE_MEMORY_BEGIN_RE = re.compile(r"LLAMA_DEVICE_MEMORY_BEGIN count=(\d+)")
DEVICE_MEMORY_END_RE = re.compile(r"LLAMA_DEVICE_MEMORY_END count=(\d+)")


def device_memory_witness(log_path):
	rows = {}
	expected = None
	complete = False
	for line in log_path.read_text(errors="replace").splitlines():
		begin = DEVICE_MEMORY_BEGIN_RE.search(line)
		end = DEVICE_MEMORY_END_RE.search(line)
		row = DEVICE_MEMORY_RE.search(line)
		if begin:
			if expected is not None or complete:
				raise RuntimeError("llama device-memory witness has multiple generations")
			expected = int(begin.group(1))
			if expected <= 0:
				raise RuntimeError("llama device-memory witness has an empty generation")
			continue
		if row:
			if expected is None or complete:
				raise RuntimeError("llama device-memory row is outside the final generation")
			device, model, context, compute, total = map(int, row.groups())
			if device in rows or total != model + context + compute:
				raise RuntimeError("llama device-memory witness has an invalid row")
			rows[device] = {
				"device": device,
				"model_bytes": model,
				"context_bytes": context,
				"compute_bytes": compute,
				"total_bytes": total,
			}
			continue
		if end:
			if expected is None or complete or int(end.group(1)) != expected:
				raise RuntimeError("llama device-memory witness has an invalid terminator")
			if len(rows) != expected or sorted(rows) != list(range(expected)):
				raise RuntimeError("llama device-memory witness is incomplete")
			complete = True
	if not complete:
		raise RuntimeError("llama device-memory final witness is missing")
	return [rows[index] for index in range(expected)]


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


def make_trace(path, requests):
	rows = [{
		"record": "capture_header", "schema": "capture_trace/v1",
		"tag": "zc0-light-gate",
	}]
	base = "ZC0 cache harness truth fixture. "
	for seq in range(requests):
		rows.append({
		# Two milliseconds preserves a saturated two-slot workload while
		# remaining above HTTP/event-loop ordering jitter on the gate host.
		"record": "request", "seq": seq, "t_arrival_ms": seq * 2.0,
			"total_ms": 100.0, "status": 200, "method": "POST",
			"path": "/completion", "stream": True,
			"body": {
				"prompt": base + ("alpha beta gamma delta " * ((seq % 4) + 1)),
				"n_predict": 6, "stream": True, "cache_prompt": True,
			},
			"fingerprint": {
				"system_prefix_sha": "zc0", "system_chars": 32,
				"prompt_chars": 64 * ((seq % 4) + 1), "n_messages": 0,
				"n_predict": 6, "tools": 0,
			},
			"server": {"timings": {"predicted_n": 6}},
		})
	path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def executable_provenance(path):
	resolved = path.resolve(strict=True)
	digest = hashlib.sha256()
	with resolved.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return {
		"resolved_path": str(resolved),
		"sha256": digest.hexdigest(),
		"size_bytes": resolved.stat().st_size,
	}


def directory_bytes(path):
	total = 0
	for root, _, files in os.walk(path):
		for name in files:
			try:
				total += os.stat(os.path.join(root, name), follow_symlinks=False).st_size
			except FileNotFoundError:
				# Atomic store replacement can retire a name during the walk.
				pass
	return total


def capstone_kv_cache_args(extra):
	"""Make the static cell explicit; a VBR cell must override both K and V."""
	has_k = any(value in ("-ctk", "--cache-type-k") for value in extra)
	has_v = any(value in ("-ctv", "--cache-type-v") for value in extra)
	if has_k != has_v:
		raise RuntimeError("capstone KV override must specify both K and V types")
	return [] if has_k else ["-ctk", "f16", "-ctv", "f16"]


class NvmlProcessInfoV3(ctypes.Structure):
	_fields_ = [
		("pid", ctypes.c_uint),
		("used_gpu_memory", ctypes.c_ulonglong),
		("gpu_instance_id", ctypes.c_uint),
		("compute_instance_id", ctypes.c_uint),
	]


class ResourceSampler:
	"""Out-of-process capstone resource witness; never runs in llama-server."""

	def __init__(self, pid, state_home):
		self.pid = pid
		self.state_home = state_home
		self.stop_event = threading.Event()
		self.peak_rss_bytes = 0
		self.peak_vram_bytes = 0
		self.max_state_dir_bytes = 0
		self.samples = 0
		self.rss_supported = False
		self.vram_supported = False
		self.vram_process_samples = 0
		self.error = None
		self.sample_ordinal = 0
		self.nvml = None
		self.nvml_handles = []
		try:
			self._init_nvml()
		except Exception as error:
			self.error = error
		self.thread = threading.Thread(target=self._run, name="zc6-resource-sampler")

	def _init_nvml(self):
		self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
		self.nvml.nvmlInit_v2.restype = ctypes.c_int
		self.nvml.nvmlShutdown.restype = ctypes.c_int
		self.nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
		self.nvml.nvmlDeviceGetCount_v2.restype = ctypes.c_int
		self.nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
			ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
		self.nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
		self.nvml.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
			ctypes.c_void_p,
			ctypes.POINTER(ctypes.c_uint),
			ctypes.POINTER(NvmlProcessInfoV3),
		]
		self.nvml.nvmlDeviceGetComputeRunningProcesses_v3.restype = ctypes.c_int
		if self.nvml.nvmlInit_v2() != 0:
			raise RuntimeError("nvmlInit_v2 failed")
		count = ctypes.c_uint()
		if self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
			raise RuntimeError("NVML reported no devices")
		for index in range(count.value):
			handle = ctypes.c_void_p()
			if self.nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
				raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
			self.nvml_handles.append(handle)

	def start(self):
		self.thread.start()

	def _sample_rss(self):
		status = pathlib.Path(f"/proc/{self.pid}/status")
		try:
			for line in status.read_text().splitlines():
				if line.startswith("VmRSS:"):
					parts = line.split()
					if len(parts) != 3 or parts[2] != "kB":
						raise RuntimeError("unsupported /proc VmRSS format")
					self.rss_supported = True
					self.peak_rss_bytes = max(self.peak_rss_bytes, int(parts[1]) * 1024)
					return
		except FileNotFoundError:
			return

	def _sample_vram(self):
		if self.nvml is None:
			return
		found = False
		used = 0
		for handle in self.nvml_handles:
			count = ctypes.c_uint(0)
			result = self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(
				handle, ctypes.byref(count), None)
			if result not in (0, 7):
				raise RuntimeError(f"NVML process census failed: {result}")
			if count.value == 0:
				continue
			entries = (NvmlProcessInfoV3 * count.value)()
			result = self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(
				handle, ctypes.byref(count), entries)
			if result != 0:
				raise RuntimeError(f"NVML process census retry failed: {result}")
			for index in range(count.value):
				if entries[index].pid != self.pid:
					continue
				value = entries[index].used_gpu_memory
				if value == (1 << 64) - 1:
					raise RuntimeError("NVML process memory unavailable")
				used += value
				found = True
		if found:
			self.vram_supported = True
			self.vram_process_samples += 1
			self.peak_vram_bytes = max(self.peak_vram_bytes, used)

	def _sample(self):
		# NVML is an in-process library query, so short arms get a genuine process
		# high-water witness instead of one integer-MiB nvidia-smi sample near exit.
		self._sample_vram()
		self._sample_rss()
		if self.sample_ordinal % 5 == 0:
			self.max_state_dir_bytes = max(
				self.max_state_dir_bytes, directory_bytes(self.state_home))
		self.samples += 1
		self.sample_ordinal += 1

	def _run(self):
		try:
			while not self.stop_event.is_set():
				self._sample()
				self.stop_event.wait(0.020)
		except Exception as error:
			self.error = error

	def finish(self):
		self.stop_event.set()
		self.thread.join(timeout=10)
		if self.thread.is_alive():
			raise RuntimeError("resource sampler did not stop")
		try:
			if self.error is not None:
				raise RuntimeError(f"resource sampler failed: {self.error}")
			if (not self.rss_supported or not self.vram_supported or
					self.vram_process_samples == 0):
				raise RuntimeError(
					"resource sampler is unsupported (requires Linux /proc and NVML v3)")
		finally:
			if self.nvml is not None:
				self.nvml.nvmlShutdown()
				self.nvml = None
		final_state = directory_bytes(self.state_home)
		self.max_state_dir_bytes = max(self.max_state_dir_bytes, final_state)
		return {
			"peak_rss_bytes": self.peak_rss_bytes,
			"peak_vram_bytes": self.peak_vram_bytes,
			"final_state_dir_bytes": final_state,
			"max_state_dir_bytes": self.max_state_dir_bytes,
			"samples": self.samples,
			"vram_process_samples": self.vram_process_samples,
			"vram_sampler": "nvml_process_bytes_20ms",
			"vram_poll_interval_ms": 20,
			"rss_supported": self.rss_supported,
			"vram_supported": self.vram_supported,
		}


def run_arm(args, name, server, parallel, extra, serialized=False,
		capstone_mode=None, session_id=None, executable=None, launch_order=None):
	log_path = args.workdir / f"{name}.server.log"
	scorecard = args.workdir / f"{name}.json"
	log = log_path.open("w")
	state_home = args.workdir / "state" / name
	if state_home.exists():
		shutil.rmtree(state_home)
	state_home.mkdir(parents=True, mode=0o700, exist_ok=True)
	state_home.chmod(0o700)
	kv_cache_args = (capstone_kv_cache_args(extra) if capstone_mode else
		["-ctk", "f16", "-ctv", "f16"])
	command = [
		str(server), "-m", str(args.model), "--host", "127.0.0.1",
		"--port", str(args.port), "-ngl", "99", "-c", "2048",
		"-np", str(parallel), "-fa", "on", *kv_cache_args,
		"--slots", *extra,
	]
	if capstone_mode:
		# Own affinity without a wrapper so executable provenance hashes the
		# actual llama-server binary used by every arm.
		command = ["/usr/bin/taskset", "-c", "0-11", *command]
	environment = dict(os.environ)
	environment["LLAMA_STATE_HOME"] = str(state_home)
	if capstone_mode:
		environment["LLAMA_DEVICE_MEMORY_WITNESS"] = "1"
	process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
		env=environment)
	sampler = ResourceSampler(process.pid, state_home) if capstone_mode else None
	if sampler:
		sampler.start()
	replay_succeeded = False
	primary_error = None
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
		replay_succeeded = True
	except BaseException as error:
		primary_error = error
	finally:
		cleanup_error = None
		try:
			stop_server(process, log)
		except BaseException as error:
			cleanup_error = error
		resources = None
		if sampler:
			try:
				# Join after server shutdown so the final state-directory sample
				# includes clean shutdown publication.  This is unconditional: a
				# failed readiness/replay must not strand a non-daemon thread.
				resources = sampler.finish()
			except BaseException as error:
				if cleanup_error is None:
					cleanup_error = error
	if primary_error is not None:
		raise primary_error.with_traceback(primary_error.__traceback__)
	if cleanup_error is not None:
		raise cleanup_error.with_traceback(cleanup_error.__traceback__)
	if capstone_mode and replay_succeeded:
		resources["llama_device_memory"] = device_memory_witness(log_path)
		resources["llama_device_memory_witness"] = "llama_owned_buffers_v1"
		card = json.loads(scorecard.read_text())
		card["capstone_arm"] = {
			"mode": capstone_mode,
			"session_id": session_id,
			"launch_order": list(launch_order),
			"executable": executable,
			"cpu_affinity": "0-11",
			"resources": resources,
		}
		scorecard.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
	return scorecard, log_path


def run_capstone_session(args):
	master = executable_provenance(args.master_server)
	branch = executable_provenance(args.branch_server)
	if ((master["sha256"], master["size_bytes"]) !=
			(branch["sha256"], branch["size_bytes"])):
		raise SystemExit("ZC6 capstone arms must use one executable identity")
	if args.capstone_extra_server_arg:
		if args.master_extra_server_arg or args.branch_extra_server_arg:
			raise SystemExit(
				"use only --capstone-extra-server-arg for a ZC6 capstone session")
		common = list(args.capstone_extra_server_arg)
	else:
		if args.master_extra_server_arg != args.branch_extra_server_arg:
			raise SystemExit("ZC6 capstone arms must use identical non-mode server arguments")
		common = list(args.branch_extra_server_arg)
	if "--cache-optimizer" in common:
		raise SystemExit("ZC6 capstone runner owns --cache-optimizer")
	session_id = args.capstone_session_id or args.workdir.resolve().name
	if args.primary_order:
		orders = tuple(args.primary_order.split(","))
	else:
		orders = (("auto", "learn", "baseline") if args.reverse_primary_order else
			("baseline", "learn", "auto"))
	if len(orders) != 3 or set(orders) != {"baseline", "learn", "auto"}:
		raise SystemExit("--primary-order must be one permutation of baseline,learn,auto")
	results = {}
	for serialized, shape, parallel in (
			(False, "concurrent", 2), (True, "serialized", 1)):
		for mode in orders:
			extra = list(common)
			# Exercise the candidate auto implementation explicitly until ZC6's
			# resource endpoint qualifies the separate absent-mode rollout.
			extra.extend(("--cache-optimizer", mode))
			name = f"capstone-{shape}-{mode}"
			results[(shape, mode)], _ = run_arm(
				args, name, args.branch_server, parallel, extra,
				serialized=serialized, capstone_mode=mode,
				session_id=session_id, executable=branch, launch_order=orders)
	print(json.dumps({
		"schema": "zc6-capstone-session/v1",
		"session_id": session_id,
		"launch_order": list(orders),
		"executable": branch,
		"scorecards": {
			shape: {mode: str(results[(shape, mode)]) for mode in ("baseline", "learn", "auto")}
			for shape in ("concurrent", "serialized")
		},
	}, sort_keys=True))
	print("ZC0_LIGHT_GATE PASS capstone_three_arm_session")


def self_test_cleanup():
	with tempfile.TemporaryDirectory() as directory:
		args = types.SimpleNamespace(
			workdir=pathlib.Path(directory), model=pathlib.Path(directory) / "missing.gguf",
			port=1, boot_timeout=2.0, bench=pathlib.Path(__file__).resolve().parent,
			trace=pathlib.Path(directory) / "missing-trace.jsonl", request_timeout=1.0,
		)
		try:
			run_arm(
				args, "cleanup-failure", pathlib.Path(sys.executable), 1, [],
				capstone_mode="baseline", session_id="cleanup",
				executable={"sha256": "0" * 64, "size_bytes": 1},
				launch_order=("baseline", "learn", "auto"))
		except RuntimeError as error:
			if "server exited early" not in str(error):
				raise AssertionError("cleanup replaced the primary readiness error") from error
		else:
			raise AssertionError("failed server unexpectedly reached replay")
	if any(thread.name == "zc6-resource-sampler" and thread.is_alive()
			for thread in threading.enumerate()):
		raise AssertionError("failed arm stranded its resource sampler")
	self_test_device_memory()
	assert capstone_kv_cache_args([]) == ["-ctk", "f16", "-ctv", "f16"]
	assert capstone_kv_cache_args(
		["-ctk", "q8_0", "-ctv", "q8_0"]) == []
	try:
		capstone_kv_cache_args(["-ctk", "q8_0"])
	except RuntimeError:
		pass
	else:
		raise AssertionError("one-sided capstone KV override was accepted")
	print("ZC0_LIGHT_GATE cleanup_failure PASS")


def self_test_device_memory():
	row0 = "LLAMA_DEVICE_MEMORY device=0 model=7 context=2 compute=1 total=10\n"
	row1 = "LLAMA_DEVICE_MEMORY device=1 model=5 context=3 compute=2 total=10\n"
	valid = "LLAMA_DEVICE_MEMORY_BEGIN count=2\n" + row0 + row1 + \
		"LLAMA_DEVICE_MEMORY_END count=2\n"
	with tempfile.TemporaryDirectory() as directory:
		path = pathlib.Path(directory) / "memory.log"
		path.write_text(valid)
		assert len(device_memory_witness(path)) == 2
		invalid = (
			row0,
			valid + "LLAMA_DEVICE_MEMORY_BEGIN count=2\n" + row0,
			"LLAMA_DEVICE_MEMORY_BEGIN count=2\n" + row0 + row0 +
				"LLAMA_DEVICE_MEMORY_END count=2\n",
			"LLAMA_DEVICE_MEMORY_BEGIN count=2\n" + row0 +
				"LLAMA_DEVICE_MEMORY_END count=2\n",
			valid + valid,
		)
		for contents in invalid:
			path.write_text(contents)
			try:
				device_memory_witness(path)
			except RuntimeError:
				continue
			raise AssertionError("invalid device-memory generation was accepted")


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
		raise SystemExit("default optimizer output diverged from explicit off: "
			f"absent={absent_rows} explicit={explicit_rows}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--master-server", type=pathlib.Path)
	parser.add_argument("--branch-server", type=pathlib.Path)
	parser.add_argument("--model", type=pathlib.Path)
	parser.add_argument("--workdir", type=pathlib.Path)
	parser.add_argument("--port", type=int, default=8081)
	parser.add_argument("--boot-timeout", type=float, default=180.0)
	parser.add_argument("--request-timeout", type=float, default=180.0)
	parser.add_argument("--requests", type=int, default=4)
	parser.add_argument("--master-extra-server-arg", action="append", default=[])
	parser.add_argument("--branch-extra-server-arg", action="append", default=[])
	parser.add_argument("--capstone-extra-server-arg", action="append", default=[],
		help="one common non-mode argument list for all ZC6 capstone arms")
	parser.add_argument("--reverse-primary-order", action="store_true",
		help="run explicit auto before baseline in the two capstone pairs")
	parser.add_argument("--primary-order",
		help="ZC6 three-arm order, exactly one comma-separated permutation of baseline,learn,auto")
	parser.add_argument("--capstone-pairs-only", action="store_true",
		help="run only fresh baseline/learn/auto concurrent+serialized arms")
	parser.add_argument("--capstone-session-id",
		help="stable restart/session identity embedded in every capstone scorecard")
	parser.add_argument("--self-test-cleanup", action="store_true")
	args = parser.parse_args()
	if args.self_test_cleanup:
		self_test_cleanup()
		return
	if not all((args.master_server, args.branch_server, args.model, args.workdir)):
		parser.error("--master-server, --branch-server, --model, and --workdir are required")
	args.bench = pathlib.Path(__file__).resolve().parent
	args.workdir.mkdir(parents=True, exist_ok=True)
	args.trace = args.workdir / "zc0-truth.jsonl"
	if args.requests < 4:
		parser.error("--requests must be at least 4")
	make_trace(args.trace, args.requests)
	if args.capstone_pairs_only:
		run_capstone_session(args)
		return

	if args.reverse_primary_order:
		parity, _ = run_arm(args, "branch-no-features", args.branch_server, 2,
			args.branch_extra_server_arg)
		base, _ = run_arm(args, "baseline-master", args.master_server, 2,
			args.master_extra_server_arg)
	else:
		base, _ = run_arm(args, "baseline-master", args.master_server, 2,
			args.master_extra_server_arg)
		parity, _ = run_arm(args, "branch-no-features", args.branch_server, 2,
			args.branch_extra_server_arg)
	if not args.capstone_pairs_only:
		active, active_log = run_arm(args, "branch-active-unfitted", args.branch_server, 2,
			[*args.branch_extra_server_arg,
			 "--cache-optimizer", "off", "--cache-debug", "--cache-lifecycle",
			 "--cache-plan-authority", "lru"])
		null, _ = run_arm(args, "baseline-concurrent-null", args.master_server, 2,
			args.master_extra_server_arg)
	if args.reverse_primary_order:
		config_absent, _ = run_arm(
			args, "branch-serialized-absent", args.branch_server, 1,
			args.branch_extra_server_arg, serialized=True)
		serial_base, _ = run_arm(
			args, "baseline-serialized", args.master_server, 1,
			args.master_extra_server_arg, serialized=True)
	else:
		serial_base, _ = run_arm(
			args, "baseline-serialized", args.master_server, 1,
			args.master_extra_server_arg, serialized=True)
		config_absent, _ = run_arm(
			args, "branch-serialized-absent", args.branch_server, 1,
			args.branch_extra_server_arg, serialized=True)
	serial_null, _ = run_arm(
		args, "baseline-serialized-null", args.master_server, 1,
		args.master_extra_server_arg, serialized=True)
	config_off, _ = run_arm(
		args, "branch-serialized-explicit-off", args.branch_server, 1,
		[*args.branch_extra_server_arg, "--cache-optimizer", "off"], serialized=True)
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
	if not expected.issubset(reasons):
		raise SystemExit("ZC0 light gate missed the active-unfitted negative controls: "
			f"expected={sorted(expected)} actual={sorted(reasons)}")
	print("ZC0_LIGHT_GATE PASS parity=null serialized=null active_unfitted=rejected "
		"optimizer_default_output=explicit_off")


if __name__ == "__main__":
	main()
