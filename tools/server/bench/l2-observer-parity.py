#!/usr/bin/env python3
# D-S GATE L2: observer parity (the B-a invariant, end-to-end). Certifies that the
# --cache-debug shadow observer (D-S1..D-S7: accounting, budget, lease, retention,
# yield planner, schema-4 record) changes NOTHING that ships. Three arms under one
# pinned deterministic schedule (greedy, seed, -np 1, fixed batch=ubatch, spec off,
# VBR freeze, ctx-checkpoints so the edit engages RESTORE — not cold-only):
#   OFF-A  : --cache-debug absent            (reference)
#   OFF-B  : --cache-debug absent, identical (null arm — proves harness determinism)
#   ON     : --cache-debug present, identical (observer fully exercised)
# Gate: OFF-A == OFF-B (else INFRA_INVALID); ON emits CACHE_PLAN/BUDGET/YIELD while
# OFF emits none and both arms actually restore (else INFRA_INVALID); then
# OFF-A == ON over canonical content + per-token top-logprob record + normalized
# decode/VBR schedule trace == PASS. Triage per L2 rule: divergent trace = scheduling
# perturbation (observer reordered work); identical trace + divergent logits = state
# perturbation (observer touched shipped state). Neither may happen under B-a.
import argparse, hashlib, json, os, re, subprocess, sys, time, urllib.request

def sh(cmd, timeout=600):
	return subprocess.run(cmd, shell=True, capture_output=True, text=True,
	                      timeout=timeout)

def http(base, path, body=None, timeout=300):
	if body is None:
		req = urllib.request.Request(base + path)
	else:
		req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
			headers={"Content-Type": "application/json"}, method="POST")
	with urllib.request.urlopen(req, timeout=timeout) as r:
		return json.loads(r.read())

class Server:
	def __init__(self, args, port, tag, cache_debug, workdir):
		self.args, self.port, self.tag = args, port, tag
		self.cache_debug = cache_debug
		self.base = f"http://127.0.0.1:{port}"
		self.log = os.path.join(workdir, f"parity_{tag}.log")
		self.trace_prefix = os.path.join(workdir, f"parity_{tag}.vbrtrace")
		self.proc = None

	def start(self):
		sh(f"fuser -k {self.port}/tcp 2>/dev/null; sleep 1")
		# Wait for the previous arm's 27B footprint to drain before asserting a free GPU.
		used = self.args.vram_guard_mib + 1
		for _ in range(15):
			used = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
			              ).stdout.split()[0])
			if used <= self.args.vram_guard_mib:
				break
			time.sleep(2)
		else:
			raise RuntimeError(f"VRAM busy: {used} MiB > guard {self.args.vram_guard_mib}")
		sh(f"rm -f {self.trace_prefix}* 2>/dev/null")  # no stale VBR trace from a prior run
		env = os.environ.copy()
		env.update({"VBR_FREEZE": "1", "VBR_BUDGET_MIB": str(self.args.vbr_budget_mib),
		            "VBR_TRACE": self.trace_prefix})
		cmd = (f"{self.args.server_bin} -m {self.args.model} -ngl 99 "
		       f"-c {self.args.ctx} -b {self.args.batch} -ub {self.args.batch} -np 1 "
		       f"-ctk vbr -fa on -lv 4 --slots --slot-save-path {self.args.slot_save} "
		       f"--ctx-checkpoints 8 --checkpoint-min-step 0 "
		       f"{'--cache-debug ' if self.cache_debug else ''}"
		       f"--port {self.port}")
		self.proc = subprocess.Popen(cmd, shell=True, env=env,
			stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
			stdin=subprocess.DEVNULL)
		for _ in range(90):
			try:
				if http(self.base, "/health", timeout=5).get("status") == "ok":
					return
			except Exception:
				pass
			time.sleep(1)
		raise RuntimeError(f"server {self.tag} failed health; tail: "
		                   + sh(f"tail -3 {self.log}").stdout)

	def stop(self):
		sh(f"fuser -k {self.port}/tcp 2>/dev/null; sleep 1")

# ---------------- workload (reuse-engaging: prime primes, edit restores) --------

def build_cases(cycles):
	base = " ".join(
		f"Ledger row {i}: account {i%13} moved {i*7} units in period {i%5}."
		for i in range(120))
	cases = []
	for c in range(cycles):
		prime = base + f" Session {c} marker alpha."
		edit = base[: int(len(base) * 0.7)] + f" Session {c} divergent beta tail."
		cases.append((f"case{c}", prime, edit))
	return cases

def run_workload(srv):
	rows = []
	for name, prime, edit in build_cases(srv.args.cycles):
		for tag, prompt in (("prime", prime), ("edit", edit)):
			r = http(srv.base, "/completion",
				{"prompt": prompt, "n_predict": srv.args.n_predict,
				 "temperature": 0.0, "seed": 7, "n_probs": 1,
				 "cache_prompt": True})
			top = []
			for tk in (r.get("completion_probabilities") or []):
				pr = (tk.get("top_probs") or tk.get("probs") or [])
				if pr:
					# repr() of the raw double keeps every bit — the run is bit-deterministic
					# under greedy+seed+VBR-freeze (the OFF-A==OFF-B null arm proves it), so no
					# rounding is warranted for a byte-identity gate.
					top.append((pr[0].get("id", pr[0].get("tok_str", "")),
					            repr(float(pr[0].get("logprob", pr[0].get("prob", 0.0))))))
			rows.append({
				"case": name, "tag": tag,
				"content_sha": hashlib.sha256((r.get("content") or "").encode()).hexdigest(),
				"logprob_sha": hashlib.sha256(json.dumps(top, sort_keys=True).encode()).hexdigest(),
			})
	return rows

# ---------------- traces + record counts ----------------

DECODE_RE = re.compile(r"cached n_tokens = (\d+), memory_seq_rm \[(\d+), end\)")

def schedule_trace(srv):
	decisions = []
	for line in open(srv.log, errors="replace"):
		m = DECODE_RE.search(line)
		if m:
			decisions.append(f"decode cached={m.group(1)} rm_from={m.group(2)}")
		if "VBR_RETIER_PREFLIGHT" in line:
			decisions.append("preflight " + line.split("VBR_RETIER_PREFLIGHT", 1)[1].strip())
		if "created context checkpoint" in line:
			decisions.append("ckpt_create " +
				re.sub(r".*created context checkpoint", "", line).strip())
		if "restored context checkpoint" in line:
			decisions.append("ckpt_restore")
	vbr = {}
	for suffix in (".base", ".swa", ""):
		p = srv.trace_prefix + suffix
		if os.path.exists(p):
			vbr[suffix or ".single"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
	return {"decisions": decisions, "vbr_trace_sha": vbr}

def record_counts(srv):
	c = {"CACHE_PLAN": 0, "CACHE_BUDGET": 0, "CACHE_YIELD": 0}
	for line in open(srv.log, errors="replace"):
		for k in c:
			if f"{k} {{" in line:
				c[k] += 1
	return c

def slots_have_cache_plan(srv):
	# Structural channel: the finalized B0 record rides on /slots.cache_plan and is
	# "Only ever non-null under --cache-debug" (server-context.cpp). Robust to log
	# wording, and proves the silent-when-off negation on the feature's own surface.
	try:
		data = http(srv.base, "/slots", timeout=30)
	except Exception:
		return None
	slots = data.get("slots", data) if isinstance(data, dict) else data
	return any(isinstance(s, dict) and s.get("cache_plan") is not None
	           for s in (slots or []))

# ---------------- comparison ----------------

def diff_rows(a, b):
	for i, (x, y) in enumerate(zip(a, b)):
		if x != y:
			return i, x, y
	if len(a) != len(b):
		return min(len(a), len(b)), "len=%d" % len(a), "len=%d" % len(b)
	return None

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--server-bin", required=True)
	ap.add_argument("--model", required=True)
	ap.add_argument("--workdir", default="/root/paritywork")
	ap.add_argument("--slot-save", default="/root/parityslots")
	ap.add_argument("--ctx", type=int, default=4096)
	ap.add_argument("--batch", type=int, default=512)
	ap.add_argument("--cycles", type=int, default=3)
	ap.add_argument("--n-predict", type=int, default=24)
	ap.add_argument("--vbr-budget-mib", type=int, default=256)
	ap.add_argument("--vram-guard-mib", type=int, default=2000)
	ap.add_argument("--base-port", type=int, default=8240)
	args = ap.parse_args()
	os.makedirs(args.workdir, exist_ok=True)
	os.makedirs(args.slot_save, exist_ok=True)

	arms = {}
	for i, (tag, dbg) in enumerate((("offA", False), ("offB", False), ("on", True))):
		srv = Server(args, args.base_port + i, tag, dbg, args.workdir)
		srv.start()
		try:
			rows = run_workload(srv)
			slots = slots_have_cache_plan(srv)   # /slots must be read while live
		finally:
			srv.stop()
		# Server is down: the log + VBR trace files are complete and flushed, so no
		# buffered ON-only event can land after the snapshot and escape comparison.
		arms[tag] = {"rows": rows, "slots_cache_plan": slots,
		             "trace": schedule_trace(srv), "records": record_counts(srv)}
		with open(os.path.join(args.workdir, f"parity_{tag}.json"), "w") as f:
			json.dump(arms[tag], f, indent=1, sort_keys=True)

	# ---- null arm: OFF-A vs OFF-B bit-equal, else INFRA_INVALID ----
	null_rows = diff_rows(arms["offA"]["rows"], arms["offB"]["rows"])
	null_trace = arms["offA"]["trace"] != arms["offB"]["trace"]
	if null_rows or null_trace:
		print("L2_PARITY=INFRA_INVALID null_arm_mismatch "
		      f"rows_diff={null_rows} trace_diff={null_trace}")
		return 2

	# ---- observer exercised ON (1:1:1 per request), silent both OFF arms ----
	# Exactly one CACHE_PLAN/BUDGET/YIELD per request (single-GPU: one CACHE_BUDGET
	# per device == per request). "at least one" would let the observer drop all but
	# the first request and still pass. Both OFF arms must be fully silent.
	n_req = args.cycles * 2
	on_rec = arms["on"]["records"]
	if any(on_rec[k] != n_req for k in on_rec):
		print(f"L2_PARITY=INFRA_INVALID observer_not_1to1 on_records={on_rec} n_req={n_req}")
		return 2
	for off in ("offA", "offB"):
		if any(arms[off]["records"].values()):
			print(f"L2_PARITY=INFRA_INVALID observer_leaked_when_off "
			      f"{off}_records={arms[off]['records']}")
			return 2

	# ---- structural channel: /slots.cache_plan present iff observer on ----
	# --slots is mandatory, so an unavailable/malformed endpoint (None) is itself a
	# failure, not a reason to fall back to logs alone.
	on_slots = arms["on"]["slots_cache_plan"]
	off_slots = (arms["offA"]["slots_cache_plan"], arms["offB"]["slots_cache_plan"])
	if on_slots is not True or any(s is not False for s in off_slots):
		print("L2_PARITY=INFRA_INVALID slots_cache_plan_channel "
		      f"on={on_slots} off={off_slots}")
		return 2

	# ---- restore path exercised in both arms (not cold-only) ----
	restores = {t: sum(1 for d in arms[t]["trace"]["decisions"] if d == "ckpt_restore")
	            for t in ("offA", "on")}
	if restores["offA"] == 0 or restores["on"] == 0:
		print(f"L2_PARITY=INFRA_INVALID vacuous_no_restore {restores}")
		return 2

	# ---- B-a: OFF-A vs ON — traces first (triage rule) ----
	trace_diff = arms["offA"]["trace"] != arms["on"]["trace"]
	rows_diff = diff_rows(arms["offA"]["rows"], arms["on"]["rows"])
	if rows_diff is None and not trace_diff:
		print(f"L2_PARITY=PASS off==on ({len(arms['offA']['rows'])} rows, "
		      f"content+logprobs+trace); on_records={on_rec}; "
		      f"slots_cache_plan on={on_slots} off={off_slots}; "
		      f"restores off={restores['offA']} on={restores['on']}")
		return 0
	kind = ("SCHEDULING(divergent trace)" if trace_diff
	        else "STATE(identical trace, divergent output)")
	print(f"L2_PARITY=FAIL kind={kind} first_row_diff={rows_diff} trace_diff={trace_diff}")
	return 1

if __name__ == "__main__":
	sys.exit(main())
