#!/usr/bin/env python3
# L2 triple-run gate harness (PROPOSAL §10.1 L2: deterministic replay equivalence).
#
# Every case runs as a TRIPLE under the pinned schedule (seed+greedy, -np 1, fixed
# batch/ubatch, spec decode OFF, VBR freeze mode, no shift/defrag, same build):
#   arm coldA  : fresh server, full workload, no reuse        (reference)
#   arm coldB  : identical fresh server, identical workload   (null arm)
#   arm restored: fresh server, workload with cache_prompt reuse so edits engage
#                 checkpoint restore, then identical continuations
# Null arm must be bit-equal to coldA (content + per-token top-logprob record +
# schedule trace) or the case is INFRA-INVALID, not red. Then coldA-vs-restored is
# the L2 verdict. Triage per spec: diff traces first — divergent trace = harness
# bug; identical trace + divergent logits = state bug.
#
# Schedule-trace sources (existing instrumentation only):
#   - VBR_TRACE per-child schedule files (WS-0 recorder, flush-on-write)
#   - normalized decode-batch lines extracted from the server log at -lv 4
# The harness owns launch guards: free-VRAM assertion, per-arm ports, health
# hard-assert, and end-state VRAM reporting.
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
	def __init__(self, args, port, tag, workdir):
		self.args, self.port, self.tag = args, port, tag
		self.base = f"http://127.0.0.1:{port}"
		self.log = os.path.join(workdir, f"l2_{tag}.log")
		self.trace_prefix = os.path.join(workdir, f"l2_{tag}.vbrtrace")
		self.proc = None

	def start(self):
		used = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
		              ).stdout.split()[0])
		if used > self.args.vram_guard_mib:
			raise RuntimeError(f"VRAM busy: {used} MiB > guard {self.args.vram_guard_mib}")
		sh(f"fuser -k {self.port}/tcp 2>/dev/null; sleep 1")
		env = os.environ.copy()
		env.update({"VBR_FREEZE": "1", "VBR_BUDGET_MIB": str(self.args.vbr_budget_mib),
		            "VBR_TRACE": self.trace_prefix})
		cmd = (f"{self.args.server_bin} -m {self.args.model} -ngl 99 "
		       f"-c {self.args.ctx} -b {self.args.batch} -ub {self.args.batch} -np 1 "
		       f"-ctk vbr -fa on -lv 4 --slots --slot-save-path {self.args.slot_save} "
		       f"--ctx-checkpoints 8 --checkpoint-min-step 0 "
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

# ---------------- workload ----------------

def build_cases(cycles):
	# deterministic corpus, no randomness (schedule must be pinned)
	base = " ".join(
		f"Ledger row {i}: account {i%13} moved {i*7} units in period {i%5}."
		for i in range(120))
	cases = []
	for c in range(cycles):
		prime = base + f" Session {c} marker alpha."
		edit = base[: int(len(base) * 0.7)] + f" Session {c} divergent beta tail."
		cases.append((f"case{c}", prime, edit))
	return cases

def run_workload(srv, reuse):
	rows = []
	for name, prime, edit in build_cases(srv.args.cycles):
		for tag, prompt in (("prime", prime), ("edit", edit)):
			r = http(srv.base, "/completion",
				{"prompt": prompt, "n_predict": srv.args.n_predict,
				 "temperature": 0.0, "seed": 7, "n_probs": 1,
				 "cache_prompt": bool(reuse)})
			top = []
			for tk in (r.get("completion_probabilities") or []):
				pr = (tk.get("top_probs") or tk.get("probs") or [])
				if pr:
					top.append((pr[0].get("id", pr[0].get("tok_str", "")),
					            round(float(pr[0].get("logprob", pr[0].get("prob", 0.0))), 9)))
			rows.append({
				"case": name, "tag": tag,
				"content_sha": hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:16],
				"logprob_sha": hashlib.sha256(json.dumps(top, sort_keys=True).encode()).hexdigest()[:16],
				"prompt_n": (r.get("timings") or {}).get("prompt_n"),
			})
	return rows

# ---------------- traces ----------------

DECODE_RE = re.compile(r"cached n_tokens = (\d+), memory_seq_rm \[(\d+), end\)")

def schedule_trace(srv):
	# normalized per-decision trace: decode-batch summary lines from -lv 4 log
	# (positions/reuse) + the WS-0 VBR trace files (tier waves, boundary counts).
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
		if "restored context checkpoint" in line or "cache_status" in line and "restored" in line:
			decisions.append("ckpt_restore")
	vbr = {}
	for suffix in (".base", ".swa", ""):
		p = srv.trace_prefix + suffix
		if os.path.exists(p):
			vbr[suffix or ".single"] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
	return {"decisions": decisions, "vbr_trace_sha": vbr}

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
	ap.add_argument("--workdir", default="/root/l2work")
	ap.add_argument("--slot-save", default="/root/slotsave")
	ap.add_argument("--ctx", type=int, default=4096)
	ap.add_argument("--batch", type=int, default=512)
	ap.add_argument("--cycles", type=int, default=3)
	ap.add_argument("--n-predict", type=int, default=24)
	ap.add_argument("--vbr-budget-mib", type=int, default=256)
	ap.add_argument("--vram-guard-mib", type=int, default=2000)
	ap.add_argument("--base-port", type=int, default=8220)
	args = ap.parse_args()
	os.makedirs(args.workdir, exist_ok=True)

	arms = {}
	for i, (tag, reuse) in enumerate((("coldA", False), ("coldB", False),
	                                   ("restored", True))):
		srv = Server(args, args.base_port + i, tag, args.workdir)
		srv.start()
		try:
			arms[tag] = {"rows": run_workload(srv, reuse),
			             "trace": schedule_trace(srv)}
		finally:
			srv.stop()
		with open(os.path.join(args.workdir, f"l2_{tag}.json"), "w") as f:
			json.dump(arms[tag], f, indent=1, sort_keys=True)

	# ---- null arm: coldA vs coldB must be bit-equal, else INFRA-INVALID ----
	null_rows = diff_rows(arms["coldA"]["rows"], arms["coldB"]["rows"])
	null_trace = arms["coldA"]["trace"] != arms["coldB"]["trace"]
	if null_rows or null_trace:
		print("L2_VERDICT=INFRA_INVALID null_arm_mismatch "
		      f"rows_diff={null_rows} trace_diff={null_trace}")
		return 2

	# ---- non-vacuity: the restored arm must actually restore ----
	restores = sum(1 for d in arms["restored"]["trace"]["decisions"]
	               if d.startswith("ckpt_restore"))
	if restores == 0:
		print("L2_VERDICT=INFRA_INVALID vacuous: restored arm engaged zero "
		      "checkpoint restores")
		return 2

	# ---- cold vs restored: traces first (triage rule) ----
	trace_diff = arms["coldA"]["trace"]["decisions"] != arms["restored"]["trace"]["decisions"]
	rows_diff = diff_rows(
		[{k: r[k] for k in ("case", "tag", "content_sha", "logprob_sha")}
		 for r in arms["coldA"]["rows"]],
		[{k: r[k] for k in ("case", "tag", "content_sha", "logprob_sha")}
		 for r in arms["restored"]["rows"]])
	if rows_diff is None:
		print("L2_VERDICT=PASS cold==restored (content+logprobs) with "
		      f"{len(arms['coldA']['rows'])} rows; restores_engaged={restores}; "
		      f"restored trace decisions={len(arms['restored']['trace']['decisions'])}")
		return 0
	kind = "HARNESS_BUG(divergent trace)" if trace_diff else "STATE_BUG(identical trace, divergent logits)"
	print(f"L2_VERDICT=FAIL kind={kind} first_diff={rows_diff}")
	return 1

if __name__ == "__main__":
	sys.exit(main())
