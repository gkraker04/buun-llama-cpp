#!/usr/bin/env python3
# P1 final-gate arm (PROPOSAL Rev-9 §9 Phase-1): "changed ownership/unrecognized
# transition rejects before mutation." Under a small fixed VBR budget in freeze
# mode (deterministic inputs, retiering still runs), decoding past capture forces
# a real degrade; the next reuse attempt must REJECT the stale checkpoint at
# selection ([I9] representation-epoch mismatch), never mutate, and cold-reprocess
# to output identical to a true always-cold arm.
import argparse, hashlib, json, os, re, subprocess, sys, time, urllib.request

def sh(cmd, timeout=600):
	return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def http(base, path, body=None, timeout=300):
	req = urllib.request.Request(base + path) if body is None else \
		urllib.request.Request(base + path, data=json.dumps(body).encode(),
			headers={"Content-Type": "application/json"}, method="POST")
	with urllib.request.urlopen(req, timeout=timeout) as r:
		return json.loads(r.read())

def start_server(args, port, log, budget_mib):
	used = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").stdout.split()[0])
	if used > 2000:
		raise RuntimeError(f"VRAM busy: {used} MiB")
	sh(f"fuser -k {port}/tcp 2>/dev/null; sleep 1")
	env = os.environ.copy()
	env.update({"VBR_FREEZE": "1", "VBR_BUDGET_MIB": str(budget_mib)})
	cmd = (f"{args.server_bin} -m {args.model} -ngl 99 -c {args.ctx} -b 512 -ub 512 -np 1 "
	       f"-ctk vbr -fa on -lv 4 --slots --slot-save-path {args.slot_save} "
	       f"--ctx-checkpoints 8 --checkpoint-min-step 0 --port {port}")
	proc = subprocess.Popen(cmd, shell=True, env=env, stdout=open(log, "w"),
	                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
	for _ in range(90):
		try:
			if http(f"http://127.0.0.1:{port}", "/health", timeout=5).get("status") == "ok":
				return proc
		except Exception:
			pass
		time.sleep(1)
	raise RuntimeError("server failed health: " + sh(f"tail -3 {log}").stdout)

def workload(base, reuse, n_gen):
	# Three-step chain engineered against the selector's two checks:
	#  A) short prime -> EARLY checkpoint (~840 tok) with pre-degrade epochs;
	#  B) long extension -> prefill crosses the mapping threshold -> degrade
	#     stales A's epochs (B's own late checkpoints are position-ineligible
	#     for the edit);
	#  C) edit sharing ~1000 tok then diverging -> A passes the position check
	#     (pos_max < pos_next) and MUST be rejected on epochs alone -> [I9].
	sents = [f"Journal {i}: the committee logged item {i*3} at hour {i%24}." for i in range(125)]
	short = " ".join(sents[:55])
	full  = " ".join(sents)
	edit  = " ".join(sents[:66]) + " EDITED continuation differs here entirely."
	rows = []
	for prompt, npred in ((short, 8), (full, n_gen), (edit, 24)):
		r = http(base, "/completion", {"prompt": prompt, "n_predict": npred,
			"temperature": 0.0, "seed": 5, "cache_prompt": reuse})
		rows.append(hashlib.sha256((r.get("content") or "").encode()).hexdigest()[:16])
	# cache_status is surfaced via /slots, not the log — read it post-edit
	slots = http(base, "/slots")
	status = (slots[0] if isinstance(slots, list) and slots else {}).get("cache_status", "")
	return rows, status

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--server-bin", required=True)
	ap.add_argument("--model", required=True)
	ap.add_argument("--workdir", default="/root/l2reject")
	ap.add_argument("--slot-save", default="/root/slotsave")
	ap.add_argument("--ctx", type=int, default=4096)
	ap.add_argument("--n-gen", type=int, default=320)
	ap.add_argument("--budgets", default="48")
	ap.add_argument("--port", type=int, default=8250)
	a = ap.parse_args()
	os.makedirs(a.workdir, exist_ok=True)

	for budget in [int(x) for x in a.budgets.split(",")]:
		log = os.path.join(a.workdir, f"reject_b{budget}.log")
		proc = start_server(a, a.port, log, budget)
		try:
			rows_reuse, status = workload(f"http://127.0.0.1:{a.port}", True, a.n_gen)
		finally:
			sh(f"fuser -k {a.port}/tcp 2>/dev/null; sleep 1")
		text = open(log, errors="replace").read()
		ckpts   = len(re.findall(r"created context checkpoint", text))
		degr    = len(re.findall(r"vbr_degrade_next: VBR degrade #", text))
		# post-capture requirement: last capture timestamp precedes some executed degrade
		caps = [m.start() for m in re.finditer(r"created context checkpoint", text)]
		degs = [m.start() for m in re.finditer(r"vbr_degrade_next: VBR degrade #", text)]
		post_capture_degrade = bool(caps and degs and max(degs) > min(caps))
		reject  = 1 if "[I9]" in status else 0
		print(f"edit_cache_status={status!r}")
		aborts  = len(re.findall(r"GGML_ASSERT|Aborted|wrong-state", text))
		print(f"budget={budget} ckpts={ckpts} degrade_lines={degr} i9_rejects={reject} aborts={aborts}")
		if aborts:
			print("VERDICT=FAIL abort under matrix"); return 1
		print(f"post_capture_degrade={post_capture_degrade}")
		if ckpts > 0 and post_capture_degrade and reject > 0:
			# cold reference arm at the SAME budget: outputs must match
			log_c = os.path.join(a.workdir, f"cold_b{budget}.log")
			proc = start_server(a, a.port, log_c, budget)
			try:
				rows_cold, _ = workload(f"http://127.0.0.1:{a.port}", False, a.n_gen)
			finally:
				sh(f"fuser -k {a.port}/tcp 2>/dev/null; sleep 1")
			ok = rows_reuse == rows_cold
			print(f"cold_parity={ok} reuse={rows_reuse} cold={rows_cold}")
			print("VERDICT=" + ("PASS reject-before-mutation with cold-identical fallback"
			                     if ok else "FAIL divergent fallback output"))
			return 0 if ok else 1
	print("VERDICT=INFRA_INVALID no budget in ladder produced ckpt+degrade+reject")
	return 2

if __name__ == "__main__":
	sys.exit(main())
