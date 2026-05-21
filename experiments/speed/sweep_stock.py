import subprocess, json, time, os, csv, sys

SERVER = r"G:\\hermes\\buun-llama-cpp\\build\\bin\\llama-server.exe"
TARGET = r"G:\\models\\unsloth\\Qwen3.6-27B-GGUF\\Qwen3.6-27B-Q4_K_M.gguf"
DRAFT = r"G:\\models\\gkraker04\\dflash-drafter-3.6\\dflash-draft-3.6-Q4_K_M.gguf"
MMPROJ = r"G:\\models\\unsloth\\Qwen3.6-27B-GGUF\\mmproj-BF16.gguf"
LOG_DIR = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\logs"
RESULTS_DIR = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "dflash_stock_nmax_sweep_p75.csv")
LOG_PATH = os.path.join(LOG_DIR, "dflash_stock_nmax_sweep_p75.log")
STATE_PATH = os.path.join(LOG_DIR, "sweep_stock_state.txt")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    with open(STATE_PATH, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
time.sleep(2)

proc = subprocess.Popen(
    [SERVER, "--threads", "12", "--prio", "3", "--n-predict", "32768",
     "--model", TARGET, "--spec-draft-model", DRAFT, "--spec-type", "dflash",
     "-ngl", "99", "-ngld", "99", "-cd", "256", "-b", "256", "-ub", "64",
     "-fa", "on", "-np", "1", "--fit-target", "5120", "--no-mmap",
     "--cache-type-k", "turbo2_tcq", "--cache-type-v", "turbo2_tcq",
     "--cache-type-k-draft", "turbo2_tcq", "--cache-type-v-draft", "turbo2_tcq",
     "--mmproj", MMPROJ, "--no-mmproj-offload",
     "--host", "0.0.0.0", "--port", "8081", "--api-key", "dummythicc",
     "--offline", "--jinja", "--reasoning", "off", "--log-file", LOG_PATH],
    creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
log(f"Server PID: {proc.pid}")

for i in range(120):
    time.sleep(1)
    try:
        r = subprocess.run(["curl.exe","-s","-o","nul","-w","%{http_code}","http://localhost:8081/health"],
                           capture_output=True, text=True, timeout=5)
        if r.stdout.strip() == "200":
            log(f"Ready after {i+1}s")
            break
    except:
        pass
else:
    log("SERVER FAILED")
    sys.exit(1)

all_configs = [(n_min, n_max) for n_min in range(1, 17) for n_max in range(n_min, 17)]
log(f"Configs: {len(all_configs)}")

done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                done.add((int(row[0]), int(row[1])))
    log(f"Existing: {len(done)}")

remaining = [(nm, nx) for nm, nx in all_configs if (nm, nx) not in done]
log(f"Remaining: {len(remaining)}")
if not remaining:
    log("All done!")
    sys.exit(0)

need_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
start_time = time.time()

for idx, (n_min, n_max) in enumerate(remaining):
    t0 = time.time()
    payload = json.dumps({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short essay about the history of computer processors."}
        ],
        "max_tokens": 256, "temperature": 0.6, "stream": False,
        "speculative.n_min": n_min, "speculative.n_max": n_max, "speculative.p_min": 0.75
    })
    
    r = subprocess.run(
        ["curl.exe", "-s", "-X", "POST", "http://localhost:8081/v1/chat/completions",
         "-H", "Content-Type: application/json", "-H", "Authorization: Bearer dummythicc", "-d", payload],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0
    
    try:
        data = json.loads(r.stdout)
        if "error" in data:
            log(f"[{len(done)+idx+1:3d}/{len(all_configs)}] n_min={n_min:2d}, n_max={n_max:2d}: ERROR - {str(data['error'])[:60]}")
            continue
        
        ct = data.get("usage", {}).get("completion_tokens", 0)
        tok_s = ct / elapsed if elapsed > 0 else 0
        
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["n_min", "n_max", "tokens", "tok_s"])
                need_header = False
            w.writerow([n_min, n_max, ct, round(tok_s, 2)])
        
        eta = (time.time() - start_time) / (idx + 1) * (len(remaining) - idx - 1)
        log(f"[{len(done)+idx+1:3d}/{len(all_configs)}] n_min={n_min:2d}, n_max={n_max:2d}: {ct:3d}tok @ {tok_s:5.1f}tok/s (ETA {eta/60:.0f}m)")
    except Exception as e:
        log(f"[{len(done)+idx+1:3d}/{len(all_configs)}] n_min={n_min:2d}, n_max={n_max:2d}: ERROR {str(e)[:60]}")

total = time.time() - start_time
log(f"\nDONE in {total/60:.1f}m")

with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    rows = [(int(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in reader if float(r[3]) > 0]
sorted_r = sorted(rows, key=lambda x: x[3], reverse=True)

log("=== TOP 10 ===")
for n_min, n_max, ct, ts in sorted_r[:10]:
    log(f"  n_min={n_min:2d}, n_max={n_max:2d}: {ts:5.1f} tok/s")

log(f"\nBEST: n_min={sorted_r[0][0]}, n_max={sorted_r[0][1]}: {sorted_r[0][3]:.1f} tok/s")
avg = sum(r[3] for r in rows) / len(rows)
log(f"AVG: {avg:.1f} tok/s over {len(rows)} configs")

subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
log("Server killed")
