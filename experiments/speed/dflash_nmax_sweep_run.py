import subprocess, json, time, os, csv, sys

SERVER = r"G:\hermes\buun-llama-cpp\build\bin\llama-server.exe"
TARGET = r"G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF\Ornstein3.6-27B-MTP-NSC-ACE-SABER-Q4_K_M-MTP.gguf"
DRAFT = r"G:\models\gkraker04\dflash-drafter-3.6\dflash-draft-3.6-Q4_K_M.gguf"
MMPROJ = r"G:\models\GestaltLabs\Ornstein3.6-27B-MTP-NSC-ACE-SABER-GGUF\mmproj-Ornstein3.6-27B-MTP-NSC-ACE-SABER-F16.gguf"
LOG_DIR = r"G:\hermes\buun-llama-cpp\experiments\speed\logs"
RESULTS_DIR = r"G:\hermes\buun-llama-cpp\experiments\speed\results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "dflash_nmax_sweep_p75.csv")
LOG_PATH = os.path.join(LOG_DIR, "dflash_nmax_sweep_p75.log")
STATE_PATH = os.path.join(LOG_DIR, "sweep_state.txt")

def log(msg):
    with open(STATE_PATH, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)

# Kill existing
subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
time.sleep(2)

# Start server
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

# Load existing results
done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                done.add((int(row[0]), int(row[1])))
    log(f"Existing: {len(done)} configs")

all_configs = [(n_min, n_max) for n_min in range(1, 17) for n_max in range(n_min, 17)]
remaining = [(nm, nx) for nm, nx in all_configs if (nm, nx) not in done]
log(f"Total: {len(all_configs)}, Remaining: {len(remaining)}")

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
        ct = data.get("usage", {}).get("completion_tokens", 0)
        tok_s = ct / elapsed if elapsed > 0 else 0
        
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["n_min", "n_max", "tokens", "tok_s"])
                need_header = False
            w.writerow([n_min, n_max, ct, round(tok_s, 2)])
        
        done_count = len(done) + idx + 1
        eta = (time.time() - start_time) / (idx + 1) * (len(remaining) - idx - 1)
        log(f"[{done_count}/{len(all_configs)}] n_min={n_min}, n_max={n_max}: {ct}tok @ {tok_s:.1f}tok/s (ETA {eta/60:.0f}m)")
    except Exception as e:
        log(f"[{len(done)+idx+1}/{len(all_configs)}] n_min={n_min}, n_max={n_max}: ERROR {str(e)[:60]}")

total = time.time() - start_time
log(f"\nSWEEP COMPLETE in {total/60:.1f}m")

# Top 15
with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    rows = [(int(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in reader if float(r[3]) > 0]
sorted_r = sorted(rows, key=lambda x: x[3], reverse=True)
log(f"\n=== TOP 15 ===")
for n_min, n_max, ct, ts in sorted_r[:15]:
    log(f"  n_min={n_min}, n_max={n_max}: {ts:.1f} tok/s ({ct} tok)")

subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
log("Server killed")
