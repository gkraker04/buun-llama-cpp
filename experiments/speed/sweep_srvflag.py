import subprocess
import json
import time
import csv
import sys

BAT = r"G:\hermes\buun-llama-cpp\experiments\speed\test_start_dflash.bat"
OUTPUT = r"G:\hermes\buun-llama-cpp\experiments\speed\results\srvflag_sweep.csv"
API_KEY = "dummythicc"
BASE_URL = "http://localhost:8081"
PASSES = 3

# Top configs from per-request sweep (stock Qwen3.6) - server flag test
CONFIGS = [
    (6, 8),   # 29.0 tok/s per-request
    (1, 11),  # 28.7 tok/s
    (5, 8),   # 28.0 tok/s
    (6, 14),  # 27.9 tok/s
    (1, 7),   # 27.8 tok/s
    (8, 16),  # 27.6 tok/s
    (4, 7),   # 27.6 tok/s
    (9, 11),  # 27.5 tok/s
    (13, 15), # 27.4 tok/s
    (5, 12),  # 27.4 tok/s
]

def kill_server():
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(3)

def wait_for_server(timeout=120):
    for i in range(timeout):
        time.sleep(1)
        try:
            r = subprocess.run(
                ["curl.exe", "-s", "-o", "nul", "-w", "%{http_code}", f"{BASE_URL}/health"],
                capture_output=True, text=True, timeout=5
            )
            if r.stdout.strip() == "200":
                return True
        except:
            pass
    return False

def benchmark():
    payload = json.dumps({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write about processors."}
        ],
        "max_tokens": 64,
        "temperature": 0.6,
        "stream": False
    })
    
    t0 = time.time()
    r = subprocess.run(
        ["curl.exe", "-s", "-X", "POST", f"{BASE_URL}/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {API_KEY}",
         "-d", payload],
        capture_output=True, text=True, timeout=60
    )
    elapsed = time.time() - t0
    data = json.loads(r.stdout)
    ct = data.get("usage", {}).get("completion_tokens", 0)
    return ct / elapsed if elapsed > 0 else 0

def main():
    print(f"DFlash Server-Flag Sweep")
    print(f"========================")
    print(f"Configs: {len(CONFIGS)}, Passes: {PASSES}")
    print(f"Output: {OUTPUT}")
    print()
    
    results = []
    
    for idx, (n_min, n_max) in enumerate(CONFIGS, 1):
        print(f"[{idx}/{len(CONFIGS)}] n_min={n_min}, n_max={n_max}")
        sys.stdout.flush()
        
        kill_server()
        
        proc = subprocess.Popen(
            ["cmd.exe", "/c", BAT, str(n_max), str(n_min)],
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        if not wait_for_server():
            print(f"  FAILED to start server")
            sys.stdout.flush()
            kill_server()
            continue
        
        pass_results = []
        for p in range(1, PASSES + 1):
            tok_s = benchmark()
            pass_results.append(tok_s)
            print(f"  Pass {p}: {tok_s:.1f} tok/s")
            sys.stdout.flush()
        
        avg = sum(pass_results) / len(pass_results)
        results.append((n_min, n_max, *pass_results, avg))
        print(f"  AVG: {avg:.1f} tok/s")
        sys.stdout.flush()
        
        # Write to CSV after each config
        with open(OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n_min", "n_max", "pass1", "pass2", "pass3", "avg"])
            for row in results:
                writer.writerow(row)
        
        print(f"  Written to {OUTPUT}")
        sys.stdout.flush()
        
        kill_server()
        print()
    
    print()
    print("FINAL RESULTS (sorted by speed):")
    print("=" * 50)
    with open(OUTPUT, "r") as f:
        print(f.read())
    
    kill_server()
    print("Done.")

if __name__ == "__main__":
    main()
