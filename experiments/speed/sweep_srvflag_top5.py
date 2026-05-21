import subprocess, json, time, csv

bat = r"G:\hermes\buun-llama-cpp\experiments\speed\test_start_dflash.bat"

# Top 5 configs from per-request sweep (stock Qwen3.6)
configs = [
    (6, 8),   # 29.0 tok/s
    (1, 11),  # 28.7 tok/s
    (5, 8),   # 28.0 tok/s
    (6, 14),  # 27.9 tok/s
    (1, 7),   # 27.8 tok/s
]

passes = 3
results = []
output_path = r"G:\hermes\buun-llama-cpp\experiments\speed\results\srvflag_top5_sweep.csv"

print(f"Starting server-flag sweep: {len(configs)} configs x {passes} passes")

for i, (n_min, n_max) in enumerate(configs):
    print(f"[{i+1}/{len(configs)}] n_min={n_min}, n_max={n_max}")
    
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(3)
    
    proc = subprocess.Popen(
        ["cmd.exe", "/c", bat, str(n_max), str(n_min)],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    ready = False
    for s in range(120):
        time.sleep(1)
        r = subprocess.run(
            ["curl.exe", "-s", "-o", "nul", "-w", "%{http_code}", "http://localhost:8081/health"],
            capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip() == "200":
            ready = True
            print(f"  Ready after {s+1}s")
            break
    
    if not ready:
        print("  FAILED to start")
        continue
    
    tok_counts = []
    for p in range(passes):
        payload = json.dumps({"model":"qwen","messages":[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"Write about processors."}
        ],"max_tokens":64,"temperature":0.6,"stream":False})
        
        t0 = time.time()
        r = subprocess.run(
            ["curl.exe", "-s", "-X", "POST", "http://localhost:8081/v1/chat/completions",
             "-H", "Content-Type: application/json", "-H", "Authorization: Bearer dummythicc", "-d", payload],
            capture_output=True, text=True, timeout=60
        )
        elapsed = time.time() - t0
        try:
            data = json.loads(r.stdout)
            ct = data.get("usage", {}).get("completion_tokens", 0)
            tok_s = ct / elapsed if elapsed > 0 else 0
            tok_counts.append(tok_s)
            print(f"  Pass {p+1}: {ct}tok @ {tok_s:.1f}tok/s")
        except Exception as e:
            print(f"  Pass {p+1}: FAILED - {e}")
    
    if tok_counts:
        avg = sum(tok_counts) / len(tok_counts)
        results.append((n_min, n_max, avg))
        print(f"  AVG: {avg:.1f} tok/s")
    
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(2)
    print()

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_min", "n_max", "tok_s"])
    for row in results:
        writer.writerow(row)

print(f"Saved to {output_path}")
for row in sorted(results, key=lambda x: -x[2]):
    print(f"  n_min={row[0]}, n_max={row[1]} -> {row[2]:.1f} tok/s")

subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
print("Done.")
