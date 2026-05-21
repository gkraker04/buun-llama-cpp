import subprocess
import json
import time
import csv
import sys
import os

BAT = r"G:\hermes\buun-llama-cpp\experiments\speed\test_start_dflash.bat"
OUTPUT = r"G:\hermes\buun-llama-cpp\experiments\speed\results\nmin_nmax_full_sweep.csv"
LOG_PATH = r"G:\hermes\buun-llama-cpp\experiments\speed\results\nmin_nmax_full_run.txt"
API_KEY = "dummythicc"
BASE_URL = "http://localhost:8081"
P_MIN = 0.75

# n_min 0-16, n_max 0-16
# Skip n_max < n_min (draft window can't be smaller than min)
CONFIGS = [(n_min, n_max) for n_min in range(0, 17) for n_max in range(max(n_min, 1), 17)]
PASSES = 3
GEN_TOKENS = 256


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee(LOG_PATH)
print("DFlash n_min/n_max Full Sweep (0-16, per-request params)")
print("=========================================================")
print(f"Configs: {len(CONFIGS)}  Passes: {PASSES}  Single server start (no restarts)")
print()


def kill_server():
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(3)


def wait_for_server(timeout=120):
    for i in range(timeout):
        time.sleep(1)
        try:
            r = subprocess.run(
                ["curl.exe", "-s", "-o", "nul", "-w", "%{http_code}", BASE_URL + "/health"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip() == "200":
                return True
        except:
            pass
    return False


def bench_config(n_min, n_max):
    results = []
    for p in range(PASSES):
        payload = json.dumps({
            "model": "qwen",
            "messages": [
                {"role": "user", "content": "Write a detailed paragraph about computer processors."}
            ],
            "max_tokens": GEN_TOKENS,
            "temperature": 0.6,
            "stream": False,
            "n": 1,
            "speculative.n_min": n_min,
            "speculative.n_max": n_max,
            "speculative.p_min": P_MIN,
        })

        r = subprocess.run(
            ["curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
             "-H", "Content-Type: application/json",
             "-H", "Authorization: Bearer " + API_KEY,
             "-d", payload],
            capture_output=True, text=True, timeout=60,
        )

        try:
            data = json.loads(r.stdout)
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            results.append({
                "completion_tokens": usage.get("completion_tokens", 0),
                "decode_tok_s": timings.get("predicted_per_second", 0),
                "draft_n": timings.get("draft_n", 0),
                "draft_n_accepted": timings.get("draft_n_accepted", 0),
            })
        except:
            results.append(None)
    return results


def main():
    kill_server()
    print("Starting server...")
    subprocess.Popen(
        ["cmd.exe", "/c", BAT],
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    if not wait_for_server():
        print("Server failed to start!")
        return

    print("Server ready.")
    print()

    # Warmup
    print("Warming up...")
    for i in range(3):
        payload = json.dumps({
            "model": "qwen",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32, "temperature": 0.6, "stream": False,
        })
        subprocess.run(
            ["curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
             "-H", "Content-Type: application/json",
             "-H", "Authorization: Bearer " + API_KEY, "-d", payload],
            capture_output=True, text=True, timeout=30,
        )
    print("Done.\n")

    all_results = []
    total = len(CONFIGS)

    for idx, (n_min, n_max) in enumerate(CONFIGS, 1):
        result = bench_config(n_min, n_max)

        # Filter valid results
        valid = [r for r in result if r and r["decode_tok_s"] > 0]
        if valid:
            avg_decode = sum(r["decode_tok_s"] for r in valid) / len(valid)
            avg_tokens = sum(r["completion_tokens"] for r in valid) / len(valid)
            avg_dn = sum(r["draft_n"] for r in valid) / len(valid)
            avg_da = sum(r["draft_n_accepted"] for r in valid) / len(valid)
            dr = (avg_da / avg_dn * 100) if avg_dn > 0 else 0
            row = (n_min, n_max, avg_decode, avg_tokens, avg_dn, avg_da, dr)
            all_results.append(row)
            print(f"[{idx:3d}/{total}] n_min={n_min:2d} n_max={n_max:2d} -> {avg_decode:5.1f} tok/s "
                  f"draft={avg_da:.0f}/{avg_dn:.0f} ({dr:.0f}%)")
        else:
            print(f"[{idx:3d}/{total}] n_min={n_min:2d} n_max={n_max:2d} -> FAILED")

        sys.stdout.flush()

    # Save CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_min", "n_max", "decode_tok_s", "avg_tokens",
                         "draft_n_avg", "draft_acc_avg", "draft_rate_pct"])
        for row in sorted(all_results, key=lambda x: -x[2]):
            writer.writerow(row)

    print()
    print("TOP 20 RESULTS:")
    print("=" * 70)
    print(f"{'n_min':>6} {'n_max':>6} {'Decode':>8} {'Tokens':>7} {'Draft Acc':>10}")
    print("-" * 70)
    for row in sorted(all_results, key=lambda x: -x[2])[:20]:
        print(f"{row[0]:>6} {row[1]:>6} {row[2]:>8.1f} {int(row[3]):>7} {row[5]:.0f}/{row[4]:.0f} ({row[6]:.0f}%)")

    print()
    print(f"CSV: {OUTPUT}")
    print(f"Log: {LOG_PATH}")

    kill_server()
    print("Done.")


if __name__ == "__main__":
    main()
