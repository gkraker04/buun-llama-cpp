import subprocess
import json
import time
import csv
import sys
import os
import tempfile
import datetime

BAT = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\test_start_dflash.bat"
API_KEY = "dummythicc"
BASE_URL = "http://localhost:8081"
P_MIN = 0.75

LOG_DIR = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\logs"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"full_sweep_{TIMESTAMP}.txt")
CSV_FILE = os.path.join(LOG_DIR, f"full_sweep_{TIMESTAMP}.csv")

WARMUP_RUNS = 3
MEASURED_RUNS = 5
GEN_TOKENS = 256

# n_min 0-16, n_max 1-16, n_max >= n_min
CONFIGS = [(n_min, n_max) for n_min in range(0, 17) for n_max in range(max(n_min, 1), 17)]


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee(LOG_FILE)
print("=" * 72)
print("FULL SWEEP: n_min/n_max 0-16 (server restarts)")
print(f"Configs: {len(CONFIGS)} | Warmup: {WARMUP_RUNS} | Measured: {MEASURED_RUNS} p-min={P_MIN}")
print(f"Log: {LOG_FILE}")
print(f"CSV: {CSV_FILE}")
print("=" * 72)
print()


def kill_server():
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(3)


def wait_for_server(timeout=120):
    for i in range(timeout):
        time.sleep(1)
        r = subprocess.run(
            ["curl.exe", "-s", "-o", "nul", "-w", "%{http_code}", BASE_URL + "/health"],
            capture_output=True, text=True, timeout=5,
        )
        if r.stdout.strip() == "200":
            return True
    return False


def run_completion(prompt_tokens, n_min, n_max):
    payload = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": "Write a detailed paragraph about processors."}],
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
         "-H", "Authorization: Bearer " + API_KEY, "-d", payload],
        capture_output=True, text=True, timeout=60,
    )

    try:
        data = json.loads(r.stdout)
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        return {
            "completion_tokens": usage.get("completion_tokens", 0),
            "decode_tok_s": timings.get("predicted_per_second", 0),
            "draft_n": timings.get("draft_n", 0) or 0,
            "draft_n_accepted": timings.get("draft_n_accepted", 0) or 0,
        }
    except:
        return None


def main():
    all_results = []
    total = len(CONFIGS)

    for idx, (n_min, n_max) in enumerate(CONFIGS, 1):
        print(f"[{idx}/{total}] n_min={n_min}, n_max={n_max}")
        sys.stdout.flush()

        # Kill + restart
        kill_server()
        subprocess.Popen(
            ["cmd.exe", "/c", BAT, str(n_max), str(n_min)],
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        if not wait_for_server():
            print("  SERVER FAILED — skipping")
            continue

        # Warmup
        for w in range(WARMUP_RUNS):
            run_completion(16, n_min, n_max)

        # Measured passes
        valid = []
        for p in range(MEASURED_RUNS):
            result = run_completion(16, n_min, n_max)
            if result and result["decode_tok_s"] > 0:
                valid.append(result)
                dr = (result["draft_n_accepted"] / result["draft_n"] * 100) if result["draft_n"] > 0 else 0
                print(f"  Pass {p+1}: decode={result['decode_tok_s']:.1f} tok/s "
                      f"draft={result['draft_n_accepted']}/{result['draft_n']} ({dr:.0f}%) "
                      f"tokens={result['completion_tokens']}")
            else:
                print(f"  Pass {p+1}: FAILED")
            sys.stdout.flush()

        if valid:
            avg_decode = sum(r["decode_tok_s"] for r in valid) / len(valid)
            avg_tokens = sum(r["completion_tokens"] for r in valid) / len(valid)
            avg_dn = sum(r["draft_n"] for r in valid) / len(valid)
            avg_da = sum(r["draft_n_accepted"] for r in valid) / len(valid)
            dr = (avg_da / avg_dn * 100) if avg_dn > 0 else 0
            row = (n_min, n_max, round(avg_decode, 1), round(avg_tokens, 1),
                   round(avg_dn, 1), round(avg_da, 1), round(dr, 1))
            all_results.append(row)
            print(f"  AVG: {avg_decode:.1f} tok/s draft={avg_da:.0f}/{avg_dn:.0f} ({dr:.0f}%)")

            # Write CSV after each config
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["n_min", "n_max", "decode_tok_s", "avg_tokens",
                                 "draft_n_avg", "draft_acc_avg", "draft_rate_pct"])
                for rw in sorted(all_results, key=lambda x: -x[2]):
                    writer.writerow(rw)
        else:
            print("  AVG: NO VALID RESULTS")
            row = (n_min, n_max, 0.0, 0.0, 0.0, 0.0, 0.0)
            all_results.append(row)

        print()
        sys.stdout.flush()

    # Final results
    print()
    print("=" * 72)
    print("FINAL RESULTS — Top 20")
    print("=" * 72)
    print(f"{'n_min':>6} {'n_max':>6} {'Decode':>8} {'Tokens':>7} {'Draft Rate':>10} {'Acc/Total':>12}")
    print("-" * 72)
    for row in sorted(all_results, key=lambda x: -x[2])[:20]:
        n_min, n_max, d_ts, toks, dn, da, dr = row
        print(f"{n_min:>6} {n_max:>6} {d_ts:>8.1f} {int(toks):>7} {dr:>9.1f}%  {da:.0f}/{dn:.0f}")

    print()
    print(f"Total configs tested: {len(all_results)}")
    print(f"CSV: {CSV_FILE}")
    print(f"Log: {LOG_FILE}")

    kill_server()
    print("Done.")


if __name__ == "__main__":
    main()
