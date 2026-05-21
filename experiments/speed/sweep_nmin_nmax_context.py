import subprocess
import json
import time
import csv
import sys
import random
import tempfile
import os
import datetime

BAT = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\test_start_dflash.bat"
OUTPUT = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\results\\nmin_nmax_context_sweep.csv"
LOG_DIR = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\logs"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"nmin_nmax_context_{TIMESTAMP}.txt")
API_KEY = "dummythicc"
BASE_URL = "http://localhost:8081"

P_MIN = 0.75
WARMUP_RUNS = 3
MEASURED_RUNS = 5
CONTEXT_LENGTHS = [256, 1024, 2048, 4096, 8192, 16384, 24576, 32768]
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
print("n_min/n_max x Context Sweep (server restarts)")
print(f"Configs: {len(CONFIGS)} | Contexts: {len(CONTEXT_LENGTHS)} | Warmup: {WARMUP_RUNS} | Measured: {MEASURED_RUNS}")
print(f"p_min={P_MIN}, p_min hardcoded in bat")
print(f"Log: {LOG_FILE}")
print(f"CSV: {OUTPUT}")
print("=" * 72)
print()


def kill_server():
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    time.sleep(3)


def kill_cmd(cmd_pid):
    try:
        subprocess.run(["taskkill", "/f", "/pid", str(cmd_pid)], capture_output=True, timeout=10)
    except:
        pass


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


def generate_prompt_tokens(n_tokens):
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "benchmark", "context", "filler", "text", "data", "test", "run",
        "analysis", "system", "process", "memory", "cache", "buffer",
        "compute", "kernel", "thread", "allocation", "device", "vector",
        "parallel", "token", "inference", "model", "weight", "layer",
    ]
    target_words = int(n_tokens * 0.75)
    return " ".join(random.choice(words) for _ in range(target_words))


def run_completion(prompt, max_tokens):
    payload = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
        "n": 1,
    })

    # Use temp file for large payloads to avoid Windows cmd-line length limit
    if len(payload) > 10000:
        fd, tmp = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            cmd = [
                "curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", "Authorization: Bearer " + API_KEY,
                "--data-binary", "@" + tmp,
            ]
        except Exception:
            os.close(fd)
            raise
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(tmp):
            os.unlink(tmp)
    else:
        r = subprocess.run(
            ["curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
             "-H", "Content-Type: application/json",
             "-H", "Authorization: Bearer " + API_KEY, "-d", payload],
            capture_output=True, text=True, timeout=120,
        )

    try:
        data = json.loads(r.stdout)
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        return {
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "prompt_tok_s": timings.get("prompt_per_second", 0),
            "decode_tok_s": timings.get("predicted_per_second", 0),
            "draft_n": timings.get("draft_n", 0) or 0,
            "draft_n_accepted": timings.get("draft_n_accepted", 0) or 0,
        }
    except Exception:
        return None


def main():
    all_results = []
    total = len(CONFIGS)

    for idx, (n_min, n_max) in enumerate(CONFIGS, 1):
        print(f"[{idx}/{total}] n_min={n_min}, n_max={n_max}")
        sys.stdout.flush()

        # Kill leftovers
        kill_server()

        # Start server in new console
        subprocess.Popen(
            ["wt", "-w", "0", "nt", "cmd", "/c", BAT, str(n_max), str(n_min)],
        )

        if not wait_for_server():
            print("  SERVER FAILED — skipping")
    
            print()
            continue

        # Warmup
        print("  Warmup...", end="")
        sys.stdout.flush()
        for w in range(WARMUP_RUNS):
            run_completion("Write about processors.", 64)
        print(" done")

        # Benchmark at each context
        config_results = []
        for ctx_len in CONTEXT_LENGTHS:
            ctx_results = []
            print(f"    Context {ctx_len}...", end="")
            sys.stdout.flush()

            prompt = generate_prompt_tokens(ctx_len)

            for m in range(MEASURED_RUNS):
                result = run_completion(prompt, GEN_TOKENS)
                if result:
                    ctx_results.append(result)

            if ctx_results:
                avg_decode = sum(r["decode_tok_s"] for r in ctx_results) / len(ctx_results)
                avg_tokens = sum(r["completion_tokens"] for r in ctx_results) / len(ctx_results)
                avg_dn = sum(r["draft_n"] for r in ctx_results) / len(ctx_results)
                avg_da = sum(r["draft_n_accepted"] for r in ctx_results) / len(ctx_results)
                dr = (avg_da / avg_dn * 100) if avg_dn > 0 else 0
                config_results.append((ctx_len, avg_decode, avg_tokens, avg_da, avg_dn, dr))
                print(f" decode={avg_decode:.1f} tok/s draft={avg_da:.0f}/{avg_dn:.0f} ({dr:.0f}%)")
            else:
                config_results.append((ctx_len, 0.0, 0.0, 0.0, 0.0, 0.0))
                print(" FAILED")
            sys.stdout.flush()

        if config_results:
            # Calculate per-config summary
            avg_decode_all = sum(r[1] for r in config_results) / len(config_results)
            avg_draft_acc_all = sum(r[4] for r in config_results) / len(config_results) / len(config_results)
            row = (n_min, n_max, round(avg_decode_all, 1), round(avg_draft_acc_all, 1),
                   len(config_results), len(CONTEXT_LENGTHS))
            all_results.append(row)
            print(f"  AVG: {row[2]:.1f} tok/s (across {row[3]} ctxs)")
        else:
            row = (n_min, n_max, 0.0, 0.0, 0, len(CONTEXT_LENGTHS))
            all_results.append(row)
            print(f"  AVG: 0.0 tok/s")

        print()

        # Kill server and cmd
        kill_server()


    # Save CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_min", "n_max", "avg_decode_tok_s", "avg_draft_acc", "valid_configs", "total_contexts"])
        for row in sorted(all_results, key=lambda x: -x[2]):
            writer.writerow(row)

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(f"{'n_min':>6} {'n_max':>6} {'Decode':>8} {'Draft Acc':>10} {'Valid':>7} {'Contexts':>8}")
    print("-" * 72)
    for row in sorted(all_results, key=lambda x: -x[2]):
        print(f"{row[0]:>6} {row[1]:>6} {row[2]:>8.1f} {row[3]:>10.1f}% {row[4]:>7} {row[5]:>8}")

    print()
    print(f"CSV: {OUTPUT}")
    print(f"Log: {LOG_FILE}")

    # Clean up
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True, timeout=10)
    print("Done.")


if __name__ == "__main__":
    main()
