import subprocess
import json
import time
import csv
import sys
import random
import tempfile
import os

BAT = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\test_start_dflash.bat"
OUTPUT = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\results\\dflash_proper_bench.csv"
LOG_PATH = r"G:\\hermes\\buun-llama-cpp\\experiments\\speed\\results\\dflash_full_run.txt"
API_KEY = "dummythicc"
BASE_URL = "http://localhost:8081"

WARMUP_RUNS = 3
MEASURED_RUNS = 5
CONTEXT_LENGTHS = [256, 1024, 2048, 4096, 8192, 16384, 24576, 32768]
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
print("DFlash Proper Benchmark (v2 - cache-ram=0, full log)")
print("====================================================")


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
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": False,
        "n": 1,
    }
    payload_json = json.dumps(payload)
    use_file = len(payload_json) > 10000

    if use_file:
        fd, tmp = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload_json)
            cmd = [
                "curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", "Authorization: Bearer " + API_KEY,
                "--data-binary", "@" + tmp,
            ]
        except Exception:
            os.close(fd)
            raise
    else:
        tmp = None
        cmd = [
            "curl.exe", "-s", "-X", "POST", BASE_URL + "/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Bearer " + API_KEY,
            "-d", payload_json,
        ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if tmp and os.path.exists(tmp):
        os.unlink(tmp)

    try:
        data = json.loads(r.stdout)
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        return {
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "prompt_tok_s": timings.get("prompt_per_second", 0),
            "decode_tok_s": timings.get("predicted_per_second", 0),
            "draft_n": timings.get("draft_n", 0),
            "draft_n_accepted": timings.get("draft_n_accepted", 0),
        }
    except Exception:
        return None


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

    # Warmup
    print(f"Warming up ({WARMUP_RUNS} runs)...")
    for i in range(WARMUP_RUNS):
        result = run_completion("Write about processors.", 64)
        if result:
            dr = (result["draft_n_accepted"] / result["draft_n"] * 100) if result["draft_n"] > 0 else 0
            print(f"  Warmup {i+1}: {result['decode_tok_s']:.1f} decode tok/s, "
                  f"draft {result['draft_n_accepted']}/{result['draft_n']} ({dr:.0f}%)")

    print()
    results = []

    for ctx_len in CONTEXT_LENGTHS:
        print(f"Context: {ctx_len} tokens")
        sys.stdout.flush()

        prompt = generate_prompt_tokens(ctx_len)
        ctx_results = []

        for i in range(MEASURED_RUNS):
            result = run_completion(prompt, GEN_TOKENS)
            if result:
                ctx_results.append(result)
                dr = (result["draft_n_accepted"] / result["draft_n"] * 100) if result["draft_n"] > 0 else 0
                print(f"  Run {i+1}: prompt={result['prompt_tok_s']:.0f} tok/s, "
                      f"decode={result['decode_tok_s']:.1f} tok/s, "
                      f"draft={result['draft_n_accepted']}/{result['draft_n']} ({dr:.0f}%), "
                      f"tokens={result['completion_tokens']}")
            else:
                print(f"  Run {i+1}: FAILED")
            sys.stdout.flush()

        if ctx_results:
            avg_decode = sum(r["decode_tok_s"] for r in ctx_results) / len(ctx_results)
            avg_prompt = sum(r["prompt_tok_s"] for r in ctx_results) / len(ctx_results)
            avg_tokens = sum(r["completion_tokens"] for r in ctx_results) / len(ctx_results)
            avg_dn = sum(r["draft_n"] for r in ctx_results) / len(ctx_results)
            avg_da = sum(r["draft_n_accepted"] for r in ctx_results) / len(ctx_results)
            dr = (avg_da / avg_dn * 100) if avg_dn > 0 else 0
            results.append((ctx_len, avg_prompt, avg_decode, avg_tokens, avg_dn, avg_da, dr))
            print(f"  AVG: prompt={avg_prompt:.0f} tok/s, decode={avg_decode:.1f} tok/s, "
                  f"draft={avg_da:.1f}/{avg_dn:.1f} ({dr:.0f}%)")
        print()

    # Save CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ctx_len", "prompt_tok_s", "decode_tok_s", "avg_tokens",
                         "draft_n_avg", "draft_acc_avg", "draft_rate_pct"])
        for row in results:
            writer.writerow(row)

    print("RESULTS:")
    print("=" * 80)
    print(f"{'Context':>10} {'Prompt tok/s':>12} {'Decode tok/s':>12} {'Tokens':>7} {'Draft Acc':>10}")
    print("-" * 80)
    for ctx, p_ts, d_ts, tokens, dn, da, dr in results:
        print(f"{ctx:>10} {p_ts:>12.0f} {d_ts:>12.1f} {int(tokens):>7} {da:.1f}/{dn:.1f} ({dr:.0f}%)")
    print()
    print(f"CSV: {OUTPUT}")
    print(f"Log: {LOG_PATH}")

    kill_server()
    print("Done.")

if __name__ == "__main__":
    main()
