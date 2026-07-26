#!/usr/bin/env python3
"""
Production KV Cache Sweep v2 — GTX 1660 Super (CUDA1)
======================================================
Tests all 8 KV cache types at d=16384 and d=32768, 5 rounds each.
GPU cleanup verification between EVERY benchmark.
exit code 9 → invalidates ENTIRE sweep.
OOM → valid result, record and move on.

Usage: python kv_prod_sweep_v2.py
Output: kv_prod_sweep_v2_results.jsonl + kv_prod_sweep_v2_log.txt
"""

import subprocess
import sys
import json
import time
import datetime
import os
import re

# ── Config ──────────────────────────────────────────────────────────────
LLAMA_BENCH = r"G:/hermes/buun-llama-cpp/build/bin/llama-bench.exe"
MODEL       = r"K:/models/GestaltLabs/Ornstein-3.5-9B-V2-GGUF/ornstein-v2-Q4_K_M.gguf"
OUT_DIR     = r"G:/hermes/buun-llama-cpp/experiments/1660-super"
RESULTS_JSONL = os.path.join(OUT_DIR, "kv_prod_sweep_v2_results.jsonl")
LOG_FILE      = os.path.join(OUT_DIR, "kv_prod_sweep_v2_log.txt")

KV_TYPES = ["f16", "q8_0", "q4_0", "turbo8", "turbo4", "turbo3", "turbo2", "vbr"]
DEPTHS   = [16384, 32768]

GPU_ID = 1  # CUDA1 = GTX 1660 Super

# ── Helpers ──────────────────────────────────────────────────────────────

def log(msg):
    """Print with timestamp and write to log file."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def cleanup_gpu():
    """Kill any llama-bench, wait, verify GPU is clean. Return True if clean."""
    log("--- GPU CLEANUP ---")
    # Kill all llama-bench
    taskkill_result = subprocess.run(["taskkill", "/F", "/IM", "llama-bench.exe"],
                   capture_output=True, text=False, timeout=30)
    log(f"taskkill: {taskkill_result.stdout.decode('utf-8', errors='replace').strip()}")
    time.sleep(15)

    # Verify via nvidia-smi + tasklist
    for attempt in range(5):
        # Check nvidia-smi for GPU compute apps
        gpu_clean = True
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(GPU_ID), "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=False, timeout=15
            )
            output = result.stdout.decode('utf-8', errors='replace').strip()
            if output and "llama-bench" in output:
                gpu_clean = False
                log(f"llama-bench still on GPU (attempt {attempt+1}):\n{output}")
        except subprocess.TimeoutExpired:
            log(f"nvidia-smi timeout on attempt {attempt+1}")

        # Also check tasklist for any running llama-bench (filtered, avoid encoding issues)
        task_result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-bench.exe", "/NH"],
            capture_output=True, text=False, timeout=15
        )
        task_out = task_result.stdout.decode('utf-8', errors='replace')
        if "llama-bench.exe" in task_out:
            gpu_clean = False
            log(f"llama-bench process still in tasklist (attempt {attempt+1})")

        if gpu_clean:
            log("GPU is clean.")
            return True

        # Kill again and wait longer
        subprocess.run(["taskkill", "/F", "/IM", "llama-bench.exe"],
                       capture_output=True, text=False, timeout=30)
        time.sleep(15)

    log("GPU CLEANUP FAILED after 5 attempts!")
    return False


def run_benchmark(depth, kv_type):
    """Run llama-bench for one config. Returns (exit_code, stdout, stderr)."""
    cmd = [
        LLAMA_BENCH,
        "-m", MODEL,
        "-dev", f"CUDA{GPU_ID}",
        "-ngl", "99",
        "-fa", "1",
        "-b", "64",
        "-ub", "64",
        "-n", "256",
        "-p", "0",
        "-d", str(depth),
        "--no-warmup",
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-r", "5",
        "-o", "jsonl",
        "--progress"
    ]
    cmd_str = " ".join(cmd)
    log(f"RUNNING: {cmd_str}")

    result = subprocess.run(cmd, capture_output=True, text=False, timeout=600)
    stdout = result.stdout.decode('utf-8', errors='replace')
    stderr = result.stderr.decode('utf-8', errors='replace')
    return result.returncode, stdout, stderr


def parse_results(stdout, depth, kv_type):
    """Parse llama-bench JSONL output into structured results."""
    results = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "avg_ts" in data:
                results.append(data)
        except json.JSONDecodeError:
            log(f"WARNING: Non-JSON output line: {line[:200]}")

    return results
# ── Main ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Write meta header
    meta = {
        "meta": {
            "tool": "kv_prod_sweep_v2.py",
            "timestamp": datetime.datetime.now().strftime("%a %m/%d/%Y %H:%M:%S"),
            "gpu": "GTX 1660 Super (CUDA1)",
            "model": MODEL,
            "note": "GPU cleanup verification between every benchmark. Exit code 9 invalidates entire sweep."
        }
    }
    with open(RESULTS_JSONL, "w") as f:
        f.write(json.dumps(meta) + "\n")

    # Initial cleanup
    log("=" * 60)
    log("PRODUCTION KV CACHE SWEEP v2")
    log(f"Started: {datetime.datetime.now()}")
    log(f"Configs: {len(KV_TYPES)} KV types x {len(DEPTHS)} depths = {len(KV_TYPES)*len(DEPTHS)} total")
    log(f"GPU: GTX 1660 Super (CUDA{GPU_ID})")
    log("=" * 60)

    if not cleanup_gpu():
        log("FATAL: Initial GPU cleanup failed. Aborting.")
        sys.exit(1)

    all_results = {}
    sweep_invalid = False
    oom_configs = []

    for depth in DEPTHS:
        log(f"\n{'='*60}")
        log(f"DEPTH: d={depth}")
        log(f"{'='*60}")

        depth_results = {}

        for kv_type in KV_TYPES:
            if sweep_invalid:
                log("Sweep invalidated by exit code 9. SKIPPING remaining configs.")
                break

            label = f"{depth}_{kv_type}"
            log(f"\n{'-'*50}")
            log(f"CONFIG: {label}")
            log(f"{'-'*50}")

            # Clean GPU before EVERY benchmark
            if not cleanup_gpu():
                log(f"FATAL: GPU cleanup failed before {label}. Aborting.")
                sys.exit(1)

            # Run benchmark
            try:
                exit_code, stdout, stderr = run_benchmark(depth, kv_type)
            except subprocess.TimeoutExpired:
                log(f"TIMEOUT: {label} exceeded 600s")
                result_entry = {
                    "config": {"depth": depth, "kv_type": kv_type},
                    "status": "TIMEOUT",
                    "exit_code": -1,
                    "detail": "timeout after 600s"
                }
                with open(RESULTS_JSONL, "a") as f:
                    f.write(json.dumps(result_entry) + "\n")
                depth_results[kv_type] = result_entry
                continue
            except Exception as e:
                log(f"ERROR: {label} - {e}")
                sys.exit(1)

            log(f"Exit code: {exit_code}")

            # Parse stdout for results
            results = parse_results(stdout, depth, kv_type)
            result_entry = {
                "config": {"depth": depth, "kv_type": kv_type},
                "exit_code": exit_code,
                "detail": stderr[:500] if stderr else ""
            }

            if exit_code == 9:
                log(f"*** EXIT CODE 9 - SWEEP INVALIDATED ***")
                log(f"Process Lasso killed {label}")
                log(f"Stdout tail: {stdout[-200:]}")
                log(f"Stderr: {stderr[:500]}")
                result_entry["status"] = "LASSO_KILL"
                result_entry["results"] = results
                with open(RESULTS_JSONL, "a") as f:
                    f.write(json.dumps(result_entry) + "\n")
                depth_results[kv_type] = result_entry
                sweep_invalid = True
                log("SWEEP INVALIDATED. All results discarded. Restart from f16 at d=16384.")
                break

            elif exit_code != 0:
                combined = (stderr + stdout).lower()
                is_oom = any(x in combined for x in [
                    "out of memory", "cuda error", "oom",
                    "failed to create context", "cuda_malloc",
                    "cuda out of memory"
                ])

                if is_oom:
                    log(f"OOM: {label} - valid result, recording and moving on.")
                    result_entry["status"] = "OOM"
                    result_entry["results"] = []
                    oom_configs.append(label)
                else:
                    log(f"FAILED: {label} (exit={exit_code})")
                    log(f"Stdout tail: {stdout[-300:]}")
                    log(f"Stderr: {stderr[:500]}")
                    result_entry["status"] = "FAILED"
                    result_entry["results"] = results

                with open(RESULTS_JSONL, "a") as f:
                    f.write(json.dumps(result_entry) + "\n")
                depth_results[kv_type] = result_entry
                continue

            # Success
            log(f"SUCCESS: {label}")
            if results:
                for r in results:
                    tg = r.get("avg_ts", "N/A")
                    log(f"  TG: {tg}")
                result_entry["status"] = "SUCCESS"
                result_entry["results"] = results
            else:
                log(f"  No structured results found. Raw stdout:\n{stdout[:500]}")
                result_entry["status"] = "SUCCESS_NO_DATA"
                result_entry["raw_stdout"] = stdout[:2000]
                result_entry["results"] = []

            with open(RESULTS_JSONL, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            depth_results[kv_type] = result_entry

        if sweep_invalid:
            break

        all_results[depth] = depth_results

    # Summary
    log("\n" + "=" * 60)
    log("SWEEP COMPLETE")
    log("=" * 60)

    if sweep_invalid:
        log("\n*** SWEEP INVALIDATED due to exit code 9 ***")
        log("Restart required from f16 at d=16384.")
    else:
        log("\nSummary:")
        header = f"{'KV Type':<15} {'d=16384 TG':<15} {'d=32768 TG':<15} {'d=32768 OOM?':<15}"
        log(header)
        log("-" * len(header))
        for kv_type in KV_TYPES:
            tg_16 = ""
            tg_32 = ""
            oom_32 = ""
            if 16384 in all_results and kv_type in all_results[16384]:
                r = all_results[16384][kv_type]
                if r.get("status") in ("SUCCESS", "SUCCESS_NO_DATA") and r.get("results"):
                    tg_16 = f"{r['results'][0].get('avg_ts', '?'):.2f}"
                elif r.get("status") == "OOM":
                    tg_16 = "OOM"
            if 32768 in all_results and kv_type in all_results[32768]:
                r = all_results[32768][kv_type]
                if r.get("status") in ("SUCCESS", "SUCCESS_NO_DATA") and r.get("results"):
                    tg_32 = f"{r['results'][0].get('avg_ts', '?'):.2f}"
                elif r.get("status") == "OOM":
                    oom_32 = "OOM"

            log(f"{kv_type:<15} {tg_16:<15} {tg_32:<15} {oom_32:<15}")

    log(f"\nResults saved to: {RESULTS_JSONL}")
    log(f"Log saved to: {LOG_FILE}")

    if oom_configs:
        log(f"\nOOM configs ({len(oom_configs)}):")
        for c in oom_configs:
            log(f"  - {c}")

    log(f"\nFinished: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
