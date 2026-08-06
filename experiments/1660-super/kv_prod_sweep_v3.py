#!/usr/bin/env python3
"""
Production KV Cache Sweep v3 — GTX 1660 Super (CUDA1)
=======================================================
Tests all KV cache types at d=16384 and d=32768, 5 rounds each.
All Tier 1 + Tier 2 safeguards from council decision t_366c04ae.

Safeguards:
  S1: Exit 9 = single-point retry (max 3), never invalidate sweep
  S2: Grace periods (60s/120s/300s) with verification
  S3: 8081 health gate before EVERY benchmark
  S4: Hard caps (lockfile, retries, failures, runtime, configs)
  S5: Sweep checkpointing to JSON
  S6: Process verification (PID capture, wait, CUDA release)
  S7: GPU health pre-flight (ECC, temp, mem, util)

Usage: python kv_prod_sweep_v3.py [--dry-run]
       --dry-run: Validate environment and configs, but don't run benchmarks.

Output files (in experiments/1660-super/):
  kv_prod_sweep_v3_results.jsonl    - per-config results
  kv_prod_sweep_v3_log.txt          - human-readable log
  kv_prod_sweep_v3_checkpoint.json  - resume checkpoint
  kv_prod_sweep_v3.lock             - instance lock
"""

import subprocess
import sys
import json
import time
import datetime
import os
import urllib.request
import urllib.error
import socket
import atexit

# ── Constants ──────────────────────────────────────────────────────────────
SWEEP_VERSION = "3"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LLAMA_BENCH = r"G:/hermes/buun-llama-cpp/build/bin/llama-bench.exe"
MODEL       = r"K:/models/GestaltLabs/Ornstein-3.5-9B-V2-GGUF/ornstein-v2-Q4_K_M.gguf"
OUT_DIR     = r"G:/hermes/buun-llama-cpp/experiments/1660-super"

RESULTS_JSONL   = os.path.join(OUT_DIR, f"kv_prod_sweep_v{SWEEP_VERSION}_results.jsonl")
LOG_FILE        = os.path.join(OUT_DIR, f"kv_prod_sweep_v{SWEEP_VERSION}_log.txt")
CHECKPOINT_FILE = os.path.join(OUT_DIR, f"kv_prod_sweep_v{SWEEP_VERSION}_checkpoint.json")
LOCK_FILE       = os.path.join(OUT_DIR, f"kv_prod_sweep_v{SWEEP_VERSION}.lock")

KV_TYPES = ["f16", "q8_0", "q4_0", "turbo8", "turbo4", "turbo3", "turbo2", "vbr"]
DEPTHS   = [16384, 32768]

GPU_ID = 1  # CUDA1 = GTX 1660 Super

# ── Safeguard constants ───────────────────────────────────────────────────
HEALTH_CHECK_URL      = "http://localhost:8081/health"
HEALTH_CHECK_TIMEOUT  = 5  # seconds

GRACE_NORMAL   = 60   # seconds after normal exit
GRACE_NONZERO  = 120  # seconds after non-zero exit
GRACE_GPU_ERROR = 300 # seconds after GPU-level error
GRACE_INITIAL  = 60   # initial cleanup wait

MAX_RETRIES_PER_POINT       = 3
MAX_CONSECUTIVE_FAILURES    = 5
MAX_CONSECUTIVE_EXIT9       = 3
MAX_RUNTIME_SECONDS         = 8 * 3600  # 8 hours
MAX_RUNTIME_HOURS           = 8
MAX_CONFIGS                 = 20

EXIT9_WAIT          = 120  # seconds to wait after exit 9 before retry
PID_WAIT_TIMEOUT    = 60   # seconds to wait for each PID to exit
CUDA_RELEASE_WAIT   = 5    # seconds for CUDA context cleanup
GRACEFUL_KILL_WAIT  = 30   # seconds to wait after taskkill (no /F)

GPU_TEMP_MAX        = 85   # Celsius
GPU_USED_MEM_MAX_MB = 100  # MiB — GPU should be idle (used mem < 100MB)
GPU_UTIL_MAX_PCT    = 5    # percent

# ── Runtime state ─────────────────────────────────────────────────────────
_start_time = None
_start_mono = None
_configs_run = 0
_consecutive_failures = 0
_consecutive_exit9 = 0
_dry_run = False


# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════

def log(msg):
    """Print with timestamp and write to log file."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════════
# S6: Process verification — capture PIDs, graceful kill, wait, verify
# ══════════════════════════════════════════════════════════════════════════

def get_llama_bench_pids():
    """Return set of PIDs for all llama-bench.exe processes."""
    pids = set()
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-bench.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=False, timeout=15
        )
        output = result.stdout.decode('utf-8', errors='replace').strip()
        for line in output.split("\n"):
            line = line.strip().strip('"')
            if not line or "llama-bench.exe" not in line:
                continue
            parts = line.split('","')
            if len(parts) >= 2:
                try:
                    pid_str = parts[1].strip('"')
                    pids.add(int(pid_str))
                except ValueError:
                    pass
    except Exception as e:
        log(f"WARNING: Failed to enumerate llama-bench PIDs: {e}")
    return pids


def wait_for_pids(pids, timeout=PID_WAIT_TIMEOUT):
    """Wait for all given PIDs to exit. Returns True if all exited, False if timeout."""
    if not pids:
        return True
    deadline = time.monotonic() + timeout
    remaining = set(pids)
    while time.monotonic() < deadline:
        # Check via tasklist for each PID
        for pid in list(remaining):
            try:
                check = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=False, timeout=10
                )
                output = check.stdout.decode('utf-8', errors='replace')
                if str(pid) not in output:
                    remaining.discard(pid)
            except Exception:
                pass
        if not remaining:
            return True
        time.sleep(1)
    log(f"WARNING: PIDs still running after {timeout}s: {remaining}")
    return False


def cleanup_gpu():
    """
    Full GPU cleanup with escalation protocol:
      (a) graceful taskkill (no /F) + 30s wait
      (b) verify via nvidia-smi + tasklist
      (c) taskkill /F only if still running
      (d) wait for PID exit + 5s CUDA release
    Returns True if GPU is clean after all attempts, False otherwise.
    """
    log("--- GPU CLEANUP (S6 escalation protocol) ---")

    for attempt in range(5):
        # ── Step (a): Capture PIDs, try graceful kill ──
        pids_before = get_llama_bench_pids()
        if pids_before:
            log(f"PIDs before cleanup: {pids_before}")

        # Try graceful kill first (no /F)
        graceful_result = subprocess.run(
            ["taskkill", "/IM", "llama-bench.exe"],  # no /F
            capture_output=True, text=False, timeout=30
        )
        graceful_stdout = graceful_result.stdout.decode('utf-8', errors='replace').strip()
        log(f"taskkill (graceful): {graceful_stdout}")

        # Wait for PIDs to exit
        if pids_before:
            all_exited = wait_for_pids(pids_before, GRACEFUL_KILL_WAIT)
            if not all_exited:
                log(f"Graceful kill incomplete — escalating to taskkill /F (attempt {attempt+1})")
                # ── Step (c): Force kill ──
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-bench.exe"],
                    capture_output=True, text=False, timeout=30
                )
                time.sleep(5)
                pids_after_force = get_llama_bench_pids()
                if pids_after_force:
                    wait_for_pids(pids_after_force, PID_WAIT_TIMEOUT)
        else:
            log("No llama-bench processes found.")

        # ── Step (d): CUDA release wait ──
        time.sleep(CUDA_RELEASE_WAIT)

        # ── Step (b): Verify via nvidia-smi + tasklist ──
        gpu_clean = True

        # nvidia-smi check
        try:
            smi_result = subprocess.run(
                ["nvidia-smi", "-i", str(GPU_ID), "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=False, timeout=15
            )
            smi_output = smi_result.stdout.decode('utf-8', errors='replace').strip()
            if smi_output and "llama-bench" in smi_output:
                gpu_clean = False
                log(f"llama-bench still in nvidia-smi (attempt {attempt+1}):\n{smi_output}")
        except subprocess.TimeoutExpired:
            log(f"nvidia-smi timeout on attempt {attempt+1}")

        # Tasklist check
        task_output = get_llama_bench_pids()
        if task_output:
            gpu_clean = False
            log(f"llama-bench PIDs still in tasklist (attempt {attempt+1}): {task_output}")

        if gpu_clean:
            log("GPU is clean.")
            return True

        # If not clean and we haven't force-killed yet this iteration, do it now
        log(f"GPU not clean after attempt {attempt+1}, retrying...")
        time.sleep(15)

    log("GPU CLEANUP FAILED after 5 attempts!")
    return False


# ══════════════════════════════════════════════════════════════════════════
# S7: GPU health pre-flight
# ══════════════════════════════════════════════════════════════════════════

def check_gpu_health():
    """
    S7: GPU health pre-flight checks.
    Returns (ok: bool, details: str).
    Fails if: ECC errors != 0, temp >= 85C, memory free < 100MB, util >= 5%.
    """
    log("--- GPU HEALTH PRE-FLIGHT (S7) ---")

    # Query multiple GPU attributes in one call
    query = (
        "temperature.gpu,"
        "utilization.gpu,"
        "memory.total,"
        "memory.used,"
        "ecc.errors.uncorrected.volatile.total"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(GPU_ID),
             f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=False, timeout=15
        )
        output = result.stdout.decode('utf-8', errors='replace').strip()
        if not output:
            return False, "No output from nvidia-smi query-gpu"
    except Exception as e:
        return False, f"nvidia-smi query failed: {e}"

    parts = [p.strip() for p in output.split(",")]
    if len(parts) < 5:
        return False, f"Unexpected nvidia-smi output: {output}"

    try:
        temp = float(parts[0])
        util = float(parts[1])
        mem_total = float(parts[2])  # MiB
        mem_used = float(parts[3])   # MiB
        ecc_errors = float(parts[4])
    except ValueError as e:
        return False, f"Failed to parse nvidia-smi values: {e} | raw: {output}"

    mem_free = mem_total - mem_used
    issues = []

    if ecc_errors > 0:
        issues.append(f"ECC uncorrected errors: {ecc_errors}")
    if temp >= GPU_TEMP_MAX:
        issues.append(f"GPU temp {temp}C >= {GPU_TEMP_MAX}C")
    if mem_used >= GPU_USED_MEM_MAX_MB:
        issues.append(f"GPU used memory {mem_used:.0f}MiB >= {GPU_USED_MEM_MAX_MB}MiB (should be idle)")
    if util >= GPU_UTIL_MAX_PCT:
        issues.append(f"GPU util {util}% >= {GPU_UTIL_MAX_PCT}%")

    if issues:
        detail = "; ".join(issues)
        log(f"GPU HEALTH FAILED: {detail}")
        return False, detail

    log(f"GPU health OK: temp={temp}C, util={util}%, mem_free={mem_free:.0f}MiB, ecc={ecc_errors}")
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════
# S3: Production server health gate (8081)
# ══════════════════════════════════════════════════════════════════════════

def check_8081_health():
    """
    S3: Check production server health.
    GET http://localhost:8081/health with 5s timeout.
    Returns True if 200, False otherwise.
    """
    log("--- 8081 HEALTH GATE (S3) ---")
    try:
        req = urllib.request.Request(HEALTH_CHECK_URL, method="GET")
        with urllib.request.urlopen(req, timeout=HEALTH_CHECK_TIMEOUT) as resp:
            status = resp.status
            if status == 200:
                log(f"8081 health: OK (status {status})")
                return True
            else:
                log(f"8081 health: FAILED (status {status})")
                return False
    except urllib.error.URLError as e:
        log(f"8081 health: CONNECTION FAILED — {e.reason}")
        return False
    except socket.timeout:
        log(f"8081 health: TIMEOUT after {HEALTH_CHECK_TIMEOUT}s")
        return False
    except Exception as e:
        log(f"8081 health: ERROR — {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# S4: Lockfile (max 1 instance)
# ══════════════════════════════════════════════════════════════════════════

def acquire_lock():
    """
    S4: Acquire instance lock. Returns True if acquired, False if another
    instance is running.
    """
    log("--- LOCKFILE CHECK (S4) ---")
    my_pid = os.getpid()

    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                data = json.load(f)
            old_pid = data.get("pid")
            old_host = data.get("host", "")
            if old_pid:
                # Check if that process is still alive
                check = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                    capture_output=True, text=False, timeout=10
                )
                output = check.stdout.decode('utf-8', errors='replace')
                if str(old_pid) in output:
                    log(f"Lock held by PID {old_pid} on {old_host}. Aborting.")
                    return False
                else:
                    log(f"Stale lock from PID {old_pid}. Removing.")
        except (json.JSONDecodeError, IOError) as e:
            log(f"Corrupt lockfile ({e}), removing.")
        os.remove(LOCK_FILE)

    # Write our lock
    lock_data = {
        "pid": my_pid,
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "timestamp": datetime.datetime.now().isoformat(),
        "sweep": f"kv_prod_sweep_v{SWEEP_VERSION}"
    }
    with open(LOCK_FILE, "w") as f:
        json.dump(lock_data, f)
    log(f"Lock acquired (PID {my_pid})")
    return True


def release_lock():
    """Release the lockfile if we own it."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                data = json.load(f)
            if data.get("pid") == os.getpid():
                os.remove(LOCK_FILE)
                log("Lock released.")
        except (json.JSONDecodeError, IOError):
            pass


# ══════════════════════════════════════════════════════════════════════════
# S5: Checkpointing
# ══════════════════════════════════════════════════════════════════════════

def load_checkpoint():
    """
    Load checkpoint from disk. Returns dict of config_key -> result_entry.
    If no checkpoint or corrupt, returns empty dict.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        log("No checkpoint found — starting fresh.")
        return {}

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            cp = json.load(f)
        completed = cp.get("completed", {})
        total = len(completed)
        log(f"Checkpoint loaded: {total} configs already completed.")
        return completed
    except (json.JSONDecodeError, IOError) as e:
        log(f"Corrupt checkpoint ({e}) — starting fresh.")
        return {}


def save_checkpoint(completed):
    """Save checkpoint to disk."""
    cp = {
        "sweep_version": SWEEP_VERSION,
        "model": MODEL,
        "gpu_id": GPU_ID,
        "timestamp": datetime.datetime.now().isoformat(),
        "run_started": _start_time.isoformat() if _start_time else None,
        "completed": completed
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)
    log(f"Checkpoint saved ({len(completed)} configs completed).")


# ══════════════════════════════════════════════════════════════════════════
# S2: Grace periods
# ══════════════════════════════════════════════════════════════════════════

def grace_period(exit_code, is_gpu_error=False):
    """
    S2: Apply appropriate grace period based on exit outcome.
    Normal exit: 60s
    Non-zero exit: 120s
    GPU error: 300s
    """
    if is_gpu_error:
        label = "GPU error"
        duration = GRACE_GPU_ERROR
    elif exit_code != 0:
        label = f"non-zero exit ({exit_code})"
        duration = GRACE_NONZERO
    else:
        label = "normal exit"
        duration = GRACE_NORMAL

    log(f"Grace period ({label}): {duration}s")
    time.sleep(duration)


# ══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# Pre-flight pipeline (P1 from decision)
# ══════════════════════════════════════════════════════════════════════════

def preflight_checks():
    """
    P1: Mandatory pre-flight checks in order:
      1. 8081 health check (S3)
      2. GPU health check (S7)
      3. Initial 60s wait + cleanup verification (S2)
      4. Lockfile acquisition (S4)
    Returns True if all pass, False otherwise.
    """
    log("=" * 60)
    log("PRE-FLIGHT CHECKS (P1)")
    log("=" * 60)

    # 1. 8081 health check (S3)
    if not check_8081_health():
        log("PRE-FLIGHT FAILED: 8081 not healthy. Aborting.")
        return False

    # 2. GPU health check (S7)
    ok, detail = check_gpu_health()
    if not ok:
        log(f"PRE-FLIGHT FAILED: GPU health check failed: {detail}. Aborting.")
        return False

    # 3. Initial 60s wait (S2) + cleanup verification
    log(f"--- INITIAL GRACE PERIOD: {GRACE_INITIAL}s ---")
    time.sleep(GRACE_INITIAL)
    log("--- INITIAL GPU CLEANUP ---")
    if not cleanup_gpu():
        log("PRE-FLIGHT FAILED: GPU cleanup failed. Aborting.")
        return False

    # 4. Lockfile acquisition (S4)
    if not acquire_lock():
        log("PRE-FLIGHT FAILED: Could not acquire lock. Aborting.")
        return False

    log("=" * 60)
    log("ALL PRE-FLIGHT CHECKS PASSED")
    log("=" * 60)
    return True


# ══════════════════════════════════════════════════════════════════════════
# S4: Hard runtime cap
# ══════════════════════════════════════════════════════════════════════════

def check_runtime_cap():
    """S4: Check if we've exceeded max runtime. Returns True if OK, False if over cap."""
    global _start_mono
    if _start_mono is None:
        return True
    elapsed = time.monotonic() - _start_mono
    if elapsed > MAX_RUNTIME_SECONDS:
        log(f"RUNTIME CAP: {elapsed:.0f}s exceeds {MAX_RUNTIME_SECONDS}s ({MAX_RUNTIME_HOURS}h). Aborting.")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# Main sweep logic
# ══════════════════════════════════════════════════════════════════════════

def process_config(depth, kv_type, checkpoint):
    """
    Process one config point with all safeguards.
    Returns (result_entry, should_abort).
    """
    global _configs_run, _consecutive_failures, _consecutive_exit9

    label = f"{depth}_{kv_type}"
    log(f"\n{'=' * 60}")
    log(f"CONFIG: {label}")
    log(f"{'=' * 60}")

    # Check runtime cap early
    if not check_runtime_cap():
        entry = {
            "config": {"depth": depth, "kv_type": kv_type},
            "status": "RUNTIME_CAP",
            "exit_code": -1,
            "detail": f"exceeded {MAX_RUNTIME_HOURS}h runtime"
        }
        return entry, True  # abort

    # S3: Health gate before every benchmark
    if not check_8081_health():
        entry = {
            "config": {"depth": depth, "kv_type": kv_type},
            "status": "8081_DOWN",
            "exit_code": -1,
            "detail": "8081 health check failed"
        }
        return entry, True  # abort immediately

    # S7: GPU health pre-flight
    ok, detail = check_gpu_health()
    if not ok:
        entry = {
            "config": {"depth": depth, "kv_type": kv_type},
            "status": "GPU_HEALTH_FAIL",
            "exit_code": -1,
            "detail": detail
        }
        return entry, True  # abort immediately

    # Clean GPU before this benchmark
    if not cleanup_gpu():
        entry = {
            "config": {"depth": depth, "kv_type": kv_type},
            "status": "CLEANUP_FAIL",
            "exit_code": -1,
            "detail": "GPU cleanup failed before benchmark"
        }
        return entry, True  # abort

    # ── S1: Exit 9 handling with bounded retry ──
    last_exit_code = None
    last_stdout = ""
    last_stderr = ""
    point_oom = False
    point_exit9_retries = 0

    for attempt in range(1 + MAX_RETRIES_PER_POINT):
        if attempt > 0:
            log(f"RETRY {attempt}/{MAX_RETRIES_PER_POINT} for {label}")
            time.sleep(EXIT9_WAIT)
            # Full cleanup between retries
            if not cleanup_gpu():
                log(f"GPU cleanup failed between retries for {label}. Aborting sweep.")
                entry = {
                    "config": {"depth": depth, "kv_type": kv_type},
                    "status": "CLEANUP_FAIL_RETRY",
                    "exit_code": -1,
                    "detail": "GPU cleanup failed during exit-9 retry"
                }
                return entry, True

            # Re-check health before retry
            if not check_8081_health():
                entry = {
                    "config": {"depth": depth, "kv_type": kv_type},
                    "status": "8081_DOWN_RETRY",
                    "exit_code": -1,
                    "detail": "8081 went down during retry wait"
                }
                return entry, True

        # Run benchmark
        try:
            exit_code, stdout, stderr = run_benchmark(depth, kv_type)
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT: {label} exceeded 600s")
            entry = {
                "config": {"depth": depth, "kv_type": kv_type},
                "status": "TIMEOUT",
                "exit_code": -1,
                "detail": "timeout after 600s"
            }
            return entry, False  # don't abort sweep for timeout
        except Exception as e:
            log(f"ERROR: {label} - {e}")
            entry = {
                "config": {"depth": depth, "kv_type": kv_type},
                "status": "EXCEPTION",
                "exit_code": -1,
                "detail": str(e)
            }
            return entry, True  # abort on unexpected exception

        last_exit_code = exit_code
        last_stdout = stdout
        last_stderr = stderr

        log(f"Exit code: {exit_code}")

        if exit_code == 9:
            point_exit9_retries += 1
            log(f"*** EXIT CODE 9 (attempt {attempt+1}) ***")
            log(f"Stdout tail: {stdout[-200:]}")
            log(f"Stderr: {stderr[:500]}")
            if attempt < MAX_RETRIES_PER_POINT:
                log(f"Will retry after {EXIT9_WAIT}s with full cleanup.")
                continue
            else:
                log(f"Exit 9 persisted after {MAX_RETRIES_PER_POINT} retries. Marking point failed.")
                # Will fall through to failure handling
        else:
            # Non-9 exit, don't retry
            break

    # ── Process result ──
    _configs_run += 1
    results = parse_results(last_stdout, depth, kv_type)

    entry = {
        "config": {"depth": depth, "kv_type": kv_type},
        "exit_code": last_exit_code,
        "detail": last_stderr[:500] if last_stderr else ""
    }

    if last_exit_code == 9:
        # All retries exhausted
        log(f"*** EXIT CODE 9 - POINT FAILED after {MAX_RETRIES_PER_POINT} retries ***")
        log(f"Moving to next config.")
        entry["status"] = "LASSO_KILL_FAILED"
        entry["results"] = results
        _consecutive_exit9 += 1
        _consecutive_failures += 1

        # Check consecutive exit 9 limit
        if _consecutive_exit9 >= MAX_CONSECUTIVE_EXIT9:
            log(f"*** {MAX_CONSECUTIVE_EXIT9} CONSECUTIVE EXIT-9 POINTS — ABORTING SWEEP ***")
            return entry, True

    elif last_exit_code != 0:
        combined = (last_stderr + last_stdout).lower()
        is_oom = any(x in combined for x in [
            "out of memory", "cuda error", "oom",
            "failed to create context", "cuda_malloc",
            "cuda out of memory"
        ])

        if is_oom:
            log(f"OOM: {label} — valid result, recording and moving on.")
            entry["status"] = "OOM"
            entry["results"] = []
            _consecutive_failures = 0  # OOM is expected, not a failure
            _consecutive_exit9 = 0
        else:
            log(f"FAILED: {label} (exit={last_exit_code})")
            log(f"Stdout tail: {last_stdout[-300:]}")
            log(f"Stderr: {last_stderr[:500]}")
            entry["status"] = "FAILED"
            entry["results"] = results
            _consecutive_failures += 1
            _consecutive_exit9 = 0
    else:
        # Success
        log(f"SUCCESS: {label}")
        if results:
            for r in results:
                tg = r.get("avg_ts", "N/A")
                log(f"  TG: {tg}")
            entry["status"] = "SUCCESS"
            entry["results"] = results
        else:
            log(f"  No structured results found. Raw stdout:\n{last_stdout[:500]}")
            entry["status"] = "SUCCESS_NO_DATA"
            entry["raw_stdout"] = last_stdout[:2000]
            entry["results"] = []
        _consecutive_failures = 0
        _consecutive_exit9 = 0

    # Check consecutive failures cap
    should_abort = _consecutive_failures >= MAX_CONSECUTIVE_FAILURES
    if should_abort:
        log(f"*** {MAX_CONSECUTIVE_FAILURES} CONSECUTIVE FAILURES — ABORTING SWEEP ***")
        entry["detail"] = (entry.get("detail", "") +
                           f" | Sweep aborted: {_consecutive_failures} consecutive failures")

    return entry, should_abort


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    global _start_time, _start_mono, _dry_run

    # Parse args
    _dry_run = "--dry-run" in sys.argv

    os.makedirs(OUT_DIR, exist_ok=True)

    _start_time = datetime.datetime.now()
    _start_mono = time.monotonic()

    # Register lock cleanup
    atexit.register(release_lock)

    # ── Pre-flight checks (P1) ──
    if not preflight_checks():
        log("PRE-FLIGHT FAILED. Cannot start sweep.")
        sys.exit(1)

    if _dry_run:
        log("DRY RUN: Pre-flight checks passed. Not running benchmarks.")
        log(f"Would process {len(KV_TYPES) * len(DEPTHS)} configs:")
        for d in DEPTHS:
            for k in KV_TYPES:
                log(f"  {d}_{k}")
        sys.exit(0)

    # ── Load checkpoint (S5) ──
    checkpoint_completed = load_checkpoint()

    # Write results meta header
    meta = {
        "meta": {
            "tool": f"kv_prod_sweep_v{SWEEP_VERSION}",
            "timestamp": _start_time.strftime("%a %m/%d/%Y %H:%M:%S"),
            "gpu": f"GTX 1660 Super (CUDA{GPU_ID})",
            "model": MODEL,
            "safeguards": ["S1_exit9_retry", "S2_grace_periods", "S3_8081_health_gate",
                           "S4_hard_caps", "S5_checkpointing", "S6_process_verification",
                           "S7_gpu_health_preflight"]
        }
    }
    with open(RESULTS_JSONL, "w") as f:
        f.write(json.dumps(meta) + "\n")

    # ── Sweep header ──
    total_configs = len(KV_TYPES) * len(DEPTHS)
    log("=" * 60)
    log(f"PRODUCTION KV CACHE SWEEP v{SWEEP_VERSION}")
    log(f"Started: {_start_time}")
    log(f"Configs: {len(KV_TYPES)} KV types x {len(DEPTHS)} depths = {total_configs} total")
    log(f"GPU: GTX 1660 Super (CUDA{GPU_ID})")
    log(f"Max runtime: {MAX_RUNTIME_HOURS}h")
    log("=" * 60)

    all_results = {}
    sweep_aborted = False
    final_status = "COMPLETE"

    # Build config list — flatten depth × kv_type
    configs = []
    for d in DEPTHS:
        for k in KV_TYPES:
            configs.append((d, k))

    # Apply max configs cap (S4)
    if len(configs) > MAX_CONFIGS:
        log(f"Config cap ({MAX_CONFIGS}): truncating from {len(configs)} configs")
        configs = configs[:MAX_CONFIGS]

    # ── Process each config ──
    for depth, kv_type in configs:
        label = f"{depth}_{kv_type}"

        # S5: Skip if already completed in checkpoint
        if label in checkpoint_completed:
            log(f"SKIPPING {label} — already completed in checkpoint.")
            if depth not in all_results:
                all_results[depth] = {}
            all_results[depth][kv_type] = checkpoint_completed[label]
            continue

        # Process with safeguards
        entry, should_abort = process_config(depth, kv_type, checkpoint_completed)

        # Save to results
        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if depth not in all_results:
            all_results[depth] = {}
        all_results[depth][kv_type] = entry

        # S5: Save checkpoint after each config
        checkpoint_completed[label] = entry
        save_checkpoint(checkpoint_completed)

        # Apply grace period (S2)
        if entry["status"] == "OOM" or entry["status"].startswith("LASSO_KILL"):
            # GPU-related issue — longer grace
            grace_period(entry.get("exit_code", 0), is_gpu_error=True)
        elif entry.get("exit_code", 0) != 0 and entry["status"] not in ("SUCCESS", "SUCCESS_NO_DATA"):
            grace_period(entry.get("exit_code", 0), is_gpu_error=False)
        else:
            grace_period(0, is_gpu_error=False)

        # Check if we should abort
        if should_abort:
            sweep_aborted = True
            final_status = "ABORTED"
            break

        # S4: Check runtime cap
        elapsed = time.monotonic() - _start_mono
        if elapsed > MAX_RUNTIME_SECONDS:
            log(f"RUNTIME CAP: {elapsed:.0f}s exceeds {MAX_RUNTIME_HOURS}h. Aborting sweep.")
            sweep_aborted = True
            final_status = "RUNTIME_CAP"
            break

    # ── Summary ──
    elapsed = time.monotonic() - _start_mono
    log(f"\n{'=' * 60}")
    log(f"SWEEP {final_status}")
    log(f"Elapsed: {elapsed:.0f}s (max {MAX_RUNTIME_HOURS}h)")
    log(f"Configs completed: {_configs_run}/{len(configs)}")
    log(f"{'=' * 60}")

    if final_status == "ABORTED":
        log(f"\n*** SWEEP ABORTED ***")
    elif final_status == "RUNTIME_CAP":
        log(f"\n*** SWEPT TERMINATED BY RUNTIME CAP ***")
    else:
        log(f"\nSummary:")
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

    log(f"\nResults: {RESULTS_JSONL}")
    log(f"Checkpoint: {CHECKPOINT_FILE}")
    log(f"Log: {LOG_FILE}")
    log(f"\nFinished: {datetime.datetime.now()}")

    # Release lock
    release_lock()

    if sweep_aborted:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
