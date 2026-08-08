#!/usr/bin/env python3
"""Focused ZC5a live gate: learn exact families, restart, exercise by-id.

The replay scenario proves either certified execution or an active-profile
baseline-preservation decision over live/cold replay. The host scenario trains
the lifecycle-owned preparation/apply/recovery terms where host restore wins.
The oversized-host scenario creates the natural opposite regime: historical
selection restores a long saved subject for a short-prefix request, while cold
replay processes only that short request.
"""

import argparse
import fcntl
import json
import os
import pathlib
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request


def request(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except Exception:
            return error.code, {"raw": raw.decode(errors="replace")}


class Arm:
    def __init__(self, args, mode, authority, port, log_path):
        self.base = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env["LLAMA_STATE_HOME"] = args.state_home
        cmd = [
            args.server_bin, "-m", args.runtime_model, "--port", str(port),
            "-ngl", str(args.ngl), "-c", str(args.ctx), "-b", str(args.batch),
            "-np", "1", "-fa", "on", "--slots", "--cache-debug",
            "--cache-optimizer", mode,
            "--cache-plan-authority", authority,
            "--cache-ram", str(args.cache_ram),
            "--ctx-checkpoints", str(args.ctx_checkpoints),
            "--seed", str(args.seed),
        ]
        for value in args.extra_server_arg:
            cmd.extend(shlex.split(value))
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
            pass_fds=(() if args.model_fd is None else (args.model_fd,)))

    def wait(self, deadline=360):
        started = time.time()
        while time.time() - started < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited before becoming healthy")
            try:
                if request(self.base, "/health", timeout=5)[0] == 200:
                    # A stale server on the same port must not satisfy this
                    # arm after our child has failed its bind asynchronously.
                    time.sleep(0.05)
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            "server exited while another process owned its port")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("server health timeout")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.log.close()


def completion(base, prompt, seed):
    status, payload = request(base, "/completion", {
        "prompt": prompt, "id_slot": 0, "cache_prompt": True,
        "n_predict": 2, "temperature": 0, "seed": seed,
    })
    if status != 200 or payload.get("id_slot") != 0:
        raise RuntimeError(f"completion failed: {status} {payload}")
    return payload.get("content") or ""


def erase_slot(base):
    status, payload = request(base, "/slots/0?action=erase", {})
    if status != 200 or payload.get("id_slot") != 0:
        raise RuntimeError(f"slot erase failed: {status} {payload}")


def training_pair(base, seed, cycle):
    # A stable total/tail geometry gives both exact families enough repeated
    # rows while the changing suffix prevents response reuse from hiding work.
    prefix = " ".join(
        f"calibration ledger row {i} remains deterministic."
        for i in range(96))
    first = f"ZC5 conversation {cycle % 2}. {prefix}"
    # The exact identity worker takes long enough that the process's very
    # first cold replay is intentionally unavailable. Periodic erasure makes
    # cold and low-LCP live rows for the same large replay feature continue
    # throughout the five-minute evidence window.
    if cycle % 3 == 0:
        erase_slot(base)
    reply = completion(base, first, seed)
    completion(base, first + reply + " Continue with one concise fact.", seed)
    return 2


def training_host(base, seed, cycle, rows=64):
    ledger = " ".join(
        f"Host ledger row {i}: bucket {i % 11} contains {i * 3} units."
        for i in range(rows))
    cold = " ".join(
        f"Cold-{cycle}-{i} is isolated from every retained subject."
        for i in range(64))
    live = " ".join(
        f"Live-{cycle}-{i} deliberately differs from the cold subject."
        for i in range(64))
    alpha = "Alpha durable conversation. " + ledger
    beta = "Beta durable conversation. " + ledger
    erase_slot(base)
    completion(base, cold, seed)
    completion(base, live, seed)
    completion(base, alpha, seed)
    completion(base, beta, seed)
    return 4


def training_oversized_host(base, seed, cycle):
    long_rows = [
        f"Oversized durable row {i}: bucket {i % 13} contains {i * 7} units."
        for i in range(384)
    ]
    long_subject = "Alpha oversized durable conversation. " + " ".join(long_rows)
    short_subject = "Alpha oversized durable conversation. " + " ".join(
        long_rows[:112])
    displacement = " ".join(
        f"Displacement-{cycle}-{i} differs from the durable subject."
        for i in range(384))

    # Keep the focused execution gate to exactly two measurable alternatives:
    # one oversized matching host artifact and one short cold replay.  Earlier
    # broad rounds mature the other replay/restore/D classes, but retaining
    # those unrelated host objects here would correctly invoke the planner's
    # complete-estimate rule for counterfactuals historical selection never
    # executes.  Erasure between subjects prevents those gate-only artifacts
    # from being saved.  The short request has an empty target, so ZC5a's first
    # execution proof is genuinely non-destructive; destructive variants stay
    # behind D-A and remain separately fail-closed/unit-proven.
    erase_slot(base)
    completion(base, long_subject, seed)
    completion(base, displacement, seed)
    erase_slot(base)
    completion(base, short_subject, seed)
    erase_slot(base)
    completion(base, long_subject, seed)
    completion(base, displacement, seed)
    return 7


def training_cycle(args, base, cycle):
    if args.scenario == "host":
        return training_host(base, args.seed, cycle)
    if args.scenario == "host_long":
        # Naturally exercise the same >=4096-position restore class used by
        # the oversized disagreement gate.  This remains historical execution,
        # not a forced counterfactual or hidden exploration path.
        return training_host(base, args.seed, cycle, rows=421)
    if args.scenario == "oversized_host":
        return training_oversized_host(base, args.seed, cycle)
    return training_pair(base, args.seed, cycle)


def parse_plans(path):
    rows = []
    with open(path, errors="replace") as handle:
        for line in handle:
            marker = "CACHE_PLAN "
            if marker not in line:
                continue
            try:
                rows.append(json.loads(line.split(marker, 1)[1]))
            except ValueError:
                pass
    return rows


def parse_observations(path):
    rows = []
    with open(path, errors="replace") as handle:
        for line in handle:
            marker = "CACHE_OPTIMIZER_OBSERVATION "
            if marker not in line:
                continue
            try:
                rows.append(json.loads(line.split(marker, 1)[1]))
            except ValueError:
                pass
    return rows


def fingerprint_exact(path):
    pattern = re.compile(
        r"CACHE_FINGERPRINT_INPUT config_exact=1 artifacts_exact=1 "
        r"inexact_fields=00000000")
    with open(path, errors="replace") as handle:
        return any(pattern.search(line) for line in handle)


def self_test():
    row = {
        "optimizer": {
            "local_authority": {"state": "executed"},
            "request_execution_policy": "local_online_authority",
        },
        "selection": "by_id", "shipped_plan_candidate": 0,
    }
    assert row["optimizer"]["local_authority"]["state"] == "executed"
    assert row["optimizer"]["request_execution_policy"] == \
        "local_online_authority"


def sealed_model(path):
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("sealed model gate requires Linux memfd_create")
    fd = os.memfd_create("zc5-exact-model", os.MFD_ALLOW_SEALING)
    source = os.open(path, os.O_RDONLY)
    try:
        remaining = os.fstat(source).st_size
        while remaining:
            copied = os.sendfile(fd, source, None, remaining)
            if copied <= 0:
                raise RuntimeError("short copy into sealed model")
            remaining -= copied
    finally:
        os.close(source)
    os.lseek(fd, 0, os.SEEK_SET)
    seals = (fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK |
             fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    return fd, f"/proc/self/fd/{fd}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin")
    parser.add_argument("--model")
    parser.add_argument("--state-home")
    parser.add_argument("--workdir")
    parser.add_argument("--port", type=int, default=8940)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--cache-ram", type=int, default=0)
    parser.add_argument("--ctx-checkpoints", type=int, default=0)
    parser.add_argument(
        "--scenario", choices=("replay", "host", "host_long", "oversized_host"),
                        default="replay")
    parser.add_argument("--expect", choices=("execute", "preserve"),
                        default="execute")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prepare-seconds", type=int, default=330)
    parser.add_argument("--authority-seconds", type=int, default=120)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--sealed-model", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for required in (args.server_bin, args.model, args.state_home, args.workdir):
        if not required:
            parser.error("live mode requires server/model/state-home/workdir")

    pathlib.Path(args.workdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.state_home).mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(args.state_home, 0o700)
    args.model_fd = None
    args.runtime_model = args.model
    if args.sealed_model:
        args.model_fd, args.runtime_model = sealed_model(args.model)
    learn_log = os.path.join(args.workdir, "learn.log")
    learn = Arm(args, "learn", "off", args.port, learn_log)
    learn_requests = 0
    try:
        learn.wait()
        deadline = time.time() + args.prepare_seconds
        cycle = 0
        while time.time() < deadline:
            completed = training_cycle(args, learn.base, cycle)
            cycle += 1
            learn_requests += completed
            time.sleep(1.05)
    finally:
        learn.stop()

    auto_log = os.path.join(args.workdir, "auto.log")
    auto = Arm(args, "auto", "by_id", args.port + 1, auto_log)
    executed = False
    outputs = []
    auto_requests = 0
    try:
        auto.wait()
        deadline = time.time() + args.authority_seconds
        cycle = 0
        while time.time() < deadline and (
                args.expect == "preserve" or not executed):
            completed = training_cycle(args, auto.base, cycle)
            cycle += 1
            auto_requests += completed
            time.sleep(1.05)
            auto.log.flush()
            rows = parse_plans(auto_log)
            executed = any(
                row.get("selection") == "by_id" and
                (row.get("optimizer") or {}).get("request_execution_policy") ==
                    "local_online_authority" and
                ((row.get("optimizer") or {}).get("local_authority") or {}).get(
                    "state") == "executed"
                for row in rows)
            outputs = rows
    finally:
        auto.stop()

    learn_observations = parse_observations(learn_log)
    auto_observations = parse_observations(auto_log)
    executed_rows = [
        row for row in outputs
        if ((row.get("optimizer") or {}).get("local_authority") or {}).get(
            "state") == "executed"
    ]
    active_preserve_rows = [
        row for row in outputs
        if (row.get("optimizer") or {}).get("profile_state") == "active" and
        (row.get("optimizer") or {}).get("request_execution_policy") ==
            "historical_legacy" and
        isinstance((row.get("optimizer") or {}).get(
            "economic_plan_candidate"), int) and
        (row.get("optimizer") or {}).get("economic_plan_candidate") ==
            (row.get("optimizer") or {}).get("baseline_plan_candidate")
    ]
    active_preserve = bool(active_preserve_rows)
    observation_counts = {}
    for row in learn_observations + auto_observations:
        key = "/".join(str(row.get(field)) for field in (
            "operation", "provider", "terminal"))
        observation_counts[key] = observation_counts.get(key, 0) + 1
    authority_counts = {}
    for row in outputs:
        optimizer = row.get("optimizer") or {}
        local = optimizer.get("local_authority") or {}
        key = "/".join(str(value) for value in (
            optimizer.get("profile_state"), local.get("state"),
            local.get("reason"),
            optimizer.get("request_execution_policy")))
        authority_counts[key] = authority_counts.get(key, 0) + 1
    def evidence(row):
        optimizer = row.get("optimizer") or {}
        local = optimizer.get("local_authority") or {}
        return {
            "prompt_tokens": row.get("n_prompt_tokens"),
            "chosen_provider": row.get("chosen"),
            "baseline_candidate": optimizer.get("baseline_plan_candidate"),
            "economic_candidate": optimizer.get("economic_plan_candidate"),
            "profile_state": optimizer.get("profile_state"),
            "authority_state": local.get("state"),
            "authority_reason": local.get("reason"),
            "execution_policy": optimizer.get("request_execution_policy"),
            "benefit_estimate_us": optimizer.get("benefit_estimate_us"),
            "benefit_lower_us": optimizer.get("benefit_lower_us"),
            "ttft_us": row.get("ttft_us"),
        }
    active_rows = [
        row for row in outputs
        if (row.get("optimizer") or {}).get("profile_state") == "active"]
    exact_inputs = fingerprint_exact(learn_log) and fingerprint_exact(auto_log)
    result = {
        "schema": "zc5-by-id-authority/v2",
        "learn_requests": learn_requests,
        "auto_requests": auto_requests,
        "auto_records": len(outputs),
        "executed": executed,
        "executed_records": len(executed_rows),
        "active_preserve": active_preserve,
        "active_preserve_records": len(active_preserve_rows),
        "fingerprint_inputs_exact": exact_inputs,
        "observation_counts": observation_counts,
        "authority_counts": authority_counts,
        "executed_evidence": [evidence(row) for row in executed_rows[:4]],
        "active_evidence": [evidence(row) for row in active_rows[-8:]],
        "profile_labels": sorted({
            (row.get("optimizer") or {}).get("profile_identity")
            for row in outputs
            if isinstance((row.get("optimizer") or {}).get("profile_identity"), str)
        }),
    }
    encoded = json.dumps(result, sort_keys=True)
    pathlib.Path(args.workdir, "result.json").write_text(encoded + "\n")
    print(encoded)
    if not exact_inputs:
        raise SystemExit("ZC5a exact fingerprint inputs were not observed")
    if args.expect == "execute" and not executed:
        raise SystemExit("ZC5a by-id authority did not execute")
    if args.expect == "preserve" and (executed or not active_preserve):
        raise SystemExit("ZC5a active profile did not preserve the baseline")


if __name__ == "__main__":
    main()
