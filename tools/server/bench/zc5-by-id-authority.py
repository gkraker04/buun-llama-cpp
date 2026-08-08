#!/usr/bin/env python3
"""Focused ZC5 live gate: learn exact families, restart, exercise one tier.

The replay scenario proves either certified execution or an active-profile
baseline-preservation decision over live/cold replay. The host scenario trains
the lifecycle-owned preparation/apply/recovery terms where host restore wins.
The oversized-host scenario creates the natural opposite regime: historical
selection restores a long saved subject for a short-prefix request, while cold
replay processes only that short request.
The similarity crossover creates a long live continuation plus an exact
shorter host artifact: historical policy restores the host object while
calibrated authority may rewind the resident tail in place.  Route-home uses
the same nontrivial-prefix workload under dynamic unified-KV VBR, where host
artifacts are unavailable by construction and ordinary historical traffic can
honestly train both replay alternatives.  Hybrid checkpoint models retain the
counterfactual live alternative fail-closed rather than introducing hidden
exploration merely to satisfy the gate.
LRU uses that same host/live crossover with dynamic VBR disabled and a strict
similarity threshold, so ordinary unforced traffic reaches the final idle-slot
tier and retains the same persisted replay/restore evidence currencies.
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
            "--slot-prompt-similarity", str(args.slot_prompt_similarity),
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


def completion(base, prompt, seed, forced=True):
    body = {
        "prompt": prompt, "cache_prompt": True,
        "n_predict": 2, "temperature": 0, "seed": seed,
    }
    if forced:
        body["id_slot"] = 0
    status, payload = request(base, "/completion", body)
    if status != 200 or (forced and payload.get("id_slot") != 0):
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


def training_similarity_crossover(base, seed, cycle):
    rows = [
        f"Similarity durable row {i}: bucket {i % 13} contains {i * 7} units."
        for i in range(112)
    ]
    common = "Similarity durable conversation. " + " ".join(rows)
    alpha = common + " Alpha terminal remains canonical."
    long_tail = " ".join(
        f"Beta continuation {i}: branch {i % 17} remains deliberately long."
        for i in range(256))
    beta = (common + f" Beta terminal {cycle % 2} remains distinct. " +
            long_tail)
    rewind_base = (
        f"Natural rewind family {cycle}. " + " ".join(
            f"Rewind row {i}: bucket {i % 11} is stable."
            for i in range(112)))
    rewind_long = rewind_base + " " + " ".join(
        f"Discardable tail {i}: branch {i % 19}." for i in range(256))
    # Alternate around the gate's short replay tail so authority uses
    # interpolation inside observed coverage, never a one-token extrapolation.
    rewind_return = rewind_base + (
        " Return through the short branch. Extra padding."
        if cycle % 2 else " Return.")
    displacement = " ".join(
        f"Similarity displacement {cycle}-{i} is unrelated."
        for i in range(112))

    # First execute the same deep-rewind geometry naturally while no matching
    # host artifact exists.  This is ordinary historical traffic, not forced
    # counterfactual exploration, and supplies the live-replay observations
    # needed to price the later alternative.
    erase_slot(base)
    completion(base, rewind_base, seed)
    completion(base, rewind_long, seed)
    completion(base, rewind_return, seed, forced=False)

    # A->displacement publishes the exact A host artifact.  B shares nearly
    # all of A but has a tail more than twice A's extent.  The final unforced A
    # request therefore selects B by strict similarity while f_keep < 0.5,
    # which enables the shipped host lookup.  Historical policy restores A;
    # calibrated similarity authority may instead rewind B's tail in place.
    erase_slot(base)
    completion(base, alpha, seed)
    completion(base, displacement, seed)
    completion(base, beta, seed)
    completion(base, alpha, seed, forced=False)
    return 7


def training_cycle(args, base, cycle):
    if args.scenario in (
            "similarity_crossover", "route_home_crossover", "lru_crossover"):
        return training_similarity_crossover(base, args.seed, cycle)
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


def candidate(row, candidate_id):
    if not isinstance(candidate_id, int):
        return None
    for value in row.get("candidates") or []:
        if value.get("id") == candidate_id:
            return value
    return None


def tier_execution(row, tier, scenario):
    optimizer = row.get("optimizer") or {}
    local = optimizer.get("local_authority") or {}
    authority = row.get("authority") or {}
    baseline_id = optimizer.get("baseline_plan_candidate")
    economic_id = optimizer.get("economic_plan_candidate")
    if not (row.get("selection") == tier and
            optimizer.get("request_execution_policy") ==
                "local_online_authority" and
            optimizer.get("profile_state") == "active" and
            optimizer.get("economic_disposition") ==
                "certified_improvement" and
            optimizer.get("profile_resume_origin") == "persisted" and
            local.get("state") == "executed" and
            local.get("reason") == "none" and
            local.get("certified_once") is True and
            local.get("candidate") == economic_id and
            authority.get("disagreed") is True and
            isinstance(baseline_id, int) and
            isinstance(economic_id, int) and baseline_id != economic_id and
            row.get("shipped_plan_candidate") == economic_id):
        return False
    if scenario not in (
            "similarity_crossover", "route_home_crossover", "lru_crossover"):
        return True
    if scenario == "route_home_crossover":
        baseline = candidate(row, baseline_id)
        economic = candidate(row, economic_id)
        return bool(
            baseline and economic and
            economic.get("provider") == "live_slot" and
            isinstance(economic.get("lcp_tokens"), int) and
            economic.get("lcp_tokens") > 1 and
            row.get("chosen") == "live_slot")
    baseline = candidate(row, baseline_id)
    economic = candidate(row, economic_id)
    if scenario == "lru_crossover":
        return bool(
            baseline and economic and
            baseline.get("provider") == "host_cache_entry" and
            economic.get("provider") == "live_slot" and
            isinstance(baseline.get("lcp_tokens"), int) and
            baseline.get("lcp_tokens") > 1 and
            isinstance(economic.get("lcp_tokens"), int) and
            economic.get("lcp_tokens") > 1 and
            row.get("chosen") == "live_slot")
    return bool(
        baseline and economic and
        baseline.get("provider") == "host_cache_entry" and
        economic.get("provider") == "live_slot" and
        row.get("chosen") == "live_slot")


def tier_active_preserve(row, tier, scenario):
    optimizer = row.get("optimizer") or {}
    local = optimizer.get("local_authority") or {}
    authority = row.get("authority") or {}
    baseline_id = optimizer.get("baseline_plan_candidate")
    economic_id = optimizer.get("economic_plan_candidate")
    if not (row.get("selection") == tier and
            optimizer.get("profile_state") == "active" and
            optimizer.get("profile_resume_origin") == "persisted" and
            optimizer.get("request_execution_policy") == "historical_legacy" and
            optimizer.get("economic_disposition") == "refused" and
            local.get("state") == "fallback" and
            local.get("reason") == "insufficient_confidence" and
            local.get("certified_once") is False and
            authority.get("disagreed") is False and
            isinstance(economic_id, int) and economic_id == baseline_id and
            row.get("shipped_plan_candidate") == economic_id):
        return False
    if scenario != "route_home_crossover":
        return True
    economic = candidate(row, economic_id)
    return bool(
        economic and economic.get("provider") == "live_slot" and
        isinstance(economic.get("lcp_tokens"), int) and
        economic.get("lcp_tokens") > 1 and row.get("chosen") == "live_slot")


def self_test():
    row = {
        "optimizer": {
            "local_authority": {
                "state": "executed", "reason": "none",
                "certified_once": True, "candidate": 0},
            "request_execution_policy": "local_online_authority",
            "profile_state": "active",
            "economic_disposition": "certified_improvement",
            "profile_resume_origin": "persisted",
            "baseline_plan_candidate": 1,
            "economic_plan_candidate": 0,
        },
        "authority": {"disagreed": True},
        "selection": "similarity", "chosen": "live_slot",
        "shipped_plan_candidate": 0,
        "candidates": [
            {"id": 0, "provider": "live_slot", "lcp_tokens": 64},
            {"id": 1, "provider": "host_cache_entry", "lcp_tokens": 65},
        ],
    }
    assert tier_execution(row, "similarity", "similarity_crossover")
    row["selection"] = "route_home"
    assert tier_execution(row, "route_home", "route_home_crossover")
    row["selection"] = "lru"
    assert tier_execution(row, "lru", "lru_crossover")
    row["candidates"][1]["lcp_tokens"] = 1
    assert not tier_execution(row, "lru", "lru_crossover")
    row["candidates"][1]["lcp_tokens"] = 65
    row["candidates"][0]["lcp_tokens"] = 1
    assert not tier_execution(row, "lru", "lru_crossover")
    row["candidates"][0]["lcp_tokens"] = 64
    del row["candidates"][1]["lcp_tokens"]
    assert not tier_execution(row, "lru", "lru_crossover")
    row["candidates"][1]["lcp_tokens"] = 65
    row["selection"] = "route_home"
    row["authority"]["disagreed"] = False
    assert not tier_execution(row, "route_home", "route_home_crossover")
    row["authority"]["disagreed"] = True
    row["shipped_plan_candidate"] = 1
    assert not tier_execution(row, "route_home", "route_home_crossover")
    row["shipped_plan_candidate"] = 0
    row["optimizer"]["baseline_plan_candidate"] = 0
    assert not tier_execution(row, "route_home", "route_home_crossover")
    row["optimizer"]["baseline_plan_candidate"] = 1
    row["candidates"][0]["lcp_tokens"] = 1
    assert not tier_execution(row, "route_home", "route_home_crossover")
    row["optimizer"]["local_authority"] = {
        "state": "fallback", "reason": "insufficient_confidence",
        "certified_once": False}
    row["optimizer"]["profile_state"] = "active"
    row["optimizer"]["request_execution_policy"] = "historical_legacy"
    row["optimizer"]["economic_disposition"] = "refused"
    row["optimizer"]["baseline_plan_candidate"] = 0
    row["authority"]["disagreed"] = False
    row["candidates"][0]["lcp_tokens"] = 64
    assert tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["optimizer"]["profile_resume_origin"] = "current_process"
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["optimizer"]["profile_resume_origin"] = "persisted"
    row["shipped_plan_candidate"] = 1
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["shipped_plan_candidate"] = 0
    row["optimizer"]["local_authority"]["certified_once"] = True
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["optimizer"]["local_authority"]["certified_once"] = False
    row["optimizer"]["local_authority"]["reason"] = "currency_changed"
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["optimizer"]["local_authority"]["reason"] = "insufficient_confidence"
    row["optimizer"]["economic_disposition"] = "refused_internal"
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["optimizer"]["economic_disposition"] = "refused"
    row["authority"]["disagreed"] = True
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")
    row["authority"]["disagreed"] = False
    row["candidates"][0]["lcp_tokens"] = 1
    assert not tier_active_preserve(
        row, "route_home", "route_home_crossover")


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
        "--scenario", choices=("replay", "host", "host_long", "oversized_host",
                               "similarity_crossover", "route_home_crossover",
                               "lru_crossover"),
                        default="replay")
    parser.add_argument(
        "--decision-tier", choices=("by_id", "similarity", "route_home", "lru"),
        default="by_id")
    parser.add_argument("--slot-prompt-similarity", type=float, default=0.1)
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
    auto = Arm(args, "auto", args.decision_tier, args.port + 1, auto_log)
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
            executed = any(tier_execution(
                row, args.decision_tier, args.scenario) for row in rows)
            outputs = rows
    finally:
        auto.stop()

    learn_observations = parse_observations(learn_log)
    auto_observations = parse_observations(auto_log)
    executed_rows = [
        row for row in outputs
        if tier_execution(row, args.decision_tier, args.scenario)
    ]
    active_preserve_rows = [
        row for row in outputs
        if tier_active_preserve(
            row, args.decision_tier, args.scenario)
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
        baseline = candidate(row, optimizer.get("baseline_plan_candidate"))
        economic = candidate(row, optimizer.get("economic_plan_candidate"))
        return {
            "selection": row.get("selection"),
            "prompt_tokens": row.get("n_prompt_tokens"),
            "chosen_provider": row.get("chosen"),
            "disagreed": (row.get("authority") or {}).get("disagreed"),
            "baseline_candidate": optimizer.get("baseline_plan_candidate"),
            "baseline_provider": baseline.get("provider") if baseline else None,
            "baseline_lcp_tokens": baseline.get("lcp_tokens") if baseline else None,
            "economic_candidate": optimizer.get("economic_plan_candidate"),
            "economic_provider": economic.get("provider") if economic else None,
            "economic_lcp_tokens": economic.get("lcp_tokens") if economic else None,
            "shipped_plan_candidate": row.get("shipped_plan_candidate"),
            "profile_state": optimizer.get("profile_state"),
            "profile_resume_origin": optimizer.get("profile_resume_origin"),
            "economic_disposition": optimizer.get("economic_disposition"),
            "authority_state": local.get("state"),
            "authority_reason": local.get("reason"),
            "certified_once": local.get("certified_once"),
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
        "schema": "zc5-local-authority/v3",
        "decision_tier": args.decision_tier,
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
        raise SystemExit("ZC5 exact fingerprint inputs were not observed")
    if args.expect == "execute" and not executed:
        raise SystemExit(f"ZC5 {args.decision_tier} authority did not execute")
    if args.expect == "preserve" and (executed or not active_preserve):
        raise SystemExit("ZC5 active profile did not preserve the baseline")


if __name__ == "__main__":
    main()
