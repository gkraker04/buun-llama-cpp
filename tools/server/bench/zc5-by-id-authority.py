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
import copy
import fcntl
import hashlib
import json
import math
import os
import pathlib
import random
import re
import shlex
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request


PAIRED_ARM_ORDERS = (
    ("auto", "learn", "baseline"),
    ("learn", "baseline", "auto"),
    ("baseline", "auto", "learn"),
    ("auto", "baseline", "learn"),
    ("baseline", "learn", "auto"),
    ("learn", "auto", "baseline"),
)

ZC6_CAPSTONE_LABELS = frozenset({
    "small-static-concurrent", "small-static-serialized",
    "small-vbr-concurrent", "small-vbr-serialized",
    "large-static-concurrent", "large-static-serialized",
    "large-vbr-concurrent", "large-vbr-serialized",
})

REQUEST_CURRENCY_FIELDS = (
    "schema_version", "id_task", "id_slot", "selection", "n_prompt_tokens",
    "calibration_profile",
)

REQUEST_IDENTITY_FIELDS = (
    # execution is deliberately absent: it is process-random frontier identity.
    "model", "adapter_config", "media_content", "tokenizer_template",
    "prefix_tokens",
)

CANDIDATE_CURRENCY_FIELDS = (
    "id", "target_slot_id", "origin_tier", "provider", "lcp_tokens",
    "payload_bytes", "source_id", "component_only", "sim", "f_keep",
    "spec_capable", "siblings_scanned",
    "siblings_rejected_epoch", "is_chain", "components", "generation_eval",
    "phases",
)


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
    def __init__(self, args, mode, authority, port, log_path, state_home=None):
        self.base = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env["LLAMA_STATE_HOME"] = state_home or args.state_home
        cmd = [
            args.server_bin, "-m", args.runtime_model, "--port", str(port),
            "-ngl", str(args.ngl), "-c", str(args.ctx), "-b", str(args.batch),
            "-np", "1", "-fa", "on", "--slots", "--cache-debug",
            "--cache-plan-authority", authority,
            "--cache-ram", str(args.cache_ram),
            "--ctx-checkpoints", str(args.ctx_checkpoints),
            "--slot-prompt-similarity", str(args.slot_prompt_similarity),
            "--seed", str(args.seed),
        ]
        if mode is not None:
            cmd.extend(["--cache-optimizer", mode])
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


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def bootstrap_median_lower(values, iterations=10000, seed=0x5A436):
    if not values:
        raise ValueError("empty authority benefit sample")
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(statistics.median(
            values[rng.randrange(len(values))] for _ in values))
    return percentile(samples, 0.05)


def request_currency(row):
    identity = row.get("identity") or {}
    return (
        tuple((field, row.get(field)) for field in REQUEST_CURRENCY_FIELDS),
        tuple((field, identity.get(field)) for field in REQUEST_IDENTITY_FIELDS),
    )


def candidate_currency(value):
    if not isinstance(value, dict):
        raise RuntimeError("authority counterfactual candidate is missing")
    return tuple(
        (field, tuple(value.get(field))
         if isinstance(value.get(field), list) else value.get(field))
        for field in CANDIDATE_CURRENCY_FIELDS)


def inventory_currency(row):
    values = row.get("candidates")
    if not isinstance(values, list) or not values:
        raise RuntimeError("authority counterfactual inventory is missing")
    return tuple(candidate_currency(value) for value in values)


def recency_order(row, candidate_ids=None):
    values = row.get("candidates") or []
    allowed = set(candidate_ids) if candidate_ids is not None else None
    ranked = [value for value in values
              if isinstance(value.get("t_last_used_us"), int) and
              (allowed is None or value.get("id") in allowed)]
    return tuple(value.get("id") for value in sorted(
        ranked, key=lambda value: (value["t_last_used_us"], value.get("id"))))


def projected_inventory_currency(row, candidate_ids):
    by_id = {value.get("id"): value for value in row.get("candidates") or []}
    if any(candidate_id not in by_id for candidate_id in candidate_ids):
        raise RuntimeError("authority counterfactual candidate disappeared")
    return (
        tuple(candidate_currency(by_id[candidate_id])
              for candidate_id in candidate_ids),
        recency_order(row, candidate_ids),
    )


def postdecision_publication_extras(row, candidate_ids, reference_row):
    expected = set(candidate_ids)
    reference_live = [value for value in reference_row.get("candidates") or []
                      if value.get("provider") == "live_slot"]
    extras = [value for value in row.get("candidates") or []
              if value.get("id") not in expected]
    for extra in extras:
        if (extra.get("provider") != "host_cache_entry" or
                tuple(extra.get("phases") or ()) != ("host_scan",) or
                not isinstance(extra.get("id"), int) or
                any(not isinstance(candidate_id, int) or
                    extra["id"] <= candidate_id for candidate_id in expected)):
            return False
        if not any(
                extra.get("target_slot_id") == live.get("target_slot_id") and
                extra.get("lcp_tokens") == live.get("lcp_tokens") and
                extra.get("sim") == live.get("sim") and
                extra.get("f_keep") == live.get("f_keep")
                for live in reference_live):
            return False
    return True


def prior_trajectory_currency(rows, stop):
    out = []
    for row in rows[:stop]:
        shipped = row.get("shipped_plan_candidate")
        shipped_row = candidate(row, shipped)
        if shipped_row is None:
            raise RuntimeError("authority prior shipped candidate is missing")
        out.append((
            request_currency(row), row.get("selection"), row.get("chosen"),
            shipped, inventory_currency(row),
            recency_order(row),
        ))
    return tuple(out)


ZC6_ADJUDICATED_FAILURES = frozenset({
    ("large-static-concurrent", "planner_tax", "p50"),
    ("large-static-concurrent", "total_tax", "p50"),
})


def validate_fresh_report_reductions(value):
    failures = set()
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("fresh overhead report cells are missing")
    family_passes = {family: True for family in (
        "learning_tax", "planner_tax", "total_tax")}
    schedule_pass = True
    resource_pass = True
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("fresh overhead report cell is invalid")
        label = cell.get("label")
        taxes = cell.get("taxes")
        if not isinstance(taxes, dict) or set(taxes) != set(family_passes):
            raise ValueError(f"fresh overhead tax families are incomplete for {label}")
        for family in family_passes:
            endpoints = taxes[family]
            if not isinstance(endpoints, dict) or set(endpoints) != {"p50", "p95"}:
                raise ValueError(f"fresh overhead tax endpoints are incomplete for {label}")
            for quantile in ("p50", "p95"):
                endpoint = endpoints[quantile]
                if not isinstance(endpoint, dict) or not isinstance(
                        endpoint.get("holm_pass"), bool):
                    raise ValueError(f"fresh overhead tax result is invalid for {label}")
                if endpoint["holm_pass"] is False:
                    failures.add((label, family, quantile))
                    family_passes[family] = False
        resources = cell.get("resources")
        fresh = cell.get("fresh")
        if (not isinstance(resources, dict) or
                not isinstance(resources.get("pass"), bool) or
                not isinstance(fresh, dict) or
                not isinstance(fresh.get("pass"), bool)):
            raise ValueError(f"fresh overhead cell reductions are invalid for {label}")
        resource_pass = resource_pass and resources["pass"]
        schedule_pass = schedule_pass and fresh["pass"]
    reductions = {
        "learning_tax_passed": family_passes["learning_tax"],
        "planner_tax_passed": family_passes["planner_tax"],
        "total_tax_passed": family_passes["total_tax"],
        "schedule_passed": schedule_pass,
        "resource_passed": resource_pass,
    }
    reductions["passed"] = all(reductions.values())
    reductions["noninferiority_passed"] = reductions["passed"]
    for field, expected in reductions.items():
        if value.get(field) is not expected:
            raise ValueError(f"fresh overhead report reduction disagrees for {field}")
    return frozenset(failures)


def validate_fresh_overhead_report(value, required_cell, adjudication=None,
                                   report_sha256=None):
    if not isinstance(value, dict) or value.get("schema") != "zc6-capstone-report/v1":
        raise ValueError("fresh overhead report schema is invalid")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("fresh overhead report cells are missing")
    labels = [cell.get("label") for cell in cells if isinstance(cell, dict)]
    if (len(labels) != len(ZC6_CAPSTONE_LABELS) or
            len(set(labels)) != len(labels) or
            set(labels) != ZC6_CAPSTONE_LABELS):
        raise ValueError("fresh overhead report cell census is not exact")
    report_failures = validate_fresh_report_reductions(value)
    formal_pass = all(value.get(field) is True for field in (
        "passed", "noninferiority_passed", "learning_tax_passed",
        "planner_tax_passed", "total_tax_passed", "schedule_passed",
        "resource_passed"))
    if not formal_pass:
        if not isinstance(adjudication, dict):
            raise ValueError("fresh overhead report requires an adjudication")
        if adjudication.get("schema") != "zc6-capstone-adjudication/v1":
            raise ValueError("fresh overhead adjudication schema is invalid")
        if adjudication.get("decision") != "accept_bounded_inconclusive_cell":
            raise ValueError("fresh overhead adjudication decision is invalid")
        if (not isinstance(report_sha256, str) or len(report_sha256) != 64 or
                adjudication.get("report_sha256") != report_sha256):
            raise ValueError("fresh overhead adjudication report digest mismatched")
        if adjudication.get("required_efficacy_cell") != required_cell:
            raise ValueError("fresh overhead adjudication cell mismatched")
        encoded_failures = adjudication.get("accepted_failed_endpoints")
        if not isinstance(encoded_failures, list):
            raise ValueError("fresh overhead adjudication failures are missing")
        accepted_failures = frozenset(
            tuple(item) for item in encoded_failures
            if isinstance(item, list) and len(item) == 3)
        if (accepted_failures != ZC6_ADJUDICATED_FAILURES or
                report_failures != ZC6_ADJUDICATED_FAILURES):
            raise ValueError("fresh overhead adjudication failure set mismatched")
        for field in ("learning_tax_passed", "schedule_passed", "resource_passed"):
            if value.get(field) is not True:
                raise ValueError(f"adjudicated fresh overhead report {field} is not true")
        if (value.get("passed") is not False or
                value.get("noninferiority_passed") is not False or
                value.get("planner_tax_passed") is not False or
                value.get("total_tax_passed") is not False):
            raise ValueError("adjudicated fresh overhead report disposition changed")
        matched = next((cell for cell in cells
                        if cell.get("label") == required_cell), None)
        if (matched is None or (matched.get("resources") or {}).get("pass") is not True or
                (matched.get("fresh") or {}).get("pass") is not True):
            raise ValueError("adjudicated efficacy cell evidence is not green")
        for endpoints in (matched.get("taxes") or {}).values():
            if not isinstance(endpoints, dict) or not endpoints:
                raise ValueError("adjudicated efficacy cell tax evidence is missing")
            if any(not isinstance(endpoint, dict) or endpoint.get("holm_pass") is not True
                   for endpoint in endpoints.values()):
                raise ValueError("adjudicated efficacy cell tax evidence is not green")
    if required_cell not in ZC6_CAPSTONE_LABELS:
        raise ValueError("fresh overhead cell is not a frozen ZC6 cell")
    overhead_by_cell = value.get("fresh_overhead_by_cell_us")
    if (not isinstance(overhead_by_cell, dict) or
            set(overhead_by_cell) != ZC6_CAPSTONE_LABELS):
        raise ValueError("fresh overhead per-cell census is not exact")
    for label, overhead in overhead_by_cell.items():
        if (not isinstance(overhead, (int, float)) or isinstance(overhead, bool) or
                not math.isfinite(overhead) or overhead < 0):
            raise ValueError(f"fresh overhead bound is invalid for {label}")
    overhead = overhead_by_cell[required_cell]
    if (not isinstance(overhead, (int, float)) or isinstance(overhead, bool) or
            not math.isfinite(overhead) or overhead < 0):
        raise ValueError("fresh overhead report upper bound is invalid")
    return float(overhead)


def load_fresh_overhead_report(path, required_cell, adjudication_path=None):
    try:
        report_bytes = pathlib.Path(path).read_bytes()
        value = json.loads(report_bytes)
        adjudication = (json.loads(pathlib.Path(adjudication_path).read_text())
                        if adjudication_path is not None else None)
    except (OSError, ValueError) as error:
        raise ValueError("fresh overhead report is unreadable") from error
    return validate_fresh_overhead_report(
        value, required_cell, adjudication,
        hashlib.sha256(report_bytes).hexdigest())


def paired_authority_benefits(auto_rows, learn_rows, baseline_rows,
                              tier, scenario, cycle_latency_us=None):
    if not (len(auto_rows) == len(learn_rows) == len(baseline_rows)):
        raise RuntimeError("authority counterfactual record count diverged")
    pairs = []
    for index, auto in enumerate(auto_rows):
        if not tier_execution(auto, tier, scenario):
            continue
        learn = learn_rows[index]
        baseline = baseline_rows[index]
        baseline_candidate = (auto.get("optimizer") or {}).get(
            "baseline_plan_candidate")
        auto_baseline = candidate(auto, baseline_candidate)
        if auto_baseline is None:
            raise RuntimeError("certified baseline candidate is missing")
        expected_request = request_currency(auto)
        expected_ids = tuple(value.get("id") for value in auto.get("candidates") or [])
        expected_inventory = projected_inventory_currency(auto, expected_ids)
        expected_candidate = candidate_currency(auto_baseline)
        expected_trajectory = prior_trajectory_currency(auto_rows, index)
        for name, counterfactual in (
                ("learn", learn), ("baseline", baseline)):
            counterfactual_baseline = (counterfactual.get("optimizer") or {}).get(
                "baseline_plan_candidate")
            counterfactual_legacy = (counterfactual.get("authority") or {}).get(
                "legacy_plan_candidate")
            counterfactual_row = candidate(counterfactual, baseline_candidate)
            if (request_currency(counterfactual) != expected_request or
                    prior_trajectory_currency(
                        learn_rows if name == "learn" else baseline_rows,
                        index) != expected_trajectory or
                    projected_inventory_currency(
                        counterfactual, expected_ids) != expected_inventory or
                    not postdecision_publication_extras(
                        counterfactual, expected_ids, auto) or
                    counterfactual_baseline != baseline_candidate or
                    counterfactual_legacy != baseline_candidate or
                    counterfactual.get("shipped_plan_candidate") != baseline_candidate or
                    candidate_currency(counterfactual_row) != expected_candidate):
                raise RuntimeError(
                    f"authority {name} counterfactual currency diverged")
        ttft = [row.get("ttft_us") for row in (auto, learn, baseline)]
        if not all(isinstance(value, int) and value >= 0 for value in ttft):
            raise RuntimeError("authority counterfactual is missing TTFT evidence")
        latency = ttft
        if cycle_latency_us is not None:
            if (len(cycle_latency_us) != 3 or
                    not all(values for values in cycle_latency_us)):
                raise RuntimeError("authority counterfactual cycle timing is missing")
            latency = [values[-1] for values in cycle_latency_us]
        pairs.append({
            "record_index": index,
            "auto_provider_ttft_us": ttft[0],
            "learn_provider_ttft_us": ttft[1],
            "baseline_provider_ttft_us": ttft[2],
            "auto_request_cycle_us": latency[0],
            "learn_request_cycle_us": latency[1],
            "baseline_request_cycle_us": latency[2],
            "gross_benefit_us": latency[1] - latency[0],
            "net_benefit_us": latency[2] - latency[0],
        })
    return pairs


def self_test():
    row = {
        "schema_version": 7,
        "id_task": 40,
        "id_slot": 0,
        "identity": {
            "model": 11, "execution": 101, "adapter_config": 12,
            "media_content": "unknown", "tokenizer_template": "unknown",
            "prefix_tokens": 13,
        },
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
    row = {
        "schema_version": 7,
        "id_task": 40,
        "id_slot": 0,
        "identity": {
            "model": 11, "execution": 101, "adapter_config": 12,
            "media_content": "unknown", "tokenizer_template": "unknown",
            "prefix_tokens": 13,
        },
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
        "authority": {"disagreed": True, "legacy_plan_candidate": 1},
        "selection": "lru", "chosen": "live_slot",
        "n_prompt_tokens": 65,
        "calibration_profile": "unit-profile",
        "shipped_plan_candidate": 0, "ttft_us": 1000,
        "candidates": [
            {"id": 0, "provider": "live_slot", "source_id": 0,
             "target_slot_id": 0, "origin_tier": "lru", "lcp_tokens": 64,
             "payload_bytes": "unknown", "phases": ["lru_scan"]},
            {"id": 1, "provider": "host_cache_entry", "source_id": 7,
             "target_slot_id": 0, "origin_tier": "lru", "lcp_tokens": 65,
             "payload_bytes": 4096, "phases": ["host_scan"]},
        ],
    }
    counterfactual = copy.deepcopy(row)
    counterfactual["identity"]["execution"] = 202
    counterfactual["shipped_plan_candidate"] = 1
    counterfactual["ttft_us"] = 4000
    pairs = paired_authority_benefits(
        [row], [counterfactual], [counterfactual], "lru", "lru_crossover")
    assert pairs[0]["gross_benefit_us"] == 3000
    assert pairs[0]["net_benefit_us"] == 3000
    assert bootstrap_median_lower([3000] * 8, 128) == 3000
    def counterfactual_must_fail(bad):
        try:
            paired_authority_benefits(
                [row], [bad], [counterfactual], "lru", "lru_crossover")
            raise AssertionError("mismatched counterfactual unexpectedly accepted")
        except RuntimeError:
            pass
    bad = copy.deepcopy(counterfactual)
    bad["shipped_plan_candidate"] = 0
    counterfactual_must_fail(bad)
    for field, value in (
            ("provider", "live_slot"), ("source_id", 8),
            ("target_slot_id", 1), ("lcp_tokens", 64)):
        bad = copy.deepcopy(counterfactual)
        bad["candidates"][1][field] = value
        counterfactual_must_fail(bad)
    bad = copy.deepcopy(counterfactual)
    bad["identity"]["prefix_tokens"] = 99
    counterfactual_must_fail(bad)
    bad = copy.deepcopy(counterfactual)
    bad["candidates"].append({"id": 2, "provider": "cold_replay"})
    counterfactual_must_fail(bad)

    # The scored cycle includes every pre-final request.  An unshipped
    # candidate change there is still causal divergence and must fail pairing.
    prior_auto = copy.deepcopy(counterfactual)
    prior_auto["optimizer"]["local_authority"]["state"] = "refused"
    prior_auto["ttft_us"] = 2000
    prior_counterfactual = copy.deepcopy(counterfactual)
    prior_counterfactual["optimizer"]["local_authority"]["state"] = "refused"
    assert paired_authority_benefits(
        [prior_auto, row],
        [prior_counterfactual, counterfactual],
        [prior_counterfactual, counterfactual],
        "lru", "lru_crossover")
    bad_prior = copy.deepcopy(prior_counterfactual)
    bad_prior["candidates"][0].update(
        provider="cold_replay", source_id=99, lcp_tokens=1)
    try:
        paired_authority_benefits(
            [prior_auto, row],
            [bad_prior, counterfactual],
            [prior_counterfactual, counterfactual],
            "lru", "lru_crossover")
        raise AssertionError("changed pre-final inventory unexpectedly accepted")
    except RuntimeError:
        pass

    assert len(set(PAIRED_ARM_ORDERS)) == len(PAIRED_ARM_ORDERS) == 6
    assert all(set(order) == {"auto", "learn", "baseline"}
               for order in PAIRED_ARM_ORDERS)
    assert all({order[position] for order in PAIRED_ARM_ORDERS} ==
               {"auto", "learn", "baseline"} for position in range(3))

    report = {
        "schema": "zc6-capstone-report/v1",
        "passed": True,
        "noninferiority_passed": True,
        "learning_tax_passed": True,
        "planner_tax_passed": True,
        "total_tax_passed": True,
        "schedule_passed": True,
        "resource_passed": True,
        "fresh_overhead_upper_us": 1200.0,
        "fresh_overhead_by_cell_us": {
            label: (600.0 if label == "small-static-serialized" else 1200.0)
            for label in ZC6_CAPSTONE_LABELS
        },
        "cells": [{
            "label": label,
            "resources": {"pass": True},
            "fresh": {"pass": True},
            "taxes": {family: {
                quantile: {"holm_pass": True}
                for quantile in ("p50", "p95")}
                for family in ("learning_tax", "planner_tax", "total_tax")},
        } for label in sorted(ZC6_CAPSTONE_LABELS)],
    }
    assert validate_fresh_overhead_report(
        report, "small-static-serialized") == 600.0
    for mutation in (
            lambda value: value.update(schema="wrong"),
            lambda value: value.update(planner_tax_passed=False),
            lambda value: value["fresh_overhead_by_cell_us"].update(
                {"small-static-serialized": float("nan")}),
            lambda value: value["fresh_overhead_by_cell_us"].pop(
                "small-static-serialized"),
            lambda value: value["cells"].pop(),
            lambda value: value["cells"].append(copy.deepcopy(value["cells"][0])),
            lambda value: value["cells"][0]["taxes"].clear(),
            lambda value: value["cells"][0]["taxes"]["learning_tax"]
                .pop("p95"),
            lambda value: value["cells"][0]["taxes"]["learning_tax"]
                ["p50"].update(holm_pass=False),
    ):
        bad_report = copy.deepcopy(report)
        mutation(bad_report)
        try:
            validate_fresh_overhead_report(
                bad_report, "small-static-serialized")
            raise AssertionError("tampered fresh overhead report unexpectedly accepted")
        except ValueError:
            pass
    adjudicated = copy.deepcopy(report)
    adjudicated.update(
        passed=False, noninferiority_passed=False,
        planner_tax_passed=False, total_tax_passed=False)
    failed_cell = next(cell for cell in adjudicated["cells"]
                       if cell["label"] == "large-static-concurrent")
    failed_cell["taxes"]["planner_tax"]["p50"]["holm_pass"] = False
    failed_cell["taxes"]["total_tax"]["p50"]["holm_pass"] = False
    adjudication = {
        "schema": "zc6-capstone-adjudication/v1",
        "decision": "accept_bounded_inconclusive_cell",
        "report_sha256": "a" * 64,
        "required_efficacy_cell": "small-static-serialized",
        "accepted_failed_endpoints": [list(item)
                                      for item in sorted(ZC6_ADJUDICATED_FAILURES)],
    }
    assert validate_fresh_overhead_report(
        adjudicated, "small-static-serialized", adjudication,
        "a" * 64) == 600.0
    for mutation in (
            lambda value: value["cells"][0]["taxes"]["learning_tax"].pop("p95"),
            lambda value: value["cells"][1].update(taxes={}),
    ):
        incomplete = copy.deepcopy(adjudicated)
        mutation(incomplete)
        try:
            validate_fresh_overhead_report(
                incomplete, "small-static-serialized", adjudication,
                "a" * 64)
            raise AssertionError("incomplete adjudicated report accepted")
        except ValueError:
            pass
    for mutation in (
            lambda value: value.update(report_sha256="b" * 64),
            lambda value: value.update(required_efficacy_cell="large-static-serialized"),
            lambda value: value["accepted_failed_endpoints"].pop(),
    ):
        bad_adjudication = copy.deepcopy(adjudication)
        mutation(bad_adjudication)
        try:
            validate_fresh_overhead_report(
                adjudicated, "small-static-serialized", bad_adjudication,
                "a" * 64)
            raise AssertionError("tampered fresh overhead adjudication accepted")
        except ValueError:
            pass
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


def copy_state(source, destination):
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    destination.chmod(0o700)
    return str(destination)


def run_fixed_cycles(args, name, mode, authority, port, log_path,
                     state_home, cycles):
    arm = Arm(args, mode, authority, port, log_path, state_home)
    cycle_latency_us = []
    try:
        arm.wait()
        for cycle in range(cycles):
            started = time.perf_counter_ns()
            training_cycle(args, arm.base, cycle)
            cycle_latency_us.append(
                (time.perf_counter_ns() - started + 999) // 1000)
            time.sleep(1.05)
            arm.log.flush()
    finally:
        arm.stop()
    return parse_plans(log_path), cycle_latency_us


def run_until_execution(args, name, mode, authority, port, log_path,
                        state_home, deadline_seconds):
    arm = Arm(args, mode, authority, port, log_path, state_home)
    rows = []
    requests = 0
    cycles = 0
    cycle_latency_us = []
    try:
        arm.wait()
        deadline = time.time() + deadline_seconds
        while time.time() < deadline and not any(
                tier_execution(row, args.decision_tier, args.scenario)
                for row in rows):
            started = time.perf_counter_ns()
            requests += training_cycle(args, arm.base, cycles)
            cycle_latency_us.append(
                (time.perf_counter_ns() - started + 999) // 1000)
            cycles += 1
            time.sleep(1.05)
            arm.log.flush()
            rows = parse_plans(log_path)
    finally:
        arm.stop()
    return rows, requests, cycles, cycle_latency_us


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
    parser.add_argument("--default-auto", action="store_true",
                        help="deprecated: omitted mode now means off")
    parser.add_argument("--paired-counterfactual", action="store_true",
                        help="compare copied persisted auto/learn/baseline arms")
    parser.add_argument("--paired-executions", type=int, default=8)
    parser.add_argument("--fresh-overhead-report", type=pathlib.Path)
    parser.add_argument("--fresh-overhead-adjudication", type=pathlib.Path)
    parser.add_argument("--fresh-overhead-cell", choices=sorted(ZC6_CAPSTONE_LABELS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.default_auto:
        parser.error(
            "--default-auto is retired because omitted mode now means off; "
            "the authority arm always uses explicit auto")
    for required in (args.server_bin, args.model, args.state_home, args.workdir):
        if not required:
            parser.error("live mode requires server/model/state-home/workdir")
    fresh_overhead_upper_us = None
    if args.paired_counterfactual:
        if args.expect != "execute":
            parser.error("paired counterfactual requires --expect execute")
        if args.paired_executions < 8:
            parser.error("paired counterfactual requires at least 8 executions")
        if args.fresh_overhead_report is None:
            parser.error(
                "paired counterfactual requires --fresh-overhead-report")
        if args.fresh_overhead_cell is None:
            parser.error(
                "paired counterfactual requires --fresh-overhead-cell")
        try:
            fresh_overhead_upper_us = load_fresh_overhead_report(
                args.fresh_overhead_report, args.fresh_overhead_cell,
                args.fresh_overhead_adjudication)
        except ValueError as error:
            parser.error(str(error))

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

    state_homes = {}
    if args.paired_counterfactual:
        state_homes["auto"] = copy_state(
            args.state_home,
            pathlib.Path(args.workdir, "persisted-auto-state"))

    auto_log = os.path.join(args.workdir, "auto.log")
    if args.paired_counterfactual:
        outputs, auto_requests, authority_cycles, _ = run_until_execution(
            args, "auto", "auto",
            args.decision_tier, args.port + 1, auto_log,
            state_homes["auto"], args.authority_seconds)
    else:
        auto = Arm(args, "auto",
                   args.decision_tier, args.port + 1, auto_log)
        outputs = []
        auto_requests = 0
        authority_cycles = 0
        try:
            auto.wait()
            deadline = time.time() + args.authority_seconds
            while time.time() < deadline and (
                    args.expect == "preserve" or not any(
                        tier_execution(row, args.decision_tier, args.scenario)
                        for row in outputs)):
                auto_requests += training_cycle(
                    args, auto.base, authority_cycles)
                authority_cycles += 1
                time.sleep(1.05)
                auto.log.flush()
                outputs = parse_plans(auto_log)
        finally:
            auto.stop()
    executed = any(tier_execution(
        row, args.decision_tier, args.scenario) for row in outputs)

    authority_pairs = []
    if args.paired_counterfactual and executed:
        # Discovery above advances and cleanly flushes the persisted profile.
        # Every efficacy sample below starts from a new copy of that one exact
        # mature state, so an executed decision cannot alter the next sample.
        mature_state = state_homes["auto"]
        trials = pathlib.Path(args.workdir, "paired-trials")
        trials.mkdir(mode=0o700, exist_ok=True)
        for trial in range(args.paired_executions):
            trial_root = trials / f"trial-{trial:02d}"
            trial_root.mkdir(mode=0o700)
            auto_state = copy_state(mature_state, trial_root / "auto-state")
            learn_state = copy_state(mature_state, trial_root / "learn-state")
            baseline_state = copy_state(
                mature_state, trial_root / "baseline-state")
            order = PAIRED_ARM_ORDERS[trial % len(PAIRED_ARM_ORDERS)]
            arm_specs = {
                "auto": ("auto",
                         args.decision_tier, args.port + 1, auto_state),
                "learn": ("learn", "off", args.port + 2, learn_state),
                "baseline": ("baseline", "off", args.port + 3,
                             baseline_state),
            }
            trial_rows = {}
            trial_latencies = {}
            # Discovery fixed the number of cycles before the paired sample.
            # Rotate all six arm orders so startup/thermal position cannot be
            # mistaken for policy benefit; no trial adapts to its own outcome.
            for arm_name in order:
                mode, authority, port, state = arm_specs[arm_name]
                rows, latencies = run_fixed_cycles(
                    args, f"trial-{trial}-{arm_name}", mode, authority, port,
                    str(trial_root / f"{arm_name}.log"), state,
                    authority_cycles)
                trial_rows[arm_name] = rows
                trial_latencies[arm_name] = latencies
            trial_auto = trial_rows["auto"]
            if not any(tier_execution(
                    row, args.decision_tier, args.scenario)
                    for row in trial_auto):
                raise RuntimeError("paired auto trial did not execute")
            matched = paired_authority_benefits(
                trial_auto, trial_rows["learn"], trial_rows["baseline"],
                args.decision_tier, args.scenario,
                (trial_latencies["auto"], trial_latencies["learn"],
                 trial_latencies["baseline"]))
            if len(matched) != 1:
                raise RuntimeError(
                    "paired trial must contain exactly one executed disagreement")
            matched[0]["trial"] = trial
            matched[0]["arm_order"] = list(order)
            authority_pairs.extend(matched)

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
    gross_lower_us = None
    net_lower_us = None
    headroom_required_us = None
    if args.paired_counterfactual:
        gross = [row["gross_benefit_us"] for row in authority_pairs]
        net = [row["net_benefit_us"] for row in authority_pairs]
        gross_lower_us = bootstrap_median_lower(gross)
        net_lower_us = bootstrap_median_lower(net)
        headroom_required_us = fresh_overhead_upper_us + 500.0
    result = {
        "schema": "zc5-local-authority/v3",
        "decision_tier": args.decision_tier,
        "default_auto": False,
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
        "paired_counterfactual": args.paired_counterfactual,
        "paired_executions": len(authority_pairs),
        "paired_evidence": authority_pairs[:16],
        "gross_benefit_median_us": (
            statistics.median(row["gross_benefit_us"]
                              for row in authority_pairs)
            if authority_pairs else None),
        "gross_benefit_lower_us": gross_lower_us,
        "net_benefit_median_us": (
            statistics.median(row["net_benefit_us"]
                              for row in authority_pairs)
            if authority_pairs else None),
        "net_benefit_lower_us": net_lower_us,
        "fresh_overhead_report": (
            str(args.fresh_overhead_report)
            if args.fresh_overhead_report is not None else None),
        "fresh_overhead_adjudication": (
            str(args.fresh_overhead_adjudication)
            if args.fresh_overhead_adjudication is not None else None),
        "fresh_overhead_cell": args.fresh_overhead_cell,
        "fresh_overhead_upper_us": fresh_overhead_upper_us,
        "headroom_required_us": headroom_required_us,
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
    if args.paired_counterfactual and (
            len(authority_pairs) < args.paired_executions or
            gross_lower_us <= headroom_required_us or net_lower_us <= 0):
        raise SystemExit(
            "ZC6 paired authority efficacy/headroom gate failed")


if __name__ == "__main__":
    main()
