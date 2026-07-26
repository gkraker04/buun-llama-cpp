#!/usr/bin/env python3
"""
Deployment-shaped routing-margin probe for the rolling DFlash tape.

Each scored cell is a two-request transaction on pinned slot 0:
  1. prime an exact token prefix plus --rewind disposable token IDs;
  2. replace that suffix with the case's real routing question.

The second response is admissible only when GET /slots proves an installed
rolling-window restore with the requested codec, depth, sequence and adapter
identity. Cold reprocess, checkpoint restore, wrong depth, or missing evidence
fail the cell instead of silently becoming a full-forward quality sample.
"""

import argparse
import json
import re
import time
import urllib.request


PAT = re.compile(
    r"FINAL_ACTION\s*=\s*([A-Za-z_]+)\s*;\s*"
    r"FINAL_TARGET\s*=\s*([A-Za-z0-9_:/.\-]+)\s*;\s*"
    r"SOURCE_RANK\s*=\s*(\d+)")
QUESTION_MARKER = "\n\nQUESTION:"


def request_json(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def tokenize(base_url, content, add_special, timeout):
    res = request_json(
        base_url.rstrip("/") + "/tokenize",
        {"content": content, "add_special": add_special},
        timeout)
    return res["tokens"]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def completion(base_url, prompt_tokens, n_predict, n_probs, timeout):
    return request_json(
        base_url.rstrip("/") + "/completion",
        {
            "prompt": prompt_tokens,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
            "n_predict": n_predict,
            "n_probs": n_probs,
            "post_sampling_probs": False,
        },
        timeout)


def erase_slot(base_url, timeout):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/slots/0?action=erase",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    if result.get("id_slot") != 0:
        raise RuntimeError("slot erase did not acknowledge slot 0")
    return result


def slot_zero(base_url, timeout):
    slots = request_json(base_url.rstrip("/") + "/slots", None, timeout)
    if isinstance(slots, dict) and "slots" in slots:
        slots = slots["slots"]
    for slot in slots:
        if slot.get("id") == 0 or slot.get("id_slot") == 0:
            return slot
    if len(slots) == 1:
        return slots[0]
    raise RuntimeError("GET /slots did not contain slot 0")


def probability_rows(response):
    rows = []
    for token in response.get("completion_probabilities", []):
        chosen_id = token.get("id")
        chosen_lp = token.get("logprob")
        runner_up = None
        for alt in token.get("top_logprobs", []):
            if alt.get("id") != chosen_id:
                runner_up = alt.get("logprob")
                break
        rows.append([token.get("token", ""), chosen_lp, runner_up])
    return rows


def span_metrics(rows, target):
    text = "".join(row[0] for row in rows)
    pos = text.find(target)
    if pos < 0:
        return None

    target_margins = []
    target_logprobs = []
    off = 0
    for token, chosen_lp, runner_up in rows:
        end = off + len(token)
        overlaps = end > pos and off < pos + len(target)
        off = end
        if not overlaps:
            continue
        if chosen_lp is not None:
            target_logprobs.append(chosen_lp)
        if chosen_lp is not None and runner_up is not None:
            target_margins.append(chosen_lp - runner_up)
    if not target_margins:
        return None
    return {
        "target_min_margin": min(target_margins),
        "target_mean_margin": sum(target_margins) / len(target_margins),
        "target_mean_logprob": (
            sum(target_logprobs) / len(target_logprobs)
            if target_logprobs else None),
        "target_scorable_tokens": len(target_margins),
    }


def validate_restore(slot, codec, rewind, generation_before, identity):
    evidence = slot.get("dflash_window_restore")
    if not evidence:
        raise RuntimeError("missing dflash_window_restore evidence")
    if evidence.get("result") != "installed":
        raise RuntimeError(
            "rolling restore was not installed: " + json.dumps(evidence))
    if evidence.get("codec") != codec:
        raise RuntimeError(
            f"restore codec {evidence.get('codec')} != {codec}")
    if evidence.get("depth") != rewind:
        raise RuntimeError(
            f"restore depth {evidence.get('depth')} != {rewind}")
    if evidence.get("generation", 0) <= generation_before:
        raise RuntimeError("restore evidence is stale")
    if evidence.get("identity") != identity:
        raise RuntimeError(
            f"restore identity {evidence.get('identity')!r} != {identity!r}")
    if evidence.get("frontier", -1) - evidence.get("target", -1) != rewind:
        raise RuntimeError("frontier/target evidence does not match depth")
    if evidence.get("boundary", -1) > evidence.get("target", -1):
        raise RuntimeError("restore target predates retained boundary")
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8099")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--codec", choices=("f32", "f16"), required=True)
    ap.add_argument("--rewind", type=int, default=289)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--identity", default="base:no-lora")
    args = ap.parse_args()

    if args.rewind <= 0 or args.n_probs < 2:
        raise SystemExit("--rewind must be positive and --n-probs must be >= 2")

    cases = [
        json.loads(line)
        for line in open(args.data, encoding="utf-8")
        if line.strip()
    ]
    done = set()
    try:
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass

    filler_candidates = tokenize(
        args.base_url, " telemetry checkpoint ledger", False, args.timeout)
    if not filler_candidates:
        raise SystemExit("could not tokenize disposable suffix")

    n_exact = n_error = n_unscorable = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            if case["id"] in done:
                continue

            rec = {
                "id": case["id"],
                "label": args.label,
                "codec": args.codec,
                "expected_action": case["expected_action"],
                "expected_target": case["expected_target"],
                "expected_rank": case["expected_rank"],
                "requested_rewind": args.rewind,
            }
            try:
                if QUESTION_MARKER not in case["user"]:
                    raise RuntimeError("case has no QUESTION marker")
                branch_text = (
                    case["user"].split(QUESTION_MARKER, 1)[0] + "\n\n")
                edit_tokens = tokenize(
                    args.base_url, case["user"], True, args.timeout)
                branch_tokens = tokenize(
                    args.base_url, branch_text, True, args.timeout)
                lcp = common_prefix(branch_tokens, edit_tokens)
                if lcp <= 0 or lcp >= len(edit_tokens):
                    raise RuntimeError(
                        f"invalid branch LCP {lcp}/{len(edit_tokens)}")

                filler = next(
                    (tok for tok in filler_candidates
                     if tok != edit_tokens[lcp]),
                    None)
                if filler is None:
                    raise RuntimeError(
                        "no disposable token differs at branch point")
                prime_tokens = edit_tokens[:lcp] + [filler] * args.rewind
                if common_prefix(prime_tokens, edit_tokens) != lcp:
                    raise RuntimeError("constructed prime has wrong token LCP")

                # Isolate one restore per cell. Otherwise the prime for case N
                # can itself restore a shared header from case N-1, turning an
                # F16 arm into an uncontrolled chain of lossy restores.
                erased = erase_slot(args.base_url, args.timeout)
                completion(
                    args.base_url, prime_tokens, 1, 2, args.timeout)
                prime_slot = slot_zero(args.base_url, args.timeout)
                prime_generation = (
                    prime_slot.get("dflash_window_restore") or {}
                ).get("generation", 0)

                response = completion(
                    args.base_url, edit_tokens,
                    args.max_tokens, args.n_probs, args.timeout)
                slot = slot_zero(args.base_url, args.timeout)
                evidence = validate_restore(
                    slot, args.codec, args.rewind,
                    prime_generation, args.identity)

                text = response.get("content", "")
                rows = probability_rows(response)
                parsed = PAT.search(text or "")
                rec.update(
                    raw=text,
                    prompt_tokens=response.get("tokens_evaluated"),
                    completion_tokens=response.get("tokens_predicted"),
                    lp=rows,
                    restore=evidence,
                    token_lcp=lcp,
                    prime_erased=erased.get("n_erased"),
                    prime_tokens=len(prime_tokens),
                    edit_tokens=len(edit_tokens),
                    all_output_margins=[
                        row[1] - row[2]
                        for row in rows
                        if row[1] is not None and row[2] is not None
                    ],
                    all_output_unscorable=sum(
                        row[1] is None or row[2] is None for row in rows),
                )
                rec.update(span_metrics(rows, case["expected_target"]) or {})
                if parsed:
                    action, target, rank = (
                        parsed.group(1),
                        parsed.group(2),
                        int(parsed.group(3)))
                    rec.update(
                        got_action=action,
                        got_target=target,
                        got_rank=rank,
                        ok_action=action == case["expected_action"],
                        ok_target=target == case["expected_target"],
                        ok_rank=rank == case["expected_rank"])
                    rec["exact"] = (
                        rec["ok_action"] and
                        rec["ok_target"] and
                        rec["ok_rank"])
                else:
                    rec.update(
                        exact=False,
                        ok_action=False,
                        ok_target=False,
                        ok_rank=False,
                        parse_fail=True)
                if "target_min_margin" not in rec:
                    rec["unscorable_target"] = True
                    n_unscorable += 1
            except Exception as err:
                rec.update(error=str(err), exact=False)
                n_error += 1

            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_exact += bool(rec.get("exact"))
            if i % 10 == 0:
                print(
                    f"[{i}/{len(cases)}] exact={n_exact} "
                    f"errors={n_error} unscorable={n_unscorable} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True)

    n_run = len(cases) - len(done)
    print(
        f"SUMMARY label={args.label} codec={args.codec} n={n_run} "
        f"exact={n_exact}/{n_run} errors={n_error} "
        f"unscorable_target={n_unscorable}")
    if n_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
