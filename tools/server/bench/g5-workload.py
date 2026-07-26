#!/usr/bin/env python3
"""Drive a representative DFlash workload for the Gate-5 rollback-depth observation.

Rollback depth in speculative decoding is a function of drafter acceptance, which
varies with BOTH content type and sampling temperature. A single narrow prompt set
would produce a distribution that silently over-fits one regime, so this driver
sweeps a deliberate content mix across a greedy and a sampled arm. The arms are
reported separately (they bracket production behaviour rather than averaging it away).
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    # (label, prompt) -- content types with materially different acceptance profiles
    ("code_algo",
     "Write a complete Python implementation of quicksort with an explanatory "
     "docstring and an example call. Output code only."),
    ("code_api",
     "Write a Python Flask REST endpoint that accepts JSON {name, email}, "
     "validates both fields, stores to SQLite, and returns 201 or 400. Code only."),
    ("prose_tech",
     "Explain in detail how a four-stroke internal combustion engine works, "
     "covering the intake, compression, power and exhaust strokes."),
    ("prose_creative",
     "Write an original short story about a lighthouse keeper who discovers "
     "something unexpected washed ashore after a winter storm."),
    ("structured_list",
     "List 20 common HTTP status codes. For each give the number, the standard "
     "reason phrase, and one sentence on when to use it."),
    ("repetitive",
     "Produce a multiplication table from 1x1 through 12x12, one line per "
     "product, formatted exactly as 'A x B = C'."),
]


def run(port, label, prompt, n_predict, temperature, seed, chat):
    # Instruct models MUST be driven through their chat template: a raw completion
    # prompt puts the target off-distribution, which collapses drafter acceptance and
    # would masquerade as a deep-rollback workload. Chat mode is therefore the default.
    if chat:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": temperature,
            "cache_prompt": False,
        }
        if temperature > 0:
            body["seed"] = seed
            body["top_p"] = 0.95
        url = "http://127.0.0.1:%d/v1/chat/completions" % port
    else:
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": False,
        }
        if temperature > 0:
            body["seed"] = seed
            body["top_p"] = 0.95
        url = "http://127.0.0.1:%d/completion" % port

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    tim = d.get("timings", {}) or {}
    if chat:
        txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    else:
        txt = d.get("content", "")
    return {
        "label": label,
        "temperature": temperature,
        "predicted": tim.get("predicted_n"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "sample": (txt or "")[:60].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--raw", action="store_true",
                    help="use /completion instead of the chat template (off-distribution for instruct models)")
    args = ap.parse_args()

    for temperature in (0.0, 0.7):
        arm = "greedy" if temperature == 0.0 else "sampled(t=0.7)"
        print("=== arm: %s ===" % arm, flush=True)
        for label, prompt in PROMPTS:
            try:
                r = run(args.port, label, prompt, args.n_predict, temperature,
                        args.seed, not args.raw)
                print("  %-16s predicted=%-4s tg=%6.1f t/s | %s" % (
                    r["label"], r["predicted"],
                    r["predicted_per_second"] or float("nan"), r["sample"]), flush=True)
            except Exception as e:
                print("  %-16s FAILED: %s" % (label, e), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
