#!/usr/bin/env python3
# cache-plan-trace-chat.py — B-7 trace shape 1: multi-turn chat with growing shared
# prefixes, regenerates, and mid-history branches. Drives a llama-server (run it with
# --cache-debug) so its log accumulates CACHE_PLAN records for cache-plan-replay.py.
#
# Turn-length/edit behavior can be drawn from a histogram JSON (--histogram, e.g. the
# private g5 edit histograms: {"turn_tokens": [[len, weight], ...], "p_regenerate": x,
# "p_branch": y}); without one, a modest built-in synthetic distribution is used.
# Deterministic under --seed.

import argparse
import json
import random
import sys

from cache_plan_common import make_text as _make_text, post_completion as _post

WORDS = ("state cache prefix token restore replay checkpoint slot frontier budget "
         "quantize kernel tensor stream batch decode prompt schedule memory tier").split()


def make_text(rng, n_tokens):
    return _make_text(rng, WORDS, n_tokens)


def post_completion(base, prompt, n_predict, timeout):
    t_ms, body = _post(base, prompt, n_predict, timeout)
    return t_ms, body.get("content", "")


def main():
    ap = argparse.ArgumentParser(description="chat-histogram cache-plan trace generator")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--histogram", help="optional histogram JSON (turn_tokens/p_regenerate/p_branch)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.histogram:
        with open(args.histogram) as f:
            h = json.load(f)
        lens    = [int(l) for l, _ in h["turn_tokens"]]
        weights = [float(w) for _, w in h["turn_tokens"]]
        p_regen  = float(h.get("p_regenerate", 0.15))
        p_branch = float(h.get("p_branch", 0.10))
    else:
        lens, weights = [8, 16, 32, 64, 128], [4, 6, 4, 2, 1]
        p_regen, p_branch = 0.15, 0.10

    n_req = 0
    for s in range(args.sessions):
        system = f"session {s}: " + make_text(rng, 24)
        history = [system]
        prompt_acc = system  # running prompt: appended per turn, rebuilt only on branch
        prev_prompt = None
        for t in range(args.turns):
            r = rng.random()
            if prev_prompt is not None and r < p_regen:
                prompt = prev_prompt                       # identity echo / regenerate
                kind = "regen"
            elif len(history) > 2 and r < p_regen + p_branch:
                cut = rng.randrange(1, len(history) - 1)   # branch: rewrite mid-history
                history = history[:cut]
                history.append("user: " + make_text(rng, rng.choices(lens, weights)[0]))
                prompt_acc = "\n".join(history)
                prompt = prompt_acc
                kind = "branch"
            else:
                history.append("user: " + make_text(rng, rng.choices(lens, weights)[0]))
                prompt_acc += "\n" + history[-1]
                prompt = prompt_acc
                kind = "turn"
            t_ms, content = post_completion(args.server, prompt, args.n_predict, args.timeout)
            history.append("assistant: " + content.strip()[:200])
            if kind != "regen":
                prompt_acc += "\n" + history[-1]
            prev_prompt = prompt
            n_req += 1
            print(f"TRACE shape=chat session={s} turn={t} kind={kind} t_ms={t_ms:.1f}", flush=True)

    print(f"TRACE_DONE shape=chat requests={n_req}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
