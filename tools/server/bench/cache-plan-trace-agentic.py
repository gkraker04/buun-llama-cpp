#!/usr/bin/env python3
# cache-plan-trace-agentic.py — B-7 trace shape 2: multi-session agentic contention. One
# MAIN agent grows a long tool-loop conversation while N sub-agents fire bursts of short
# summarization-style requests between its turns — deliberately displacing the main
# agent's slot/checkpoint state on a slot-starved server (run with --cache-debug and a
# small -np). The headline metrics: the main agent's per-turn latency trajectory and its
# invalidation rate (cold/restore-failed records attributable to displacement), read from
# the server log with cache-plan-replay.py; this script reports the client-side view.
# Deterministic under --seed.

import argparse
import random
import sys

from cache_plan_common import make_text as _make_text, post_completion as _post

WORDS = ("plan tool call result observe act summarize context window agent task step "
         "search read write execute verify commit branch merge test deploy").split()


def make_text(rng, n_tokens):
    return _make_text(rng, WORDS, n_tokens)


def post_completion(base, prompt, n_predict, timeout):
    t_ms, body = _post(base, prompt, n_predict, timeout)
    return t_ms, body.get("content", "")


def main():
    ap = argparse.ArgumentParser(description="agentic-contention cache-plan trace generator")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--main-turns", type=int, default=10)
    ap.add_argument("--subagents", type=int, default=3)
    ap.add_argument("--sub-burst", type=int, default=2, help="sub-agent requests between main turns")
    ap.add_argument("--main-grow", type=int, default=96, help="tokens the main history grows per turn")
    ap.add_argument("--sub-tokens", type=int, default=48, help="tokens per sub-agent prompt")
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    main_prompt = "main agent objective: " + make_text(rng, 32)
    sub_prompt = {i: f"subagent {i} standing instructions: " + make_text(rng, 16)
                  for i in range(args.subagents)}
    main_lat = []

    for turn in range(args.main_turns):
        # the main agent's turn: the prompt grows monotonically (the reusable prefix)
        main_prompt += "\nstep %d: " % turn + make_text(rng, args.main_grow)
        t_ms, content = post_completion(args.server, main_prompt,
                                        args.n_predict, args.timeout)
        main_prompt += "\nresult: " + content.strip()[:160]
        main_lat.append(t_ms)
        print(f"TRACE shape=agentic role=main turn={turn} t_ms={t_ms:.1f}", flush=True)

        # sub-agent burst: each re-summarizes ITS OWN growing transcript — high-frequency,
        # partially-overlapping prefixes that compete for the same slots
        for _ in range(args.sub_burst):
            i = rng.randrange(args.subagents)
            sub_prompt[i] += "\nsummarize progress: " + make_text(rng, args.sub_tokens)
            t_ms, _ = post_completion(args.server, sub_prompt[i],
                                      args.n_predict, args.timeout)
            print(f"TRACE shape=agentic role=sub agent={i} turn={turn} t_ms={t_ms:.1f}", flush=True)

    # client-side headline: does the main agent's latency degrade as sub-agents displace it?
    half = max(1, len(main_lat) // 2)
    first, second = main_lat[:half], main_lat[half:]
    print(f"TRACE_DONE shape=agentic main_turns={len(main_lat)} "
          f"main_t_ms_first_half={sum(first)/len(first):.1f} "
          f"main_t_ms_second_half={sum(second)/len(second):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
