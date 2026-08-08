#!/usr/bin/env python3
"""Capture deterministic concurrent append/compact/rewind cache shapes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request


def atoms(seed: int, count: int, namespace: str) -> str:
    rng = random.Random(seed)
    vocabulary = (
        "cache", "state", "token", "prefix", "branch", "agent", "tool", "result",
        "edit", "build", "test", "source", "memory", "prompt", "reply", "context",
    )
    words = " ".join(vocabulary[rng.randrange(len(vocabulary))] for _ in range(count))
    return f"{namespace} seed {seed} {words}"


def complete(target: str, model: str, messages: list[dict], seed: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "stream": False,
    }, separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"http://{target}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        payload = json.load(response)
    text = payload["choices"][0]["message"].get("content")
    if not isinstance(text, str) or not text:
        text = payload["choices"][0]["message"].get("reasoning_content")
    if not isinstance(text, str) or not text:
        raise RuntimeError("completion returned no textual reply")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="192.168.1.152:8080")
    parser.add_argument("--model", default="Qwen3.6")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--workers", type=int, choices=(1, 3), default=3,
        help="request concurrency; use 1 for deterministic output-parity arms",
    )
    args = parser.parse_args()

    common = atoms(args.seed ^ 0xCACE, 2800, "shared_system")
    chains = []
    snapshots: list[list[list[dict]]] = [[], [], []]
    for chain in range(3):
        chains.append([{
            "role": "system",
            "content": (
                "You are one worker in a concurrent coding project. Preserve exact evidence. "
                + common
                + f" worker_{chain}"
            ),
        }])

    records = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for round_id in range(12):
            for chain in range(3):
                if round_id == 7:
                    chains[chain] = [chains[chain][0], {
                        "role": "user",
                        "content": (
                            f"Compacted project state for worker {chain}. "
                            + atoms(args.seed + chain * 1000 + round_id, 700, "summary")
                        ),
                    }]
                elif round_id == 10:
                    chains[chain] = [dict(message) for message in snapshots[chain][3]]
                    chains[chain].append({
                        "role": "user",
                        "content": (
                            f"Deep rewind for worker {chain}; continue from the older branch. "
                            + atoms(args.seed + chain * 1000 + round_id, 1400, "rewind")
                        ),
                    })
                else:
                    chains[chain].append({
                        "role": "user",
                        "content": (
                            f"Worker {chain} round {round_id}; inspect and advance this branch. "
                            + atoms(args.seed + chain * 1000 + round_id, 1400, "append")
                        ),
                    })

            futures = [
                pool.submit(
                    complete, args.target, args.model,
                    [dict(message) for message in chains[chain]],
                    args.seed + chain * 100 + round_id,
                )
                for chain in range(3)
            ]
            for chain, future in enumerate(futures):
                reply = future.result()
                chains[chain].append({"role": "assistant", "content": reply})
                snapshots[chain].append([dict(message) for message in chains[chain]])
                records.append({
                    "round": round_id,
                    "chain": chain,
                    "transition": (
                        "compact" if round_id == 7 else
                        "deep_rewind" if round_id == 10 else
                        "append"
                    ),
                    "message_count": len(chains[chain]),
                    "reply_sha256": hashlib.sha256(reply.encode()).hexdigest(),
                })

    result = {
        "schema": "zc1_retention_shape_transcript/v1",
        "label": args.label,
        "seed": args.seed,
        "chains": 3,
        "rounds": 12,
        "workers": args.workers,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "records": records,
    }
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(
        f"ZC1_RETENTION_SHAPES PASS label={args.label} "
        f"requests={len(records)} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
