#!/usr/bin/env python3
"""Capture pressure-bearing concurrent grow/rewind retention shapes."""

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
        "max_tokens": 8,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "stream": False,
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"http://{target}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    text = payload["choices"][0]["message"].get("content")
    if not isinstance(text, str) or not text:
        text = payload["choices"][0]["message"].get("reasoning_content")
    if not isinstance(text, str) or not text:
        raise RuntimeError("completion returned no textual reply")
    return text


def parent_round(round_id: int) -> int | None:
    if round_id in (0, 16):
        return None
    if round_id == 12:
        return 3
    if round_id == 22:
        return 18
    return round_id - 1


def transition(round_id: int) -> str:
    if round_id == 16:
        return "compact"
    if round_id in (12, 22):
        return "deep_rewind"
    return "append"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="192.168.1.152:8080")
    parser.add_argument("--model", default="Qwen3.6")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    common = atoms(args.seed ^ 0xCACE, 2800, "shared_system")
    chains: list[list[dict]] = []
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for round_id in range(24):
            for chain in range(3):
                if round_id == 16:
                    chains[chain] = [chains[chain][0], {
                        "role": "user",
                        "content": (
                            f"Compacted project state for worker {chain}. "
                            + atoms(args.seed + chain * 1000 + round_id, 1400, "summary")
                        ),
                    }]
                elif round_id in (12, 22):
                    source_round = parent_round(round_id)
                    assert source_round is not None
                    chains[chain] = [
                        dict(message) for message in snapshots[chain][source_round]
                    ]
                    chains[chain].append({
                        "role": "user",
                        "content": (
                            f"Deep rewind for worker {chain}; continue from round {source_round}. "
                            + atoms(
                                args.seed + chain * 1000 + round_id,
                                2800,
                                "rewind",
                            )
                        ),
                    })
                else:
                    chains[chain].append({
                        "role": "user",
                        "content": (
                            f"Worker {chain} round {round_id}; inspect and advance this branch. "
                            + atoms(
                                args.seed + chain * 1000 + round_id,
                                2800,
                                "append",
                            )
                        ),
                    })

            futures = [
                pool.submit(
                    complete,
                    args.target,
                    args.model,
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
                    "parent_round": parent_round(round_id),
                    "transition": transition(round_id),
                    "message_count": len(chains[chain]),
                    "request_seed": args.seed + chain * 100 + round_id,
                    "reply_sha256": hashlib.sha256(reply.encode()).hexdigest(),
                })

    result = {
        "schema": "zc1_retention_pressure_transcript/v1",
        "label": args.label,
        "seed": args.seed,
        "chains": 3,
        "rounds": 24,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "records": records,
    }
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    print(
        f"ZC1_RETENTION_PRESSURE PASS label={args.label} "
        f"requests={len(records)} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

