#!/usr/bin/env python3
"""Drive warm conversation-edit traffic for the Gate-5 edit-regime observation.

Unlike the speculative regime (where rollback depth is hard-capped by draft length),
edit-regime rewind depth is unbounded: it is however many tokens the slot must discard
when the user regenerates, edits a turn, or branches backwards. That is the distribution
the tape's deep-window claim actually rests on.

Routing discipline: with -np 1 there is ONE slot, so a conversation's edits only land warm
if they are issued consecutively while that conversation still owns the slot. Interleaving
conversations would make every request an LCP=0 cold start and silently measure nothing.
Each conversation is therefore built and edited to completion before the next begins.
"""

import argparse
import json
import sys
import urllib.request

TOPICS = [
    ("engines", "How does a four-stroke engine work?",
     "What differs in a two-stroke?", "How does turbocharging change that?"),
    ("python", "Write a Python function to merge two sorted lists.",
     "Now make it handle duplicates.", "Add type hints and a docstring."),
    ("networks", "Explain how TCP establishes a connection.",
     "What happens on packet loss?", "How does QUIC improve on this?"),
    ("cooking", "How do I make a basic risotto?",
     "What if I have no white wine?", "How do I make it vegan?"),
    ("history", "What caused the fall of the Western Roman Empire?",
     "How did the Eastern half survive?", "What ended it in 1453?"),
    ("storage", "Explain how an SSD differs from a hard drive.",
     "What is write amplification?", "How does TRIM help?"),
    ("music", "What defines the baroque period in music?",
     "How did the classical era differ?", "Where does Beethoven sit?"),
    ("biology", "How does photosynthesis work?",
     "What limits its efficiency?", "How do C4 plants differ?"),
    ("finance", "What is compound interest?",
     "How does inflation affect it?", "What is a real rate of return?"),
    ("weather", "How do hurricanes form?",
     "Why do they weaken over land?", "How is intensity measured?"),
]

EDITS = ("regenerate", "edit_last_user", "branch_two_back", "midpoint_branch")


def chat(port, messages, n_predict, temperature, seed):
    body = {
        "messages": messages,
        "max_tokens": n_predict,
        "temperature": temperature,
        "cache_prompt": True,   # warm reuse is the whole point
        # Gemma-4 is a thinking model: with thinking on, a bounded token budget is
        # consumed entirely by reasoning_content and `content` comes back EMPTY.
        # Appending that empty reply would silently build user-only "conversations"
        # and the measured rewind depths would be meaningless.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temperature > 0:
        body["seed"] = seed
        body["top_p"] = 0.95
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    if not text:
        # fail loudly: an empty assistant turn corrupts every subsequent depth
        raise RuntimeError(
            "empty assistant content (finish_reason=%s, reasoning_len=%d) -- "
            "conversation would not accumulate" % (
                (d.get("choices") or [{}])[0].get("finish_reason"),
                len(msg.get("reasoning_content") or "")))
    return text


def run_conversation(port, topic, n_predict, temperature, seed, stats):
    label, q1, q2, q3 = topic
    msgs = []

    # --- build three turns (turn 1 is a legitimate cold start) ---
    for q in (q1, q2, q3):
        msgs.append({"role": "user", "content": q})
        a = chat(port, msgs, n_predict, temperature, seed)
        msgs.append({"role": "assistant", "content": a})
        stats["build"] += 1

    # --- warm edits, each issued while this conversation still owns the slot ---
    # regenerate: drop the last assistant turn and ask again
    regen = msgs[:-1]
    a = chat(port, regen, n_predict, temperature, seed + 1)
    stats["edit"] += 1

    # edit last user turn: rewrite turn 3's question
    edited = msgs[:-2] + [{"role": "user", "content": q3 + " Answer briefly."}]
    a = chat(port, edited, n_predict, temperature, seed)
    stats["edit"] += 1

    # branch two turns back: rewrite turn 2, discarding turns 2-3
    branch2 = msgs[:2] + [{"role": "user", "content": q2 + " Keep it short."}]
    a = chat(port, branch2, n_predict, temperature, seed)
    stats["edit"] += 1

    # midpoint branch: re-ask turn 2 differently after restoring the full thread
    _ = chat(port, msgs, n_predict, temperature, seed)          # re-warm the full thread
    stats["build"] += 1
    midpoint = msgs[:2] + [{"role": "user", "content": "Actually, summarize instead."}]
    a = chat(port, midpoint, n_predict, temperature, seed)
    stats["edit"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--conversations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    stats = {"build": 0, "edit": 0}
    half = args.conversations // 2
    for i in range(args.conversations):
        topic = TOPICS[i % len(TOPICS)]
        temperature = 0.0 if i < half else 0.7
        arm = "greedy" if temperature == 0.0 else "sampled"
        try:
            run_conversation(args.port, topic, args.n_predict,
                             temperature, args.seed + i, stats)
            print("conv %2d/%d [%s/%-8s] ok  (build=%d edit=%d)" % (
                i + 1, args.conversations, arm, topic[0],
                stats["build"], stats["edit"]), flush=True)
        except Exception as e:
            print("conv %2d FAILED: %s" % (i + 1, e), flush=True)

    print("TOTAL build_requests=%d edit_requests=%d" % (stats["build"], stats["edit"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
