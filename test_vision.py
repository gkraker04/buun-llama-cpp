#!/usr/bin/env python3
"""Test vision capabilities of OpenCode Go models."""

import base64
import json
import os
import subprocess
import sys
import time

# Read API key from .env file
def get_api_key():
    env_path = os.path.expanduser("~/AppData/Local/hermes/.env")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENCODE_GO_API_KEY=*** and "=" in line:
                return line.split("=", 1)[1].strip()
    return None

API_KEY = get_api_key()
if not API_KEY:
    print("ERROR: Could not find OPENCODE_GO_API_KEY")
    sys.exit(1)

print(f"API key loaded ({len(API_KEY)} chars)")

# Image path
IMAGE_PATH = "I:/Downloads/images.jpg"

# Read and encode image
with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Image encoded: {len(image_b64)} chars base64")

# All 19 models
MODELS = [
    "minimax-m3", "minimax-m2.7", "minimax-m2.5",
    "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5",
    "glm-5.1", "glm-5",
    "deepseek-v4-pro", "deepseek-v4-flash",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus",
    "mimo-v2-pro", "mimo-v2-omni", "mimo-v2.5-pro", "mimo-v2.5",
    "hy3-preview",
]

API_URL = "https://opencode.ai/zen/go/v1/chat/completions"

results = {}

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"{'='*60}")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    cmd = [
        "curl", "-s", "-X", "POST", API_URL,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer ***
        "-d", json.dumps(payload),
        "--max-time", "60"
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        elapsed = time.time() - start
        response_text = result.stdout
        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = None
        
        if data and "error" in data:
            error = data["error"]
            err_msg = error.get("message", str(error))[:200]
            err_type = error.get("type", "unknown")
            print(f"  ERROR ({err_type}): {err_msg}")
            results[model] = {"status": "error", "error": err_msg, "type": err_type}
        elif data and "choices" in data:
            content = data["choices"][0]["message"].get("content", "")
            reasoning = data["choices"][0]["message"].get("reasoning_content", "")
            preview = content[:200] if content else "(empty)"
            usage = data.get("usage", {})
            print(f"  OK ({elapsed:.1f}s, {usage.get('total_tokens', '?')} tokens)")
            print(f"  Response: {preview}")
            results[model] = {"status": "ok", "response": content[:200], "tokens": usage.get("total_tokens", 0), "time": elapsed}
        elif data and "type" in data and data["type"] == "error":
            err_msg = data.get("error", {}).get("message", str(data))[:200]
            print(f"  ERROR: {err_msg}")
            results[model] = {"status": "error", "error": err_msg}
        else:
            preview = response_text[:300] if response_text else "(empty response)"
            print(f"  UNKNOWN ({elapsed:.1f}s): {preview}")
            results[model] = {"status": "unknown", "raw": preview}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        results[model] = {"status": "timeout"}
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        results[model] = {"status": "exception", "error": str(e)}

# Summary
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
ok_count = sum(1 for r in results.values() if r["status"] == "ok")
err_count = sum(1 for r in results.values() if r["status"] == "error")
other_count = len(results) - ok_count - err_count

print(f"\nTotal: {len(results)} models")
print(f"  Success: {ok_count}")
print(f"  Errors:  {err_count}")
print(f"  Other:   {other_count}")

print(f"\n{'Model':<25} {'Status':<10} {'Details'}")
print("-" * 80)
for model in MODELS:
    r = results.get(model, {})
    status = r.get("status", "unknown")
    if status == "ok":
        detail = r.get("response", "")[:50]
    elif status == "error":
        detail = r.get("error", "")[:50]
    else:
        detail = str(r)[:50]
    print(f"{model:<25} {status:<10} {detail}")

# Save full results
with open("G:/hermes/buun-llama-cpp/vision_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to vision_test_results.json")
