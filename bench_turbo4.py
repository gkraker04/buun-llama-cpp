#!/usr/bin/env python3
import requests
import json

url = "http://localhost:8082/completion"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummythicc"
}

# Test 1: Short generation
data = {
    "prompt": "The capital of France is",
    "n_predict": 64,
    "temperature": 0
}

print("Benchmark 1: 64 tokens")
resp = requests.post(url, headers=headers, json=data)
result = resp.json()
timings = result["timings"]
print(f"  Prompt: {timings['prompt_n']} tokens in {timings['prompt_ms']:.0f}ms = {timings['prompt_per_second']:.1f} tok/s")
print(f"  Generation: {timings['predicted_n']} tokens in {timings['predicted_ms']:.0f}ms = {timings['predicted_per_second']:.1f} tok/s")
print(f"  Output: {result['content'][:50]}...")
print()

# Test 2: Longer generation
data["n_predict"] = 128
data["prompt"] = "Write a detailed explanation of quantum computing."

print("Benchmark 2: 128 tokens")
resp = requests.post(url, headers=headers, json=data)
result = resp.json()
timings = result["timings"]
print(f"  Prompt: {timings['prompt_n']} tokens in {timings['prompt_ms']:.0f}ms = {timings['prompt_per_second']:.1f} tok/s")
print(f"  Generation: {timings['predicted_n']} tokens in {timings['predicted_ms']:.0f}ms = {timings['predicted_per_second']:.1f} tok/s")
print(f"  Output: {result['content'][:80]}...")
