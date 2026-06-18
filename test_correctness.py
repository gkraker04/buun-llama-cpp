#!/usr/bin/env python3
import requests
import json

url = "http://localhost:8082/completion"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer ***
}

tests = [
    {"prompt": "The capital of France is", "n_predict": 32},
    {"prompt": "What is 2+2?", "n_predict": 32},
    {"prompt": "just putting a new llm model through its paces. any thoughts on some quick tests?", "n_predict": 200},
]

for i, test in enumerate(tests, 1):
    data = {**test, "temperature": 0}
    resp = requests.post(url, headers=headers, json=data)
    result = resp.json()
    timings = result["timings"]
    print(f"Test {i}: {test['prompt'][:50]}...")
    print(f"  Output: {result['content'][:150]}")
    print(f"  Speed: {timings['predicted_per_second']:.1f} tok/s")
    print()
