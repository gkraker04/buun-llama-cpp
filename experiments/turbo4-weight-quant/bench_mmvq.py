"""Benchmark turbo4 MMVQ server — correctness + speed."""
import urllib.request, json, time

URL = "http://localhost:8082/v1/completions"
HEADERS = {
    "Authorization": "Bearer dummythicc",
    "Content-Type": "application/json"
}

def complete(prompt, n_predict=16, temperature=0, cache=True):
    data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "cache_prompt": cache,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(data).encode(), headers=HEADERS, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())

# Warmup
result = complete("Hello", n_predict=1, cache=False)
print(f"Warmup OK. Output keys: {list(result.keys())}")

# Correctness test
result = complete("The capital of France is", n_predict=16)
tokens = result["usage"]["completion_tokens"]
text = result["choices"][0]["text"]
print(f"\nCorrectness: {text!r}")
print(f"  Prompt: {result['usage']['prompt_tokens']} tok, {result['timings']['prompt_per_second']:.2f} tok/s")
print(f"  Decode: {tokens} tok,  {result['timings']['predicted_per_second']:.2f} tok/s")

# Speed benchmark — 128 tokens
print("\n--- 128-token benchmark ---")
data = {
    "prompt": "The theory of relativity was developed by",
    "n_predict": 128,
    "temperature": 0,
    "cache_prompt": True,
}
t0 = time.time()
req = urllib.request.Request(
    URL, data=json.dumps(data).encode(), headers=HEADERS, method="POST"
)
resp = urllib.request.urlopen(req, timeout=300)
result = json.loads(resp.read())
elapsed = time.time() - t0

prompt_n = result["usage"]["prompt_tokens"]
completion_n = result["usage"]["completion_tokens"]
prompt_speed = result["timings"]["prompt_per_second"]
decode_speed = result["timings"]["predicted_per_second"]
output = result["choices"][0]["text"]

print(f"Output: {output[:120]!r}")
print(f"Prompt: {prompt_n} tok, {result['timings']['prompt_ms']/1000:.1f}s, {prompt_speed:.2f} tok/s")
print(f"Decode: {completion_n} tok, {result['timings']['predicted_ms']/1000:.1f}s, {decode_speed:.2f} tok/s")
print(f"Total:  {elapsed:.1f}s wall, {(prompt_n+completion_n)/elapsed:.2f} tok/s")
print(f"Correct: {'YES' if ('Einstein' in output or 'relativity' in output) else 'CHECK'}")
