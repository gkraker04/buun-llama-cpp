#!/bin/bash
# TCQ benchmark grid: turbo2_tcq + turbo3_tcq combos × context lengths
# PPL (llama-perplexity) + speed (llama-bench)
# Run on server: bash /root/llama-tcq-clean/scripts/benchmark_grid.sh

set -e

BIN="/root/llama-tcq-clean/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"
OUT="/tmp/bench_grid_results.txt"

echo "=== TCQ Benchmark Grid — $(date) ===" | tee "$OUT"
echo "" | tee -a "$OUT"

# Configs: name, -ctk flag, -ctv flag
declare -a CONFIGS=(
	"turbo3_tcq|turbo3_tcq|turbo3_tcq"
	"t2tcq-K_t3tcq-V|turbo2_tcq|turbo3_tcq"
	"t3tcq-K_t2tcq-V|turbo3_tcq|turbo2_tcq"
	"turbo2_tcq|turbo2_tcq|turbo2_tcq"
	"q8_0|q8_0|q8_0"
	"f16|f16|f16"
)

# Context lengths and chunk counts for PPL
# 64K max (128K OOMs on CPU logits buffer with 64GB RAM)
declare -a CTX_PPL=(
	"2048|64"
	"8192|8"
	"16384|4"
	"32768|4"
	"65536|4"
)

# Context lengths for speed (llama-bench can go higher since no logits buffer)
declare -a CTX_SPEED=(
	"2048"
	"8192"
	"16384"
	"32768"
	"65536"
	"131072"
)

echo "========================================" | tee -a "$OUT"
echo "PART 1: PERPLEXITY" | tee -a "$OUT"
echo "========================================" | tee -a "$OUT"

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r name ctk ctv <<< "$cfg"
	echo "" | tee -a "$OUT"
	echo "--- Config: $name (K=$ctk, V=$ctv) ---" | tee -a "$OUT"

	for ctx_info in "${CTX_PPL[@]}"; do
		IFS='|' read -r ctx chunks <<< "$ctx_info"
		ctx_k=$((ctx / 1024))
		echo -n "  PPL @${ctx_k}K (${chunks}ch): " | tee -a "$OUT"

		result=$("$BIN/llama-perplexity" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-f "$WIKI" \
			-c "$ctx" --chunks "$chunks" \
			-ngl 99 -t 1 2>&1)

		ppl=$(echo "$result" | grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')
		if [ -n "$ppl" ]; then
			echo "$ppl" | tee -a "$OUT"
		else
			# Check for OOM or error
			err=$(echo "$result" | grep -iE "bad_alloc|CUDA error|out of memory|error" | head -1)
			if [ -n "$err" ]; then
				echo "OOM/ERROR: $err" | tee -a "$OUT"
			else
				echo "FAILED (no PPL found)" | tee -a "$OUT"
			fi
			# If OOM, skip remaining contexts for this config
			if echo "$result" | grep -qiE "bad_alloc|out of memory"; then
				echo "  (skipping remaining contexts for $name due to OOM)" | tee -a "$OUT"
				break
			fi
		fi
	done
done

echo "" | tee -a "$OUT"
echo "========================================" | tee -a "$OUT"
echo "PART 2: DECODE SPEED (tok/s)" | tee -a "$OUT"
echo "========================================" | tee -a "$OUT"

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r name ctk ctv <<< "$cfg"
	echo "" | tee -a "$OUT"
	echo "--- Config: $name (K=$ctk, V=$ctv) ---" | tee -a "$OUT"

	for ctx in "${CTX_SPEED[@]}"; do
		ctx_k=$((ctx / 1024))
		echo -n "  Speed @${ctx_k}K: " | tee -a "$OUT"

		result=$("$BIN/llama-bench" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-c "$ctx" -n 64 -p 0 \
			-ngl 99 -t 1 2>&1)

		# Extract tg64 speed
		speed=$(echo "$result" | grep "tg64" | grep -oP '[0-9]+\.[0-9]+(?= \± )' | tail -1)
		if [ -n "$speed" ]; then
			echo "${speed} tok/s" | tee -a "$OUT"
		else
			err=$(echo "$result" | grep -iE "bad_alloc|CUDA error|out of memory|error" | head -1)
			if [ -n "$err" ]; then
				echo "OOM/ERROR: $err" | tee -a "$OUT"
			else
				echo "FAILED" | tee -a "$OUT"
			fi
			if echo "$result" | grep -qiE "bad_alloc|out of memory"; then
				echo "  (skipping remaining contexts for $name due to OOM)" | tee -a "$OUT"
				break
			fi
		fi
	done
done

echo "" | tee -a "$OUT"
echo "=== Grid complete — $(date) ===" | tee -a "$OUT"
echo "Results saved to $OUT"
