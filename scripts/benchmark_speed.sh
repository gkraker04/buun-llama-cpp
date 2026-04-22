#!/bin/bash
# Speed grid: decode tok/s for turbo2_tcq + turbo3_tcq combos × context lengths
# llama-bench tg64 only (no logits buffer, so 128K may work)

BIN="/root/llama-tcq-clean/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
OUT="/tmp/bench_speed_results.txt"

echo "=== Speed Grid — $(date) ===" | tee "$OUT"

declare -a CONFIGS=(
	"turbo3_tcq|turbo3_tcq|turbo3_tcq"
	"t2tcq-K_t3tcq-V|turbo2_tcq|turbo3_tcq"
	"t3tcq-K_t2tcq-V|turbo3_tcq|turbo2_tcq"
	"turbo2_tcq|turbo2_tcq|turbo2_tcq"
	"q8_0|q8_0|q8_0"
	"f16|f16|f16"
)

declare -a CONTEXTS=(2048 8192 16384 32768 65536 131072)

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r name ctk ctv <<< "$cfg"
	echo "" | tee -a "$OUT"
	echo "--- $name ---" | tee -a "$OUT"

	for ctx in "${CONTEXTS[@]}"; do
		ctx_k=$((ctx / 1024))
		echo -n "  ${ctx_k}K: " | tee -a "$OUT"

		result=$("$BIN/llama-bench" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-c "$ctx" -n 64 -p 0 \
			-ngl 99 -t 1 2>&1)

		speed=$(echo "$result" | grep "tg64" | grep -oP '[0-9]+\.[0-9]+(?= \± )' | tail -1)
		if [ -n "$speed" ]; then
			echo "${speed} tok/s" | tee -a "$OUT"
		else
			err=$(echo "$result" | grep -iE "bad_alloc|CUDA error|out of memory|failed" | head -1)
			if [ -n "$err" ]; then
				echo "OOM" | tee -a "$OUT"
				echo "  (skipping higher contexts for $name)" | tee -a "$OUT"
				break
			else
				echo "PARSE_FAIL" | tee -a "$OUT"
			fi
		fi
	done
done

echo "" | tee -a "$OUT"
echo "=== Done — $(date) ===" | tee -a "$OUT"
