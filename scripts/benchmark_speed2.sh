#!/bin/bash
# Speed grid: decode tok/s for TCQ combos × context lengths
# Uses -p to set context fill, -n 64 for decode measurement

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

declare -a PROMPTS=(512 2048 8192 16384 32768 65536)

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r name ctk ctv <<< "$cfg"
	echo "" | tee -a "$OUT"
	echo "--- $name ---" | tee -a "$OUT"

	for pp in "${PROMPTS[@]}"; do
		ctx_k=$((pp / 1024))
		if [ "$pp" -lt 1024 ]; then
			label="${pp}"
		else
			label="${ctx_k}K"
		fi
		echo -n "  pp${label}+tg64: " | tee -a "$OUT"

		result=$("$BIN/llama-bench" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-p "$pp" -n 64 \
			-ngl 99 -t 1 -r 1 2>&1)

		# Extract tg64 speed
		speed=$(echo "$result" | grep "tg64" | awk '{print $(NF-2)}')
		pp_speed=$(echo "$result" | grep "pp" | grep -v tg | awk '{print $(NF-2)}')
		if [ -n "$speed" ]; then
			echo "decode=${speed} tok/s, prefill=${pp_speed} tok/s" | tee -a "$OUT"
		else
			err=$(echo "$result" | grep -iE "bad_alloc|CUDA error|out of memory|failed" | head -1)
			if [ -n "$err" ]; then
				echo "OOM" | tee -a "$OUT"
				echo "  (skipping higher contexts for $name)" | tee -a "$OUT"
				break
			else
				echo "PARSE_FAIL" | tee -a "$OUT"
				echo "$result" >> /tmp/bench_speed_debug.txt
			fi
		fi
	done
done

echo "" | tee -a "$OUT"
echo "=== Done — $(date) ===" | tee -a "$OUT"
