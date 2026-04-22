#!/bin/bash
# Speed grid: save raw llama-bench markdown output
BIN="/root/llama-tcq-clean/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
OUT="/tmp/bench_speed_raw.txt"

echo "=== Speed Grid — $(date) ===" > "$OUT"

declare -a CONFIGS=(
	"turbo3_tcq|turbo3_tcq|turbo3_tcq"
	"turbo2_tcq|turbo3_tcq|turbo2_tcq"
	"turbo3_tcq|turbo2_tcq|turbo3_tcq"
	"turbo2_tcq|turbo2_tcq|turbo2_tcq"
	"q8_0|q8_0|q8_0"
	"f16|f16|f16"
)

declare -a PROMPTS=(512 2048 8192 16384 32768 65536)

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r ctk ctv name <<< "$cfg"
	echo "" >> "$OUT"
	echo "--- K=$ctk V=$ctv ---" >> "$OUT"
	echo "Running K=$ctk V=$ctv ..." >&2

	for pp in "${PROMPTS[@]}"; do
		echo "  pp=$pp ..." >&2
		"$BIN/llama-bench" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-p "$pp" -n 64 \
			-ngl 99 -t 1 -r 1 >> "$OUT" 2>&1
		rc=$?
		if [ $rc -ne 0 ]; then
			echo "  FAILED (rc=$rc), skipping higher contexts" >> "$OUT"
			echo "  FAILED, skipping higher" >&2
			break
		fi
	done
done

echo "" >> "$OUT"
echo "=== Done — $(date) ===" >> "$OUT"
echo "All done" >&2
