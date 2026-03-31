#!/bin/bash
# Re-run mixed configs with BEST codebooks (not compiled-in)
# Best 2-bit: 100-iter numpy (32.1% MSE red, best @64K)
# Best 3-bit: finetuned 50-iter (best @64K from earlier grid)

BIN="/root/llama-tcq-clean/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"
OUT="/tmp/bench_best_codebooks.txt"

CB2="/tmp/tcq_2bit_100iter_s99.bin"
CB3="/tmp/cb_50iter_finetuned.bin"

echo "=== Best Codebook Grid — $(date) ===" | tee "$OUT"
echo "2-bit CB: $CB2 (100-iter numpy, 32.1% MSE red)" | tee -a "$OUT"
echo "3-bit CB: $CB3 (finetuned 50-iter)" | tee -a "$OUT"

declare -a CONFIGS=(
	"turbo3_tcq(best)|turbo3_tcq|turbo3_tcq|$CB3|"
	"turbo2_tcq(best)|turbo2_tcq|turbo2_tcq||$CB2"
	"t2K(best)/t3V(best)|turbo2_tcq|turbo3_tcq|$CB3|$CB2"
	"t3K(best)/t2V(best)|turbo3_tcq|turbo2_tcq|$CB3|$CB2"
)

declare -a CTX=(
	"2048|64"
	"8192|8"
	"32768|4"
	"65536|4"
)

for cfg in "${CONFIGS[@]}"; do
	IFS='|' read -r name ctk ctv cb3_path cb2_path <<< "$cfg"
	echo "" | tee -a "$OUT"
	echo "--- $name (K=$ctk, V=$ctv) ---" | tee -a "$OUT"

	for ctx_info in "${CTX[@]}"; do
		IFS='|' read -r ctx chunks <<< "$ctx_info"
		ctx_k=$((ctx / 1024))
		echo -n "  PPL @${ctx_k}K (${chunks}ch): " | tee -a "$OUT"

		env_vars=""
		if [ -n "$cb3_path" ]; then
			env_vars="TURBO_TCQ_CB=$cb3_path"
		fi
		if [ -n "$cb2_path" ]; then
			env_vars="$env_vars TURBO_TCQ_CB2=$cb2_path"
		fi

		result=$(env $env_vars "$BIN/llama-perplexity" \
			-m "$MODEL" \
			-ctk "$ctk" -ctv "$ctv" \
			-f "$WIKI" \
			-c "$ctx" --chunks "$chunks" \
			-ngl 99 -t 1 2>&1)

		ppl=$(echo "$result" | grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')
		if [ -n "$ppl" ]; then
			echo "$ppl" | tee -a "$OUT"
		else
			echo "FAILED" | tee -a "$OUT"
			echo "$result" | tail -3 >> "$OUT"
		fi
	done
done

echo "" | tee -a "$OUT"
echo "=== Done — $(date) ===" | tee -a "$OUT"
