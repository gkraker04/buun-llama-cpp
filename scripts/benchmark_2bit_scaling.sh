#!/bin/bash
# 2-bit TCQ scaling law grid: different codebooks × context lengths
# Tests how MSE reduction interacts with context length for PPL

BIN="/root/llama-tcq-clean/build/bin"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"
OUT="/tmp/bench_2bit_scaling.txt"

echo "=== 2-bit TCQ Scaling Law Grid — $(date) ===" | tee "$OUT"

# Codebooks: name|path (empty path = compiled-in)
declare -a CODEBOOKS=(
	"3iter_numpy(0.7%)|/tmp/tcq_2bit_3iter_s99.bin"
	"10iter_numpy(13%)|/tmp/tcq_2bit_10iter_s99.bin"
	"30iter_numpy(25.5%)|/tmp/tcq_2bit_30iter_s99.bin"
	"50iter_numpy(28.5%)|/tmp/tcq_2bit_50iter_s99.bin"
	"100iter_numpy(32.1%)|/tmp/tcq_2bit_100iter_s99.bin"
	"compiled_in(~33%)|"
	"cuda_200iter(34.9%)|/tmp/tcq_2bit_cuda_200iter.bin"
)

# Context lengths and chunk counts
declare -a CTX=(
	"2048|64"
	"8192|8"
	"32768|4"
	"65536|4"
)

for cb in "${CODEBOOKS[@]}"; do
	IFS='|' read -r name cb_path <<< "$cb"
	echo "" | tee -a "$OUT"
	echo "--- Codebook: $name ---" | tee -a "$OUT"

	for ctx_info in "${CTX[@]}"; do
		IFS='|' read -r ctx chunks <<< "$ctx_info"
		ctx_k=$((ctx / 1024))
		echo -n "  PPL @${ctx_k}K (${chunks}ch): " | tee -a "$OUT"

		if [ -n "$cb_path" ]; then
			result=$(TURBO_TCQ_CB2="$cb_path" "$BIN/llama-perplexity" \
				-m "$MODEL" \
				-ctk turbo2_tcq -ctv turbo2_tcq \
				-f "$WIKI" \
				-c "$ctx" --chunks "$chunks" \
				-ngl 99 -t 1 2>&1)
		else
			result=$("$BIN/llama-perplexity" \
				-m "$MODEL" \
				-ctk turbo2_tcq -ctv turbo2_tcq \
				-f "$WIKI" \
				-c "$ctx" --chunks "$chunks" \
				-ngl 99 -t 1 2>&1)
		fi

		# Check for codebook loading confirmation
		cb_loaded=$(echo "$result" | grep "TCQ2" | head -2)
		if [ -n "$cb_path" ] && [ -z "$cb_loaded" ]; then
			echo "WARNING: codebook not loaded!" | tee -a "$OUT"
		fi

		ppl=$(echo "$result" | grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')
		if [ -n "$ppl" ]; then
			echo "$ppl" | tee -a "$OUT"
		else
			echo "FAILED" | tee -a "$OUT"
			echo "$result" | tail -5 >> "$OUT"
		fi
	done
done

echo "" | tee -a "$OUT"
echo "=== Done — $(date) ===" | tee -a "$OUT"
