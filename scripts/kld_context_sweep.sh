#!/bin/bash
# KLD alpha sweep at 8K and 32K — focused around 2K optima
set -e

PPLBIN="build/bin/llama-perplexity"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"
FA="-fa on"
CHUNKS=4
OUTFILE="kld_context_sweep_results.txt"

CB2="codebooks/2bit/tcq_2bit_cuda_200iter.bin"
CB3="codebooks/3bit/cb_50iter_finetuned.bin"

echo "=== KLD Context Sweep $(date) ===" | tee "$OUTFILE"
echo "Chunks: $CHUNKS" | tee -a "$OUTFILE"

run_kld() {
	local ctk=$1 ctv=$2 ak=$3 av=$4 label=$5 extra_env=$6 ctx=$7
	echo -n "$label ctx=$ctx (αK=$ak αV=$av): " | tee -a "$OUTFILE"
	local result=$(env TURBO_TCQ_ALPHA="$ak" TURBO_TCQ_ALPHA_V="$av" $extra_env \
		$PPLBIN -m "$MODEL" -ctk "$ctk" -ctv "$ctv" -c $ctx $FA --chunks $CHUNKS \
		--kl-divergence --kl-divergence-base "/tmp/f16_base_ctx${ctx}.logits" -f "$WIKI" 2>&1)

	local layers=$(echo "$result" | grep "offloaded" | grep -o '[0-9]*/[0-9]*')
	if echo "$layers" | grep -q "^0/"; then
		echo "FATAL: 0 layers offloaded!" | tee -a "$OUTFILE"
		return 1
	fi

	local ppl=$(echo "$result" | grep "Mean PPL(Q)" | head -1 | awk '{print $4}')
	local kld=$(echo "$result" | grep "Mean    KLD:" | awk '{print $3}')
	local median_kld=$(echo "$result" | grep "Median  KLD:" | awk '{print $3}')
	local same_top=$(echo "$result" | grep "Same top p:" | awk '{print $4}')

	echo "PPL=$ppl  KLD=$kld  median=$median_kld  same_top=$same_top" | tee -a "$OUTFILE"
}

run_turbo4() {
	local alpha=$1 ctx=$2
	echo -n "turbo4 ctx=$ctx (α=$alpha): " | tee -a "$OUTFILE"
	local result=$(env TURBO4_ALPHA="$alpha" \
		$PPLBIN -m "$MODEL" -ctk turbo4 -ctv turbo4 -c $ctx $FA --chunks $CHUNKS \
		--kl-divergence --kl-divergence-base "/tmp/f16_base_ctx${ctx}.logits" -f "$WIKI" 2>&1)

	local layers=$(echo "$result" | grep "offloaded" | grep -o '[0-9]*/[0-9]*')
	if echo "$layers" | grep -q "^0/"; then
		echo "FATAL: 0 layers offloaded!" | tee -a "$OUTFILE"
		return 1
	fi

	local ppl=$(echo "$result" | grep "Mean PPL(Q)" | head -1 | awk '{print $4}')
	local kld=$(echo "$result" | grep "Mean    KLD:" | awk '{print $3}')
	local median_kld=$(echo "$result" | grep "Median  KLD:" | awk '{print $3}')
	local same_top=$(echo "$result" | grep "Same top p:" | awk '{print $4}')

	echo "PPL=$ppl  KLD=$kld  median=$median_kld  same_top=$same_top" | tee -a "$OUTFILE"
}

AV_POINTS="1.00 1.02 1.04 1.06 1.08 1.10 1.20"

for ctx in 8192 32768; do
	echo "" | tee -a "$OUTFILE"
	echo "=== Context $ctx ===" | tee -a "$OUTFILE"

	# Generate f16 base logits for this context
	echo "--- f16 base logits (ctx=$ctx) ---" | tee -a "$OUTFILE"
	$PPLBIN -m "$MODEL" -ctk f16 -ctv f16 -c $ctx $FA --chunks $CHUNKS \
		--save-all-logits "/tmp/f16_base_ctx${ctx}.logits" -f "$WIKI" 2>&1 | grep "Final estimate" | tee -a "$OUTFILE"

	# q8_0 baseline
	echo -n "q8_0 ctx=$ctx: " | tee -a "$OUTFILE"
	result=$(env $PPLBIN -m "$MODEL" -ctk q8_0 -ctv q8_0 -c $ctx $FA --chunks $CHUNKS \
		--kl-divergence --kl-divergence-base "/tmp/f16_base_ctx${ctx}.logits" -f "$WIKI" 2>&1)
	ppl=$(echo "$result" | grep "Mean PPL(Q)" | head -1 | awk '{print $4}')
	kld=$(echo "$result" | grep "Mean    KLD:" | awk '{print $3}')
	echo "PPL=$ppl  KLD=$kld" | tee -a "$OUTFILE"

	# turbo2_tcq αV sweep
	echo "--- turbo2_tcq αV sweep ---" | tee -a "$OUTFILE"
	for av in $AV_POINTS; do
		run_kld "turbo2_tcq" "turbo2_tcq" "1.0" "$av" "t2tcq" "TURBO_TCQ_CB2=$CB2" "$ctx"
	done

	# turbo3_tcq αV sweep
	echo "--- turbo3_tcq αV sweep ---" | tee -a "$OUTFILE"
	for av in $AV_POINTS; do
		run_kld "turbo3_tcq" "turbo3_tcq" "1.0" "$av" "t3tcq" "TURBO_TCQ_CB=$CB3" "$ctx"
	done

	# turbo4 α sweep
	echo "--- turbo4 α sweep ---" | tee -a "$OUTFILE"
	for a in $AV_POINTS; do
		run_turbo4 "$a" "$ctx"
	done
done

echo "" | tee -a "$OUTFILE"
echo "=== Context sweep complete $(date) ===" | tee -a "$OUTFILE"
