#!/bin/bash
# KLD alpha sweep for turbo2_tcq, turbo3_tcq, turbo4
# Finds KLD-optimal temperature scaling for each quant type.
# Run from the build directory root (e.g., /root/exp-kld-sweep)
set -e

PPLBIN="build/bin/llama-perplexity"
MODEL="/root/Qwen3.5-27B-heretic.Q6_K.gguf"
WIKI="/root/wikitext-2-raw/wiki.test.raw"
CTX=2048
CHUNKS=8
FA="-fa on"
BASE_LOGITS="/tmp/f16_base_sweep.logits"
OUTFILE="kld_sweep_results.txt"

CB2="codebooks/2bit/tcq_2bit_cuda_200iter.bin"
CB3="codebooks/3bit/cb_50iter_finetuned.bin"

echo "=== KLD Alpha Sweep $(date) ===" | tee "$OUTFILE"
echo "Model: $MODEL" | tee -a "$OUTFILE"
echo "Context: $CTX, Chunks: $CHUNKS" | tee -a "$OUTFILE"
echo "2-bit codebook: $CB2" | tee -a "$OUTFILE"
echo "3-bit codebook: $CB3" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

# Verify GPU offloading works
echo "--- Verifying GPU offloading ---" | tee -a "$OUTFILE"
OFFLOAD=$($PPLBIN -m "$MODEL" -ctk f16 -ctv f16 -c $CTX $FA --chunks 1 -f "$WIKI" 2>&1 | grep "offloaded")
echo "$OFFLOAD" | tee -a "$OUTFILE"
if echo "$OFFLOAD" | grep -q "offloaded 0/"; then
	echo "FATAL: 0 layers offloaded! Aborting." | tee -a "$OUTFILE"
	exit 1
fi

# Step 1: Generate f16 base logits
echo "--- Generating f16 base logits ---" | tee -a "$OUTFILE"
$PPLBIN -m "$MODEL" -ctk f16 -ctv f16 -c $CTX $FA --chunks $CHUNKS \
	--save-all-logits "$BASE_LOGITS" -f "$WIKI" 2>&1 | grep "Final estimate" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

run_kld() {
	local ctk=$1 ctv=$2 ak=$3 av=$4 label=$5 extra_env=$6
	echo -n "$label (αK=$ak αV=$av): " | tee -a "$OUTFILE"
	local result=$(env TURBO_TCQ_ALPHA="$ak" TURBO_TCQ_ALPHA_V="$av" $extra_env \
		$PPLBIN -m "$MODEL" -ctk "$ctk" -ctv "$ctv" -c $CTX $FA --chunks $CHUNKS \
		--kl-divergence --kl-divergence-base "$BASE_LOGITS" -f "$WIKI" 2>&1)

	# Verify GPU offload
	local layers=$(echo "$result" | grep "offloaded" | grep -o '[0-9]*/[0-9]*')
	if echo "$layers" | grep -q "^0/"; then
		echo "FATAL: 0 layers offloaded!" | tee -a "$OUTFILE"
		return 1
	fi

	# Verify buffer size (catch CPU fallback)
	local ksize=$(echo "$result" | grep "K ($ctk)" | grep -o 'K ([^)]*): *[0-9.]* MiB' | grep -o '[0-9.]* MiB')

	local ppl=$(echo "$result" | grep "Mean PPL(Q)" | head -1 | awk '{print $4}')
	local kld=$(echo "$result" | grep "Mean    KLD:" | awk '{print $3}')
	local median_kld=$(echo "$result" | grep "Median  KLD:" | awk '{print $3}')
	local same_top=$(echo "$result" | grep "Same top p:" | awk '{print $4}')
	local rms_dp=$(echo "$result" | grep "RMS" | awk '{print $4}')

	echo "PPL=$ppl  KLD=$kld  median=$median_kld  same_top=$same_top  rms_dp=$rms_dp  K=$ksize" | tee -a "$OUTFILE"
}

# Step 2: q8_0 baseline
echo "=== q8_0 baseline ===" | tee -a "$OUTFILE"
run_kld "q8_0" "q8_0" "1.0" "1.0" "q8_0" ""
echo "" | tee -a "$OUTFILE"

# Step 3: turbo2_tcq αV sweep (αK=1.0 fixed)
echo "=== turbo2_tcq αV sweep (αK=1.0, CUDA 200iter codebook) ===" | tee -a "$OUTFILE"
for av in 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.20 1.25 1.30; do
	run_kld "turbo2_tcq" "turbo2_tcq" "1.0" "$av" "t2tcq" "TURBO_TCQ_CB2=$CB2"
done
echo "" | tee -a "$OUTFILE"

# Step 4: turbo2_tcq αK sweep (αV=best from step 3, read from results)
# For now sweep αK with αV=1.0 (no V scaling) to isolate K effect
echo "=== turbo2_tcq αK sweep (αV=1.0, CUDA 200iter codebook) ===" | tee -a "$OUTFILE"
for ak in 1.00 1.02 1.04 1.06 1.08 1.10 1.14 1.20; do
	run_kld "turbo2_tcq" "turbo2_tcq" "$ak" "1.0" "t2tcq" "TURBO_TCQ_CB2=$CB2"
done
echo "" | tee -a "$OUTFILE"

# Step 5: turbo3_tcq αV sweep (αK=1.0 fixed)
echo "=== turbo3_tcq αV sweep (αK=1.0, CUDA finetuned codebook) ===" | tee -a "$OUTFILE"
for av in 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.20 1.25 1.30; do
	run_kld "turbo3_tcq" "turbo3_tcq" "1.0" "$av" "t3tcq" "TURBO_TCQ_CB=$CB3"
done
echo "" | tee -a "$OUTFILE"

# Step 6: turbo3_tcq αK sweep (αV=1.0)
echo "=== turbo3_tcq αK sweep (αV=1.0, CUDA finetuned codebook) ===" | tee -a "$OUTFILE"
for ak in 1.00 1.02 1.04 1.06 1.08 1.10 1.14 1.20; do
	run_kld "turbo3_tcq" "turbo3_tcq" "$ak" "1.0" "t3tcq" "TURBO_TCQ_CB=$CB3"
done
echo "" | tee -a "$OUTFILE"

# Step 7: turbo4 α sweep (single α for both K and V)
echo "=== turbo4 α sweep (single α, PolarQuant centroids) ===" | tee -a "$OUTFILE"
for a in 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.20 1.25 1.30; do
	echo -n "turbo4 (α=$a): " | tee -a "$OUTFILE"
	result=$(env TURBO4_ALPHA="$a" \
		$PPLBIN -m "$MODEL" -ctk turbo4 -ctv turbo4 -c $CTX $FA --chunks $CHUNKS \
		--kl-divergence --kl-divergence-base "$BASE_LOGITS" -f "$WIKI" 2>&1)

	layers=$(echo "$result" | grep "offloaded" | grep -o '[0-9]*/[0-9]*')
	if echo "$layers" | grep -q "^0/"; then
		echo "FATAL: 0 layers offloaded!" | tee -a "$OUTFILE"
		continue
	fi

	ppl=$(echo "$result" | grep "Mean PPL(Q)" | head -1 | awk '{print $4}')
	kld=$(echo "$result" | grep "Mean    KLD:" | awk '{print $3}')
	median_kld=$(echo "$result" | grep "Median  KLD:" | awk '{print $3}')
	same_top=$(echo "$result" | grep "Same top p:" | awk '{print $4}')
	rms_dp=$(echo "$result" | grep "RMS" | awk '{print $4}')

	echo "PPL=$ppl  KLD=$kld  median=$median_kld  same_top=$same_top  rms_dp=$rms_dp" | tee -a "$OUTFILE"
done
echo "" | tee -a "$OUTFILE"

echo "=== Sweep complete $(date) ===" | tee -a "$OUTFILE"
