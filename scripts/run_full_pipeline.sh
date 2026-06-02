#!/usr/bin/env bash
# Top-level pipeline:
#   1. Llama-3.1-8B pipeline (SFT -> DPO -> eval -> DAPO -> eval)
#   2. DAPO-variant multi-seed n=5 (v2, leak-only, v3) -> paired Wilcoxon vs v4.1
# Each sub-script is idempotent and skips done stages.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] ===== STAGE A: Llama-3.1-8B pipeline ====="
bash scripts/run_llama_pipeline.sh 2>&1 | tee -a logs/full_pipeline_llama.log
LLAMA_EXIT=${PIPESTATUS[0]}
echo "[$(ts)] Llama pipeline exit=$LLAMA_EXIT"

echo "[$(ts)] ===== STAGE B: DAPO-variant multi-seed n=5 ====="
bash scripts/run_dapo_variants_multiseed.sh 2>&1 | tee -a logs/full_pipeline_dapo_variants.log
DV_EXIT=${PIPESTATUS[0]}
echo "[$(ts)] DAPO-variant multi-seed exit=$DV_EXIT"

echo "[$(ts)] [DONE] full pipeline complete (llama=$LLAMA_EXIT dapo_variants=$DV_EXIT)"
