#!/usr/bin/env bash
# Wait for the Llama pipeline (Stage A) to complete, then run:
#   (i)   on-policy distillation on Qwen3-8B starting from v4.1 SFT+DPO
#   (ii)  DAPO-variants multi-seed (deferred via /tmp/pl_skip_dapo_variants flag)
# This is the replacement for the original Stage B.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Wait for full_pipeline_v2 background task to finish (Llama Stage A).
echo "[$(ts)] waiting for runs/phase3_llama_sft_dpo/scored.jsonl OR for full_pipeline_v2.log to settle"
LAST_LOG_MTIME=0
SETTLE_NEEDED=180
while true; do
  if [ -f runs/phase3_llama_sft_dpo/scored.jsonl ] && \
     [ "$(wc -l < runs/phase3_llama_sft_dpo/scored.jsonl)" -ge 108 ]; then
    if [ -f runs/phase3_llama_dapo_v1_step35/scored.jsonl ] && \
       [ "$(wc -l < runs/phase3_llama_dapo_v1_step35/scored.jsonl)" -ge 108 ]; then
      echo "[$(ts)] both Llama SFT+DPO and Llama DAPO eval complete"
      break
    fi
    # Llama DAPO may not finish (OOM). Watch log mtime.
    log_mtime=$(stat -c %Y logs/full_pipeline_v2.log 2>/dev/null || echo 0)
    now=$(date -u +%s)
    if [ $((now - log_mtime)) -ge $SETTLE_NEEDED ]; then
      # log hasn't changed in 3 min AND we have SFT+DPO eval. Likely DAPO failed or done.
      echo "[$(ts)] full_pipeline_v2.log idle ${SETTLE_NEEDED}s; treating as complete"
      break
    fi
  fi
  sleep 30
done

echo "[$(ts)] ===== POST-LLAMA STAGE 1: on-policy distillation on Qwen3-8B ====="
# Make sure no vLLM is running
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 5

PL_ONPOLICY_ITERS=3 \
PL_ONPOLICY_BASE=runs/qwen_sft_dpo_v4_1_merged \
PL_ONPOLICY_N_SAMPLES=1 \
PL_ONPOLICY_TEMP=1.0 \
bash scripts/run_onpolicy_distill.sh 2>&1 | tee logs/onpolicy_main.log | tail -50
echo "[$(ts)] on-policy distillation exit=${PIPESTATUS[0]}"

echo "[$(ts)] ===== POST-LLAMA STAGE 2: DAPO-variants multi-seed (lifting skip flag) ====="
rm -f /tmp/pl_skip_dapo_variants
bash scripts/run_dapo_variants_multiseed.sh 2>&1 | tee logs/dapo_variants_main.log | tail -30

echo "[$(ts)] [DONE] post-Llama pipeline complete"
