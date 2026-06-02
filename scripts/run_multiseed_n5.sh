#!/usr/bin/env bash
# Extend the n=3 multi-rollout summaries to n=5 by adding seed4 and seed5
# for both the v4.1 DPO endpoint and the DAPO-v1 step_35 checkpoint, then
# run the paired Wilcoxon test.
#
# NOTE: NVML is broken on this box (590 kernel vs 595 userspace). Process
# detection uses lsof, not nvidia-smi.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=300

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

start_vllm() {
  local model_path="$1"
  local log_path="$2"
  echo "[vllm] starting $model_path -> $log_path"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT \
    >"$log_path" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm] died"; tail -30 "$log_path"; return 1; fi
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; tail -30 "$log_path"; return 1; fi
  done
  echo "[vllm] ready after ${t}s"
}

# 1. v4.1 endpoint multi-seed (add seed4, seed5)
kill_vllm
start_vllm "runs/qwen_sft_dpo_v4_1_merged" "logs/vllm_v4_1_multiseed.log" || exit 1
/home/ubuntu/aoi-env/bin/python scripts/multi_rollout_eval.py \
  --base runs/phase2_trained_v4_1 \
  --extra-seeds 4 \
  --counterparty claude-sonnet \
  --parallel 4 2>&1 | tee logs/multiseed_v4_1.log | tail -30

# 2. DAPO-v1 step_35 multi-seed (add seed4, seed5)
kill_vllm
start_vllm "runs/qwen_dapo_v1_step35_merged" "logs/vllm_dapo_v1_multiseed.log" || exit 1
/home/ubuntu/aoi-env/bin/python scripts/multi_rollout_eval.py \
  --base runs/phase3_dapo_v1_step35 \
  --extra-seeds 4 \
  --counterparty claude-sonnet \
  --parallel 4 2>&1 | tee logs/multiseed_dapo_v1.log | tail -30

# 3. Stop vLLM
kill_vllm

# 4. Paired Wilcoxon at n=5
echo "===== paired Wilcoxon (n=5) ====="
/home/ubuntu/aoi-env/bin/python scripts/paired_seed_test.py \
  --a runs/phase2_trained_v4_1 --b runs/phase3_dapo_v1_step35 \
  --a-seeds 5 --b-seeds 5 2>&1 | tee logs/paired_seed_test_n5.log
