#!/usr/bin/env bash
# Chain to run after v4.1 multi-seed trajectories complete:
# - kill v4.1 vLLM
# - start DAPO-v1 step35 vLLM
# - generate DAPO-v1 seed4+seed5 trajectories
# - kill DAPO-v1 vLLM
# - score all 4 new seed dirs (v4.1 seed4/5, dapo seed4/5)
# - run paired Wilcoxon at n=5
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=420

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

wait_vllm_ready() {
  local pid_pattern="$1"; local log="$2"
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! pgrep -f "$pid_pattern" >/dev/null; then
      echo "[vllm] DIED while waiting for ready"; tail -30 "$log"; return 1
    fi
    if [ $t -ge $WAIT_TIMEOUT ]; then
      echo "[vllm] timeout ${t}s"; tail -30 "$log"; return 1
    fi
  done
  echo "[vllm] READY after ${t}s"
}

start_vllm() {
  local model_path="$1"; local log="$2"
  echo "[vllm] starting $model_path"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT >"$log" 2>&1 &
}

# 1. Kill v4.1 vLLM
kill_vllm

# 2. Start DAPO-v1 step35 vLLM
start_vllm "runs/qwen_dapo_v1_step35_merged" "logs/vllm_dapo_v1_step35_multiseed.log"
wait_vllm_ready "model runs/qwen_dapo_v1_step35_merged" "logs/vllm_dapo_v1_step35_multiseed.log" || exit 1

# 3. Generate DAPO-v1 seed4+seed5 trajectories
/home/ubuntu/polar-env/bin/python scripts/run_traj_only.py \
  --seed-dirs runs/phase3_dapo_v1_step35_seed4 runs/phase3_dapo_v1_step35_seed5 \
  --subject qwen-8b-local --counterparty claude-sonnet --parallel 4 \
  2>&1 | tee logs/multiseed_dapo_v1_traj.log | tail -30

# 4. Kill DAPO-v1 vLLM (free GPU for Llama SFT later)
kill_vllm

# 5. Score all 4 new seed dirs (OpenRouter judge)
/home/ubuntu/polar-env/bin/python scripts/score_only.py \
  --seed-dirs runs/phase2_trained_v4_1_seed4 runs/phase2_trained_v4_1_seed5 \
              runs/phase3_dapo_v1_step35_seed4 runs/phase3_dapo_v1_step35_seed5 \
  --parallel 8 2>&1 | tee logs/multiseed_scoring.log | tail -30

# 6. Paired Wilcoxon at n=5
echo "===== paired Wilcoxon n=5 ====="
/home/ubuntu/polar-env/bin/python scripts/paired_seed_test.py \
  --a runs/phase2_trained_v4_1 --b runs/phase3_dapo_v1_step35 \
  --a-seeds 5 --b-seeds 5 2>&1 | tee logs/paired_seed_test_n5.log

echo "[DONE] multiseed n=5 pipeline complete"
