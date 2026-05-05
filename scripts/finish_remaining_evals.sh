#!/usr/bin/env bash
# Finish the 2 remaining evals that the orchestrator failed on (vllm OOM
# from zombie engine cores). vllm with v1_step30 is already running and the
# step_30 eval is in progress as PID 4141571. After it finishes, kill vllm,
# swap to v3_step30 model, run the v3 eval.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=300

kill_vllm_thoroughly() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F, 'tolower($2) ~ /vllm/ {gsub(/ /,"",$1); print $1}'); do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 5
}

start_vllm() {
  local model_path="$1"
  local log_path="$2"
  echo "[vllm] starting $model_path"
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT \
    >"$log_path" 2>&1 &
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; return 1; fi
  done
  echo "[vllm] ready after ${t}s"
}

# 1. Wait for step_30 eval to finish (PID 4141571 from main thread)
echo "[1/2] waiting on step_30 eval pid 4141571"
while kill -0 4141571 2>/dev/null; do
  sleep 30
done
echo "[1/2] step_30 done"

# 2. Restart vllm with v3_step30 checkpoint
kill_vllm_thoroughly
start_vllm runs/qwen_dapo_v3_step30_merged logs/vllm_phase3_dapo_v3_step30.log

# Backup broken eval, run v3_step_30
mv runs/phase3_dapo_v3_step30 runs/phase3_dapo_v3_step30_BROKEN_AUTH401 2>/dev/null
mkdir -p runs/phase3_dapo_v3_step30
PL_STEP=30 python3 scripts/run_phase3_dapo_v3.py 2>&1 | tee logs/phase3_dapo_v3_step30_rerun.log

echo "All re-runs complete."
