#!/usr/bin/env bash
# Orchestrate Phase 2 re-eval on expanded 30-item set across all 5 variants.
# Harness has resume logic, so only the 6 new items fire per variant.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=180

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  # Kill any VLLM::EngineCore subprocesses that survived the API server
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F, 'tolower($2) ~ /vllm/ {gsub(/ /,"",$1); print $1}' \
    | xargs -r kill -9 2>/dev/null || true
  # Wait for GPU to actually free
  for _ in $(seq 1 30); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free" ] && [ "$free" -ge 80000 ]; then
      break
    fi
    sleep 2
  done
  sleep 3
}

start_vllm() {
  local model_path="$1"
  local log_path="$2"
  echo "[vllm] starting $model_path -> $log_path"
  nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.55 \
    --port $PORT \
    >"$log_path" 2>&1 &
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 3
    t=$((t+3))
    if [ $t -ge $WAIT_TIMEOUT ]; then
      echo "[vllm] timeout waiting for $model_path"; return 1
    fi
  done
  echo "[vllm] ready ($model_path) after ${t}s"
}

declare -a VARIANTS=(
  "trained|runs/qwen_sft_dpo_merged|run_phase2_trained.py|score_phase2_trained.py"
  "trained_v1|runs/qwen_sft_dpo_v1_merged|run_phase2_trained_v1.py|score_phase2_trained_v1.py"
  "trained_v1_lite|runs/qwen_sft_dpo_v1_lite_merged|run_phase2_trained_v1_lite.py|score_phase2_trained_v1_lite.py"
  "trained_v2|runs/qwen_sft_dpo_v2_merged|run_phase2_trained_v2.py|score_phase2_trained_v2.py"
)

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model run score <<<"$entry"
  echo "===== $name ====="
  kill_vllm
  start_vllm "$model" "runs/vllm_${name}_v03.log" || { echo "[fail] $name vllm start"; continue; }
  echo "[harness] python3 scripts/$run"
  python3 "scripts/$run" 2>&1 | tail -20
  echo "[score] python3 scripts/$score"
  python3 "scripts/$score" 2>&1 | tail -20
  echo "===== $name done ====="
done

kill_vllm
echo "[v03] all variants done"
