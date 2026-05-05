#!/usr/bin/env bash
# Re-run the 5 evaluations that all-errored on Anthropic API 401 during a
# transient auth outage. Their original scored.jsonl has 108 single-turn
# trajectories instead of the multi-turn results the paper reports.
#
# Backed-up originals -> runs/<name>_BROKEN_AUTH401/.
#
# Serially swaps the vLLM model between checkpoints (only one GPU available,
# competing with other workloads).
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=300
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.25}

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  sleep 3
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
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $PORT \
    >"$log_path" 2>&1 &
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; return 1; fi
  done
  echo "[vllm] ready after ${t}s"
}

backup_and_clear() {
  local d="$1"
  if [ -d "$d" ] && [ ! -d "${d}_BROKEN_AUTH401" ]; then
    mv "$d" "${d}_BROKEN_AUTH401"
  fi
  mkdir -p "$d"
}

eval_pair() {
  # eval_pair <model_path> <out_dir> <run_script> [extra_env]
  local model_path="$1" out_dir="$2" run_script="$3" extra_env="${4:-}"
  echo "=== $out_dir ==="
  kill_vllm
  start_vllm "$model_path" "logs/vllm_$(basename "$out_dir").log" || return 1
  backup_and_clear "$out_dir"
  env $extra_env python3 "$run_script" 2>&1 | tee "logs/$(basename "$out_dir")_rerun.log"
  echo "=== done $out_dir ==="
}

# Order: smallest+most critical first
eval_pair runs/qwen_sft_dpo_v4_1_merged   runs/phase2_trained_v4_1     scripts/run_phase2_trained_v4_1.py
eval_pair runs/qwen_sft_dpo_v4_merged     runs/phase2_trained_v4       scripts/run_phase2_trained_v4.py
eval_pair runs/qwen_dapo_v1_step30_merged runs/phase3_dapo_v1_step30   scripts/run_phase3_dapo_v1_step30.py
eval_pair runs/qwen_dapo_v3_step30_merged runs/phase3_dapo_v3_step30 scripts/run_phase3_dapo_v3.py "PL_STEP=30"

echo "All re-runs complete."
