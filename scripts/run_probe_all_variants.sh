#!/usr/bin/env bash
# Authoring-to-principal probe across baseline + 5 trained variants.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=180

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F, 'tolower($2) ~ /vllm/ {gsub(/ /,"",$1); print $1}' \
    | xargs -r kill -9 2>/dev/null || true
  for _ in $(seq 1 30); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free" ] && [ "$free" -ge 80000 ]; then break; fi
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

# Each variant writes to its own probe_auth_to_principal_<name> dir.
declare -a VARIANTS=(
  "baseline|Qwen/Qwen3-8B"
  "trained|runs/qwen_sft_dpo_merged"
  "trained_v1|runs/qwen_sft_dpo_v1_merged"
  "trained_v1_lite|runs/qwen_sft_dpo_v1_lite_merged"
  "trained_v2|runs/qwen_sft_dpo_v2_merged"
  "trained_v21|runs/qwen_sft_dpo_v21_merged"
)

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model <<<"$entry"
  out_dir="runs/probe_auth_to_principal_${name}"
  echo "===== $name ====="
  kill_vllm
  start_vllm "$model" "runs/vllm_probe_${name}.log" || { echo "[fail] $name vllm start"; continue; }
  mkdir -p "$out_dir"
  python3 - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
PROBE_IDS = {"pb-author-to-principal-01", "pb-author-to-principal-02"}
all_items = load_items("items/v0")
items = [i for i in all_items if i.id in PROBE_IDS]
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted"],
    out_path="${out_dir}/trajectories.jsonl",
    counterparty_spec="claude-sonnet",
    parallel=4,
)
PYEOF
  python3 - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
score_grid(
    trajectory_jsonl="${out_dir}/trajectories.jsonl",
    items=load_items("items/v0"),
    out_path="${out_dir}/scored.jsonl",
    resume=True,
    parallel=6,
)
PYEOF
  echo "===== $name done ====="
done

kill_vllm
echo "[probe] all variants done"
