#!/usr/bin/env bash
# Run v22 across the author-to-principal probe AND phase2 full eval.
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
      echo "[vllm] timeout"; return 1
    fi
  done
  echo "[vllm] ready after ${t}s"
}

kill_vllm
start_vllm "runs/qwen_sft_dpo_v22_merged" "runs/vllm_v22.log" || exit 1

# === Probe: 2 author-to-principal items, 2 arms
mkdir -p runs/probe_auth_to_principal_trained_v22
python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
PROBE_IDS = {"pb-author-to-principal-01", "pb-author-to-principal-02"}
items = [i for i in load_items("items/v0") if i.id in PROBE_IDS]
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted"],
    out_path="runs/probe_auth_to_principal_trained_v22/trajectories.jsonl",
    counterparty_spec="claude-sonnet",
    parallel=4,
)
PYEOF

python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
score_grid(
    trajectory_jsonl="runs/probe_auth_to_principal_trained_v22/trajectories.jsonl",
    items=load_items("items/v0"),
    out_path="runs/probe_auth_to_principal_trained_v22/scored.jsonl",
    resume=True,
    parallel=6,
)
PYEOF

# === Full phase2 eval on v22: 30 items, 2 arms
mkdir -p runs/phase2_trained_v22
python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
# Exclude the 2 author-to-principal probe items from the phase2 set — they
# stay as a separate probe per §4.2; n=30 headline set is unchanged.
PROBE_IDS = {"pb-author-to-principal-01", "pb-author-to-principal-02"}
items = [i for i in load_items("items/v0") if i.id not in PROBE_IDS]
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted"],
    out_path="runs/phase2_trained_v22/trajectories.jsonl",
    counterparty_spec="claude-sonnet",
    parallel=4,
)
PYEOF

python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
score_grid(
    trajectory_jsonl="runs/phase2_trained_v22/trajectories.jsonl",
    items=load_items("items/v0"),
    out_path="runs/phase2_trained_v22/scored.jsonl",
    resume=True,
    parallel=6,
)
PYEOF

kill_vllm
echo "[v22] probe + phase2 done"
