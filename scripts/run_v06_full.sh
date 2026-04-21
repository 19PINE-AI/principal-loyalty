#!/usr/bin/env bash
# v0.6 architectural-fix pipeline:
#   1. Build dpo_v06.jsonl (sentinel-rewritten v2 + v05 produce pairs)
#   2. Train + merge v06
#   3. Run probe (6-item reader_is_principal) under sentinel-enabled Agent
#   4. Run phase2 (30 items) under sentinel-enabled Agent
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
    sleep 3; t=$((t+3))
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; return 1; fi
  done
  echo "[vllm] ready after ${t}s"
}

run_probe_on() {
  local name="$1"
  local out_dir="runs/probe_auth_to_principal_${name}_v06"
  mkdir -p "$out_dir"
  python3 - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = [i for i in load_items("items/v0") if i.reader_is_principal]
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
}

run_phase2_on_v06() {
  mkdir -p runs/phase2_trained_v06
  python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = [i for i in load_items("items/v0") if not i.reader_is_principal]
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted"],
    out_path="runs/phase2_trained_v06/trajectories.jsonl",
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
    trajectory_jsonl="runs/phase2_trained_v06/trajectories.jsonl",
    items=load_items("items/v0"),
    out_path="runs/phase2_trained_v06/scored.jsonl",
    resume=True,
    parallel=6,
)
PYEOF
}

# === Step 1: build pairs ===
python3 scripts/build_dpo_v06.py

# === Step 2: train + merge ===
PL_DPO_PATH=data/dpo_v06.jsonl PL_DPO_DIR=runs/qwen_dpo_v06 \
  python3 scripts/train_qwen_dpo.py
python3 scripts/merge_lora.py \
  --adapter runs/qwen_dpo_v06 \
  --out runs/qwen_sft_dpo_v06_merged

# === Step 3: probe + phase2 on v06 ===
kill_vllm
start_vllm "runs/qwen_sft_dpo_v06_merged" "runs/vllm_v06_main.log" || exit 1
run_probe_on "trained_v06"
run_phase2_on_v06

kill_vllm
echo "[v06] pipeline done"
