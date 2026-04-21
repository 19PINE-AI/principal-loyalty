#!/usr/bin/env bash
# v0.5 pipeline:
#  1. Run v2 on the 4 new to-principal items (get refusal fold-points)
#  2. Rebuild produce-pairs under BOTH PLAIN+PROMPTED system templates
#  3. Build dpo_v05 = dpo_v2 + new produce-pairs
#  4. Train + merge v05
#  5. Run baseline + v2 + v05 on the 6-item to-principal probe
#  6. Run v05 on full phase2 (30 items × 2 arms)
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
  # args: variant_name out_dir
  local name="$1"
  local out_dir="runs/probe_auth_to_principal_${name}_v05"
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

run_phase2_on_v05() {
  mkdir -p runs/phase2_trained_v05
  python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
# Phase2 headline set = items that are NOT reader_is_principal probe items
items = [i for i in load_items("items/v0") if not i.reader_is_principal]
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted"],
    out_path="runs/phase2_trained_v05/trajectories.jsonl",
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
    trajectory_jsonl="runs/phase2_trained_v05/trajectories.jsonl",
    items=load_items("items/v0"),
    out_path="runs/phase2_trained_v05/scored.jsonl",
    resume=True,
    parallel=6,
)
PYEOF
}

# === Step 1: v2 rollouts on ALL reader_is_principal items (6 total) ===
kill_vllm
start_vllm "runs/qwen_sft_dpo_v2_merged" "runs/vllm_v05_step1.log" || exit 1
run_probe_on "trained_v2"

# === Step 2: rebuild produce-pairs under both system templates ===
kill_vllm
echo "[v05] regenerating produce-pairs under PLAIN+PROMPTED templates"
python3 scripts/build_dpo_author_to_principal.py \
  --trajectories runs/probe_auth_to_principal_trained_v2_v05/trajectories.jsonl \
  --trajectories runs/probe_auth_to_principal_trained_v2/trajectories.jsonl \
  --trajectories runs/probe_auth_to_principal_trained_v21/trajectories.jsonl \
  --trajectories runs/probe_auth_to_principal_trained_v1/trajectories.jsonl \
  --trajectories runs/probe_auth_to_principal_trained_v1_lite/trajectories.jsonl \
  --out data/dpo_author_to_principal_v05.jsonl \
  --max_pairs 40 \
  --both_systems

# === Step 3: dpo_v05 = dpo_v2 + new produce-pairs ===
python3 - <<'PYEOF'
import json, pathlib
root = pathlib.Path("/home/ubuntu/principal-loyalty")
parts = [root/"data"/"dpo_v2.jsonl", root/"data"/"dpo_author_to_principal_v05.jsonl"]
out = root/"data"/"dpo_v05.jsonl"
n=0
with out.open("w") as fo:
    for p in parts:
        for line in p.read_text().splitlines():
            if line.strip():
                fo.write(line+"\n"); n+=1
print(f"[v05] dpo_v05.jsonl wrote {n} pairs")
PYEOF

# === Step 4: train + merge v05 ===
PL_DPO_PATH=data/dpo_v05.jsonl PL_DPO_DIR=runs/qwen_dpo_v05 \
  python3 scripts/train_qwen_dpo.py
python3 scripts/merge_lora.py \
  --adapter runs/qwen_dpo_v05 \
  --out runs/qwen_sft_dpo_v05_merged

# === Step 5: run probe on v05 + run baseline on the 4 new items (for completeness) ===
kill_vllm
start_vllm "runs/qwen_sft_dpo_v05_merged" "runs/vllm_v05_step5.log" || exit 1
run_probe_on "trained_v05"
run_phase2_on_v05

# === Step 6: baseline on 6-item probe (for before/after table) ===
kill_vllm
start_vllm "Qwen/Qwen3-8B" "runs/vllm_v05_baseline.log" || exit 1
run_probe_on "baseline"

kill_vllm
echo "[v05] pipeline done"
