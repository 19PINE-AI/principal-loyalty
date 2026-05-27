#!/usr/bin/env bash
# Cross-base per-token KL with SAME-FAMILY teachers (Llama-3.1-70B-AWQ teacher
# for Llama-3.1-8B student; Mixtral-8x7B-AWQ teacher for Mistral-7B student).
# Same-family tokenizers eliminate the cross-vocab issue noted earlier.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

STUDENT_PORT=8000
TEACHER_PORT=8001

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do
    kill -9 $p 2>/dev/null || true
  done
  sleep 8
}

wait_for_gpu() {
  local needed="${1:-30}"
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    if [ "$free" -ge "$needed" ]; then echo "[gpu] free=${free}GB OK"; return 0; fi
    echo "[gpu] free=${free}GB need ${needed}GB; sleeping 60s"; sleep 60
  done
}

# Variant table: (base_name, student_base, teacher_model, teacher_vendor_spec)
# We'll do Llama first (most likely to fit), then Mistral if GPU allows.

run_xbase() {
  local NAME="$1"
  local STUDENT_BASE="$2"
  local TEACHER_MODEL="$3"
  local STUDENT_SERVED_NAME="$4"
  local TEACHER_VENDOR="$5"
  local TEACHER_GPU_UTIL="$6"

  local TRAJ="data/pertoken_kl/${NAME}_iter1_trajectories.jsonl"
  local TOPK="data/pertoken_kl/${NAME}_iter1_topk.jsonl"
  local ADAPTER="runs/${NAME}_pertoken_kl_iter1"
  local MERGED="runs/${NAME}_pertoken_kl_iter1_merged"
  local EVAL_DIR="runs/phase5_pertoken_kl_${NAME}_iter1"

  echo "===== ${NAME} ====="

  # 1. Reuse existing on-policy trajectories from Tier B4 if available
  local EXISTING_TRAJ="data/onpolicy_${NAME}_iter1_trajectories.jsonl"
  if [ ! -f "$TRAJ" ] && [ -f "$EXISTING_TRAJ" ]; then
    cp "$EXISTING_TRAJ" "$TRAJ"
    echo "[${NAME}] copied existing trajectories from $EXISTING_TRAJ"
  fi
  if [ ! -f "$TRAJ" ]; then
    echo "[${NAME}] no trajectories — skipping (need Tier B4 trajectories)"
    return 0
  fi
  echo "[${NAME}] $(wc -l < $TRAJ) trajectories"

  # 2. Bring up teacher vLLM, collect per-token top-K
  if [ ! -f "$TOPK" ] || [ "$(wc -l < $TOPK)" -lt 30 ]; then
    kill_vllm
    wait_for_gpu 40
    echo "[vllm-teacher] starting $TEACHER_MODEL on :$TEACHER_PORT"
    # --enforce-eager + bumped engine-ready timeout to avoid CUDA-graph
    # capture timing out on large AWQ models (see project_awq32b_vllm memory).
    VLLM_ENGINE_READY_TIMEOUT_S=1800 \
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$TEACHER_MODEL" \
      --served-model-name "$TEACHER_VENDOR" \
      --dtype auto --quantization awq --max-model-len 8192 \
      --gpu-memory-utilization "$TEACHER_GPU_UTIL" \
      --enforce-eager \
      --port $TEACHER_PORT > "logs/vllm_teacher_${NAME}.log" 2>&1 &
    local pid=$!
    local t=0
    until curl -sf http://localhost:$TEACHER_PORT/v1/models >/dev/null 2>&1; do
      sleep 15; t=$((t+15))
      if ! kill -0 $pid 2>/dev/null; then echo "[teacher] DIED"; tail -20 logs/vllm_teacher_${NAME}.log; return 1; fi
      if [ $t -ge 1500 ]; then echo "[teacher] timeout"; return 1; fi
    done
    echo "[vllm-teacher] ready after ${t}s"

    # Collect: pertoken_kl_collect.py needs adaptation for non-Qwen teachers
    # We hardcode the teacher URL/model via env-var override
    TEACHER_URL="http://localhost:$TEACHER_PORT/v1/completions" \
    TEACHER_MODEL_NAME="$TEACHER_VENDOR" \
    TOKENIZER_NAME="$STUDENT_BASE" \
    /home/ubuntu/aoi-env/bin/python scripts/pertoken_kl_collect.py \
      --trajectories "$TRAJ" \
      --out "$TOPK" \
      --top-k 20 --parallel 4 --max-turns-per-item 4 2>&1 \
      | tee "logs/pertoken_kl_${NAME}_collect.log" | tail -10
  fi
  echo "[${NAME}] $(wc -l < $TOPK) topk records"
  if [ "$(wc -l < $TOPK)" -lt 30 ]; then echo "[${NAME}] too few records; skip"; return 0; fi

  # 3. Train per-token KL on student base
  if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
    kill_vllm
    wait_for_gpu 22
    PL_MODEL_ID="$STUDENT_BASE" \
    PL_KL_PATH="$TOPK" \
    PL_OUT_DIR="$ADAPTER" \
    PL_KL_EPOCHS=3 PL_KL_LR=5e-5 PL_KL_BS=1 PL_KL_GRAD_ACCUM=8 PL_KL_MAX_LEN=3000 \
    /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
      | tee "logs/pertoken_kl_${NAME}_train.log" | tail -30
  fi

  # 4. Merge
  if [ ! -d "$MERGED" ]; then
    /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
      --adapter "$ADAPTER" --base "$STUDENT_BASE" --out "$MERGED" 2>&1 | tail -3
    /home/ubuntu/polar-env/bin/python -c "
import json, os
p = '$MERGED/tokenizer_config.json'
if os.path.exists(p):
    c = json.load(open(p))
    if 'extra_special_tokens' in c:
        del c['extra_special_tokens']
        json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
"
  fi

  # 5. Eval (served as 'qwen-8b-local' so harness routing works)
  mkdir -p "$EVAL_DIR"
  if [ ! -f "$EVAL_DIR/scored.jsonl" ] || [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -lt 100 ]; then
    kill_vllm
    wait_for_gpu 22
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$MERGED" \
      --served-model-name "$STUDENT_SERVED_NAME" qwen-8b-local \
      --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.20 \
      --port $STUDENT_PORT >"logs/vllm_eval_${NAME}.log" 2>&1 &
    local pid=$!
    local t=0
    until curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; do
      sleep 10; t=$((t+10))
      if ! kill -0 $pid 2>/dev/null; then echo "[student] DIED"; return 1; fi
      if [ $t -ge 300 ]; then return 1; fi
    done
    /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF
    /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$EVAL_DIR/trajectories.jsonl" --require 100 --allow-error-frac 0.05
    /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
           items=items, out_path="$EVAL_DIR/scored.jsonl",
           resume=True, parallel=4)
PYEOF
    kill_vllm
  fi

  # 6. Summary
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$EVAL_DIR/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] ${NAME}_pertoken_kl: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
}

# Run Llama family (Llama-3.1-70B-Instruct-AWQ teacher → Llama-3.1-8B student).
# 70B-AWQ weights eat ~35GB; bumped util 0.40 -> 0.60 to leave headroom for KV.
# STUDENT_SERVED_NAME is Qwen/Qwen3-8B so the harness's qwen-8b-local vendor
# (which expects model="Qwen/Qwen3-8B") can route to it.
run_xbase llama runs/llama_sft_dpo_v4_1_merged \
  hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  Qwen/Qwen3-8B \
  llama-70b-teacher \
  0.60

# Mistral variant deferred: pertoken_kl_collect.py returns 0 records on
# mistral trajectories (silent filter — needs debugging).

echo "[DONE] Cross-base per-token KL complete"
