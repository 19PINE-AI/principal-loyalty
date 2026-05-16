#!/usr/bin/env bash
# K=3 iterations of per-token KL distillation.
# Each iter:
#   - Sample student trajectories under current iter's model (vLLM, PLAIN arm)
#   - Bring up Qwen3-32B-AWQ teacher (vLLM port 8001), collect per-token top-K
#   - Train QLoRA on Qwen3-8B current ckpt + per-token KL loss
#   - Merge LoRA, eval on 108-grid with claude-sonnet counterparty
set -uo pipefail
cd /home/ubuntu/principal-loyalty

STUDENT_PORT=8000
TEACHER_PORT=8001

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  # Engine cores set process title "VLLM::EngineCore" — find by /proc/*/comm and kill
  for pid in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do
    kill -9 $pid 2>/dev/null || true
  done
  sleep 8
  # Verify free GPU before returning
  local free_gb=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))")
  echo "[kill_vllm] post-kill free=${free_gb}GB"
}

wait_for_gpu() {
  local needed_gb="${1:-20}"
  while :; do
    local free_gb=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))")
    if [ "$free_gb" -ge "$needed_gb" ]; then
      echo "[wait_for_gpu] free=${free_gb}GB OK"
      return 0
    fi
    echo "[wait_for_gpu] free=${free_gb}GB, need ${needed_gb}GB; sleeping 30s"
    sleep 30
  done
}

start_student_vllm() {
  local model="$1"; local log="$2"
  local attempt=0; local max_attempts=5
  while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt+1))
    wait_for_gpu 22
    echo "[vllm-student] attempt ${attempt}/${max_attempts}: starting $model on :$STUDENT_PORT"
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$model" \
      --served-model-name Qwen/Qwen3-8B qwen-8b-local \
      --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.20 \
      --port $STUDENT_PORT >"$log" 2>&1 &
    local pid=$!
    local t=0
    local ready=0
    while [ $t -lt 240 ]; do
      sleep 10; t=$((t+10))
      if curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; then
        echo "[vllm-student] READY after ${t}s (attempt ${attempt})"; ready=1; break
      fi
      if ! kill -0 $pid 2>/dev/null; then break; fi
    done
    [ $ready -eq 1 ] && return 0
    echo "[vllm-student] attempt ${attempt} failed; cleanup + retry"
    kill -9 $pid 2>/dev/null
    for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
    sleep 20
  done
  echo "[vllm-student] gave up after ${max_attempts} attempts"; return 1
}

start_teacher_vllm() {
  local attempt=0; local max_attempts=5
  while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt+1))
    wait_for_gpu 26
    echo "[vllm-teacher] attempt ${attempt}/${max_attempts}: Qwen3-32B-AWQ on :$TEACHER_PORT"
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --served-model-name Qwen/Qwen3-32B qwen-32b-local \
    --dtype auto --quantization awq --max-model-len 8192 \
    --gpu-memory-utilization 0.25 \
    --port $TEACHER_PORT >"logs/vllm_qwen32b_K3.log" 2>&1 &
    local pid=$!
    local t=0
    local ready=0
    while [ $t -lt 300 ]; do
      sleep 10; t=$((t+10))
      if curl -sf http://localhost:$TEACHER_PORT/v1/models >/dev/null 2>&1; then
        echo "[vllm-teacher] READY after ${t}s (attempt ${attempt})"; ready=1; break
      fi
      if ! kill -0 $pid 2>/dev/null; then break; fi
    done
    [ $ready -eq 1 ] && return 0
    echo "[vllm-teacher] attempt ${attempt} failed; cleanup + retry"
    kill -9 $pid 2>/dev/null
    for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
    sleep 20
  done
  echo "[vllm-teacher] gave up after ${max_attempts} attempts"; return 1
}

for ITER in 2 3; do
  PREV=$((ITER - 1))
  PREV_MERGED="runs/qwen_pertoken_kl_iter${PREV}_merged"
  ITER_TRAJ="data/pertoken_kl/iter${ITER}_trajectories.jsonl"
  ITER_TOPK="data/pertoken_kl/iter${ITER}_topk.jsonl"
  ITER_ADAPTER="runs/qwen_pertoken_kl_iter${ITER}"
  ITER_MERGED="runs/qwen_pertoken_kl_iter${ITER}_merged"
  ITER_EVAL="runs/phase5_pertoken_kl_iter${ITER}"

  echo "===== ITER ${ITER} (from iter${PREV}) ====="

  # 1. Sample student trajectories under prev iter's checkpoint
  if [ ! -f "$ITER_TRAJ" ] || [ "$(wc -l < $ITER_TRAJ)" -lt 25 ]; then
    kill_vllm
    start_student_vllm "$PREV_MERGED" "logs/vllm_pertoken_kl_iter${ITER}_sample.log" || { echo "[fatal] sample vLLM failed"; exit 1; }
    # Use existing onpolicy_distill_iter.py to sample student trajectories; it produces
    # data/pertoken_kl/iter${ITER}_trajectories.jsonl as a side effect (we only need traj,
    # not SFT/DPO files — but the script generates all of them as a unit).
    OUT_PREFIX="data/pertoken_kl/iter${ITER}_sample"
    /home/ubuntu/aoi-env/bin/python scripts/onpolicy_distill_iter.py \
      --out-prefix "$OUT_PREFIX" \
      --n-samples 1 --temperature 1.0 --parallel 4 \
      --max-turns-per-item 4 \
      --teacher-spec claude-sonnet \
      --student-arm plain --teacher-arm scaffolded 2>&1 \
      | tee "logs/pertoken_kl_iter${ITER}_sample.log" | tail -15
    # The trajectories.jsonl file produced is what we want
    cp "${OUT_PREFIX}_trajectories.jsonl" "$ITER_TRAJ"
  fi
  echo "[iter${ITER}] $(wc -l < $ITER_TRAJ) student trajectories"

  # 2. Bring up teacher, collect per-token top-K
  if [ ! -f "$ITER_TOPK" ] || [ "$(wc -l < $ITER_TOPK)" -lt 90 ]; then
    kill_vllm
    start_teacher_vllm || { echo "[fatal] teacher vLLM failed"; exit 1; }
    /home/ubuntu/aoi-env/bin/python scripts/pertoken_kl_collect.py \
      --trajectories "$ITER_TRAJ" \
      --out "$ITER_TOPK" \
      --top-k 20 --parallel 4 --max-turns-per-item 4 2>&1 \
      | tee "logs/pertoken_kl_iter${ITER}_collect.log" | tail -10
  fi
  echo "[iter${ITER}] $(wc -l < $ITER_TOPK) topk records"

  # 3. Train QLoRA + per-token KL loss
  if [ ! -d "$ITER_ADAPTER" ] || [ -z "$(ls $ITER_ADAPTER 2>/dev/null)" ]; then
    kill_vllm  # free GPU for training
    wait_for_gpu 20  # don't start training until we have ~20GB free
    PL_MODEL_ID="$PREV_MERGED" \
    PL_KL_PATH="$ITER_TOPK" \
    PL_OUT_DIR="$ITER_ADAPTER" \
    PL_KL_EPOCHS=3 \
    PL_KL_LR=5e-5 \
    PL_KL_BS=1 \
    PL_KL_GRAD_ACCUM=8 \
    PL_KL_MAX_LEN=4096 \
    /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
      | tee "logs/pertoken_kl_iter${ITER}_train.log" | tail -30
  fi

  # 4. Merge
  if [ ! -d "$ITER_MERGED" ]; then
    /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
      --adapter "$ITER_ADAPTER" --base "$PREV_MERGED" --out "$ITER_MERGED" 2>&1 | tail -3
    /home/ubuntu/polar-env/bin/python -c "
import json, os
p = '$ITER_MERGED/tokenizer_config.json'
if os.path.exists(p):
    c = json.load(open(p))
    if 'extra_special_tokens' in c:
        del c['extra_special_tokens']
        json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
        print('stripped extra_special_tokens')
"
  fi

  # 5. Eval
  mkdir -p "$ITER_EVAL"
  if [ ! -f "$ITER_EVAL/scored.jsonl" ] || [ "$(wc -l < $ITER_EVAL/scored.jsonl)" -lt 100 ]; then
    kill_vllm
    wait_for_gpu 15
    start_student_vllm "$ITER_MERGED" "logs/vllm_pertoken_kl_iter${ITER}_eval.log" || { echo "[fatal] iter${ITER} eval vLLM failed"; exit 1; }
    /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$ITER_EVAL/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
PYEOF
    /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py \
      "$ITER_EVAL/trajectories.jsonl" --require 100 || true
    /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$ITER_EVAL/trajectories.jsonl",
           items=items, out_path="$ITER_EVAL/scored.jsonl",
           resume=True, parallel=4)
PYEOF
    kill_vllm
  fi

  # 6. Summary
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$ITER_EVAL/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] pertoken_kl_iter${ITER}: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
done

echo "[DONE] K=3 per-token KL iterations complete"
