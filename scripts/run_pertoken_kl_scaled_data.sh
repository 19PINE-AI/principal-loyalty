#!/usr/bin/env bash
# Stage 5: per-token KL with 3x more on-policy data (n_samples=3 instead of 1
# → ~93 trajectories × ~4 turns = ~372 records vs original 113).
# Tests whether the held-out generalization gap (40.3% vs training 30.6%)
# closes with more data — the main caveat on Variant 3 iter1.
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
  local needed="${1:-22}"
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    if [ "$free" -ge "$needed" ]; then echo "[gpu] ready free=${free}GB"; return 0; fi
    echo "[gpu] free=${free}GB need ${needed}GB; sleeping 60s"; sleep 60
  done
}

start_vllm() {
  local model="$1"; local port="$2"; local util="$3"; local served="$4"; local log="$5"
  local attempt=0; local max=5
  while [ $attempt -lt $max ]; do
    attempt=$((attempt+1))
    echo "[vllm] attempt ${attempt}: $model on :$port"
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$model" --served-model-name $served \
      --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization "$util" \
      --port $port >"$log" 2>&1 &
    local pid=$!
    local t=0
    while [ $t -lt 240 ]; do
      sleep 10; t=$((t+10))
      curl -sf http://localhost:$port/v1/models >/dev/null 2>&1 && { echo "[vllm] ready ${t}s"; return 0; }
      kill -0 $pid 2>/dev/null || break
    done
    kill -9 $pid 2>/dev/null
    sleep 20
  done
  return 1
}

DATA_PREFIX="data/pertoken_kl/scaled3x_iter1"
ADAPTER="runs/qwen_pertoken_kl_scaled3x_iter1"
MERGED="runs/qwen_pertoken_kl_scaled3x_iter1_merged"
EVAL_DIR="runs/phase5_pertoken_kl_scaled3x_iter1"

# Step 1: sample 3x more student trajectories from v4.1 base (matched setting)
SAMPLE_TRAJ="${DATA_PREFIX}_trajectories.jsonl"
if [ ! -f "$SAMPLE_TRAJ" ] || [ "$(wc -l < $SAMPLE_TRAJ)" -lt 80 ]; then
  kill_vllm
  wait_for_gpu 22
  start_vllm runs/qwen_sft_dpo_v4_1_merged 8000 0.20 "Qwen/Qwen3-8B qwen-8b-local" logs/vllm_scaled_sample.log || { echo "[fatal] sample vLLM failed"; exit 1; }
  /home/ubuntu/aoi-env/bin/python scripts/onpolicy_distill_iter.py \
    --out-prefix "$DATA_PREFIX" \
    --n-samples 3 --temperature 1.0 --parallel 4 \
    --max-turns-per-item 4 \
    --teacher-spec claude-sonnet \
    --student-arm plain --teacher-arm scaffolded 2>&1 \
    | tee logs/pertoken_kl_scaled_sample.log | tail -15
fi
echo "[scaled3x] $(wc -l < $SAMPLE_TRAJ) trajectories"

# Step 2: collect per-token top-K from Qwen3-32B teacher
TOPK="${DATA_PREFIX}_topk.jsonl"
if [ ! -f "$TOPK" ] || [ "$(wc -l < $TOPK)" -lt 200 ]; then
  kill_vllm
  wait_for_gpu 30
  echo "[teacher] starting Qwen3-32B-AWQ"
  VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --served-model-name Qwen/Qwen3-32B qwen-32b-local \
    --dtype auto --quantization awq --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --enforce-eager \
    --port 8001 > logs/vllm_scaled_teacher.log 2>&1 &
  TEACHER_PID=$!
  t=0; until curl -sf http://localhost:8001/v1/models >/dev/null 2>&1; do sleep 15; t=$((t+15)); if ! kill -0 $TEACHER_PID 2>/dev/null; then echo "[teacher] DIED"; tail -20 logs/vllm_scaled_teacher.log; exit 1; fi; if [ $t -ge 900 ]; then exit 1; fi; done

  /home/ubuntu/aoi-env/bin/python scripts/pertoken_kl_collect.py \
    --trajectories "$SAMPLE_TRAJ" \
    --out "$TOPK" \
    --top-k 20 --parallel 4 --max-turns-per-item 4 2>&1 \
    | tee logs/pertoken_kl_scaled_collect.log | tail -10
fi
echo "[scaled3x] $(wc -l < $TOPK) topk records"

# Step 3: train per-token KL
if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
  kill_vllm
  wait_for_gpu 22
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  PL_MODEL_ID=runs/qwen_sft_dpo_v4_1_merged \
  PL_KL_PATH="$TOPK" \
  PL_OUT_DIR="$ADAPTER" \
  PL_KL_EPOCHS=3 PL_KL_LR=5e-5 PL_KL_BS=1 PL_KL_GRAD_ACCUM=8 PL_KL_MAX_LEN=3500 \
  /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
    | tee logs/pertoken_kl_scaled_train.log | tail -30
fi

# Step 4: merge
if [ ! -d "$MERGED" ]; then
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base runs/qwen_sft_dpo_v4_1_merged --out "$MERGED" 2>&1 | tail -3
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

# Step 5: eval on training set + held-out v0_75 (to test if gap closes)
for items_set in v0 v0_75; do
  OUT="${EVAL_DIR}"
  if [ "$items_set" = "v0_75" ]; then OUT="${EVAL_DIR}_heldout_v0_75"; fi
  if [ -f "$OUT/scored.jsonl" ] && [ "$(wc -l < $OUT/scored.jsonl)" -ge 60 ]; then echo "[skip] $OUT"; continue; fi
  mkdir -p "$OUT"
  kill_vllm
  wait_for_gpu 22
  start_vllm "$MERGED" 8000 0.20 "Qwen/Qwen3-8B qwen-8b-local" logs/vllm_scaled_eval.log || { echo "[fatal] eval vLLM"; exit 1; }
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/$items_set")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$OUT/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$OUT/trajectories.jsonl" --require 60 --allow-error-frac 0.05
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/$items_set")
score_grid(trajectory_jsonl="$OUT/trajectories.jsonl",
           items=items, out_path="$OUT/scored.jsonl",
           resume=True, parallel=4)
PYEOF
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$OUT/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] scaled3x $items_set: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
done

kill_vllm
echo "[DONE] Scaled (3x) per-token KL complete"
