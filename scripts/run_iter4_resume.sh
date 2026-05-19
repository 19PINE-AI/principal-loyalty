#!/usr/bin/env bash
# Resume iter4 from stage 2: teacher vLLM already running (PID 139122).
# Wait for /v1/models, run collect, then continue through train/merge/eval.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

STUDENT_PORT=8000
TEACHER_PORT=8001

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
  sleep 8
}

wait_for_gpu() {
  local needed="${1:-22}"
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    [ "$free" -ge "$needed" ] && { echo "[gpu] free=${free}GB OK"; return 0; }
    echo "[gpu] free=${free}GB need ${needed}GB; sleeping 60s"; sleep 60
  done
}

PREV_MERGED="runs/qwen_pertoken_kl_iter3_merged"
TRAJ="data/pertoken_kl/iter4_trajectories.jsonl"
TOPK="data/pertoken_kl/iter4_topk.jsonl"
ADAPTER="runs/qwen_pertoken_kl_iter4"
MERGED="runs/qwen_pertoken_kl_iter4_merged"
EVAL_DIR="runs/phase5_pertoken_kl_iter4"

# Copy from stage 1 output if not already done
if [ ! -f "$TRAJ" ]; then
  cp data/pertoken_kl/iter4_sample_trajectories.jsonl "$TRAJ"
fi
echo "[iter4-resume] $(wc -l < $TRAJ) student trajectories"

# Stage 2: wait for teacher to be ready (already started, extended timeout)
echo "[iter4-resume] waiting up to 900s for teacher vLLM at :$TEACHER_PORT"
t=0
until curl -sf http://localhost:$TEACHER_PORT/v1/models >/dev/null 2>&1; do
  sleep 15
  t=$((t+15))
  if [ $t -ge 900 ]; then echo "[fatal] teacher timeout"; exit 1; fi
  echo "[wait] ${t}s elapsed"
done
echo "[iter4-resume] teacher ready"

if [ ! -f "$TOPK" ] || [ "$(wc -l < $TOPK)" -lt 90 ]; then
  /home/ubuntu/aoi-env/bin/python scripts/pertoken_kl_collect.py \
    --trajectories "$TRAJ" --out "$TOPK" --top-k 20 --parallel 4 --max-turns-per-item 4 2>&1 \
    | tee logs/pertoken_kl_iter4_collect.log | tail -10
fi
echo "[iter4-resume] $(wc -l < $TOPK) topk records"

# Stage 3: train
if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
  kill_vllm
  wait_for_gpu 22
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  PL_MODEL_ID="$PREV_MERGED" PL_KL_PATH="$TOPK" PL_OUT_DIR="$ADAPTER" \
    PL_KL_EPOCHS=3 PL_KL_LR=5e-5 PL_KL_BS=1 PL_KL_GRAD_ACCUM=8 PL_KL_MAX_LEN=3000 \
    /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
    | tee logs/pertoken_kl_iter4_train.log | tail -20
fi

# Stage 4: merge
if [ ! -d "$MERGED" ]; then
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base "$PREV_MERGED" --out "$MERGED" 2>&1 | tail -3
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

# Stage 5: eval (parallel=1 per GPU-contention rule)
mkdir -p "$EVAL_DIR"
if [ ! -f "$EVAL_DIR/scored.jsonl" ] || [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -lt 100 ]; then
  kill_vllm
  wait_for_gpu 22
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
    --port $STUDENT_PORT > logs/vllm_iter4_eval.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; do sleep 10; t=$((t+10)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 600 ]; then exit 1; fi; done

  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"], arms=["plain","prompted","scaffolded"],
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
fi

/home/ubuntu/polar-env/bin/python -c "
import json
rows=[json.loads(l) for l in open('$EVAL_DIR/scored.jsonl')]
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak=sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound=sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi=sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] pertoken_kl_iter4: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
kill_vllm
echo "[DONE] iter4 resume"
