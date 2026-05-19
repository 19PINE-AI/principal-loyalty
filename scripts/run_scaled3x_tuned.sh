#!/usr/bin/env bash
# Tuned scaled3x: same 372 records but with epochs=9 (3x more passes) to
# disentangle data-limit from undertraining. Direct test of the negative
# scaling result documented in the paper.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

PORT=8000
PREV_BASE="runs/qwen_sft_dpo_v4_1_merged"
TOPK="data/pertoken_kl/scaled3x_iter1_topk.jsonl"
ADAPTER="runs/qwen_pertoken_kl_scaled3x_tuned"
MERGED="runs/qwen_pertoken_kl_scaled3x_tuned_merged"
EVAL_DIR="runs/phase5_pertoken_kl_scaled3x_tuned"

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

[ -f "$TOPK" ] || { echo "[fatal] missing $TOPK"; exit 1; }
N_RECORDS=$(wc -l < "$TOPK")
echo "[scaled3x-tuned] training on $N_RECORDS records, 9 epochs (vs original 3)"

if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
  kill_vllm
  wait_for_gpu 22
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  PL_MODEL_ID="$PREV_BASE" PL_KL_PATH="$TOPK" PL_OUT_DIR="$ADAPTER" \
    PL_KL_EPOCHS=9 PL_KL_LR=5e-5 PL_KL_BS=1 PL_KL_GRAD_ACCUM=8 PL_KL_MAX_LEN=3500 \
    /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
    | tee logs/pertoken_kl_scaled3x_tuned_train.log | tail -30
fi

if [ ! -d "$MERGED" ]; then
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base "$PREV_BASE" --out "$MERGED" 2>&1 | tail -3
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

mkdir -p "$EVAL_DIR"
if [ ! -f "$EVAL_DIR/scored.jsonl" ] || [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -lt 100 ]; then
  kill_vllm
  wait_for_gpu 22
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
    --port $PORT > logs/vllm_scaled3x_tuned_eval.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do sleep 10; t=$((t+10)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 300 ]; then exit 1; fi; done

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
print(f'[summary] scaled3x_tuned (9 epochs): harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
kill_vllm
echo "[DONE] scaled3x_tuned"
