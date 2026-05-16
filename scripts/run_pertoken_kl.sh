#!/usr/bin/env bash
# Tier D10: per-token KL distillation with Qwen3-32B teacher (canonical
# Thinking Machines / DeepSeek V4 algorithm, top-K approximation).
#
# Pipeline:
#   1. (precondition) data/pertoken_kl/iter1_topk.jsonl exists, collected by
#      scripts/pertoken_kl_collect.py against the Qwen3-32B-AWQ teacher on
#      port 8001.
#   2. Train: QLoRA on Qwen3-8B v4.1 base, per-token KL loss against teacher
#      top-20 distribution. Saves adapter at runs/qwen_pertoken_kl_iter1.
#   3. Merge: merge LoRA into base, strip extra_special_tokens.
#   4. Eval: serve student vLLM on port 8000, run 108-grid with claude-sonnet
#      counterparty, score with gpt-5-mini judge.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

BASE="runs/qwen_sft_dpo_v4_1_merged"
KL_DATA="data/pertoken_kl/iter1_topk.jsonl"
ADAPTER="runs/qwen_pertoken_kl_iter1"
MERGED="runs/qwen_pertoken_kl_iter1_merged"
EVAL_DIR="runs/phase5_pertoken_kl_iter1"
PORT=8000

kill_vllm_8000() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  ps aux | grep -E "EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 6
}

start_student_vllm() {
  local model_path="$1"; local log="$2"
  echo "[vllm] starting student $model_path on :$PORT"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --port $PORT >"$log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm] DIED"; tail -30 "$log"; return 1; fi
    if [ $t -ge 300 ]; then echo "[vllm] timeout"; tail -30 "$log"; return 1; fi
  done
  echo "[vllm] READY after ${t}s"
}

# Step 1: check data
N_KL=$(wc -l < "$KL_DATA" 2>/dev/null || echo 0)
if [ "$N_KL" -lt 30 ]; then
  echo "[fatal] $KL_DATA has only $N_KL records — run pertoken_kl_collect.py first"
  exit 1
fi
echo "[step1] $N_KL KL records ready"

# Step 2: train
if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
  echo "[step2] training per-token KL adapter from $BASE"
  PL_MODEL_ID="$BASE" \
  PL_KL_PATH="$KL_DATA" \
  PL_OUT_DIR="$ADAPTER" \
  PL_KL_EPOCHS=3 \
  PL_KL_LR=5e-5 \
  PL_KL_BS=1 \
  PL_KL_GRAD_ACCUM=8 \
  PL_KL_MAX_LEN=4096 \
  /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
    | tee "logs/pertoken_kl_train.log" | tail -30
else
  echo "[step2] adapter exists at $ADAPTER — skipping training"
fi

# Step 3: merge
if [ ! -d "$MERGED" ]; then
  echo "[step3] merging LoRA into $MERGED"
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base "$BASE" --out "$MERGED" 2>&1 | tail -5
  /home/ubuntu/polar-env/bin/python -c "
import json, os
p = '$MERGED/tokenizer_config.json'
if os.path.exists(p):
    c = json.load(open(p))
    if 'extra_special_tokens' in c:
        del c['extra_special_tokens']
        json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
        print('stripped extra_special_tokens')
"
fi

# Step 4: eval
mkdir -p "$EVAL_DIR"
if [ -f "$EVAL_DIR/scored.jsonl" ] && [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -ge 100 ]; then
  echo "[step4] eval already complete"
else
  kill_vllm_8000
  start_student_vllm "$MERGED" "logs/vllm_pertoken_kl_eval.log" || { echo "[fatal] vLLM failed"; exit 1; }
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py \
    "$EVAL_DIR/trajectories.jsonl" --require 100 || { echo "[fatal] audit failed"; exit 1; }
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
           items=items, out_path="$EVAL_DIR/scored.jsonl",
           resume=True, parallel=4)
PYEOF
  kill_vllm_8000
fi

# Step 5: summarize
/home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$EVAL_DIR/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'pertoken_kl_iter1: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"

echo "[DONE] Tier D10 per-token KL iter1 complete"
