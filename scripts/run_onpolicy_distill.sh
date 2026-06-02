#!/usr/bin/env bash
# Run K iterations of on-policy distillation on Qwen3-8B starting from the
# v4.1 SFT+DPO endpoint. Each iteration:
#   1. Start vLLM serving current student (port 8000, name 'qwen-8b-local')
#   2. Run scripts/onpolicy_distill_iter.py — sample, score, build pairs
#   3. Kill vLLM
#   4. DPO update: train LoRA on top of current student using new pairs
#   5. Merge to runs/qwen_onpolicy_iter${K}_merged/
#   6. Eval on full grid (single seed)
set -uo pipefail
cd /home/ubuntu/principal-loyalty

K_TOTAL=${PL_ONPOLICY_ITERS:-3}
START_BASE=${PL_ONPOLICY_BASE:-runs/qwen_sft_dpo_v4_1_merged}
N_SAMPLES=${PL_ONPOLICY_N_SAMPLES:-1}
TEMP=${PL_ONPOLICY_TEMP:-1.0}
PORT=8000
WAIT_TIMEOUT=300

mkdir -p logs data runs

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

start_vllm() {
  local model_path="$1"; local log="$2"
  echo "[vllm] starting $model_path"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT >"$log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm] DIED"; tail -30 "$log"; return 1; fi
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; tail -30 "$log"; return 1; fi
  done
  echo "[vllm] READY after ${t}s"
}

PREV_MERGED="$START_BASE"
TEACHER_CACHE="data/onpolicy_teacher_cache.jsonl"

for K in $(seq 1 $K_TOTAL); do
  echo "===== [iter $K/$K_TOTAL] start (base=$PREV_MERGED) ====="

  PAIRS="data/onpolicy_dpo_iter${K}.jsonl"
  ADAPTER_DIR="runs/qwen_onpolicy_iter${K}"
  MERGED_DIR="runs/qwen_onpolicy_iter${K}_merged"
  EVAL_DIR="runs/phase5_onpolicy_iter${K}"

  if [ -f "$EVAL_DIR/scored.jsonl" ] && [ "$(wc -l < "$EVAL_DIR/scored.jsonl")" -ge 108 ]; then
    echo "  [skip] iter $K eval already complete"
    PREV_MERGED="$MERGED_DIR"
    continue
  fi

  # Start vLLM serving current student
  kill_vllm
  start_vllm "$PREV_MERGED" "logs/vllm_onpolicy_iter${K}_sample.log" || exit 1

  # Sample + score + build pairs
  if [ ! -f "$PAIRS" ]; then
    /home/ubuntu/polar-env/bin/python scripts/onpolicy_distill_iter.py \
      --out-pairs "$PAIRS" \
      --teacher-cache "$TEACHER_CACHE" \
      --n-samples "$N_SAMPLES" \
      --temperature "$TEMP" \
      --parallel 4 \
      2>&1 | tee "logs/onpolicy_iter${K}_sample.log" | tail -30
  else
    echo "  [skip] $PAIRS already exists"
  fi

  N_PAIRS=$(wc -l < "$PAIRS")
  echo "  iter $K: $N_PAIRS pairs"
  if [ "$N_PAIRS" -lt 5 ]; then
    echo "  [stop] iter $K produced too few pairs (<5); halting on-policy loop"
    kill_vllm
    break
  fi

  # Kill vLLM (free GPU for DPO training)
  kill_vllm

  # DPO update on top of PREV_MERGED
  if [ ! -d "$ADAPTER_DIR" ] || [ -z "$(ls "$ADAPTER_DIR" 2>/dev/null)" ]; then
    PL_MODEL_ID="$PREV_MERGED" \
    PL_DPO_PATH="$PAIRS" \
    PL_SFT_DIR="$PREV_MERGED" \
    PL_DPO_DIR="$ADAPTER_DIR" \
    PL_DPO_EPOCHS=3 \
    /home/ubuntu/polar-env/bin/python scripts/train_qwen_dpo.py 2>&1 \
      | tee "logs/onpolicy_iter${K}_train.log" | tail -30
  fi

  # Merge adapter
  if [ ! -d "$MERGED_DIR" ]; then
    /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
      --adapter "$ADAPTER_DIR" \
      --base "$PREV_MERGED" \
      --out "$MERGED_DIR" 2>&1 | tail -5
    /home/ubuntu/polar-env/bin/python -c "
import json, os
p = '$MERGED_DIR/tokenizer_config.json'
if os.path.exists(p):
    c = json.load(open(p))
    if 'extra_special_tokens' in c:
        del c['extra_special_tokens']
        json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
        print(f'patched {p}')
"
  fi

  # Eval
  start_vllm "$MERGED_DIR" "logs/vllm_onpolicy_iter${K}_eval.log" || exit 1
  mkdir -p "$EVAL_DIR"
  /home/ubuntu/polar-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
score_grid(trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
           items=items, out_path="$EVAL_DIR/scored.jsonl",
           resume=True, parallel=8)
PYEOF
  kill_vllm

  /home/ubuntu/polar-env/bin/python -c "
import json
n=harm=leak=bound=mi=0
with open('$EVAL_DIR/scored.jsonl') as f:
    for line in f:
        r=json.loads(line); n+=1
        h = r.get('harm') or {}
        harm += int(bool(h.get('harm_fire')))
        leak += int((r.get('leak_rate') or 0)>0)
        bound += int(bool(h.get('leaked_private_bound')))
        mi += int(bool(h.get('missed_instruction')))
print(f'onpolicy_iter${K}: harm={harm}/{n} leak={leak}/{n} bound={bound} MI={mi}')
"

  PREV_MERGED="$MERGED_DIR"
done

echo "[DONE] on-policy distillation $K_TOTAL iterations complete"
echo "Final headline path: runs/phase5_onpolicy_iter${K_TOTAL}/scored.jsonl"
