#!/usr/bin/env bash
# On-policy distillation iter1 on the Mistral-7B-Instruct-v0.3 SFT+DPO base
# and the Llama-3.1-8B-Instruct SFT+DPO base. Same recipe as Qwen3-8B iter1:
# per-turn SFT on teacher-completed-at-student-state.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
WAIT_TIMEOUT=300

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  ps aux | grep -E "EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 6
}

start_vllm() {
  local model_path="$1"; local log="$2"
  echo "[vllm] starting $model_path"
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
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; tail -30 "$log"; return 1; fi
  done
  echo "[vllm] READY after ${t}s"
}

# Variant table: (base_name, base_dir, sft_data_path, adapter_dir, merged_dir, eval_dir)
declare -A BASES
BASES[mistral]="runs/mistral_sft_dpo_v4_1_merged"
BASES[llama]="runs/llama_sft_dpo_v4_1_merged"

for NAME in mistral llama; do
  BASE="${BASES[$NAME]}"
  OUT_PREFIX="data/onpolicy_${NAME}_iter1"
  ADAPTER="runs/${NAME}_onpolicy_sft_iter1"
  MERGED="runs/${NAME}_onpolicy_sft_iter1_merged"
  EVAL_DIR="runs/phase5_onpolicy_${NAME}_sft_iter1"

  echo "===== [${NAME}] base=${BASE} ====="

  if [ -f "$EVAL_DIR/scored.jsonl" ] && [ "$(wc -l < "$EVAL_DIR/scored.jsonl")" -ge 100 ]; then
    echo "[skip] ${NAME} already complete"; continue
  fi

  # 1. Sample student trajectories under base, get teacher completions
  kill_vllm
  start_vllm "$BASE" "logs/vllm_onpolicy_${NAME}_sample.log" || { echo "[fatal] ${NAME} sample vLLM failed"; continue; }
  if [ ! -f "${OUT_PREFIX}_sft.jsonl" ]; then
    /home/ubuntu/polar-env/bin/python scripts/onpolicy_distill_iter.py \
      --out-prefix "$OUT_PREFIX" \
      --n-samples 1 --temperature 1.0 --parallel 4 \
      --max-turns-per-item 4 \
      --teacher-spec claude-sonnet \
      --student-arm plain --teacher-arm scaffolded \
      2>&1 | tee "logs/onpolicy_${NAME}_sample.log" | tail -15
  fi
  N_SFT=$(wc -l < "${OUT_PREFIX}_sft.jsonl")
  echo "[${NAME}] $N_SFT SFT points"
  if [ "$N_SFT" -lt 10 ]; then
    echo "[stop] ${NAME} too few points; skip"; kill_vllm; continue
  fi

  # 2. Train SFT
  kill_vllm
  if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
    PL_MODEL_ID="$BASE" \
    PL_SFT_PATH="${OUT_PREFIX}_sft.jsonl" \
    PL_OUT_DIR="$ADAPTER" \
    PL_SFT_EPOCHS=2 \
    PL_SFT_LR=5e-5 \
    /home/ubuntu/polar-env/bin/python scripts/train_qwen_sft_onpolicy.py 2>&1 \
      | tee "logs/onpolicy_${NAME}_train.log" | tail -8
  fi

  # 3. Merge
  if [ ! -d "$MERGED" ]; then
    /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
      --adapter "$ADAPTER" --base "$BASE" --out "$MERGED" 2>&1 | tail -3
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

  # 4. Eval
  start_vllm "$MERGED" "logs/vllm_onpolicy_${NAME}_eval.log" || { echo "[fatal] ${NAME} eval vLLM failed"; continue; }
  mkdir -p "$EVAL_DIR"
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
    "$EVAL_DIR/trajectories.jsonl" --require 100 || true
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

  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$EVAL_DIR/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'${NAME}_onpolicy_iter1: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
done

echo "[DONE] cross-base on-policy iter1 complete"
