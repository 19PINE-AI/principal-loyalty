#!/usr/bin/env bash
# Llama-3.1-8B cross-family arm of the recipe: SFT (sft_v4) -> DPO (v4_combined)
# -> eval -> DAPO (v1 reward) -> eval. Mirrors the Mistral run that landed
# at 27/108 harm (SFT+DPO) and regressed to 52/108 under DAPO-v1.
#
# Stages can be resumed individually by skipping ones whose output dirs
# already exist.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

LLAMA_BASE="NousResearch/Meta-Llama-3.1-8B-Instruct"
PORT=8000
WAIT_TIMEOUT=300

mkdir -p logs runs

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

start_vllm() {
  local model_path="$1"; local log_path="$2"
  echo "[vllm] starting $model_path"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT >"$log_path" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 5; t=$((t+5))
    if ! kill -0 $pid 2>/dev/null; then tail -30 "$log_path"; return 1; fi
    if [ $t -ge $WAIT_TIMEOUT ]; then tail -30 "$log_path"; return 1; fi
  done
  echo "[vllm] ready after ${t}s"
}

# ---- SFT ----
if [ ! -d "runs/llama_sft_v4" ] || [ -z "$(ls runs/llama_sft_v4 2>/dev/null)" ]; then
  echo "===== Llama SFT ====="
  kill_vllm
  PL_MODEL_ID="$LLAMA_BASE" \
  PL_SFT_PATH=data/sft_v4.jsonl \
  PL_OUT_DIR=runs/llama_sft_v4 \
  PL_SFT_EPOCHS=3 \
  /home/ubuntu/polar-env/bin/python scripts/train_qwen_sft.py 2>&1 | tee logs/llama_sft_v4.log | tail -20
else
  echo "[skip] Llama SFT (exists)"
fi

# ---- DPO ----
if [ ! -d "runs/llama_dpo_v4_1" ] || [ -z "$(ls runs/llama_dpo_v4_1 2>/dev/null)" ]; then
  echo "===== Llama DPO v4.1 (combined pairs) ====="
  kill_vllm
  PL_MODEL_ID="$LLAMA_BASE" \
  PL_DPO_PATH=data/dpo_v4_combined.jsonl \
  PL_SFT_DIR=runs/llama_sft_v4 \
  PL_DPO_DIR=runs/llama_dpo_v4_1 \
  PL_DPO_EPOCHS=3 \
  /home/ubuntu/polar-env/bin/python scripts/train_qwen_dpo.py 2>&1 | tee logs/llama_dpo_v4_1.log | tail -20
else
  echo "[skip] Llama DPO (exists)"
fi

# ---- Merge ----
if [ ! -d "runs/llama_sft_dpo_v4_1_merged" ]; then
  echo "===== Llama merge LoRA ====="
  kill_vllm
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter runs/llama_dpo_v4_1 \
    --base "$LLAMA_BASE" \
    --out runs/llama_sft_dpo_v4_1_merged 2>&1 | tee logs/llama_merge.log | tail -10
fi

# ---- Eval SFT+DPO ----
EVAL_DIR="runs/phase3_llama_sft_dpo"
if [ ! -f "$EVAL_DIR/scored.jsonl" ]; then
  echo "===== Llama SFT+DPO eval ====="
  kill_vllm
  start_vllm "runs/llama_sft_dpo_v4_1_merged" "logs/vllm_llama_sftdpo.log" || exit 1
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
fi

# ---- DAPO ----
# Reuse the merged-user-roles parquet from the Mistral run; Llama tolerates
# merged consecutive same-role messages without breaking its chat template.
if [ ! -d "runs/llama_dapo_v1" ] || [ -z "$(ls -d runs/llama_dapo_v1/global_step_* 2>/dev/null)" ]; then
  echo "===== Llama DAPO v1 ====="
  kill_vllm
  PL_MODEL_PATH=runs/llama_sft_dpo_v4_1_merged \
  PL_DAPO_DIR=runs/llama_dapo_v1 \
  bash scripts/run_dapo_mistral.sh 2>&1 | tee logs/llama_dapo_v1.log | tail -50
fi

# ---- DAPO eval ----
EVAL_DAPO="runs/phase3_llama_dapo_v1_step35"
if [ ! -f "$EVAL_DAPO/scored.jsonl" ]; then
  echo "===== Llama DAPO eval ====="
  kill_vllm
  STEP=$(ls -d runs/llama_dapo_v1/global_step_* 2>/dev/null | sort -V | tail -1 | sed 's|.*global_step_||')
  ADAPTER="runs/llama_dapo_v1/global_step_${STEP}/actor/lora_adapter"
  MERGED="runs/llama_dapo_v1_step${STEP}_merged"
  if [ ! -d "$MERGED" ]; then
    /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
      --adapter "$ADAPTER" \
      --base runs/llama_sft_dpo_v4_1_merged \
      --out "$MERGED" 2>&1 | tail -5
  fi
  start_vllm "$MERGED" "logs/vllm_llama_dapo_step${STEP}.log" || exit 1
  mkdir -p "$EVAL_DAPO"
  /home/ubuntu/polar-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
from src.scorer import score_grid

items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DAPO/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
score_grid(trajectory_jsonl="$EVAL_DAPO/trajectories.jsonl",
           items=items, out_path="$EVAL_DAPO/scored.jsonl",
           resume=True, parallel=8)
PYEOF
fi

kill_vllm

# ---- Headlines ----
echo "===== Llama pipeline headlines ====="
for ed in "$EVAL_DIR" "$EVAL_DAPO"; do
  [ -f "$ed/scored.jsonl" ] || continue
  /home/ubuntu/polar-env/bin/python -c "
import json
errs=harm=leak=bound=mi=0; turns={}; n=0
with open('$ed/scored.jsonl') as f:
    for line in f:
        r=json.loads(line); n+=1
        errs += int('error' in (r.get('early_end_reason') or '').lower())
        h = r.get('harm') or {}
        harm += int(bool(h.get('harm_fire')))
        leak += int((r.get('leak_rate') or 0)>0)
        bound += int(bool(h.get('leaked_private_bound')))
        mi += int(bool(h.get('missed_instruction')))
        t = r.get('turns') or 0; turns[t] = turns.get(t,0)+1
print(f'$ed: harm={harm}/{n} leak={leak}/{n} bound={bound} MI={mi} errs={errs} turns={turns}')
"
done
