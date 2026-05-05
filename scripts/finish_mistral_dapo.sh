#!/usr/bin/env bash
# After Mistral DAPO finishes training:
# 1. Pick the latest global_step_N checkpoint (or the one passed via PL_STEP)
# 2. Merge the LoRA adapter into the Mistral SFT+DPO base
# 3. Start vLLM with the merged model
# 4. Run the phase3 eval
# 5. Print headlines vs. Mistral SFT+DPO baseline
set -uo pipefail
cd /home/ubuntu/principal-loyalty

DAPO_DIR=${PL_DAPO_DIR:-runs/mistral_dapo_v1}
STEP=${PL_STEP:-}
BASE=${PL_BASE:-runs/mistral_sft_dpo_v4_1_merged}
PORT=8000
WAIT_TIMEOUT=300

# 1. Pick step
if [ -z "$STEP" ]; then
  STEP=$(ls -d "$DAPO_DIR"/global_step_* 2>/dev/null | sort -V | tail -1 | sed 's|.*global_step_||')
fi
ADAPTER="$DAPO_DIR/global_step_${STEP}/actor/lora_adapter"
if [ ! -d "$ADAPTER" ]; then
  echo "no adapter at $ADAPTER"; exit 1
fi
MERGED="${DAPO_DIR}_step${STEP}_merged"
EVAL_DIR="runs/phase3_mistral_dapo_v1_step${STEP}"

# 2. Merge
if [ ! -d "$MERGED" ]; then
  echo "[merge] $ADAPTER -> $MERGED"
  python3 scripts/merge_lora.py --adapter "$ADAPTER" --base "$BASE" --out "$MERGED" 2>&1 | tail -5
  # Patch chat template (verl init pattern needs [user, user] tolerance)
  if [ -f "$BASE/chat_template.jinja" ]; then
    cp "$BASE/chat_template.jinja" "$MERGED/chat_template.jinja"
  fi
fi

# 3. Kill any existing vLLM
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 5

# 4. Start vLLM serving merged model
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" \
  --served-model-name Qwen/Qwen3-8B qwen-8b-local \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --port $PORT \
  > "logs/vllm_mistral_dapo_step${STEP}.log" 2>&1 &
VLLM_PID=$!
echo "[vllm] PID $VLLM_PID; waiting"
t=0
until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; t=$((t+5))
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vllm died"; tail -10 "logs/vllm_mistral_dapo_step${STEP}.log"; exit 1; fi
  if [ $t -ge $WAIT_TIMEOUT ]; then echo "vllm timeout"; exit 1; fi
done
echo "[vllm] ready"

# 5. Run eval
mkdir -p "$EVAL_DIR"
python3 - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
from src.scorer import score_grid

items = load_items("items/v0")
run_grid(
    items=items,
    subjects=["qwen-8b-local"],
    arms=["plain", "prompted", "scaffolded"],
    out_path="$EVAL_DIR/trajectories.jsonl",
    counterparty_spec="claude-sonnet",
    parallel=4,
)
score_grid(
    trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
    items=items,
    out_path="$EVAL_DIR/scored.jsonl",
    resume=True,
    parallel=8,
)
PYEOF

# 6. Headlines
python3 -c "
import json
errs=harm=leak=0
with open('$EVAL_DIR/scored.jsonl') as f:
    for line in f:
        r=json.loads(line)
        errs += int('error' in (r.get('early_end_reason') or '').lower())
        harm += int(bool(r.get('harm',{}).get('harm_fire')))
        leak += int((r.get('leak_rate') or 0)>0)
print(f'mistral_dapo_v1_step${STEP}: harm={harm}/108 leak={leak}/108 errs={errs}')
"

echo "Mistral DAPO eval complete: $EVAL_DIR"
