#!/usr/bin/env bash
# Eval Llama-3.1-8B per-token KL iter3 on held-out items/v0_75.
# Tests whether cross-family per-token KL generalizes to fresh items.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

# Use a separate vLLM port to avoid conflict with the iter4 pipeline
PORT=8003
MERGED="runs/llama_pertoken_kl_iter3_merged"
EVAL_DIR="runs/phase5_pertoken_kl_llama_iter3_heldout"

mkdir -p "$EVAL_DIR"

# Wait for GPU to be free enough for an 8B model on a separate port.
while :; do
  free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
  [ "$free" -ge 22 ] && { echo "[gpu] free=${free}GB OK"; break; }
  echo "[gpu] free=${free}GB need 22GB; sleeping 60s"; sleep 60
done

# Start a separate student vLLM on port 8003 (so it doesn't conflict
# with iter4's port-8000 pipeline)
nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local-heldout \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
  --port $PORT > logs/vllm_llama_iter3_heldout.log 2>&1 &
pid=$!
t=0
until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
  sleep 10; t=$((t+10))
  if ! kill -0 $pid 2>/dev/null; then echo "[fatal] vllm died"; tail -20 logs/vllm_llama_iter3_heldout.log; exit 1; fi
  if [ $t -ge 600 ]; then echo "[fatal] vllm timeout"; exit 1; fi
done

# Eval on items/v0_75 using a custom vendor spec that points to port 8003
/home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys, os
sys.path.insert(0, ".")
# Monkeypatch qwen-8b-local to use port 8003
import src.vendors as V
orig_get_vendor = V.get_vendor
def patched(spec):
    if spec == "qwen-8b-local":
        return V.OpenAICompatVendor(
            model="Qwen/Qwen3-8B",
            base_url="http://localhost:$PORT/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="qwen-8b-local",
        )
    return orig_get_vendor(spec)
V.get_vendor = patched

from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0_75")
run_grid(items=items, subjects=["qwen-8b-local"], arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF

/home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py \
  "$EVAL_DIR/trajectories.jsonl" --require 60 --allow-error-frac 0.10

/home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0_75")
score_grid(trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
           items=items, out_path="$EVAL_DIR/scored.jsonl",
           resume=True, parallel=4)
PYEOF

/home/ubuntu/polar-env/bin/python -c "
import json
rows = []
for line in open('$EVAL_DIR/scored.jsonl'):
    try: rows.append(json.loads(line))
    except: pass
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak=sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
mi=sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] Llama iter3 heldout: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi}')
"

# Cleanup
kill $pid 2>/dev/null || true
echo "[DONE] llama iter3 heldout"
