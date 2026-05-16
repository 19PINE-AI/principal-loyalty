#!/usr/bin/env bash
# Robust Qwen3-32B-AWQ + v4 teacher validation with retry on vLLM death.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

OUT_DIR=runs/phase4_qwen32b_teacher_eval
PORT=8001
mkdir -p "$OUT_DIR" logs

start_teacher() {
  pkill -9 -f "vllm.entrypoints.*Qwen3-32B" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
  sleep 8
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    [ "$free" -ge 30 ] && { echo "[gpu] free=${free}GB OK"; break; }
    echo "[gpu] free=${free}GB; sleeping 60s"; sleep 60
  done
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --served-model-name Qwen/Qwen3-32B qwen-32b-local \
    --dtype auto --quantization awq --max-model-len 4096 \
    --gpu-memory-utilization 0.30 --enforce-eager \
    --port $PORT > logs/vllm_qwen32b_robust.log 2>&1 &
  local pid=$!
  local t=0
  while [ $t -lt 240 ]; do
    sleep 10; t=$((t+10))
    curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "[teacher] ready ${t}s"; return 0; }
    kill -0 $pid 2>/dev/null || break
  done
  kill -9 $pid 2>/dev/null
  return 1
}

attempts=0
while [ $attempts -lt 5 ]; do
  attempts=$((attempts+1))
  echo "===== attempt $attempts/5 ====="
  rm -f "$OUT_DIR/trajectories.jsonl"

  if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    start_teacher || { echo "[teacher] start failed; retry"; continue; }
  fi

  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-32b-local"],
         arms=["plain", "prompted", "scaffolded"],
         out_path="$OUT_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=2)
PYEOF

  # Audit gate FAIL-FAST
  if ! /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$OUT_DIR/trajectories.jsonl" --require 100; then
    echo "[ERR] attempt $attempts audit FAILED — vLLM probably died; restart"
    rm -f "$OUT_DIR/trajectories.jsonl"
    pkill -9 -f "vllm.entrypoints.*Qwen3-32B" 2>/dev/null
    sleep 10
    continue
  fi

  # Audit passed — score and exit
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$OUT_DIR/trajectories.jsonl",
           items=items, out_path="$OUT_DIR/scored.jsonl",
           resume=True, parallel=4)
PYEOF

  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$OUT_DIR/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'qwen3-32b-awq+v4 teacher: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
print(f'gold claude-sonnet+v4: 21/108 17/5/21')
"

  pkill -9 -f "vllm.entrypoints.*Qwen3-32B" 2>/dev/null
  echo "[DONE] teacher validation"
  exit 0
done

echo "[FAIL] teacher validation exhausted ${attempts} attempts"
exit 1
