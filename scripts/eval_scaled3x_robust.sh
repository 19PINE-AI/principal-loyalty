#!/usr/bin/env bash
# Robust eval for the scaled3x checkpoint (already merged). Same chunked-
# retry pattern as Stage 2: keeps successful trajectories across cycles.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

MERGED="runs/qwen_pertoken_kl_scaled3x_iter1_merged"
OUT_DIR="runs/phase5_pertoken_kl_scaled3x_iter1"
PORT=8000
mkdir -p "$OUT_DIR" logs

start_student() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
  sleep 8
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    [ "$free" -ge 25 ] && { echo "[gpu] free=${free}GB OK"; break; }
    echo "[gpu] free=${free}GB; sleeping 60s"; sleep 60
  done
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 6144 --enforce-eager \
    --gpu-memory-utilization 0.20 \
    --port $PORT > logs/vllm_scaled3x_eval.log 2>&1 &
  local pid=$!
  local t=0
  while [ $t -lt 240 ]; do
    sleep 10; t=$((t+10))
    curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "[student] ready ${t}s"; return 0; }
    kill -0 $pid 2>/dev/null || break
  done
  kill -9 $pid 2>/dev/null
  return 1
}

prune_errors() {
  local file="$1"
  [ -f "$file" ] || return 0
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import json
path = "$file"
keep = []
removed = 0
for line in open(path):
    d = json.loads(line)
    end = d.get("early_end_reason", "") or ""
    if d.get("n_agent_turns", 0) == 0 or "error" in end.lower():
        removed += 1
        continue
    keep.append(line)
with open(path, "w") as f:
    for l in keep:
        f.write(l)
print(f"[prune] kept {len(keep)}, removed {removed}")
PYEOF
}

run_eval_set() {
  local items_dir="$1"; local out="$2"; local need="$3"
  mkdir -p "$out"
  cycle=0; max_cycles=6
  while [ $cycle -lt $max_cycles ]; do
    cycle=$((cycle+1))
    prune_errors "$out/trajectories.jsonl"
    n_done=$(wc -l < "$out/trajectories.jsonl" 2>/dev/null || echo 0)
    echo "===== ($items_dir) cycle ${cycle}: have ${n_done} ====="
    [ "$n_done" -ge "$need" ] && break
    if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
      start_student || { sleep 30; continue; }
    fi
    /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("$items_dir")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$out/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=2)
PYEOF
  done
  prune_errors "$out/trajectories.jsonl"
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("$items_dir")
score_grid(trajectory_jsonl="$out/trajectories.jsonl",
           items=items, out_path="$out/scored.jsonl",
           resume=True, parallel=4)
PYEOF
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$out/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] scaled3x $items_dir: n={n} harm={harm} leak={leak} bound={bound} MI={mi}')
"
}

# Training set (v0)
run_eval_set "items/v0" "$OUT_DIR" 100

# Held-out (v0_75)
run_eval_set "items/v0_75" "${OUT_DIR}_heldout_v0_75" 60

pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null
echo "[DONE] scaled3x eval"
