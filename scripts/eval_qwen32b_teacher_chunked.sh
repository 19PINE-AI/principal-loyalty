#!/usr/bin/env bash
# Chunked Qwen3-32B-AWQ teacher eval: keeps successful trajectories across
# retries, only re-runs failed ones. Survives vLLM-death events by saving
# progress in chunks.
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
    [ "$free" -ge 35 ] && { echo "[gpu] free=${free}GB OK"; break; }
    echo "[gpu] free=${free}GB; sleeping 60s"; sleep 60
  done
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --served-model-name Qwen/Qwen3-32B qwen-32b-local \
    --dtype auto --quantization awq --max-model-len 4096 \
    --gpu-memory-utilization 0.25 --enforce-eager \
    --port $PORT > logs/vllm_qwen32b_chunked.log 2>&1 &
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

# Remove only errored trajectories — keep successful ones for resume
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

cycle=0
max_cycles=8
while [ $cycle -lt $max_cycles ]; do
  cycle=$((cycle+1))
  prune_errors "$OUT_DIR/trajectories.jsonl"
  n_done=$(wc -l < "$OUT_DIR/trajectories.jsonl" 2>/dev/null || echo 0)
  echo "===== cycle ${cycle}: have ${n_done}/108 ====="
  [ "$n_done" -ge 32 ] && break  # 32/36 scaffolded enough for headline

  if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
    start_teacher || { echo "[teacher] start failed"; sleep 20; continue; }
  fi

  # run_grid with resume=True via existing file — harness skips done items
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-32b-local"],
         arms=["scaffolded"],
         out_path="$OUT_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF
done

prune_errors "$OUT_DIR/trajectories.jsonl"
n_final=$(wc -l < "$OUT_DIR/trajectories.jsonl" 2>/dev/null || echo 0)
echo "[chunked] final n=${n_final}/108"

if [ "$n_final" -lt 24 ]; then
  echo "[FAIL] insufficient valid trajectories after ${max_cycles} cycles"
  exit 1
fi

# Score
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
echo "[DONE] chunked teacher validation"
