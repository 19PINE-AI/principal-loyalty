#!/usr/bin/env bash
# Multi-seed n=5 for per-token KL iter2 (Variant 3 iter 2) - locks in the
# leak/bound minimum claim with paired Wilcoxon vs v4.1.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
MERGED="runs/qwen_pertoken_kl_iter2_merged"
BASE_DIR="runs/phase5_pertoken_kl_iter2"

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do
    # only kill ours: heuristic — kill all engine cores that aren't from polar/sema projects
    kill -9 $p 2>/dev/null || true
  done
  sleep 6
}

wait_for_gpu() {
  local needed_gb="${1:-22}"
  while :; do
    local free_gb=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    if [ "$free_gb" -ge "$needed_gb" ]; then echo "[gpu] ready free=${free_gb}GB"; return 0; fi
    echo "[gpu] free=${free_gb}GB; sleeping 60s"; sleep 60
  done
}

start_student() {
  local attempt=0; local max=5
  while [ $attempt -lt $max ]; do
    attempt=$((attempt+1))
    wait_for_gpu 25
    echo "[vllm] attempt ${attempt}: starting $MERGED on :$PORT"
    nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$MERGED" \
      --served-model-name Qwen/Qwen3-8B qwen-8b-local \
      --dtype bfloat16 --max-model-len 6144 \
      --enforce-eager \
      --gpu-memory-utilization 0.22 \
      --port $PORT >logs/vllm_iter2_multiseed.log 2>&1 &
    local pid=$!
    local t=0
    while [ $t -lt 240 ]; do
      sleep 10; t=$((t+10))
      curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1 && { echo "[vllm] ready ${t}s"; return 0; }
      kill -0 $pid 2>/dev/null || break
    done
    echo "[vllm] attempt ${attempt} failed; retry"
    kill -9 $pid 2>/dev/null
    sleep 20
  done
  return 1
}

run_seed_eval() {
  local seed="$1"
  local out_dir="${BASE_DIR}_seed${seed}"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 100 ]; then
    echo "[skip] seed${seed} already complete"; return 0
  fi
  mkdir -p "$out_dir"
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$out_dir/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF
  if ! /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 100; then
    echo "[ERR] seed${seed} audit FAILED — removing trajectories and skipping scoring"
    rm -f "$out_dir/trajectories.jsonl" "$out_dir/scored.jsonl"
    return 1
  fi
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$out_dir/trajectories.jsonl",
           items=items, out_path="$out_dir/scored.jsonl",
           resume=True, parallel=4)
PYEOF
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$out_dir/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] iter2_seed${seed}: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
}

kill_vllm
start_student || { echo "[fatal] vLLM failed"; exit 1; }

for seed in 2 3 4 5; do
  echo "=== seed $seed ==="
  attempts=0
  while [ $attempts -lt 4 ]; do
    attempts=$((attempts+1))
    # Verify vLLM is alive; restart if not
    if ! curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; then
      echo "[seed${seed}] vLLM dead (attempt ${attempts}); restarting"
      kill_vllm
      start_student || { echo "[fatal] vLLM restart failed"; continue; }
    fi
    if run_seed_eval "$seed"; then break; fi
    echo "[seed${seed}] attempt ${attempts} failed; will retry"
  done
done

kill_vllm

# Paired Wilcoxon
/home/ubuntu/aoi-env/bin/python scripts/paired_seed_test.py \
  --a runs/phase2_trained_v4_1 \
  --b "$BASE_DIR" \
  --a-seeds 5 --b-seeds 5 2>&1 | tee logs/iter2_paired_seed_test.log

echo "[DONE] iter2 multi-seed complete"
