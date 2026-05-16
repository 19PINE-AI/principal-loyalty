#!/usr/bin/env bash
# Rigor experiments for per-token KL iter1 (Tier D10):
#   - 4 extra seeds for multi-seed n=5 paired Wilcoxon vs v4.1
#   - Held-out items/v0_75/ (24 items × 3 arms = 72 trajectories)
#   - Counterparty robustness: GPT-5, Gemini-3-flash
# All evals share one student vLLM on port 8000.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

PORT=8000
MERGED="runs/qwen_pertoken_kl_iter1_merged"
BASE_DIR="runs/phase5_pertoken_kl_iter1"

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore\|EngineCore" 2>/dev/null || true
  sleep 6
}

start_student() {
  echo "[vllm] starting student $MERGED on :$PORT"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --port $PORT >"logs/vllm_pertoken_kl_validation.log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm] DIED"; tail -30 logs/vllm_pertoken_kl_validation.log; return 1; fi
    if [ $t -ge 300 ]; then echo "[vllm] timeout"; return 1; fi
  done
  echo "[vllm] READY after ${t}s"
}

run_eval() {
  local out_dir="$1"; local counterparty="$2"; local items_dir="$3"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 60 ]; then
    echo "[skip] $out_dir already has $(wc -l < $out_dir/scored.jsonl) scored"
    return 0
  fi
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("$items_dir")
run_grid(items=items, subjects=["qwen-8b-local"],
         arms=["plain","prompted","scaffolded"],
         out_path="$out_dir/trajectories.jsonl",
         counterparty_spec="$counterparty", parallel=4)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 60 || true
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("$items_dir")
score_grid(trajectory_jsonl="$out_dir/trajectories.jsonl",
           items=items, out_path="$out_dir/scored.jsonl",
           resume=True, parallel=4)
PYEOF
  # Summary
  /home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('$out_dir/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] $out_dir: harm={harm}/{n} leak={leak} bound={bound} MI={mi}')
"
}

# Start student vLLM ONCE
kill_vllm
start_student || { echo "[fatal] student vLLM failed"; exit 1; }

# === 1. Multi-seed n=5 (4 extra seeds) ===
echo "=== Multi-seed seeds 2-5 ==="
for seed in 2 3 4 5; do
  run_eval "${BASE_DIR}_seed${seed}" "claude-sonnet" "items/v0"
done

# === 2. Held-out items/v0_75/ ===
echo "=== Held-out items/v0_75 ==="
run_eval "${BASE_DIR}_heldout_v0_75" "claude-sonnet" "items/v0_75"

# === 3. Counterparty robustness ===
echo "=== Counterparty GPT-5 ==="
run_eval "${BASE_DIR}_cp_gpt5" "gpt-5" "items/v0"

echo "=== Counterparty Gemini-3-flash ==="
run_eval "${BASE_DIR}_cp_gemini" "gemini-3-flash" "items/v0"

kill_vllm

# === 4. Paired Wilcoxon multi-seed test ===
echo "=== Paired Wilcoxon test ==="
/home/ubuntu/aoi-env/bin/python scripts/paired_seed_test.py \
  --a runs/phase2_trained_v4_1 \
  --b "$BASE_DIR" \
  --a-seeds 5 --b-seeds 5 2>&1 | tee logs/pertoken_kl_paired_seed_test.log

echo "[DONE] Tier D10 validation complete"
