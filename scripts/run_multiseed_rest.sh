#!/usr/bin/env bash
# Complete multi-seed n=5 coverage: the remaining 5 calibrated vendors.
# Already done: claude-sonnet, deepseek, gemini-3-flash, gpt-5, qwen-27b,
#               claude-opus, glm-4.6, gpt-5-mini (8 vendors)
# This script: gemini-3p1-flash-lite, mistral-large, llama-70b,
#              gemini-2.5-flash, qwen-32b (5 vendors)
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_seed() {
  local subj="$1"; local label="$2"; local seed="$3"
  local out_dir="runs/phase4_promptv4_${label}_seed${seed}"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 90 ]; then
    echo "[skip] $label seed$seed already complete"; return 0
  fi
  echo "===== $label seed=$seed ====="
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["$subj"],
         arms=["plain","prompted","scaffolded"],
         out_path="$out_dir/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 90 --allow-error-frac 0.10 || \
    echo "[warn] $label seed$seed audit FAILED — scoring anyway"
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
rows = []
for line in open('$out_dir/scored.jsonl'):
    try: rows.append(json.loads(line))
    except: pass
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
print(f'[summary] $label seed$seed: harm={harm}/{n} ({100*harm/n:.0f}%)')
"
}

for spec_label in "gemini-3p1-flash-lite:gemini3p1_lite" "mistral-large:mistral_large" "llama-70b:llama70b" "gemini-2.5-flash:gemini25flash" "qwen-32b:qwen32b_openrouter"; do
  spec="${spec_label%%:*}"
  label="${spec_label##*:}"
  for seed in 2 3 4 5; do
    run_seed "$spec" "$label" "$seed"
  done
done

echo "[DONE] multi-seed-rest"
