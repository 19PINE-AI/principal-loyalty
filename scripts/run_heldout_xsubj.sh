#!/usr/bin/env bash
# Tier B.4: held-out v0_75 cross-vendor — does the bimodal pattern hold
# on 24 fresh items? Use the 5 vendors that bracket the gap.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_one() {
  local subj="$1"; local label="$2"
  local out_dir="runs/phase4_promptv4_${label}_heldout"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 60 ]; then
    echo "[skip] $label heldout complete"; return 0
  fi
  echo "===== $label heldout (subj=$subj) ====="
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0_75")
run_grid(items=items, subjects=["$subj"],
         arms=["plain","prompted","scaffolded"],
         out_path="$out_dir/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=4)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 60 --allow-error-frac 0.10 || \
    echo "[warn] $label audit FAILED — will score anyway"
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0_75")
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
print(f'[summary] $label heldout: harm={harm}/{n} ({100*harm/n:.0f}%)')
"
}

# 5 vendors that bracket the gap
run_one "deepseek"        "deepseek"
run_one "gemini-3-flash"  "gemini3flash"
run_one "claude-sonnet"   "claude"
run_one "gpt-5"           "gpt5"
run_one "qwen-27b"        "qwen27b"

echo "[DONE] heldout cross-vendor"
