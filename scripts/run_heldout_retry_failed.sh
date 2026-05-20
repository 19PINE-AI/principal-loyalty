#!/usr/bin/env bash
# Retry the two held-out runs that failed due to vendor key / provider issues:
#   gpt-5-nano  -- now routed via OpenRouter (vendor spec updated)
#   mistral-large -- prior 'google-vert' routing failure was transient
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
    { echo "[fatal] $label audit FAILED on retry"; return 1; }
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
leak=sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
mi=sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] $label heldout: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi}')
"
}

# Wait for in-flight heldout expansion pipeline (if any) to finish first
while pgrep -f "run_heldout_xsubj_more.sh" >/dev/null 2>&1; do
  echo "[wait] heldout_xsubj_more.sh still running; sleeping 30s"
  sleep 30
done
echo "[ok] no in-flight pipeline"

run_one "gpt-5-nano"     "gpt5_nano"
run_one "mistral-large"  "mistral_large"

echo "[DONE] heldout retry"
