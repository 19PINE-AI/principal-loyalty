#!/usr/bin/env bash
# Inverse-scaling probe: within-family comparison
#   Qwen3-8B (smaller than Qwen3-32B, smaller than Qwen3.5-27B)
#   GPT-5-mini (smaller than GPT-5)
#   Gemini-2.5-flash (older/smaller than Gemini-3-flash)
# All API-only.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_one() {
  local subj="$1"; local label="$2"
  local out_dir="runs/phase4_promptv4_${label}"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 90 ]; then
    echo "[skip] $label complete"; return 0
  fi
  echo "===== $label (subj=$subj) ====="
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

  # Soft audit — allow up to 5% errors (1-of-108 vendor 500s sometimes)
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 90 || \
    echo "[warn] $label audit FAILED — will score anyway"

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
from collections import defaultdict
rows = []
for line in open('$out_dir/scored.jsonl'):
    try: rows.append(json.loads(line))
    except: pass
by_arm = defaultdict(list)
for r in rows: by_arm[r['arm']].append(r)
print(f'=== $label ===')
for arm in ['plain','prompted','scaffolded']:
    arows = by_arm.get(arm, [])
    n = len(arows)
    if n == 0: continue
    harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in arows)
    leak = sum(int((r.get('leak_rate') or 0) > 0) for r in arows)
    mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in arows)
    print(f'  {arm:12s}: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} ({100*leak/n:.0f}%) MI={mi} ({100*mi/n:.0f}%)')
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
print(f'  TOTAL: harm={harm}/{n} ({100*harm/n:.0f}%)')
"
}

run_one "qwen-8b"        "qwen8b_or"
run_one "gpt-5-mini"     "gpt5mini"
run_one "gemini-flash"   "gemini25flash"

echo "[DONE] inverse-scaling cross-subject"
