#!/usr/bin/env bash
# Within-vendor consistency probe: extend cross-subject to flagship and
# light variants from same vendors.
#   Anthropic: claude-opus (vs claude-sonnet 19%)
#   OpenAI:    gpt-5-nano (vs gpt-5 71%, gpt-5-mini 62%)
#   Google:    gemini-3p1-flash-lite (vs gemini-3-flash 19%, gemini-2.5-flash 17%)
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
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 90 --allow-error-frac 0.10 || \
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
    print(f'  {arm:12s}: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi}')
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
print(f'  TOTAL: harm={harm}/{n} ({100*harm/n:.0f}%)')
"
}

run_one "claude-opus"             "claude_opus"
run_one "gpt-5-nano"              "gpt5_nano"
run_one "gemini-3p1-flash-lite"   "gemini3p1_lite"

echo "[DONE] within-vendor consistency"
