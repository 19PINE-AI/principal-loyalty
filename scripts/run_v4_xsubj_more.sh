#!/usr/bin/env bash
# Extend cross-subject v4 prompt evaluation to additional API-only vendors
# (kimi, deepseek, qwen-32b via OpenRouter) to strengthen the bimodal
# intrinsic-difference finding.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_one() {
  local subj="$1"; local label="$2"
  local out_dir="runs/phase4_promptv4_${label}"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 100 ]; then
    echo "[skip] $label: complete"; return 0
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

  if ! /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 100; then
    echo "[ERR] $label audit FAILED"; return 1
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
from collections import defaultdict
rows=[json.loads(l) for l in open('$out_dir/scored.jsonl')]
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
    print(f'  {arm:12s}: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi} ({100*mi/n:.0f}%)')
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
print(f'  TOTAL: harm={harm}/{n} ({100*harm/n:.0f}%)')
"
}

run_one "kimi"             "kimi"
run_one "deepseek"         "deepseek"
run_one "qwen-32b"         "qwen32b_openrouter"

echo "[DONE] extended cross-subject v4 eval"
