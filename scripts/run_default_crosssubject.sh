#!/usr/bin/env bash
# Default-prompt 108-grid for GPT-5 and Qwen-27B (no v4 system prompt).
# Establishes baseline to attribute the v4 over-refusal: is it v4-induced
# or intrinsic to these vendors? Test: plain arm only is the "default" baseline.
# 36 items × 1 plain arm = 36 trajectories per subject.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_one() {
  local subj="$1"; local label="$2"
  local out_dir="runs/phase4_default_${label}"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 30 ]; then
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

  if ! /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 30; then
    echo "[ERR] $label audit FAILED"
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
from collections import defaultdict
rows=[json.loads(l) for l in open('$out_dir/scored.jsonl')]
by_arm = defaultdict(list)
for r in rows:
    by_arm[r['arm']].append(r)
print(f'=== $label ===')
for arm in ['plain','prompted','scaffolded']:
    arows = by_arm.get(arm, [])
    n = len(arows)
    if n == 0: continue
    harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in arows)
    leak = sum(int((r.get('leak_rate') or 0) > 0) for r in arows)
    mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in arows)
    print(f'  {arm}: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} ({100*leak/n:.0f}%) MI={mi} ({100*mi/n:.0f}%)')
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
print(f'  TOTAL: n={n} harm={harm} ({100*harm/n:.0f}%)')
"
}

# Wait — actually we want BOTH plain-only (default baseline) AND comparison with phase4_promptv4_*.
# The harness applies system prompts via "arm" parameter; the SAME run gives us plain/prompted/scaffolded.
# That means the existing phase4_promptv4_gpt5 already has the plain arm (no v4) data!
# Let me check.

# Actually for v4 prompt to be active in any arm, it has to be PART of the system prompt for that arm.
# Looking at src/agent.py: plain = PLAIN_SYSTEM, prompted = PROMPTED_SYSTEM (v4), scaffolded = SCAFFOLDED_HEADER (v4 + sentinel).
# So phase4_promptv4_gpt5/scored.jsonl already has 36 plain-arm trajectories that are the DEFAULT prompt!
# No need to re-run.

# Let me just analyze the existing data.
echo "[no-op] data already in phase4_promptv4_* — use scripts/analyze_arm_breakdown.py"
