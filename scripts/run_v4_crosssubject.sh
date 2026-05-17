#!/usr/bin/env bash
# Full 108-grid v4 prompt eval across multiple frontier subjects.
# All subjects are API-only — no local vLLM, no GPU contention.
# Counterparty: claude-sonnet (default).
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

run_one() {
  local subj="$1"; local label="$2"
  local out_dir="runs/phase4_promptv4_${label}"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/scored.jsonl" ] && [ "$(wc -l < $out_dir/scored.jsonl)" -ge 100 ]; then
    echo "[skip] $label: already complete ($(wc -l < $out_dir/scored.jsonl) rows)"
    return 0
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
rows=[json.loads(l) for l in open('$out_dir/scored.jsonl')]
n=len(rows); harm=sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak=sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
bound=sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rows)
mi=sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] $label: n={n} harm={harm} leak={leak} bound={bound} MI={mi}')
"
}

# Run cross-subject in sequence (could parallelize but APIs may rate-limit)
run_one "gpt-5"           "gpt5"
run_one "gemini-3-flash"  "gemini3flash"
run_one "qwen-27b"        "qwen27b"

echo "[DONE] cross-subject v4 prompt eval"
