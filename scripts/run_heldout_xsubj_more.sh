#!/usr/bin/env bash
# Tier B.4 expansion: extend held-out v0_75 cross-vendor to the remaining 9 of 14 vendors.
# Reuses scripts/run_heldout_xsubj.sh logic. Already done: deepseek, gemini-3-flash,
# claude-sonnet, gpt-5, qwen-27b.
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
leak=sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
mi=sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
print(f'[summary] $label heldout: n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi}')
"
}

# 9 more vendors
# Calibrated cluster (6): claude-opus, mistral-large, llama-70b, gemini-3p1-flash-lite,
# gpt-5-nano, qwen3-32b. Already done: claude-sonnet, deepseek, gemini-3-flash.
run_one "claude-opus"            "claude_opus"
run_one "mistral-large"          "mistral_large"
run_one "llama-70b"              "llama70b"
run_one "gemini-3p1-flash-lite"  "gemini3p1_lite"
run_one "gpt-5-nano"             "gpt5_nano"
run_one "qwen-32b"               "qwen32b_openrouter"

# Intermediate (1): GLM-4.6
run_one "glm-4.6"                "glm46"

# Over-refuse cluster (2): gpt-5-mini, qwen3.5-27b. Already done: gpt-5, qwen-27b -- wait,
# qwen-27b IS qwen3.5-27b. So just gpt-5-mini needed for over-refuse.
run_one "gpt-5-mini"             "gpt5mini"

echo "[DONE] heldout cross-vendor expansion"
