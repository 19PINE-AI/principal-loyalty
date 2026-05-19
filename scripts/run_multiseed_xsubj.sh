#!/usr/bin/env bash
# Multi-seed cross-subject v4 prompt evaluation. Re-evaluate each vendor at
# 4 additional seeds to get error bars on the calibrated / over-refusal
# classification. Pure API, no GPU.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

# (vendor_spec, label_in_existing_run, friendly_name)
declare -A VENDORS
VENDORS[claude-sonnet]="frontier:Claude-Sonnet"
VENDORS[gpt-5]="gpt5:GPT-5"
VENDORS[gemini-3-flash]="gemini3flash:Gemini-3-flash"
VENDORS[deepseek]="deepseek:DeepSeek"
VENDORS[qwen-27b]="qwen27b:Qwen3.5-27B"

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
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$out_dir/trajectories.jsonl" --require 90 --allow-error-frac 0.10 || true
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

# Run 4 extra seeds for each of 5 vendors
for subj_spec in claude-sonnet gpt-5 gemini-3-flash deepseek qwen-27b; do
  for seed in 2 3 4 5; do
    case $subj_spec in
      claude-sonnet)   label="claude" ;;
      gpt-5)           label="gpt5" ;;
      gemini-3-flash)  label="gemini3flash" ;;
      deepseek)        label="deepseek" ;;
      qwen-27b)        label="qwen27b" ;;
    esac
    run_seed "$subj_spec" "$label" "$seed"
  done
done

echo "[DONE] multi-seed cross-subject"
