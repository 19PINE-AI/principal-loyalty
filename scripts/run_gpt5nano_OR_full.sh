#!/usr/bin/env bash
# Re-eval gpt-5-nano on the full 108-grid via OpenRouter routing, to
# determine whether the prior 14% direct-OpenAI number reproduces under
# the new routing path (spot check showed 67% on 12 items).
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs runs/phase4_promptv4_gpt5_nano

/home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, '.')
from src.harness import run_grid
from src.items import load_items
items = load_items('items/v0')
print(f'Re-evaluating gpt-5-nano (n={len(items)} items x 3 arms) via OpenRouter')
run_grid(items=items, subjects=['gpt-5-nano'],
         arms=['plain','prompted','scaffolded'],
         out_path='runs/phase4_promptv4_gpt5_nano/trajectories.jsonl',
         counterparty_spec='claude-sonnet', parallel=4)
PYEOF

/home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py \
  runs/phase4_promptv4_gpt5_nano/trajectories.jsonl --require 100 --allow-error-frac 0.05 || \
  { echo "[fatal] audit failed"; exit 1; }

/home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, '.')
from src.items import load_items
from src.scorer import score_grid
items = load_items('items/v0')
score_grid(trajectory_jsonl='runs/phase4_promptv4_gpt5_nano/trajectories.jsonl',
           items=items, out_path='runs/phase4_promptv4_gpt5_nano/scored.jsonl',
           resume=True, parallel=4)
PYEOF

/home/ubuntu/polar-env/bin/python -c "
import json
rows = [json.loads(l) for l in open('runs/phase4_promptv4_gpt5_nano/scored.jsonl')]
n = len(rows)
harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rows)
leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rows)
mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rows)
from collections import Counter
by_arm = {}
for r in rows:
    by_arm.setdefault(r['arm'], []).append(r)
print(f'[summary] gpt-5-nano via OpenRouter (training): n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} MI={mi}')
for arm in ['plain', 'prompted', 'scaffolded']:
    arows = by_arm.get(arm, [])
    if not arows: continue
    a_harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in arows)
    print(f'  {arm:12s}: n={len(arows)} harm={a_harm} ({100*a_harm/len(arows):.0f}%)')
"
echo "[DONE] gpt-5-nano OpenRouter full"
