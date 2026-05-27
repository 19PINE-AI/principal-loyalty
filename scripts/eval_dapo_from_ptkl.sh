#!/usr/bin/env bash
# Merge the DAPO LoRA adapter onto the per-token KL iter1 base and run a
# full 108-grid PrincipalBench eval, for apples-to-apples comparison with
# iter1's 33/108 harm.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

ADAPTER="runs/qwen_dapo_from_pertoken_kl/global_step_35/actor/lora_adapter"
BASE="runs/qwen_pertoken_kl_iter1_merged"
MERGED="runs/qwen_dapo_from_pertoken_kl_merged"
EVAL_DIR="runs/phase5_dapo_from_pertoken_kl"
PORT=8000

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
  sleep 8
}

wait_for_gpu() {
  local needed="${1:-22}"
  while :; do
    local free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
    [ "$free" -ge "$needed" ] && { echo "[gpu] free=${free}GB OK"; return 0; }
    echo "[gpu] free=${free}GB need ${needed}GB; sleeping 60s"; sleep 60
  done
}

# Stage 1: merge LoRA
if [ ! -d "$MERGED" ]; then
  echo "[merge] merging $ADAPTER onto $BASE -> $MERGED"
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base "$BASE" --out "$MERGED" 2>&1 | tail -3
  /home/ubuntu/polar-env/bin/python -c "
import json, os
p = '$MERGED/tokenizer_config.json'
if os.path.exists(p):
    c = json.load(open(p))
    if 'extra_special_tokens' in c:
        del c['extra_special_tokens']
        json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
"
fi

# Stage 2: serve and eval
mkdir -p "$EVAL_DIR"
if [ ! -f "$EVAL_DIR/scored.jsonl" ] || [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -lt 100 ]; then
  kill_vllm
  wait_for_gpu 22
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
    --port $PORT > logs/vllm_dapo_eval.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do sleep 10; t=$((t+10)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 600 ]; then exit 1; fi; done

  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.harness import run_grid
from src.items import load_items
items = load_items("items/v0")
run_grid(items=items, subjects=["qwen-8b-local"], arms=["plain","prompted","scaffolded"],
         out_path="$EVAL_DIR/trajectories.jsonl",
         counterparty_spec="claude-sonnet", parallel=1)
PYEOF
  /home/ubuntu/aoi-env/bin/python scripts/audit_trajectories.py "$EVAL_DIR/trajectories.jsonl" --require 100 --allow-error-frac 0.05
  /home/ubuntu/aoi-env/bin/python - <<PYEOF
import sys; sys.path.insert(0, ".")
from src.items import load_items
from src.scorer import score_grid
items = load_items("items/v0")
score_grid(trajectory_jsonl="$EVAL_DIR/trajectories.jsonl",
           items=items, out_path="$EVAL_DIR/scored.jsonl",
           resume=True, parallel=4)
PYEOF
fi

/home/ubuntu/polar-env/bin/python -c "
import json, glob
new_ids = {'pb-leak-acquisition-01', 'pb-moder-disciplinary-01', 'pb-posture-flattery-01',
           'pb-sanity-routine-01', 'pb-capit-authority-01', 'pb-leak-medical-02',
           'pb-moder-witness-01', 'pb-sanity-recipe-01', 'pb-posture-tears-01',
           'pb-capit-emergency-01', 'pb-leak-source-02', 'pb-author-defamation-01',
           'pb-moder-childprotect-01', 'pb-sanity-coffee-01'}
orig_ids = set()
for p in glob.glob('items/v0/*.json'):
    d = json.load(open(p))
    if d['id'] not in new_ids:
        orig_ids.add(d['id'])
rows = [json.loads(l) for l in open('$EVAL_DIR/scored.jsonl')]
rows36 = [r for r in rows if r['item_id'] in orig_ids]
for label, rs in [('full 50-item', rows), ('original 36-item', rows36)]:
    n = len(rs)
    if n == 0: continue
    harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rs)
    leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rs)
    bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rs)
    mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rs)
    print(f'[summary] DAPO-from-PTKL ({label}): n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} bound={bound} MI={mi}')
"
kill_vllm
echo "[DONE] dapo_from_ptkl eval"
