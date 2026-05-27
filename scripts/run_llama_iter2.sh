#!/usr/bin/env bash
# Llama per-token KL iter2: sample from llama_pertoken_kl_iter1_merged,
# collect Llama-70B-AWQ teacher logprobs, train, eval. Tests whether the
# K=2 saturation observed for Qwen3 generalizes to Llama-3.1.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

STUDENT_PORT=8000
TEACHER_PORT=8001

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

PREV_MERGED="runs/llama_pertoken_kl_iter1_merged"
TRAJ="data/pertoken_kl/llama_iter2_trajectories.jsonl"
TOPK="data/pertoken_kl/llama_iter2_topk.jsonl"
ADAPTER="runs/llama_pertoken_kl_iter2"
MERGED="runs/llama_pertoken_kl_iter2_merged"
EVAL_DIR="runs/phase5_pertoken_kl_llama_iter2"

# 1. Sample student under iter1 checkpoint
if [ ! -f "$TRAJ" ] || [ "$(wc -l < $TRAJ)" -lt 25 ]; then
  kill_vllm
  wait_for_gpu 22
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$PREV_MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
    --port $STUDENT_PORT > logs/vllm_llama_iter2_sample.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; do sleep 10; t=$((t+10)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 600 ]; then exit 1; fi; done

  OUT_PREFIX="data/pertoken_kl/llama_iter2_sample"
  /home/ubuntu/aoi-env/bin/python scripts/onpolicy_distill_iter.py \
    --out-prefix "$OUT_PREFIX" --n-samples 1 --temperature 1.0 --parallel 4 \
    --max-turns-per-item 4 --teacher-spec claude-sonnet \
    --student-arm plain --teacher-arm scaffolded 2>&1 | tee logs/llama_iter2_sample.log | tail -10
  cp "${OUT_PREFIX}_trajectories.jsonl" "$TRAJ"
fi
echo "[llama-iter2] $(wc -l < $TRAJ) student trajectories"

# 2. Collect Llama-70B-AWQ teacher top-K
if [ ! -f "$TOPK" ] || [ "$(wc -l < $TOPK)" -lt 30 ]; then
  kill_vllm
  wait_for_gpu 50
  VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --served-model-name llama-70b-teacher \
    --dtype auto --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.60 \
    --enforce-eager \
    --port $TEACHER_PORT > logs/vllm_llama_iter2_teacher.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$TEACHER_PORT/v1/models >/dev/null 2>&1; do sleep 15; t=$((t+15)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 1500 ]; then exit 1; fi; done

  TEACHER_URL="http://localhost:$TEACHER_PORT/v1/completions" \
  TEACHER_MODEL_NAME="llama-70b-teacher" \
  TOKENIZER_NAME="runs/llama_pertoken_kl_iter1_merged" \
  /home/ubuntu/aoi-env/bin/python scripts/pertoken_kl_collect.py \
    --trajectories "$TRAJ" --out "$TOPK" --top-k 20 --parallel 4 --max-turns-per-item 4 2>&1 \
    | tee logs/llama_iter2_collect.log | tail -10
fi
echo "[llama-iter2] $(wc -l < $TOPK) topk records"

# 3. Train
if [ ! -d "$ADAPTER" ] || [ -z "$(ls $ADAPTER 2>/dev/null)" ]; then
  kill_vllm
  wait_for_gpu 22
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  PL_MODEL_ID="$PREV_MERGED" PL_KL_PATH="$TOPK" PL_OUT_DIR="$ADAPTER" \
    PL_KL_EPOCHS=3 PL_KL_LR=5e-5 PL_KL_BS=1 PL_KL_GRAD_ACCUM=8 PL_KL_MAX_LEN=3000 \
    /home/ubuntu/polar-env/bin/python -u scripts/train_pertoken_kl.py 2>&1 \
    | tee logs/llama_iter2_train.log | tail -20
fi

# 4. Merge
if [ ! -d "$MERGED" ]; then
  /home/ubuntu/polar-env/bin/python scripts/merge_lora.py \
    --adapter "$ADAPTER" --base "$PREV_MERGED" --out "$MERGED" 2>&1 | tail -3
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

# 5. Eval (served as Qwen/Qwen3-8B so harness routes correctly)
mkdir -p "$EVAL_DIR"
if [ ! -f "$EVAL_DIR/scored.jsonl" ] || [ "$(wc -l < $EVAL_DIR/scored.jsonl)" -lt 100 ]; then
  kill_vllm
  wait_for_gpu 22
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.20 \
    --port $STUDENT_PORT > logs/vllm_llama_iter2_eval.log 2>&1 &
  pid=$!
  t=0; until curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; do sleep 10; t=$((t+10)); if ! kill -0 $pid 2>/dev/null; then exit 1; fi; if [ $t -ge 600 ]; then exit 1; fi; done

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
    harm = sum(int(bool((r.get('harm') or {}).get('harm_fire'))) for r in rs)
    leak = sum(int((r.get('leak_rate') or 0) > 0) for r in rs)
    bound = sum(int(bool((r.get('harm') or {}).get('leaked_private_bound'))) for r in rs)
    mi = sum(int(bool((r.get('harm') or {}).get('missed_instruction'))) for r in rs)
    print(f'[summary] llama_pertoken_kl_iter2 ({label}): n={n} harm={harm} ({100*harm/n:.0f}%) leak={leak} bound={bound} MI={mi}')
"
kill_vllm
echo "[DONE] llama-iter2"
