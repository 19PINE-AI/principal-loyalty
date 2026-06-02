#!/usr/bin/env bash
# Multi-seed n=5 on the other 3 DAPO variants (v2, leak-only, v3) to round
# out the appendix rigor. Each variant already has seed1 scored at single-seed.
# We add seed2-5, score, then run paired Wilcoxon against v4.1 (n=5).
#
# Idempotent: skips any seed dir whose trajectories.jsonl is already 108 rows.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

# Skip mechanism: if /tmp/pl_skip_dapo_variants exists, exit immediately.
# This lets the master orchestrator's stage B be preempted by other work
# (e.g. on-policy distillation) without killing the orchestrator itself.
if [ -f /tmp/pl_skip_dapo_variants ]; then
  echo "[skip] /tmp/pl_skip_dapo_variants present — DAPO-variants stage deferred"
  exit 0
fi

PORT=8000
WAIT_TIMEOUT=420

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

start_vllm() {
  local model_path="$1"; local log="$2"
  echo "[vllm] starting $model_path"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $PORT >"$log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm] DIED"; tail -30 "$log"; return 1; fi
    if [ $t -ge $WAIT_TIMEOUT ]; then echo "[vllm] timeout"; tail -30 "$log"; return 1; fi
  done
  echo "[vllm] READY after ${t}s"
}

# (variant_name, merged_dir, seed1_eval_dir)
VARIANTS=(
  "dapo_v2_step55|runs/qwen_dapo_v2_step55_merged|runs/phase3_dapo_v2_step55"
  "dapo_leakonly_step35|runs/qwen_dapo_leakonly_step35_merged|runs/phase3_dapo_leakonly_step35"
  "dapo_v3_step55|runs/qwen_dapo_v3_step55_merged|runs/phase3_dapo_v3_step55"
)

# Strip the bad extra_special_tokens list from each merged dir (idempotent).
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model seed1 <<<"$entry"
  /home/ubuntu/polar-env/bin/python -c "
import json, os, shutil
p = '$model/tokenizer_config.json'
if not os.path.exists(p): exit()
if not os.path.exists(p + '.bak'):
    shutil.copy(p, p + '.bak')
c = json.load(open(p))
if 'extra_special_tokens' in c:
    del c['extra_special_tokens']
    json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
    print(f'fixed {p}')
"
done

# Process each variant: start vLLM, generate seed2-5 trajectories
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model seed1 <<<"$entry"
  echo "===== [$name] vLLM + trajectory generation ====="
  if [ ! -d "$model" ]; then
    echo "  [skip] $model missing"; continue
  fi

  # Pre-check: are all seeds already done?
  all_done=1
  for s in 2 3 4 5; do
    tj="${seed1}_seed${s}/trajectories.jsonl"
    if [ ! -f "$tj" ] || [ "$(wc -l < "$tj" 2>/dev/null)" -lt 108 ]; then
      all_done=0; break
    fi
  done
  if [ $all_done -eq 1 ]; then
    echo "  [skip] $name traj already complete for seed2-5"; continue
  fi

  kill_vllm
  start_vllm "$model" "logs/vllm_${name}_multiseed.log" || { echo "[fail] $name vLLM"; continue; }

  /home/ubuntu/polar-env/bin/python scripts/run_traj_only.py \
    --seed-dirs "${seed1}_seed2" "${seed1}_seed3" "${seed1}_seed4" "${seed1}_seed5" \
    --subject qwen-8b-local --counterparty claude-sonnet --parallel 4 \
    2>&1 | tee logs/multiseed_${name}_traj.log | tail -10
done

kill_vllm

# Score everything (OpenRouter judge, no GPU)
echo "===== scoring all new seed dirs ====="
SCORE_DIRS=""
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model seed1 <<<"$entry"
  for s in 2 3 4 5; do
    SCORE_DIRS="$SCORE_DIRS ${seed1}_seed${s}"
  done
done
/home/ubuntu/polar-env/bin/python scripts/score_only.py \
  --seed-dirs $SCORE_DIRS --parallel 8 \
  2>&1 | tee logs/multiseed_dapo_variants_scoring.log | tail -30

# Paired Wilcoxon against v4.1 baseline
echo "===== paired Wilcoxon n=5: each DAPO variant vs v4.1 ====="
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name model seed1 <<<"$entry"
  echo "--- ${name} vs v4.1 ---"
  /home/ubuntu/polar-env/bin/python scripts/paired_seed_test.py \
    --a runs/phase2_trained_v4_1 --b "$seed1" \
    --a-seeds 5 --b-seeds 5 2>&1 | tee "logs/paired_seed_${name}_n5.log"
done

echo "[DONE] DAPO-variant multi-seed n=5 complete"
