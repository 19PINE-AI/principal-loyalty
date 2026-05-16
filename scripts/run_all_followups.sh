#!/usr/bin/env bash
# Orchestrator: wait for stage 1 (iter2 multi-seed) then fire stages 2-5 in
# sequence. Each stage has GPU-wait gates and retry logic.
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

wait_for_pid_done() {
  local pid="$1"; local name="$2"
  echo "[orchestrator] waiting for $name (pid=$pid)"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[orchestrator] $name finished"
}

# Wait for stage 1 (iter2 multi-seed) — caller passes its pid via env
[ -n "${STAGE1_PID:-}" ] && wait_for_pid_done "$STAGE1_PID" "stage1_iter2_multiseed"

# Stage 2: Qwen3-32B teacher validation
echo "===== STAGE 2: Qwen3-32B teacher validation ====="
bash scripts/eval_qwen32b_teacher.sh 2>&1 | tee logs/stage2_teacher_validation.log || echo "[stage2 failed]"

# Stage 3: DAPO from per-token KL iter1
echo "===== STAGE 3: DAPO from per-token KL iter1 ====="
bash scripts/run_dapo_from_pertoken_kl.sh 2>&1 | tee logs/stage3_dapo_from_ptkl.log || echo "[stage3 failed]"

# Stage 4: cross-base same-family
echo "===== STAGE 4: cross-base same-family ====="
bash scripts/run_pertoken_kl_xbase_v2.sh 2>&1 | tee logs/stage4_xbase.log || echo "[stage4 failed]"

# Stage 5: scaled 3x data
echo "===== STAGE 5: scaled 3x on-policy data ====="
bash scripts/run_pertoken_kl_scaled_data.sh 2>&1 | tee logs/stage5_scaled.log || echo "[stage5 failed]"

echo "[ALL DONE] All 5 follow-ups complete (or skipped on failure)"
