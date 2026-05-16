#!/usr/bin/env bash
# v2 orchestrator: waits for stage 1 (iter2 multiseed) and runs stages 2-5
# with proper GPU gating between each stage (since stage 1 ate vLLM contention).
set -uo pipefail
cd /home/ubuntu/principal-loyalty
mkdir -p logs

STAGE1_PID="${STAGE1_PID:-}"
if [ -n "$STAGE1_PID" ]; then
  echo "[v2] waiting for stage 1 (pid=$STAGE1_PID)"
  while kill -0 "$STAGE1_PID" 2>/dev/null; do sleep 60; done
  echo "[v2] stage 1 finished"
fi

# Stage 2: Qwen3-32B + v4 teacher validation
echo "===== v2 STAGE 2: Qwen3-32B teacher validation ====="
pkill -9 -f "vllm.entrypoints" 2>/dev/null
for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
while :; do
  free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
  [ "$free" -ge 30 ] && break
  echo "[v2-stage2] free=${free}GB; sleeping 60s"; sleep 60
done
bash scripts/eval_qwen32b_teacher.sh 2>&1 | tee logs/stage2_teacher_v2.log
echo "===== v2 STAGE 2 DONE ====="

# Stage 3: DAPO from per-token KL iter1
echo "===== v2 STAGE 3: DAPO from per-token KL iter1 ====="
pkill -9 -f "vllm.entrypoints" 2>/dev/null
for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
while :; do
  free=$(/home/ubuntu/aoi-env/bin/python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null || echo 0)
  [ "$free" -ge 60 ] && break
  echo "[v2-stage3] free=${free}GB; sleeping 60s"; sleep 60
done
bash scripts/run_dapo_from_pertoken_kl.sh 2>&1 | tee logs/stage3_dapo_v2.log
echo "===== v2 STAGE 3 DONE ====="

# Stage 4: cross-base same-family (Llama-3.1-70B teacher + Mixtral teacher)
echo "===== v2 STAGE 4: cross-base same-family ====="
pkill -9 -f "vllm.entrypoints" 2>/dev/null
for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
bash scripts/run_pertoken_kl_xbase_v2.sh 2>&1 | tee logs/stage4_xbase_v2.log
echo "===== v2 STAGE 4 DONE ====="

# Stage 5: scaled 3x data
echo "===== v2 STAGE 5: scaled 3x on-policy data ====="
pkill -9 -f "vllm.entrypoints" 2>/dev/null
for p in $(ps -e -o pid=,comm= | awk '/EngineCore/{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 10
bash scripts/run_pertoken_kl_scaled_data.sh 2>&1 | tee logs/stage5_scaled_v2.log
echo "===== v2 STAGE 5 DONE ====="

echo "[v2 ALL DONE]"
