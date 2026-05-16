#!/usr/bin/env bash
# Start vLLM serving Qwen3-32B-AWQ on port 8001 as 'qwen-32b-local'.
# This is the OPEN-WEIGHT TEACHER for on-policy distillation, parallel to the
# claude-sonnet API teacher. Qwen3-32B-AWQ at 4-bit AWQ fits in ~22 GB on
# this shared GPU.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

MODEL="Qwen/Qwen3-32B-AWQ"
PORT=${PL_TEACHER_PORT:-8001}
LOG=logs/vllm_qwen32b_teacher.log
WAIT_TIMEOUT=420

mkdir -p logs

# Kill anything on the teacher port
pkill -9 -f "api_server --model.*Qwen3-32B-AWQ.*port $PORT" 2>/dev/null || true
sleep 3

# Start vLLM with AWQ quantization
nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen/Qwen3-32B qwen-32b-local \
  --dtype auto \
  --quantization awq \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.30 \
  --port $PORT \
  > "$LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > /tmp/qwen32b_teacher.pid
echo "[teacher] PID $VLLM_PID on port $PORT (Qwen3-32B-AWQ)"

t=0
until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
  sleep 10; t=$((t+10))
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[teacher] DIED"; tail -30 "$LOG"; exit 1
  fi
  if [ $t -ge $WAIT_TIMEOUT ]; then
    echo "[teacher] timeout ${t}s"; tail -30 "$LOG"; exit 1
  fi
done
echo "[teacher] READY after ${t}s on port $PORT"
