#!/usr/bin/env bash
# Per-token KL distillation on Mistral-7B and Llama-3.1-8B SFT+DPO bases
# (parallel to the Qwen3-8B iter1 result). Tests whether per-token KL changes
# the "on-policy direction depends on base position" finding from Variant B.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

STUDENT_PORT=8000
TEACHER_PORT=8001

kill_vllm() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore\|EngineCore" 2>/dev/null || true
  sleep 6
}

start_student_vllm() {
  local model="$1"; local log="$2"
  echo "[vllm-student] starting $model on :$STUDENT_PORT"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --served-model-name Qwen/Qwen3-8B qwen-8b-local \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --port $STUDENT_PORT >"$log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$STUDENT_PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm-student] DIED"; tail -30 "$log"; return 1; fi
    if [ $t -ge 300 ]; then echo "[vllm-student] timeout"; return 1; fi
  done
  echo "[vllm-student] READY after ${t}s"
}

start_teacher_vllm() {
  echo "[vllm-teacher] starting Qwen3-32B-AWQ on :$TEACHER_PORT"
  nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --served-model-name Qwen/Qwen3-32B qwen-32b-local \
    --dtype auto --quantization awq --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --port $TEACHER_PORT >"logs/vllm_qwen32b_xbase.log" 2>&1 &
  local pid=$!
  local t=0
  until curl -sf http://localhost:$TEACHER_PORT/v1/models >/dev/null 2>&1; do
    sleep 10; t=$((t+10))
    if ! kill -0 $pid 2>/dev/null; then echo "[vllm-teacher] DIED"; tail -30 logs/vllm_qwen32b_xbase.log; return 1; fi
    if [ $t -ge 300 ]; then echo "[vllm-teacher] timeout"; return 1; fi
  done
  echo "[vllm-teacher] READY after ${t}s"
}

# WARNING: Cross-base per-token KL uses a Qwen3 teacher distilling INTO Mistral/Llama students.
# Token vocabularies are DIFFERENT between Qwen3 and Mistral/Llama. The teacher's top-K teacher
# tokens cannot be directly applied to the student's vocab. We use Qwen3 tokenizer for both
# sides via cross-tokenization (re-encode teacher logprobs to student vocab) — but this is lossy.
# As an alternative, we could re-collect using a same-family teacher (Mistral-Instruct-large for
# Mistral student, Llama-3.1-70B for Llama student). Given GPU constraints, we just attempt the
# Qwen3 teacher direction and accept the cross-tokenizer noise — the result is informative as
# an upper bound on what per-token KL can do for non-Qwen students under our setup.
#
# For now, only run if the bases are Qwen-family (which Mistral and Llama are NOT).
# So Tier B4 with per-token KL is DEFERRED pending same-family teachers.
echo "[note] cross-base per-token KL deferred: requires same-family teacher per base"
echo "       (Qwen3 teacher's vocab doesn't match Mistral or Llama vocab)"
echo "       Falling back to per-turn SFT comparison already documented in Tier B4."
echo "[DONE] Cross-base per-token KL skipped"
