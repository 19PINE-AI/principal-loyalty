#!/usr/bin/env bash
# Start vLLM serving the untrained Llama-3.1-8B-Instruct, then run the
# 108-rollout baseline grid, then stop vLLM. NVML is broken on this box
# (driver 590 kernel vs 595 userspace mismatch), so process detection
# uses lsof against /dev/nvidia* rather than nvidia-smi.
set -uo pipefail
cd /home/ubuntu/principal-loyalty

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
PORT=8000
WAIT_TIMEOUT=420
LOG="logs/vllm_llama_baseline.log"
EVAL_DIR="runs/phase3_baseline_llama"

mkdir -p "$EVAL_DIR" logs

# Kill any existing vLLM
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3

# Start vLLM
nohup /home/ubuntu/aoi-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen/Qwen3-8B qwen-8b-local \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --port $PORT \
  > "$LOG" 2>&1 &
VLLM_PID=$!
echo "[vllm] PID $VLLM_PID; waiting for readiness"

t=0
until curl -sf http://localhost:$PORT/v1/models >/dev/null 2>&1; do
  sleep 5; t=$((t+5))
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[vllm] died before ready"; tail -30 "$LOG"; exit 1
  fi
  if [ $t -ge $WAIT_TIMEOUT ]; then
    echo "[vllm] timeout after ${t}s"; tail -30 "$LOG"; exit 1
  fi
done
echo "[vllm] ready after ${t}s"

# Run the eval
/home/ubuntu/polar-env/bin/python scripts/run_phase3_baseline_llama.py 2>&1 | tee "$EVAL_DIR/run.log"
EVAL_STATUS=${PIPESTATUS[0]}

# Headlines
/home/ubuntu/polar-env/bin/python -c "
import json
errs=harm=leak=bound=mi=0; turns={}; n=0
with open('$EVAL_DIR/scored.jsonl') as f:
    for line in f:
        r=json.loads(line); n+=1
        errs += int('error' in (r.get('early_end_reason') or '').lower())
        h = r.get('harm') or {}
        sub = h.get('subflags') or {}
        harm += int(bool(h.get('harm_fire')))
        leak += int((r.get('leak_rate') or 0)>0)
        bound += int(bool(sub.get('leaked_private_bound')))
        mi += int(bool(sub.get('missed_instruction')))
        t = r.get('turns') or 0; turns[t] = turns.get(t,0)+1
print(f'llama_baseline: harm={harm}/{n} leak={leak}/{n} bound={bound} MI={mi} errs={errs} turns={turns}')
"

# Cleanup
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true

exit $EVAL_STATUS
