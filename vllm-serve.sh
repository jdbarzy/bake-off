#!/usr/bin/env bash
# Example: start two vLLM instances on a model host, one per GPU.
# Run this ON the model host (ssh in first), ideally inside tmux.
#
# Startup is SEQUENTIAL on purpose: two first-time vLLM instances starting at
# once deadlock on shared torch.compile / JIT caches. We bring one fully up,
# then start the next. --enforce-eager skips torch.compile/CUDA-graph capture
# (fine for short Q&A prompts) for fast, deadlock-free startup.
set -euo pipefail

MODEL_B="NousResearch/Meta-Llama-3.1-8B-Instruct"   # ungated mirror of meta-llama/Llama-3.1-8B-Instruct
MODEL_C="Qwen/Qwen2.5-7B-Instruct"
MAXLEN=8192
VLLM="${HOME}/vllm-env/bin/vllm"   # vLLM lives in a venv (Ubuntu 24.04 PEP 668)

wait_up() {  # wait_up <port> <name>
  local port="$1" name="$2"
  echo "Waiting for ${name} on :${port} (up to ~10 min)..."
  for _ in $(seq 1 120); do
    if [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "http://localhost:${port}/v1/models")" = "200" ]; then
      echo "  ${name} is UP on :${port}"
      return 0
    fi
    sleep 5
  done
  echo "  WARNING: ${name} did not come up on :${port}; check vllm-${port}.log"
  return 1
}

echo "Starting ${MODEL_B} on GPU 0, port 8002..."
CUDA_VISIBLE_DEVICES=0 "${VLLM}" serve "${MODEL_B}" \
  --port 8002 \
  --max-model-len "${MAXLEN}" \
  --enforce-eager \
  > vllm-8002.log 2>&1 &
wait_up 8002 "${MODEL_B}" || true

echo "Starting ${MODEL_C} on GPU 1, port 8003..."
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${MODEL_C}" \
  --port 8003 \
  --max-model-len "${MAXLEN}" \
  --enforce-eager \
  > vllm-8003.log 2>&1 &
wait_up 8003 "${MODEL_C}" || true

echo "Done. Both vLLM instances should be up. Logs: vllm-8002.log vllm-8003.log"
echo "Verify both GPUs are in use: nvidia-smi"
