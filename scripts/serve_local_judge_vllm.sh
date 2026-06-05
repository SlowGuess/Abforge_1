#!/usr/bin/env bash
set -euo pipefail

# Starts an OpenAI-compatible local judge endpoint. The reward servers connect
# to it through JUDGE_API_BASE=http://127.0.0.1:${PORT}/v1.

: "${JUDGE_MODEL:?Set JUDGE_MODEL to the local model to serve as the judge}"
MODEL=${JUDGE_MODEL}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "$@"
