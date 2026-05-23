#!/usr/bin/env bash
# vllm-gemma4.service ExecStart wrapper.
# Why a wrapper, not inline ExecStart=?
#   systemd's ExecStart parser unquotes nested JSON in arguments unpredictably.
#   --speculative-config takes a JSON blob like {"model": "...", "num_speculative_tokens": 5}.
#   Building that string in bash and exec'ing vllm directly is the only way to
#   keep the JSON intact across systemd → /bin/sh → vllm argv.

set -euo pipefail

LLM_OPS_DIR="${LLM_OPS_DIR:-/home/haru/Desktop/LLM-OPS}"
MODEL="${MODEL:-BCCard/gemma-4-31B-it-FP8-Dynamic}"
ASSISTANT_MODEL="${ASSISTANT_MODEL:-google/gemma-4-31B-it-assistant}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it}"

# Activate serving venv
# shellcheck disable=SC1091
source "${LLM_OPS_DIR}/.venv/bin/activate"

# CUDA env (defaults so this script also works standalone outside systemd)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"

# Speculative decoding config (MTP draft head, num_speculative_tokens=5).
# 모델 카드 권장값(4) 대신 5를 쓰는 이유는 docs/adr/003-mtp-num-spec-tokens-5.md 참고.
SPEC_CONFIG=$(cat <<EOF
{"model": "${ASSISTANT_MODEL}", "num_speculative_tokens": 5, "method": "mtp"}
EOF
)

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --async-scheduling \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --speculative-config "${SPEC_CONFIG}"
