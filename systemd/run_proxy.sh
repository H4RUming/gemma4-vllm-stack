#!/usr/bin/env bash
# vllm-proxy.service ExecStart wrapper.

set -euo pipefail

LLM_OPS_DIR="${LLM_OPS_DIR:-/home/haru/Desktop/LLM-OPS}"

# Activate proxy venv (no torch, no CUDA)
# shellcheck disable=SC1091
source "${LLM_OPS_DIR}/.venv-proxy/bin/activate"

# Tokenizer cache; share with the serving env to avoid double downloads.
export HF_HOME="${HF_HOME:-${LLM_OPS_DIR}/hf-cache}"

cd "${LLM_OPS_DIR}/proxy"
exec python app.py
