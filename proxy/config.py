"""Static configuration for the LLM-OPS logging proxy.

Values are sourced from environment variables when present, otherwise from the
defaults below. Production overrides live in the systemd unit / .env file.
"""

import os
from pathlib import Path


PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")

# Daily-rotated JSONL request log lives here.
LOG_DIR = Path(os.getenv("LOG_DIR", "/home/haru/Desktop/LLM-OPS/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# served-model-name as exposed to clients. Held constant across model migrations.
MODEL_NAME = os.getenv("MODEL_NAME", "BCCard/gemma-4-31B-it-FP8-Dynamic")

# Sampling defaults injected when a client omits them.
# 클라이언트가 값을 보내면 덮어쓰지 않는다 — setdefault만 사용.
SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}

# Thinking mode: ON unless the client explicitly turns it off.
THINKING_DEFAULT = True

# Admin endpoints that must never be reachable from outside.
BLOCKED_PATHS = {
    "/v1/load_lora_adapter",
    "/v1/unload_lora_adapter",
    "/reset_prefix_cache",
}

# httpx upstream timeout. Long enough to cover 256K-context generations.
UPSTREAM_TIMEOUT_SECONDS = int(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "600"))
