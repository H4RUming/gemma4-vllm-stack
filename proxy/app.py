"""FastAPI logging proxy — SKELETON.

This file is a skeleton scaffold. The production catch-all handler lives in the
deployed `~/Desktop/LLM-OPS/proxy/main.py`; port it here, preserving:

- Streaming pass-through with SSE chunk accumulation for the log row.
- Non-streaming JSON branch.
- BLOCKED_PATHS rejection.
- Background-task JSONL logging via `write_log_row`.
- Thinking-mode normalization + sampling default injection on
  `/v1/chat/completions` via `adapt_request`.

What this skeleton intentionally does NOT include:

- Auth, rate limiting, multi-tenant routing — out of scope (see ADR-005).
- Persistence beyond JSONL — out of scope.

The lifespan opens a single shared httpx.AsyncClient bound to the upstream so
keep-alive connections are reused across requests.
"""

from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from config import (
    BLOCKED_PATHS,
    PROXY_HOST,
    PROXY_PORT,
    UPSTREAM_TIMEOUT_SECONDS,
    VLLM_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open a shared httpx client for the lifetime of the process."""
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    async with httpx.AsyncClient(
        base_url=VLLM_URL,
        timeout=UPSTREAM_TIMEOUT_SECONDS,
        limits=limits,
    ) as client:
        app.state.upstream = client
        yield


app = FastAPI(lifespan=lifespan)


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def catch_all(full_path: str):
    """Transparent proxy with logging.

    TODO: port the production handler. The handler must:
      1. Reject requests whose path matches BLOCKED_PATHS with a 403.
      2. Read the request body. If Content-Type is application/json, parse it.
      3. For `/v1/chat/completions`, call adapt_request() on the parsed body
         BEFORE forwarding (thinking normalization + sampling defaults).
      4. Detect `stream: true`. If streaming, forward upstream as a streaming
         response and accumulate `content` + `reasoning_content` chunks for
         the log row as they pass through. If non-streaming, forward and
         capture the full JSON response.
      5. Capture start/end timestamps. Assemble a log row via make_log_row()
         and schedule it as a FastAPI BackgroundTask running write_log_row.
         Log-write failures MUST NOT propagate to the response path.
    """
    # Sketch only — replace with the real implementation.
    _ = BLOCKED_PATHS  # silence "unused import" until the handler lands
    raise NotImplementedError(
        "proxy.app.catch_all is a skeleton; port the production handler from "
        "~/Desktop/LLM-OPS/proxy/main.py before deploying."
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=PROXY_HOST,
        port=PROXY_PORT,
        access_log=False,
    )
