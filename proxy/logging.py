"""Daily-rotated JSONL request log.

Design constraints:

- One row per request, append-only, line-delimited JSON.
- File path rotates by UTC date: requests-YYYY-MM-DD.jsonl
- Writing the log row MUST NOT affect the request path. Wrap the actual file
  write in try/except; on failure, print to stderr (systemd captures it) and
  drop the row.
- I/O via anyio so the write can be scheduled as a FastAPI background task
  without blocking the event loop.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio

from config import LOG_DIR


def _log_path_for_today() -> Path:
    """Return today's JSONL log path (UTC)."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return LOG_DIR / f"requests-{today}.jsonl"


def make_log_row(
    *,
    ts: str,
    ip: str,
    method: str,
    path: str,
    status: int,
    streamed: bool,
    model: str,
    tokens_in: int,
    tokens_out: int,
    tokens_reasoning: int,
    reasoning_effort: str,
    latency_ms: int,
    request: Any,
    response: Any,
    reasoning_text: str,
) -> dict:
    """Canonical log row shape. Kwargs-only to prevent positional drift."""
    return {
        "ts": ts,
        "ip": ip,
        "method": method,
        "path": path,
        "status": status,
        "streamed": streamed,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_reasoning": tokens_reasoning,
        "reasoning_effort": reasoning_effort,
        "latency_ms": latency_ms,
        "request": request,
        "response": response,
        "reasoning_text": reasoning_text,
    }


async def write_log_row(row: dict) -> None:
    """Append one JSONL line. Never raises into the request path."""
    try:
        path = _log_path_for_today()
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with await anyio.open_file(path, mode="a", encoding="utf-8") as f:
            await f.write(line)
    except Exception as exc:  # noqa: BLE001 — 의도적으로 광범위. 로그 실패가 응답을 막아선 안 됨.
        print(f"[proxy.logging] write failed: {exc!r}", file=sys.stderr, flush=True)
