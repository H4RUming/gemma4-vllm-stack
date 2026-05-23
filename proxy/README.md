# LLM-OPS logging proxy

Async FastAPI proxy that sits in front of vLLM at `0.0.0.0:8000` and forwards to `127.0.0.1:8001`.

## What it does

1. Accepts every OpenAI-compatible endpoint via a catch-all route.
2. Normalizes thinking-mode parameters across the four client encodings (`reasoning_effort`, `extra_body.thinking`, `thinking_config`, canonical `chat_template_kwargs.enable_thinking`). Default ON.
3. Injects sampling defaults (`temperature=1.0, top_p=0.95, top_k=64`) when the client omits them — never overwrites a client value.
4. Streams the upstream response back transparently while accumulating content + reasoning for the log row.
5. Appends one JSONL row per request to `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl` as a FastAPI background task.
6. Rejects `BLOCKED_PATHS` (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/reset_prefix_cache`) with a 403.
7. Maintains a single shared `httpx.AsyncClient` (keep-alive, 100 max connections) for the lifetime of the process.

## What it doesn't do

- No auth or API keys — access control lives at the network layer (Tailnet).
- No rate limiting.
- No multi-tenant routing — single model.
- No persistence beyond the daily JSONL log.

## Layout

```
proxy/
├── README.md          ← this file
├── app.py             ← FastAPI app + catch-all route (SKELETON — port from production)
├── adapters.py        ← normalize_thinking, inject_sampling_defaults, adapt_request
├── logging.py         ← make_log_row, write_log_row, _log_path_for_today
├── config.py          ← static config + env-var overrides
└── requirements.txt   ← fastapi, httpx, uvicorn[standard], transformers, tokenizers, anyio
```

## Dependencies

Intentionally tiny. No torch, no vLLM, no CUDA. Boots in seconds; ~400 MB resident.

```
fastapi>=0.115
httpx>=0.27
uvicorn[standard]>=0.30
transformers>=5.5
tokenizers>=0.20
anyio>=4
```

`transformers` and `tokenizers` are here only to count tokens on the request/response side for the log row — neither imports torch.

## Run locally for development

```bash
uv venv .venv-proxy --python 3.12
uv pip install --python .venv-proxy/bin/python -r requirements.txt
.venv-proxy/bin/python app.py
```

The proxy will try to forward to `http://127.0.0.1:8001` (override with `VLLM_URL=...`). For development without a real vLLM running, point `VLLM_URL` at a local mock or comment out the forward and return canned responses.

## Example JSONL row

```json
{
  "ts": "2026-05-09T10:14:32.118Z",
  "ip": "100.64.1.7",
  "method": "POST",
  "path": "/v1/chat/completions",
  "status": 200,
  "streamed": true,
  "model": "gemma-4-31b-it",
  "tokens_in": 412,
  "tokens_out": 798,
  "tokens_reasoning": 245,
  "reasoning_effort": "high",
  "latency_ms": 7912,
  "request": { "messages": [ ... ], "stream": true },
  "response": "...",
  "reasoning_text": "..."
}
```

See `docs/architecture.md` for `jq` query examples against the log file.
