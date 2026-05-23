# Architecture

This document describes the runtime topology, dependency hygiene, request flow, and observability for the Gemma 4 31B serving stack.

---

## Components

### `vllm-gemma4.service`

Long-running vLLM OpenAI-compatible server bound to `127.0.0.1:8001`. Loads `BCCard/gemma-4-31B-it-FP8-Dynamic` with an MTP draft head and FP8 KV cache.

Key serving flags:

| Flag                                | Value                                   | Purpose                                                                 |
| ----------------------------------- | --------------------------------------- | ----------------------------------------------------------------------- |
| `--served-model-name`               | `gemma-4-31b-it`                        | Stable name kept across model migrations so clients don't change.       |
| `--host` / `--port`                 | `127.0.0.1` / `8001`                    | Loopback only; proxy is the public surface.                             |
| `--max-model-len`                   | `262144`                                | 256K per-request context cap.                                           |
| `--gpu-memory-utilization`          | `0.90`                                  | Leaves headroom for CUDA graphs + FlashInfer JIT scratch.               |
| `--max-num-batched-tokens`          | `32768`                                 | Bounds the per-step prefill chunk; trades ~14% KV pool for predictable activation memory. See [ADR-007](adr/007-max-num-batched-tokens-32768.md). |
| `--kv-cache-dtype`                  | `fp8`                                   | Standard FP8 e4m3, not TurboQuant. See [ADR-001](adr/001-fp8-kv-over-turboquant.md). |
| `--async-scheduling`                | (flag)                                  | Overlaps prefill + decode; lower TTFT under concurrency.                |
| `--reasoning-parser`                | `gemma4`                                | Splits `<think>…</think>` into the `reasoning_content` channel.         |
| `--tool-call-parser`                | `gemma4`                                | Parses Gemma 4's function-call markup.                                  |
| `--enable-auto-tool-choice`         | (flag)                                  | Server-side choice over which tool to call when client passes a list.   |
| `--speculative-config`              | (JSON, see `run_vllm.sh`)               | `num_speculative_tokens=5`, draft `gemma-4-31B-it-assistant`. See [ADR-003](adr/003-mtp-num-spec-tokens-5.md). |

### `vllm-proxy.service`

Async FastAPI proxy bound to `0.0.0.0:8000`. Lives in `.venv-proxy` (no torch, no CUDA). Code is versioned in a [separate repository](https://github.com/H4RUming/gemma4-vllm-proxy) and cloned to `${LLM_OPS_DIR}/proxy/` at deploy time; this repo keeps [proxy/README.md](../proxy/README.md) as the interface specification. Three responsibilities:

1. **Normalize thinking-mode parameters.** Accept any of `reasoning_effort`, `extra_body.thinking`, `thinking_config`, or canonical `chat_template_kwargs.enable_thinking`; collapse to the canonical form before forwarding. Default ON.
2. **Inject sampling defaults the client omits.** `temperature=1.0, top_p=0.95, top_k=64` via `setdefault` — never overwrite a client value.
3. **Log every request and response** to a daily-rotated JSONL file at `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl`. The log write is scheduled as a background task so it cannot delay the response.

Additionally, the proxy enforces `BLOCKED_PATHS` (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/reset_prefix_cache`) with a 403 — these admin endpoints should never be reachable from outside.

### systemd unit graph

```
network-online.target
        │
        ▼
nvidia-persistenced.service
        │ After=
        ▼
vllm-gemma4.service ────── Requires= ◄──── vllm-proxy.service
                                                      │
                                              (FastAPI on :8000)
```

`vllm-proxy.service` declares `Requires=vllm-gemma4.service`, so a manual restart of the serving unit cascades — there is no window where the proxy can forward to a dead upstream.

`TimeoutStartSec=600` on the serving unit is intentional: first launch after a vLLM upgrade or model change spends 3–5 minutes on FlashInfer SM120 JIT compilation and CUDA graph capture before the OpenAI HTTP endpoint binds.

---

## Dependency hygiene — three venvs

| venv           | Purpose                          | Notable contents                                              | Why isolated                                                                                  |
| -------------- | -------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `.venv`        | vLLM serving                     | `vllm` (nightly cu129), `torch 2.11.0+cu129`, `flashinfer`, `transformers 5.8.0` | Heavy CUDA stack; the canonical environment for `vllm-gemma4.service`.            |
| `.venv-proxy`  | FastAPI logging proxy            | `fastapi`, `httpx`, `uvicorn[standard]`, `transformers`, `tokenizers`, `anyio` | No torch. Boots in seconds, ~400 MB RAM. Iteration on the proxy never risks the serving env. |
| `.venv-quant`  | Quantization / model authoring   | `llmcompressor`, older torch/transformers                     | `llmcompressor` aggressively pins old `torch`/`transformers`; installed in `.venv` it silently downgrades vLLM. See [ADR-006](adr/006-llmcompressor-venv-isolation.md). |

The operational discipline: `which python` before any `uv pip install` related to quantization. If it points at `.venv`, stop.

---

## Data flow

```
   client (chat completion request, possibly stream=true)
        │
        ▼  POST /v1/chat/completions
   ┌──────────────────────────────────────────┐
   │  proxy (FastAPI, :8000)                  │
   │  ├── normalize thinking params           │
   │  ├── inject sampling defaults            │
   │  └── forward via shared httpx client     │
   └──────────────────────────────────────────┘
        │
        ▼  POST 127.0.0.1:8001/v1/chat/completions
   ┌──────────────────────────────────────────┐
   │  vLLM (:8001)                            │
   │  ├── MTP draft proposes k=5 tokens       │
   │  ├── target verifies in parallel         │
   │  ├── FP8 KV cache (~495K-token pool)     │
   │  └── stream tokens back over SSE         │
   └──────────────────────────────────────────┘
        │
        ▼  SSE chunks (proxy accumulates content + reasoning)
   ┌──────────────────────────────────────────┐
   │  proxy assembles log row                 │
   │  schedules background.add_task(...)      │
   │  returns SSE / JSON to client            │
   └──────────────────────────────────────────┘
        │
        ▼  (background, off the request path)
   logs/requests-YYYY-MM-DD.jsonl  (append-only)
```

---

## Observability

Each request appends one JSONL row.

Sample row (formatted for readability — actual rows are one line):

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

Example queries against a day's log:

```bash
# Requests per hour
jq -r '.ts[:13]' logs/requests-2026-05-09.jsonl | sort | uniq -c

# p95 end-to-end latency (ms)
jq -s 'map(.latency_ms) | sort | .[(length * 0.95 | floor)]' \
  logs/requests-2026-05-09.jsonl

# Top users by output tokens
jq -r '"\(.ip)\t\(.tokens_out)"' logs/requests-2026-05-09.jsonl \
  | awk '{a[$1]+=$2} END {for (k in a) printf "%s\t%d\n", k, a[k]}' \
  | sort -k2 -nr | head
```
