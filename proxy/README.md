# LLM-OPS logging proxy — interface specification

> **Code lives in a separate repository.**
> Repository: [gemma4-vllm-proxy](https://github.com/H4RUming/gemma4-vllm-proxy)
>
> This directory keeps only the interface specification — what the proxy does, the request/response contract, and the JSONL log schema — so the stack documentation here stays self-contained. The implementation is versioned independently so its dependencies (`fastapi`, `httpx`, `uvicorn`) can iterate without touching the serving environment.

Async FastAPI proxy that sits in front of vLLM at `0.0.0.0:8000` and forwards to `127.0.0.1:8001`. Runs as `vllm-proxy.service` ([systemd unit](../systemd/vllm-proxy.service), [wrapper](../systemd/run_proxy.sh)). Design rationale is in [ADR-005](../docs/adr/005-logging-proxy-separation.md).

---

## Responsibilities

The proxy MUST:

1. Accept every OpenAI-compatible endpoint transparently via a catch-all route.
2. Normalize thinking-mode parameters across the four client encodings (`reasoning_effort`, `extra_body.thinking`, `thinking_config`, canonical `chat_template_kwargs.enable_thinking`) into the canonical form before forwarding. Default ON.
3. Inject sampling defaults (`temperature=1.0, top_p=0.95, top_k=64`) when the client omits them. Use `setdefault` — never overwrite a client value.
4. Stream the upstream response back transparently while accumulating content + reasoning for the log row.
5. Append one JSONL row per request to `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl` as a FastAPI background task. Log-write failures MUST NOT propagate to the response path.
6. Reject `BLOCKED_PATHS` (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/reset_prefix_cache`) with a 403.
7. Maintain a single shared `httpx.AsyncClient` (keep-alive, 100 max connections) for the lifetime of the process.

The proxy MUST NOT:

- Provide auth, API keys, or rate limiting — access control lives at the network layer.
- Persist anything beyond the daily JSONL log.
- Block or buffer streaming chunks beyond what's needed to accumulate the log row.
- Implement multi-tenant routing or model fan-out.

---

## Thinking-mode precedence (highest wins)

```
chat_template_kwargs.enable_thinking
    > reasoning_effort
        > extra_body.thinking
            > thinking_config
                > default ON
```

The canonical form (`chat_template_kwargs.enable_thinking: bool`) is always written before the request leaves the proxy. The reasoning-effort label returned by the normalizer (`"high"` / `"none"`) is used for the log row, not for routing.

---

## Configuration surface

| Constant                   | Default                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `PROXY_HOST`               | `0.0.0.0`                                                  |
| `PROXY_PORT`               | `8000`                                                     |
| `VLLM_URL`                 | `http://127.0.0.1:8001`                                    |
| `LOG_DIR`                  | `/home/haru/Desktop/LLM-OPS/logs`                          |
| `MODEL_NAME`               | `BCCard/gemma-4-31B-it-FP8-Dynamic`                        |
| `SAMPLING_DEFAULTS`        | `{temperature: 1.0, top_p: 0.95, top_k: 64}`               |
| `THINKING_DEFAULT`         | `True`                                                     |
| `BLOCKED_PATHS`            | `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/reset_prefix_cache` |
| `UPSTREAM_TIMEOUT_SECONDS` | `600`                                                      |

All are overridable via environment variables. The systemd unit and `.env.example` document the intended overrides per host.

---

## JSONL log schema

One row per request, append-only, line-delimited JSON. File path rotates by UTC date.

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
  "request": { "messages": [ ], "stream": true },
  "response": "...",
  "reasoning_text": "..."
}
```

`jq` query examples in [docs/architecture.md](../docs/architecture.md#observability).

---

## Deployment fit

The serving stack expects the proxy code to be cloned to `${LLM_OPS_DIR}/proxy/` and a `.venv-proxy` to be created next to it. The wrapper script ([systemd/run_proxy.sh](../systemd/run_proxy.sh)) does:

```bash
source "${LLM_OPS_DIR}/.venv-proxy/bin/activate"
cd "${LLM_OPS_DIR}/proxy"
exec python app.py
```

So as long as the separate proxy repository is cloned into that path, no changes are needed on the serving side. See [docs/deployment.md §8](../docs/deployment.md) for the install sequence.

---

## Dependency profile

Intentionally tiny. No torch, no vLLM, no CUDA. Boots in seconds; ~400 MB resident.

```
fastapi>=0.115
httpx>=0.27
uvicorn[standard]>=0.30
transformers>=5.5
tokenizers>=0.20
anyio>=4
```

`transformers` and `tokenizers` are only used for token counting on the log row — neither imports torch.
