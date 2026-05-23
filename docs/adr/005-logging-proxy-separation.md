# ADR-005: Hand-rolled FastAPI proxy in its own process and venv

## Status

Accepted, 2026-04-30.

## Context

vLLM's OpenAI-compatible server logs to stdout but does not provide:

- Per-request JSONL with response bodies, reasoning text, and per-user attribution
- IP / source tracking
- Token-count breakdowns (`tokens_in`, `tokens_out`, `tokens_reasoning`) in a machine-queryable schema
- End-to-end latency in milliseconds
- Sampling-default injection (so clients can omit `temperature`/`top_p`/`top_k` and still get our defaults)
- Thinking-mode parameter normalization across the four client encodings vLLM accepts

These are non-negotiable for the lab: we need to answer "who used how many tokens last week" and "what was the actual prompt + reasoning + response" for support and capacity planning.

## Options considered

### LiteLLM

Mature, multi-provider router. Pulls in a wide dependency tree and a SQL backend by default; the OSS proxy is opinionated about its schema and admin UI. For a ~10-user single-model lab it's heavyweight, and customizing the log shape means fighting its defaults.

### Langfuse

Excellent for traced LLM apps with span structure, but expects each application to instrument itself or sit behind an SDK. As a transparent OpenAI proxy it requires extra glue, and the storage layer (Postgres + ClickHouse) is more infrastructure than the use case warrants.

### Helicone

OpenAI-compatible proxy with hosted + self-host modes. Self-host pulls in Supabase. Schema and dashboard are good but not customizable to the depth we needed (e.g., capturing the `reasoning_content` channel as its own field, not mixed into the response body).

### Hand-rolled FastAPI

~300 lines of Python, owns the log schema, no external storage, no admin UI, easy to audit. Deploys as a single systemd unit alongside vLLM.

### Modify vLLM directly

Patching vLLM's server to add per-request JSONL logging means maintaining a fork across nightly upgrades — a recurring cost for a feature that's better expressed as a separate concern.

## Decision

Hand-rolled FastAPI proxy in a separate process and a separate venv (`.venv-proxy`).

## Why minimal

- ~10-user lab; no rate limiting, no auth, no multi-tenant routing needed.
- Owning the log schema is the point of the project. A third-party schema would either need translation or constrain queries.
- ~300 lines is auditable in one sitting and reviewable in PRs.

## Why a separate venv

- vLLM ships heavy CUDA dependencies. Restarting the proxy to deploy a logging change shouldn't risk loading the wrong torch/flashinfer.
- The proxy boots in seconds (~400 MB RAM, no torch import). vLLM boots in 3–5 minutes on first launch. Decoupling iteration speeds is a clean win.
- The proxy's dependencies (`fastapi`, `httpx`, `uvicorn`, `transformers`, `tokenizers`, `anyio`) have no overlap with vLLM's heavy stack; no risk of accidental version drift.

## Why systemd `Requires=`, not a single supervisor

`vllm-proxy.service` declares `Requires=vllm-gemma4.service`. When the serving unit restarts, systemd cascades the proxy restart automatically — no window where the proxy can forward to a dead upstream. A single supervising process (e.g., one Python script that starts both) would re-implement this dependency logic and re-implement systemd's journal integration. Not worth it for this scale.

## Implementation contract

The proxy MUST:

1. Accept all OpenAI-compatible endpoints transparently via a catch-all route.
2. Normalize the four thinking-mode encodings (`reasoning_effort`, `extra_body.thinking`, `thinking_config`, `chat_template_kwargs.enable_thinking`) into the canonical `chat_template_kwargs.enable_thinking` boolean before forwarding. Default ON.
3. Inject sampling defaults (`temperature=1.0, top_p=0.95, top_k=64`) via `setdefault` — never overwrite a client value.
4. Stream the upstream response back transparently while accumulating content + reasoning for logging.
5. Append one JSONL row per request to `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl` as a background task. Log write failures MUST NOT affect the request path.

The proxy MUST NOT:

- Provide auth, API keys, or rate limiting (the proxy listens on the lab's Tailnet only; access control lives at the network layer).
- Persist anything beyond JSONL logs.
- Block or buffer streaming chunks beyond accumulating them for the log row.
- Implement multi-tenant routing or model fan-out.

## Consequences

- We own the log schema and can extend it without coordination.
- Proxy iteration is decoupled from vLLM upgrades.
- Two services to operate instead of one, mitigated by systemd cascade.
- No off-the-shelf dashboard; reads use `jq` (samples in `docs/architecture.md`).

## Revisit conditions

- User count grows beyond ~50 (start needing rate limits, auth, multi-tenant).
- Log volume exceeds what `jq` over JSONL handles comfortably (move to DuckDB-over-files or ClickHouse).
- A third-party proxy adds the missing capabilities (canonical thinking normalization, our exact log schema) and matures.

## References

- LiteLLM proxy: https://docs.litellm.ai/docs/simple_proxy
- Langfuse: https://langfuse.com
- Helicone: https://helicone.ai
- FastAPI background tasks: https://fastapi.tiangolo.com/tutorial/background-tasks/
