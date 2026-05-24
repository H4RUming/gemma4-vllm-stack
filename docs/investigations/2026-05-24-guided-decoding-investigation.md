# vLLM Gemma 4 chat completions guided_decoding → response_format migration

**Date:** 2026-05-24
**Service:** `vllm-gemma4.service` (BCCard/gemma-4-31B-it-FP8-Dynamic)
**Host:** hai-server (RTX PRO 6000 Blackwell 96GB, SM12.0, CUDA 12.9)
**vLLM version:** 0.20.2rc1.dev128+g0b9997135-4d7a6c8f

## Trigger

A new downstream client (Japanese-learning companion app, in design) needs token-level output constraint to guarantee that AI replies stay within the learner's known-kana set. Plan was to use vLLM's `guided_choice` / `guided_regex` extension fields on `/v1/chat/completions`.

Initial probe returned 200 OK with natural model output regardless of the constraint passed, with no error and no log warning. Investigation needed to determine whether the issue was on the client, proxy, or engine side, and to find a working path for structured-output constraint on the existing serving stack.

## Initial state

- Endpoint: `http://hai-server:8000/v1/chat/completions` (proxy) → `127.0.0.1:8001` (vLLM)
- Proxy: `H4RUming/gemma4-vllm-proxy` (async FastAPI logging proxy, normalizes thinking-mode encodings, forwards rest verbatim)
- Served model name: `gemma-4-31b-it`
- Engine: `--reasoning-parser gemma4`, `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, `--async-scheduling`, `--speculative-config` with draft model `google/gemma-4-31B-it-assistant` and `num_speculative_tokens=5`

## Probe design

Test prompt was chosen to be unambiguous about whether the constraint was applied: a question with a single correct natural answer (`"1 더하기 1은?"`), and a `guided_choice` set to options the model would never pick on its own:

```bash
curl -sS -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role": "user", "content": "1 더하기 1은?"}],
    "guided_choice": ["わからない", "むずかしい"],
    "max_tokens": 500,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected if constraint is applied: `content` is exactly `"わからない"` or `"むずかしい"`.
Observed: `content` is `"1 더하기 1은 **2**입니다."`. The constraint is being silently ignored.

No 4xx, no warning. Identical behavior whether the request goes through the proxy on :8000 or hits vLLM directly on :8001.

## Hypothesis chain

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | `max_tokens=20` was cut off during reasoning, content never reached | Raise `max_tokens` to 500, disable thinking via `chat_template_kwargs.enable_thinking=false` | Natural answer in `content`, `reasoning=null`. Ruled out. |
| 2 | Proxy strips unknown top-level fields | Hit `localhost:8001` (vLLM direct) with same payload | Identical response. Proxy innocent. |
| 3 | Speculative decoding bypasses `LogitsProcessor` during draft+verify | Restart vLLM without `--speculative-config`, retest. Process confirmed restarted (fingerprint hash `5106ce54` → `4d7a6c8f`, `ps -ef` confirms flag absent) | Same natural answer. Ruled out. |
| 4 | `--enable-auto-tool-choice` sampling path overrides user constraint | Add `"tool_choice": "none"` to request | Same natural answer. Ruled out. |
| 5 | Default backend not selected at engine init | Add explicit `"guided_decoding_backend": "outlines"` to request | Same natural answer. Ruled out. |

## Smoking gun

A deliberately malformed `guided_regex` was sent to determine whether the engine was parsing the parameter at all. A working pipeline would reject the pattern at compile time and return 4xx with a regex compilation error.

```bash
curl -sS -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role": "user", "content": "test"}],
    "guided_regex": "[[[INVALID(regex",
    "max_tokens": 10
  }'
```

Response:

```
HTTP 200 OK
content: "Test received! I am working correctly. How can"
```

A malformed regex produced no error, and the model generated a normal greeting. This is conclusive: **vLLM is not seeing the `guided_*` fields at all on `/v1/chat/completions` in this version**. The fields are being dropped before reaching any parser or grammar compilation path. All five hypotheses above were probing the wrong layer.

## Root cause

vLLM 0.20.x has consolidated chat-completions structured-output handling on the standard OpenAI `response_format` field with JSON schema. The `guided_choice` / `guided_regex` / `guided_grammar` / `guided_json` extension fields are no longer wired through the chat completions request handler. The request body passes JSON schema validation at the HTTP layer (no 4xx), but the guided_* fields are not present in the parsed `ChatCompletionRequest` model and never reach the engine.

`/v1/completions` (non-chat) still appears to accept guided_*, though for our test prompt it returned degenerate output (`"額額額額…"`) because raw text completions on an instruction-tuned model without the chat template cause the model to drift. That endpoint was not investigated further — the chat endpoint is the integration target.

## Resolution

Test of standard `response_format` with json_schema on `/v1/chat/completions`:

```bash
curl -sS -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role": "user", "content": "1 더하기 1은? わからない 또는 むずかしい 중에 답해."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "answer",
        "schema": {
          "type": "object",
          "properties": {
            "answer": {"type": "string", "enum": ["わからない", "むずかしい"]}
          },
          "required": ["answer"]
        }
      }
    },
    "max_tokens": 80,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Response:

```
HTTP 200 OK
content: "{ \"answer\": \"むずかしい\" }"
```

Exact enum enforcement confirmed. JSON parses cleanly to one of the schema's allowed values. The constraint is applied at the token level by the xgrammar backend compiling the JSON schema into an FSM.

## Implementation pattern for downstream clients

Vocabulary-restricted output (e.g., "only kana the learner has studied") is expressed as a `pattern` field on a string property:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "companion_reply",
      "schema": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string",
            "pattern": "^[<allowed-character-class>、。！？\\s]+$"
          }
        },
        "required": ["text"],
        "additionalProperties": false
      }
    }
  }
}
```

The `<allowed-character-class>` is built per request from the application's state (e.g., the learner's known-kana set). The wrapping JSON object also enables future extension fields (mood, gesture, tool hints) without changing the constraint mechanism. Client parses `json.loads(content)["text"]` to extract the constrained text.

This is the recommended path for any structured-output requirement on this serving stack going forward. `guided_*` fields should not be used on `/v1/chat/completions`.

## Remaining work

1. **Benchmark `response_format` overhead.** Measure TTFT and TPS for matched workloads with and without `response_format` on representative kana-restricted patterns. Expected: small overhead from FSM construction (one-time per unique schema) and per-token mask application.

2. **Vocab-leakage benchmark suite.** For the Japanese companion app: run a fixed conversation dataset with (a) prompt-only constraint baseline, (b) `response_format` with pattern-restricted schema, and record leak rate, latency, fluency proxies. Provides the trade-off data point for the app's portfolio narrative.

3. **Confirm `/v1/completions` guided_choice status.** Lower priority — chat endpoint is the actual integration target — but worth a clean test (with chat template applied client-side) for completeness if the API surface picture ever needs documenting fully.

4. **Track upstream consolidation.** Monitor whether `chat_template_kwargs` or other vLLM extension fields go through similar migrations in future releases. Add to the same upstream-tracking list as the Gemma 4 attention-backend lock (see `2026-05-23-prefill-investigation.md`).

## References

- OpenAI API reference, `response_format` field specification (`json_schema` type)
- vLLM structured outputs documentation (xgrammar backend)
- `H4RUming/gemma4-vllm-proxy` — proxy implementation, confirmed pass-through behavior in this investigation (no fix needed proxy-side)
- `2026-05-23-prefill-investigation.md` — adjacent investigation; configuration snapshot below matches that document's post-change state

## Configuration after this investigation

No serving-side configuration change resulted. The fix is entirely client-side: emit `response_format` with json_schema on `/v1/chat/completions` instead of `guided_*` extension fields.

Current vLLM `ExecStart` remains unchanged from the 2026-05-23 prefill investigation:

```
vllm serve BCCard/gemma-4-31B-it-FP8-Dynamic \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --served-model-name gemma-4-31b-it \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --async-scheduling \
  --speculative-config '{"model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}'
```
