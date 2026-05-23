# ADR-006: Three venvs — never install `llmcompressor` in `.venv`

## Status

Accepted, 2026-04-22.

## Context

`llmcompressor` is the canonical tool for producing FP8 / W8A8 / other quantized checkpoints of large models. It is also unusually aggressive about pinning `torch` and `transformers` to versions that are typically *behind* current vLLM nightly.

Installing `llmcompressor` into the serving venv (`.venv`) silently downgrades `torch` and `transformers` to versions vLLM cannot import. The serving unit then refuses to start, and recovery requires:

1. Identify which packages were downgraded (not always obvious from `uv pip list` diffs).
2. Re-resolve `vllm` + `torch` + `flashinfer` against the nightly cu129 index.
3. Re-test the deterministic Korean prompt to confirm no garbage-output regression.
4. Re-run a reboot test before declaring the env clean.

First time we hit this it cost ~90 minutes. Twice would be unacceptable.

## Decision

Operate three separate venvs in `~/Desktop/LLM-OPS`, each with a single role.

| venv          | Role                              | Notable contents                                                              | Never installed                  |
| ------------- | --------------------------------- | ----------------------------------------------------------------------------- | -------------------------------- |
| `.venv`       | vLLM serving (`vllm-gemma4.service`) | `vllm` (nightly cu129), `torch 2.11.0+cu129`, `flashinfer`, `transformers 5.8.0` | `llmcompressor`, any quant tool |
| `.venv-proxy` | FastAPI logging proxy (`vllm-proxy.service`) | `fastapi`, `httpx`, `uvicorn[standard]`, `transformers`, `tokenizers`, `anyio` | `torch`, `vllm`, `llmcompressor` |
| `.venv-quant` | Quantization / model authoring (ad-hoc, no service) | `llmcompressor`, older `torch`/`transformers` that `llmcompressor` requires | `vllm`                            |

`.venv-quant` is intentionally not wired to any systemd unit — it's a workspace for one-off quantization runs, not a serving environment.

## Operational note

**Always check `which python` before any `uv pip install` related to quantization.**

```bash
$ which python
/home/haru/Desktop/LLM-OPS/.venv-quant/bin/python   # OK
```

If `which python` points at `.venv` (serving) or `.venv-proxy`, stop. The two ways this slips through:

- `source .venv/bin/activate` left over from a previous session.
- IDE / agent shells that auto-activate the project venv (`.venv`) on directory entry.

Either way, the prompt prefix usually shows it. If you're using `uv pip install --python` explicitly with the full path to the right venv's `python`, you cannot accidentally cross-contaminate.

## Revisit conditions

- `llmcompressor` stops pinning torch/transformers aggressively (e.g., declares loose lower bounds compatible with vLLM nightly). Then `.venv-quant` becomes redundant.
- The lab stops producing its own quantizations (i.e., consumes upstream FP8 checkpoints exclusively). Then `.venv-quant` can be removed entirely.

## References

- `llmcompressor`: https://github.com/vllm-project/llm-compressor
- vLLM dependency surface: https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt
