# ADR-001: FP8 KV cache, defer TurboQuant indefinitely

## Status

Accepted, 2026-05-08.

## Context

At 256K context (`--max-model-len 262144`), the KV cache dominates GPU memory. Two compression options were considered to expand the KV pool and therefore concurrency:

1. **Standard FP8 KV** (`--kv-cache-dtype fp8`, e4m3) — supported broadly across vLLM backends; ~2x KV memory savings vs BF16.
2. **TurboQuant** — vLLM's newer 4-bit KV scheme, ~4x savings on supported backends.

The target ceiling is the guaranteed concurrency vLLM reports at the configured `--max-model-len`. With BF16 KV the pool is small enough that 256K requests block each other immediately; with FP8 it opens to ~1.89x at the current chunk-size setting (see [ADR-007](007-max-num-batched-tokens-32768.md)), which is comfortable headroom for a ~10-user lab.

## Investigation

### TurboQuant blockers on Gemma 4

Two independent issues block TurboQuant for this model, observed on vLLM 0.20.1rc1.dev119 (2026-04-29):

```
ValueError: Selected backend AttentionBackendEnum.TURBOQUANT is not valid for this configuration.
Reason: ['kv_cache_dtype not supported', 'partial multimodal token full attention not supported']
```

Root causes:

1. **Heterogeneous head dimensions.** Gemma 4 alternates 256-dim local attention with 512-dim global attention. The backend selector falls through to `TRITON_ATTN`, which does not implement `kv_cache_dtype` for TurboQuant.
2. **Multimodal full-attention propagation.** vLLM PR #40534 added `use_bidirectional_attention='vision'`, which propagates `use_mm_prefix=True` onto full-attention layers. TurboQuant's preconditions reject the resulting "partial multimodal token full attention" configuration.

Both are upstream-architectural issues, not flags we can tune around from the user side.

### Standard FP8 KV results

With `--kv-cache-dtype fp8`, `--max-model-len 262144`, and `--max-num-batched-tokens 32768` (see [ADR-007](007-max-num-batched-tokens-32768.md)):

- KV cache size: **495,400 tokens** (block_size=16, num_gpu_blocks=35,447, Available KV mem 43.27 GiB; measured 2026-05-23 on vllm-gemma4.service)
- Guaranteed concurrency at 256K context: **~1.89x** (`495,400 / 262,144`)
- Decode ITL improvement vs BF16 KV: **~32%** (consistent with vLLM's published Gemma 4 numbers)
- No measurable accuracy regression on GSM8K Platinum (target stays at 0.977; the model-card-reported figure also holds)

The pool size is sensitive to `--max-num-batched-tokens`, `--max-model-len`, and `--gpu-memory-utilization`. Reproduce on the running service:

```bash
curl -s http://127.0.0.1:8001/metrics \
  | grep -E 'num_gpu_blocks|kv_cache_memory_bytes'
```

Earlier drafts of the stack documentation cited an 853K-token pool / 3.26x concurrency figure; that number did not correspond to any configuration actually run on this hardware and has been corrected here.

### Concurrency utility analysis

The lab averages 5–10% of the pool over a day. The marginal value of going from ~1.89x to a hypothetical ~10x guaranteed concurrency is low: the lab is not bottlenecked there, and going further means TurboQuant patches + retest cycles that we can't afford to absorb on a working production stack.

If the lab grows past ~20 concurrent users, or if a workload pattern emerges that pins the pool above 50%, the calculus changes. Until then the value is hypothetical.

## Decision

Use `--kv-cache-dtype fp8`. Do not attempt to enable TurboQuant on Gemma 4.

## Consequences

- KV pool sized to 495,400 tokens; ~1.89x guaranteed concurrency at 256K context (see [ADR-007](007-max-num-batched-tokens-32768.md) for why the pool size is what it is).
- ~32% decode latency reduction inherited essentially for free.
- No GSM8K Platinum regression observed.
- Lose the 4-bit KV option until vLLM resolves the TurboQuant + TRITON_ATTN gap *and* the multimodal full-attention propagation.

## Revisit conditions

- vLLM's TurboQuant supports `TRITON_ATTN` backend (or Gemma 4's hetero-head backend gains TurboQuant support).
- vLLM PR #40534's `use_mm_prefix=True` propagation is changed so full-attention layers are not flagged as multimodal-partial.
- The lab's average KV pool utilization climbs above 50% sustained.

## References

- vLLM PR #38479 — FP8 KV cache support and the Gemma 4 recipe.
- vLLM PR #40534 — `use_bidirectional_attention='vision'` and `use_mm_prefix` propagation.
- Gemma 4 vLLM recipe: https://docs.vllm.ai/projects/recipes/en/latest/google/gemma-4.html
