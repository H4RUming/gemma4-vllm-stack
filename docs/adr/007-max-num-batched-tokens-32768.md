# ADR-007: `--max-num-batched-tokens 32768` (prefill chunk size)

## Status

Accepted, 2026-05-23.

Trigger for this change is documented in [docs/investigations/2026-05-23-prefill-investigation.md](../investigations/2026-05-23-prefill-investigation.md) — the workload-shift incident that surfaced the prefill bottleneck this ADR responds to.

## Context

vLLM uses `--max-num-batched-tokens` to bound the per-step token budget shared between prefill and decode under chunked prefill. With no explicit value, vLLM picks a default sized to keep large prefills monolithic, which packs full prompts into fewer steps at the cost of more activation memory per step. At high `--gpu-memory-utilization` (we run at 0.90 on a 96 GB Blackwell), the activation budget directly competes with the KV pool — every GiB of headroom reserved for a larger chunk is a GiB the KV pool cannot use.

Observed before this change:

```
num_gpu_blocks    : 36,057
block_size        : 16
KV pool           : 576,912 tokens
```

Observed after setting `--max-num-batched-tokens 32768`:

```
num_gpu_blocks    : 35,447
block_size        : 16
KV pool           : 495,400 tokens
Available KV mem  : 43.27 GiB
```

KV pool shrank by ~14% (~81,500 tokens). In exchange, the per-step activation budget is bounded, which gives more predictable prefill latency under bursty traffic (long-prompt requests no longer monopolize a step) and leaves more headroom for the speculative-decoding draft model + CUDA Graph captures.

## Decision

Pass `--max-num-batched-tokens 32768` explicitly in `run_vllm.sh`. Document KV pool as a measured value rather than a fixed promise.

## Consequences

- **KV pool**: 495,400 tokens (down from 576,912 pre-change, far from the 853K previously documented in earlier drafts of this stack's README — that figure did not correspond to any configuration this stack has actually run; corrected as part of this ADR).
- **Guaranteed concurrency at 256K context (`--max-model-len 262144`)**: `495,400 / 262,144 ≈ 1.89x`. Smaller than the 3.26x the README previously claimed; the earlier figure was tied to the (incorrect) 853K pool number.
- **Prefill behavior**: long prompts are sliced into 32K-token chunks; per-step latency becomes a function of chunk size rather than prompt size.
- **Memory headroom**: the activation budget shaved from a "fit any prompt in one step" assumption goes toward CUDA Graphs and the MTP draft model.

## Measurement reproduction

```bash
curl -s http://127.0.0.1:8001/metrics \
  | grep -E 'num_gpu_blocks|kv_cache_memory_bytes'
```

Multiply `num_gpu_blocks` by `block_size` (default 16 for Gemma 4) to get pool tokens. Divide by `--max-model-len` to get guaranteed concurrency.

## Relationship to other ADRs

- [ADR-001](001-fp8-kv-over-turboquant.md) — FP8 KV cache decision. That ADR's "853K-token pool, 3.26x concurrency" figures predate this change and have been corrected to point here.
- [ADR-003](003-mtp-num-spec-tokens-5.md) — MTP draft selection. The headroom freed by the smaller chunk size is part of what makes `num_speculative_tokens=5` viable on a 96 GB card.

## Revisit conditions

- `--max-model-len` changes (the concurrency derivation moves).
- `--gpu-memory-utilization` changes (the activation/KV split moves).
- vLLM ships a chunked-prefill heuristic that picks a chunk size adaptive to live KV pressure, in which case an explicit value becomes unnecessary.
- A workload mix shifts dominantly to short prompts (<2K tokens), where chunking cost is pure overhead.

## References

- vLLM chunked prefill: https://docs.vllm.ai/en/latest/usage/engine_args.html#cmdoption-max-num-batched-tokens
- vLLM Prometheus metrics endpoint: `/metrics` on the vLLM HTTP port.
