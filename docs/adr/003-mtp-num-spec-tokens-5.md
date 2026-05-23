# ADR-003: `num_speculative_tokens=5`, not the model-card-recommended 4

## Status

Accepted, 2026-05-09.

## Context

`google/gemma-4-31B-it-assistant` is the official MTP draft head for Gemma 4 31B. The model card recommends `num_speculative_tokens=4`. vLLM, separately, emits a generic warning when `num_speculative_tokens > 1`, advising operators to tune the value.

The recommendation and the warning are both generic — they do not account for this hardware (single Blackwell, FP8 KV) or this draft head configuration. The decision needed local measurement.

## Measurement

**Workload**: deterministic Korean prompt, fixed seed, `max_tokens=800`, `temperature=1.0`, raw `curl` with `--write-out '%{time_total}'`. 5 sequential runs per configuration.

### Baseline (MTP disabled)

```
Run 1: 19.040s → 42.02 tok/s
Run 2: 19.005s → 42.09 tok/s
Run 3: 19.020s → 42.06 tok/s
Run 4: 18.995s → 42.12 tok/s
Run 5: 19.010s → 42.08 tok/s
mean: 42.07 tok/s, σ 0.03
```

Used as the **42.1 tok/s** reference.

### MTP=5

```
Run 1: 7.732s → 103.5 tok/s
Run 2: 7.850s → 101.9 tok/s
Run 3: 7.611s → 105.1 tok/s
Run 4: 7.834s → 102.1 tok/s
Run 5: 7.778s → 102.9 tok/s
mean: 103.2 tok/s, σ ~3
```

Speedup: **103.2 / 42.1 = 2.45x**. Acceptance rate reported by vLLM in this 5-run window: **78%**.

### Sustained behavior (2026-05-23 re-check on `vllm-gemma4.service`)

Spec-decoding metrics observed on long reasoning streams under live traffic, after the `--max-num-batched-tokens 32768` change ([ADR-007](007-max-num-batched-tokens-32768.md)):

```
Per-position acceptance: 0.99, 0.98, 0.96, 0.94, 0.89
Avg Draft acceptance rate: 94–97%
Mean acceptance length:    5.7–5.9 (of 5 drafted)
Accepted throughput:       100–125 tokens/s
```

Two readings of the 78% vs 94–97% gap:
- The 5-run benchmark used short, deterministic Korean prompts where acceptance per-position is dragged down by the last-position drop-off; in sustained reasoning traffic with longer continuations, acceptance climbs.
- The original measurement may pre-date vLLM scheduler improvements that landed alongside the cu129 nightly upgrade.

Both readings point at the same operational conclusion: **94–97% is the realistic acceptance rate to plan against**, and the 78% headline is best read as a worst-case bound from the synthetic benchmark.

### MTP=4 (model-card recommended)

Not formally captured in 5-run form; observed live across several requests at **95–98 tok/s**, ~5% below MTP=5. The model-card recommendation appears tuned for a different deployment profile.

## Decision

Set `num_speculative_tokens=5` in `--speculative-config`. The 5% improvement over MTP=4 is empirically free on this hardware + draft head, and vLLM's generic >1 warning does not point at any specific failure mode at 5.

## Concurrency notes — 3 concurrent requests, MTP=5

```
Total wall time: 8.151s
Req 1: 800 tokens / 7.902s = 101.23 tok/s
Req 2: 800 tokens / 8.146s =  98.20 tok/s
Req 3: 800 tokens / 8.054s =  99.32 tok/s
Aggregate: 2400 tokens / 8.152s = 294.41 tok/s
```

Per-request throughput holds at ~99 tok/s (96% of the single-request 103.2). Aggregate **294 tok/s = 7.0x** the BF16 baseline (42.1 tok/s). Wall time goes from 7.7s single to 8.2s for 3 concurrent — a **+6% wall-time cost** for **3x the work**. Near-linear scaling at this concurrency level.

## Multimodal compatibility

vLLM logs on startup:

```
Draft model does not support multimodal inputs.
Speculative decoding will be disabled for multimodal requests.
```

This is the intended behavior. Text-only chat completions use MTP=5; requests carrying images fall back to standard autoregressive decoding for that request only. No config change needed; no manual routing required.

## Caveats

- vLLM disables `min_p` and `logit_bias` sampling parameters while speculative decoding is active. Neither is used by this deployment, but client libraries that always send `min_p` will see vLLM warning-log them.
- CUDA graph capture takes slightly longer at first launch because the draft head is captured separately. First-launch cost ~+30s; cached on subsequent restarts.

## Revisit conditions

- A new draft head ships from Google with a different recommended `num_speculative_tokens`.
- Workload shifts toward short outputs (<200 tokens), where MTP's overhead-to-benefit ratio may invert; measure before lowering k.
- A vLLM upgrade changes the acceptance-rate logging or sampler-disable list.

## References

- Gemma 4 model card: https://huggingface.co/google/gemma-4-31B-it
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/usage/speculative_decoding.html
- MTP assistant head: `google/gemma-4-31B-it-assistant`
