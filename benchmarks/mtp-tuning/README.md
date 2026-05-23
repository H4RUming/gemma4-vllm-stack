# MTP tuning — `num_speculative_tokens` sweep

Supporting measurements for [ADR-003](../../docs/adr/003-mtp-num-spec-tokens-5.md).

## Configurations tested

| Config | num_speculative_tokens | Draft head                              |
| ------ | ---------------------- | --------------------------------------- |
| A      | disabled               | —                                       |
| B      | 4 (model-card rec)     | `google/gemma-4-31B-it-assistant`       |
| C      | 5 (chosen)             | `google/gemma-4-31B-it-assistant`       |

## Workload

Same as `benchmarks/README.md`:

- Deterministic Korean prompt, fixed seed
- `max_tokens=800`, `temperature=1.0`
- 5 sequential runs per configuration
- Raw `curl` with `%{time_total}`

## Results

### Config A — baseline (MTP disabled)

```
Run 1: 19.040s → 42.02 tok/s
Run 2: 19.005s → 42.09 tok/s
Run 3: 19.020s → 42.06 tok/s
Run 4: 18.995s → 42.12 tok/s
Run 5: 19.010s → 42.08 tok/s
```

mean: **42.1 tok/s**, σ 0.03.

### Config B — MTP=4 (model-card recommended)

Not formally captured in 5-run form. Observed live across several requests at **95–98 tok/s** — consistently ~5% below Config C. The model-card recommendation appears tuned for a different hardware / draft profile.

### Config C — MTP=5 (chosen)

```
Run 1: 7.732s → 103.5 tok/s
Run 2: 7.850s → 101.9 tok/s
Run 3: 7.611s → 105.1 tok/s
Run 4: 7.834s → 102.1 tok/s
Run 5: 7.778s → 102.9 tok/s
```

mean: **103.2 tok/s**, σ ~3.

## Speedup

- Config C / Config A: **103.2 / 42.1 = 2.45x**
- Acceptance rate (vLLM-reported): **78%**

## Discussion

- vLLM emits a generic warning at `num_speculative_tokens > 1`. The warning does not account for this hardware (single Blackwell, FP8 KV) or this draft head — it's a default-conservative nudge.
- Config C (5) beats Config B (4) by ~5%, consistently. The 5% improvement is free on this stack.
- Going higher than 5 was not tested in this sweep; ADR-003 leaves that as a revisit condition if a future draft head changes the accept-rate profile.

## Caveats

- Single workload (Korean reasoning-style prompt, 800-token decode).
- Single prompt within that workload (deterministic seed).
- Single hardware configuration (RTX PRO 6000 Blackwell, FP8 KV).
- A different language, prompt style, or output length could shift the acceptance-rate vs `num_speculative_tokens` curve. The 5 chosen here is empirically best for the lab's dominant workload, not necessarily universally.
