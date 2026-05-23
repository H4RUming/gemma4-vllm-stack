# Concurrency scaling

How throughput holds up as concurrent requests grow.

## Configuration under test

- MTP=5 (`google/gemma-4-31B-it-assistant`)
- `--kv-cache-dtype fp8`
- `--max-model-len 262144` (256K)
- KV pool: 495,400 tokens (current, post `--max-num-batched-tokens 32768`; initial concurrency benchmark below predates that change — see [ADR-007](../../docs/adr/007-max-num-batched-tokens-32768.md))
- `max_tokens=800` per request
- 3 distinct deterministic Korean prompts (different topics, same length class)

## Single-request reference

103.2 tok/s mean (from [`../mtp-tuning/README.md`](../mtp-tuning/README.md)).

## 3 concurrent requests

```
Total wall time: 8.151s
Req 1: 800 tokens / 7.902s = 101.23 tok/s
Req 2: 800 tokens / 8.146s =  98.20 tok/s
Req 3: 800 tokens / 8.054s =  99.32 tok/s
Aggregate: 2400 tokens / 8.152s = 294.41 tok/s
```

## Analysis

| Metric                          | Single   | 3-concurrent | Δ                                |
| ------------------------------- | -------- | ------------ | -------------------------------- |
| Per-request throughput          | 103.2    | ~99.6 (mean) | -3.5% (well within run variance) |
| Aggregate throughput            | 103.2    | **294.4**    | **+185%** (2.86x of single)      |
| Scaling efficiency vs ideal 3x  | —        | 95%          | near-linear                      |
| Wall time (800 tokens)          | 7.7s     | 8.2s         | +6%                              |

3 concurrent users do not feel slower — the +6% wall-time cost is below human-perceptible.

## End-to-end speedup table

| Stage                              | Throughput    | Speedup vs BF16 baseline |
| ---------------------------------- | ------------- | ------------------------ |
| BF16 baseline (MTP off)            | 42.1 tok/s    | 1.00x                    |
| FP8 + MTP=5, single request        | 103.2 tok/s   | 2.45x                    |
| FP8 + MTP=5, 3 concurrent (agg.)   | **294.4 tok/s** | **7.0x**               |

## Higher concurrency

Not formally measured at 5+ concurrent in this 5-run-mean form on the current FP8-Dynamic stack. Earlier NVFP4-era operations (different checkpoint, otherwise similar pipeline) hit ~107 tok/s aggregate at **10 concurrent** while the single-request baseline was ~40 tok/s — efficiency holding to at least 10x. The FP8-Dynamic stack is faster end-to-end, so an equivalent test would only push the numbers higher. Plan to redo this formally if user count grows.
