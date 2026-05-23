# vLLM Gemma 4 prefill performance investigation

**Date:** 2026-05-23
**Service:** `vllm-gemma4.service` (BCCard/gemma-4-31B-it-FP8-Dynamic)
**Host:** hai-server (RTX PRO 6000 Blackwell 96GB, SM12.0, CUDA 12.9)

## Trigger

User-visible text generation latency on the gemma-4-31b-it endpoint started degrading around 2026-05-20. Investigation also covered prefix cache hit rate, which had been suspected of being lower than expected for the long-session OpenClaw workload.

## Latency baseline

Per-day stats from `logs/requests-*.jsonl` (`/v1/chat/completions`, status 200):

| Window | p50 latency | p95 latency | TPS (p50) | p50 input tokens |
|---|---|---|---|---|
| 2026-05-15 – 2026-05-19 | ~6 s | ~37 s | 90 tok/s | 3,256 |
| 2026-05-20 – 2026-05-23 | 13 s | 106 s | 25 tok/s | 37,511 |

Median input grew 11× across the boundary. Output length did not grow; if anything, p50 reasoning tokens fell (321 → 255). The slowdown is dominated by prefill cost, not generation.

## Workload composition change on 2026-05-20

Three classifications based on the system-prompt prefix:

- `openclaw` — "You are a personal assistant running inside OpenClaw…"
- `summarizer` — "You are a context summarization assistant…"
- `other` — everything else

Per-day request counts:

```
date         openclaw  oc_distinct_sys  summarizer  other
2026-05-19         85               47          43    182
2026-05-20         57               29          29      0
2026-05-21         61               31          25      0
2026-05-22         93               53          46      0
2026-05-23         44               23          19      0
```

The `other` category dropped to zero on 2026-05-20 — the only remaining client is `100.106.44.41`. Workload mix is now OpenClaw (heavy thinking, ~38k input) plus periodic compaction calls from the summarizer.

GPU-time attribution over 5/20–5/23:

| Class | Requests | Share of input tokens | Share of wall-time |
|---|---|---|---|
| openclaw | 255 | 91% | 27% |
| summarizer | 119 | 9% | 73% |

OpenClaw drives prefill volume. The summarizer drives total occupancy because each call decodes 5k–13k output tokens.

## Prefix cache investigation

Cumulative vLLM Prometheus counters at investigation time:

```
vllm:prefix_cache_queries_total  4,242,324
vllm:prefix_cache_hits_total     1,333,728
                                 = 31.4% hit rate
vllm:external_prefix_cache_*     0   (no KV connector)
vllm:mm_cache_hits_total         0   (no MM)
```

For a workload dominated by an OpenClaw client that maintains long-running sessions, hit rate should be 70%+ steady state. 31% is anomalous.

Root cause: the OpenClaw system prompt is not stable across calls.

```
distinct OpenClaw system prompts (2026-05-22 + 2026-05-23): 75
system prompt length range: 43,543 – 45,064 chars
first divergence between two variants: char 227 of 44,060
```

Divergence point is inside the tool-list block. Variants differ in tool ordering, line wording, or which tools are listed:

```
…exactly as listed.
- read: Read file contents
- write: Create or overwrite files
                                ↑ divergence here
variant A: \n- edit: Make precise edits to files\n- exec: …
variant B: \nTOOLS.md does not control tool availability; …
```

The downstream effect: of a ~44k-char system prompt, only the first ~227 chars are guaranteed cache-hits across calls. Everything after that hashes differently per request and re-enters prefill.

Within-session prefix overlap on consecutive OpenClaw requests (paired ≤60 s apart, n=78):
- p10 = 0.2%
- p50 = 34.3%
- p90 = 100%

Half of consecutive turns in the same session share less than 35% prefix with the previous turn. Expected steady-state for a stable serializer is ≥95%.

The breakage started 2026-05-11:

```
date         openclaw  oc_distinct_sys
2026-05-08         73                2
2026-05-09         32                2
2026-05-10         49                2
2026-05-11         74               51   ← regression
2026-05-12         94               79
…
```

Through 2026-05-10, two distinct system-prompt hashes covered all OpenClaw calls. From 2026-05-11 onward, roughly 60–80% of daily calls produce a unique hash. Whatever changed in the OpenClaw client on that date introduced non-determinism into the tool-list serialization.

## Infrastructure analysis

vLLM is forced onto the Triton attention backend by Gemma 4's heterogeneous head dimensions:

```
vllm/config:101  Gemma4 model has heterogeneous head dimensions
                 (head_dim=256, global_head_dim=512).
                 Forcing TRITON_ATTN backend to prevent
                 mixed-backend numerical divergence.
```

The lock condition (verified against vllm-project/vllm issue #42068): `head_dim != global_head_dim AND max(head_dim, global_head_dim) > 256`. Gemma 4 satisfies both. The lock is defensive (prevents sliding-window vs. full-attention layers from using different backends and accumulating numerical divergence), not driven by fundamental kernel incompatibility. There is no escape hatch for the target model — `SpeculativeConfig.attention_backend` only overrides the drafter.

Measured prefill rate over 136 post-restart requests at the time of investigation:

```
vllm:request_prefill_time_seconds_sum        671.11
vllm:request_prefill_kv_computed_tokens_sum  2,955,917
                                             ≈ 4,401 tok/s
```

Queue time was effectively zero (`request_queue_time_seconds_sum` ≈ 0.001s over 136 requests) — the scheduler is not the bottleneck. GPU memory utilization steady at 89/96 GB. No backpressure.

Additional contributors observed in the journal:
- `max_num_batched_tokens=8192` (default). A 38k input is split into five prefill chunks.
- `compile_ranges_endpoints=[8192]`. torch.compile/Inductor specializes for `[1, 8192]` and `[8193, max_num_batched_tokens]`.
- Triton kernel JIT warnings (`kernel_unified_attention`, `_compute_slot_mapping_kernel`, `eagle_prepare_*`, `expand_kernel`, `reduce_segments`) firing on the first request after each restart when a new shape combination is hit.
- Speculative decoding (MTP, `num_speculative_tokens=5`, draft = `google/gemma-4-31B-it-assistant`) is enabled. Affects decode acceptance (~95% sustained on stable content) but is irrelevant to prefill cost.

## Change applied

`run_vllm.sh`:

```diff
   --max-model-len 262144 \
+  --max-num-batched-tokens 32768 \
   --kv-cache-dtype fp8 \
```

Rationale: cut a 38k input from 5 prefill chunks to 2, amortizing kernel-launch overhead and giving Inductor a larger compile range to specialize against. Risk: increased activation memory at peak.

Service restarted 2026-05-23 18:17:52 KST. First boot took ~2m 43s because of recompilation for the new chunk size; subsequent restarts will hit the AOT cache.

Post-restart state:

```
GPU KV cache size:      495,400 tokens     (was ~576k)
num_gpu_blocks:         35,447             (was 36,057, −1.7%)
Available KV memory:    43.27 GiB
GPU memory used:        89.3 GB / 96 GB    (unchanged)
compile_ranges_endpoints: [32768]          (auto-extended)
```

No OOM. KV capacity remained well above the 400k-token target for the actual workload.

## Result

Two real requests after the restart, ~40k input each, full thinking-mode output. vLLM metrics:

```
prefill_kv_computed_tokens_sum  209,501
prefill_time_sum                51.21 s
                                ≈ 4,090 tok/s
```

Per-token prefill rate did not improve. Both qualifying requests came in at 22.1 s wall time for ~40k input + ~500 output, matching the pre-change profile.

Interpretation: on this hardware/model combination, Triton attention is compute-bound on per-token work. Kernel-launch overhead on Blackwell is in the µs range, so amortizing it across larger chunks does not materially reduce wall time. The change is benign — KV capacity is still ample, no OOM, no regression — but it does not recover the latency gap.

The change is being kept in place. If a future vLLM release lifts the Gemma 4 TRITON_ATTN lock and FlashAttention/FlashInfer becomes available, the larger chunk size should start delivering visible gains.

## Remaining work

Infrastructure side is at its floor for the current vLLM + Gemma 4 + Blackwell combination. Two genuine levers remain:

1. **OpenClaw client — deterministic system-prompt serialization.** Sort the tool list, freeze the wording, and stop emitting dynamic content above char 227. Expected effect: per-call cache hit on the static 15k-token system prompt and ~15k-token rolling history block, dropping the actual prefill working set from ~28k tokens to ~5–10k tokens. At the observed 4,090 tok/s Triton rate, p50 prefill goes from ~7 s to ~1.5 s. This is the single biggest available win and does not require any vLLM change.

2. **Track upstream.** Monitor vllm-project/vllm issues #38887 and #42068 for movement on the Gemma 4 attention-backend lock. A clean unlock would roughly double prefill throughput on this hardware.

Lower-value follow-ups, not pursued:
- Triton kernel warmup script for new shapes after restart (eliminates ~1–2 s spike on first request only).
- Reducing `--max-model-len` from 262144 to ~131072 (frees minor KV memory; only worth it if we confirm we never actually serve contexts above 128k).
- Disabling speculative decoding to measure its actual prefill contribution. Hypothesis is "none," but unverified.

## Configuration after this change

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

## References

- vllm-project/vllm#42068 — Gemma 4 + DFlash incompatible; TRITON_ATTN forced
- vllm-project/vllm#38887 — Gemma 4 E4B extremely slow on forced TRITON_ATTN fallback
- vLLM blog, 2026-03-04 — Triton attention backend deep dive
- `vllm/config/compilation.py:554` — `compile_ranges_endpoints` semantics
