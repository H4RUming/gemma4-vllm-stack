# Benchmarks

Performance evidence for the decisions in `docs/adr/`.

## Headline numbers

| Configuration                          | Throughput      | vs baseline | Notes                  |
| -------------------------------------- | --------------- | ----------- | ---------------------- |
| Baseline (MTP disabled)                | 42.1 tok/s (σ 0.03) | 1.00x   | reference              |
| MTP=5, single request                  | 103.2 tok/s (σ ~3)  | **2.45x** | initial 5-run benchmark, 78% acceptance |
| MTP=5, 3 concurrent (per-request)      | ~99 tok/s       | 2.35x       | 96% of single-request  |
| MTP=5, 3 concurrent (aggregate)        | **294 tok/s**   | **7.0x**    | wall time +6% vs single |
| MTP=5, sustained (live, 2026-05-23)    | 100–125 tok/s   | 2.4–3.0x    | acceptance 94–97%, mean accept length 5.7–5.9 of 5 |

## Methodology

- **Workload**: deterministic Korean prompt (fixed seed), `max_tokens=800`, `temperature=1.0`.
- **Tool**: raw `curl --write-out '%{time_total}'`. `vllm bench serve` had compatibility issues with chat-completions + the random-dataset workload, so raw curl was preferred — every measurement here goes through the exact same path a real client would.
- **Repetitions**: 5 sequential runs per configuration. Reported value is the mean; σ noted when relevant.
- **Variance**: low across the board. Baseline σ is 0.03 tok/s (a few hundredths of a percent); MTP variance is higher (~3 tok/s) because acceptance-rate fluctuations show up in wall-clock.

### Single-request 5-run loop

```bash
PROMPT='한국어로 피보나치 수열을 10개 출력해줘. 각 숫자에 짧은 설명을 한국어로 한 줄씩 붙여.'
for i in 1 2 3 4 5; do
  curl -s -o /dev/null -w "Run $i: %{time_total}s\n" \
    http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$PROMPT" '{
          model: "gemma-4-31b-it",
          messages: [{role:"user", content:$p}],
          max_tokens: 800,
          temperature: 1.0,
          seed: 42
        }')"
done
```

Throughput per run = `800 / time_total`.

### 3-concurrent measurement

```bash
PROMPTS=(
  '한국어로 피보나치 수열을 10개 출력...'
  '한국어로 소수를 10개 출력...'
  '한국어로 완전수를 10개 출력...'
)
mkdir -p /tmp/conc
START=$(date +%s.%N)
for i in 0 1 2; do
  ( curl -s -o /tmp/conc/req$i.json -w "Req $((i+1)): %{time_total}s\n" \
      http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "$(jq -n --arg p "${PROMPTS[$i]}" '{
            model: "gemma-4-31b-it",
            messages: [{role:"user", content:$p}],
            max_tokens: 800,
            temperature: 1.0,
            seed: 42
          }')" ) &
done
wait
END=$(date +%s.%N)
echo "Total wall: $(echo "$END - $START" | bc)s"
```

## Files

- [`mtp-tuning/`](mtp-tuning/README.md) — `num_speculative_tokens` sweep (disabled / 4 / 5). References [ADR-003](../docs/adr/003-mtp-num-spec-tokens-5.md).
- [`concurrency/`](concurrency/README.md) — 3-concurrent scaling measurements.

## What's not measured

- **TTFT (time-to-first-token)** — no streaming-aware harness yet; current measurements are end-to-end.
- **Tail latency at >5 concurrent** — qualitatively measured during operations (no degradation observed), but not captured in a 5-run form.
- **Long context >128K** — workload is 800-token decode with short prompts; we don't have a benchmark prompt that exercises the full 256K window.
- **Thinking-mode throughput** — measurements above used the default ON behavior. Comparing thinking ON vs OFF systematically is open work.
