# ADR-004: Migrate from NVFP4 to BCCard FP8-Dynamic

## Status

Accepted, 2026-05-12.

## Context

The initial production deployment chose `nvidia/Gemma-4-31B-IT-NVFP4` over the FP8-Dynamic checkpoint because of a then-open garbage-output bug in llmcompressor-quantized FP8 Gemma 4 models — affected outputs degenerated to repeated tokens like `"a a a a a"`. See vLLM Issue #39407.

By early May 2026 the bug had a candidate fix landed in vLLM nightly. We needed to decide whether the migration was worth the operational cost (re-download, re-test, served-model-name continuity).

## Investigation

### Garbage-output check

Deterministic Korean prompt: "한국어로 피보나치 수열을 10개 출력해줘. 각 숫자에 짧은 설명을 한국어로 한 줄씩 붙여."

- **`BCCard/gemma-4-31B-it-FP8-Dynamic`** on vLLM nightly `0.20.2rc1.dev128`: clean output, ten correct Fibonacci numbers, each with a one-line Korean explanation. No `"a a a"` pattern across 5 repeats with different seeds.
- The bug is patched for **FP8-Dynamic** specifically. FP8_BLOCK (a sibling quantization scheme) was **not retested** in this round.

### Accuracy

GSM8K Platinum, 0-shot, deterministic decoding, official reference scoring:

| Checkpoint                                | GSM8K Platinum |
| ----------------------------------------- | -------------- |
| BF16 baseline                             | 0.976          |
| **BCCard/gemma-4-31B-it-FP8-Dynamic**     | **0.977**      |
| nvidia/Gemma-4-31B-IT-NVFP4 (legacy)      | ~0.94          |

FP8-Dynamic is within noise of BF16. NVFP4 sat ~3.5 points lower — a measurable, consistent gap.

### Speed

Same workload as ADR-003 (800-token decode, deterministic Korean prompt, single GPU):

| Checkpoint              | Baseline | MTP=5  |
| ----------------------- | -------- | ------ |
| NVFP4 (legacy)          | ~40      | n/a    |
| FP8-Dynamic (chosen)    | 42.1     | 103.2  |

NVFP4 MTP was never enabled in production because the garbage-output bug took precedence.

### Memory

- Weights: NVFP4 **17 GB** → FP8-Dynamic **33 GB** (+16 GB)
- KV pool: NVFP4 52K → FP8-Dynamic ~50K-ish raw tokens at BF16, but with `--kv-cache-dtype fp8` we still hit the configured **853K-token pool** because pool size is bounded by `gpu-memory-utilization` and KV dtype rather than weight footprint.

Net: +16 GB of VRAM goes to weights; the operational KV envelope is unchanged.

## Decision

Migrate to `BCCard/gemma-4-31B-it-FP8-Dynamic`. Keep `--served-model-name gemma-4-31b-it` so no client changes.

## Consequences

- Accuracy gain: +0.037 GSM8K Platinum (NVFP4 → FP8-Dynamic), within noise of BF16.
- Speed gain: 42.1 vs ~40 baseline; unlocked the MTP=5 → 103.2 tok/s path (NVFP4 was never MTP-tested in production).
- Memory cost: +16 GB weights (33 GB total), well within 96 GB VRAM.
- **Zero client impact**: `served-model-name` unchanged.

## Notes on bug history

The vLLM #39407 garbage-output bug applies to llmcompressor-quantized Gemma 4 FP8 models historically. The fix in nightly ≥ `0.20.2rc1.dev128` was verified empirically against the **FP8-Dynamic** checkpoint produced by BCCard. **FP8_BLOCK** is a separate scheme and was not retested here — do not assume the patch covers it without re-running the garbage-output check.

## Revisit conditions

- A higher-accuracy quantization for Gemma 4 31B ships (FP4 variant that matches BF16, INT4-AWQ, etc.) with stable vLLM support.
- The `BCCard/gemma-4-31B-it-FP8-Dynamic` checkpoint regresses on any subsequent vLLM upgrade (re-run the deterministic Korean Fibonacci check as the canary).
- A new draft head is published that requires a specific target-checkpoint pairing.

## References

- vLLM Issue #39407 — llmcompressor FP8 garbage-output bug history.
- `BCCard/gemma-4-31B-it-FP8-Dynamic` — https://huggingface.co/BCCard/gemma-4-31B-it-FP8-Dynamic
- `nvidia/Gemma-4-31B-IT-NVFP4` (legacy) — https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
- GSM8K Platinum: https://github.com/openai/grade-school-math
