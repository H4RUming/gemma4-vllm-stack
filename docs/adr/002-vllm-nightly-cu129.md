# ADR-002: vLLM nightly cu129 wheels, pinned install command

## Status

Accepted, 2026-05-05.

## Context

The system runs CUDA Toolkit 12.9 because FlashInfer's SM120 (Blackwell) JIT compilation requires it. The driver (595.58.03) advertises CUDA 13.2 compatibility, so in principle we could install CUDA 13 alongside — but doing so creates two cohabiting toolkits to maintain and an unclear interaction with FlashInfer's compile-time CUDA version detection.

The PyPI stable `vllm` wheel is built against CUDA 13. Verifying:

```
$ ldd .venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so | grep cudart
        libcudart.so.13 => not found
```

That is a hard failure at import time. Three options were considered.

## Investigation

### Option A — install CUDA 13 alongside 12.9

Pros: use stock PyPI wheels, no nightly index plumbing.
Cons: two toolkits to track on upgrades; FlashInfer's behavior when both are present is fragile (it picks up the first on PATH, and our systemd unit explicitly *doesn't* put CUDA on PATH); doubles disk footprint of the toolkit.

### Option B — `nvidia-cuda-runtime-cu13` PyPI package

Pros: lighter than a full toolkit install.
Cons: only ships the runtime, not `nvcc`/headers, so FlashInfer JIT still requires `cuda-toolkit-12-9` *and* the cu13 runtime — three CUDA installs in play. Increases the number of moving pieces without removing any.

### Option C — vLLM nightly cu129 wheels

Pros: matches the installed toolkit exactly; single CUDA install; vLLM publishes these wheels on every nightly so they track upstream.
Cons: must pass two `--extra-index-url` flags and `--index-strategy unsafe-best-match` on every install; "nightly" framing is slightly unsettling for production but the cadence is daily and the version is pinnable.

## Decision

Use vLLM nightly cu129 wheels from `https://wheels.vllm.ai/nightly/cu129`. Pin the install command in `docs/deployment.md` §6 and reproduce it in the README's quick-reproduce block.

Reference install:

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm
```

This resolves `vllm`, `torch`, and `flashinfer` from the nightly indices together. `--index-strategy unsafe-best-match` is the load-bearing flag — uv's default `first-match` will pick `torch` from PyPI (cu13) and silently break the install.

## Required env vars

The systemd unit injects all three; an interactive shell does not need them but if you reproduce the install manually with `uv run`, export them:

```
TORCH_CUDA_ARCH_LIST=12.0
CUDA_ARCHITECTURES=120
CUDA_HOME=/usr/local/cuda-12.9
```

Without `TORCH_CUDA_ARCH_LIST=12.0` / `CUDA_ARCHITECTURES=120`, FlashInfer falls back to a wider arch list and either builds slower or produces "No kernel image available for SM120" at runtime.

## Revisit conditions

- vLLM publishes stable cu129 wheels on PyPI (then we can drop the nightly indices).
- The host migrates to CUDA 13 toolkit (then stable PyPI wheels work directly).
- vLLM's nightly index URL pattern changes (the URL is the only durable contract).

## References

- vLLM nightly index: https://wheels.vllm.ai/nightly/cu129
- PyTorch nightly cu129 index: https://download.pytorch.org/whl/nightly/cu129
- FlashInfer SM120 JIT: https://github.com/flashinfer-ai/flashinfer
