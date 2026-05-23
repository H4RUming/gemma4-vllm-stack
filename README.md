# vllm-gemma4-31b-fp8

[![English](https://img.shields.io/badge/English-blue?style=flat-square)](README.md) [![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-lightgrey?style=flat-square)](README.ko.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-lightgrey?style=flat-square)](README.ja.md)

vLLM serving stack for Gemma 4 31B on RTX PRO 6000 Blackwell, with FP8 KV cache, MTP speculative decoding, and an async FastAPI logging proxy in front.

---

## Results

Single GPU, 800-token decode, deterministic Korean prompt, 5-run mean.

| Configuration               | Throughput      | vs baseline | Notes                              |
| --------------------------- | --------------- | ----------- | ---------------------------------- |
| Baseline (MTP disabled)     | 42.1 tok/s (σ 0.03) | 1.00x   | reference                          |
| MTP=5, single request       | 103.2 tok/s (σ ~3)  | **2.45x** | initial 5-run benchmark            |
| MTP=5, 3 concurrent (per req) | ~99 tok/s     | 2.35x       | 96% of single — near-linear        |
| MTP=5, 3 concurrent (aggregate) | **294 tok/s** | **7.0x** | wall time 8.2s vs 7.7s single (+6%) |
| MTP=5, sustained (live traffic) | 100–125 tok/s | 2.4–3.0x | per-position acceptance 0.99/0.98/0.96/0.94/0.89; avg draft acceptance **94–97%** (re-measured 2026-05-23) |

MTP=4 (model-card recommended) measured at 95–98 tok/s, ~5% below the chosen MTP=5. See [ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md).

---

## Memory profile

- Model weights: **~33 GB** (BCCard FP8-Dynamic, BF16 + F8_E4M3 mixed); ~89 GB GPU residency including weights + KV pool + CUDA Graphs + draft model
- KV cache pool: **495,400 tokens** (`--kv-cache-dtype fp8`, `block_size=16`, `num_gpu_blocks=35,447`; measured 2026-05-23, post-`--max-num-batched-tokens` change — see [ADR-007](docs/adr/007-max-num-batched-tokens-32768.md))
- Max context per request: **262,144** (256K, `--max-model-len`)
- Guaranteed concurrency at 256K context: **~1.89x** (`495,400 / 262,144`); pool size is config-dependent and should be re-derived after any flag change
- Decode ITL improvement vs BF16 KV: ~32% (per vLLM's published Gemma 4 numbers)

Reproduce KV pool on the running service:

```bash
curl -s http://127.0.0.1:8001/metrics | grep -E 'num_gpu_blocks|kv_cache_memory_bytes'
```

---

## Stack

| Layer              | Choice                                          | Why                                                                                       |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| GPU                | RTX PRO 6000 Blackwell, 96 GB VRAM (SM120)      | Single-card 31B serving with 256K context fits.                                           |
| Driver             | NVIDIA 595.58.03 (advertises CUDA 13.2)         | Stable Blackwell driver; CUDA 13 toolkit not required because vLLM ships cu129 wheels.    |
| CUDA Toolkit       | 12.9                                            | Required for FlashInfer SM120 JIT compilation.                                            |
| vLLM               | nightly cu129 (≥ 0.20.2rc1.dev128)              | PyPI stable wheel links libcudart.so.13; nightly cu129 matches the installed toolkit. See [ADR-002](docs/adr/002-vllm-nightly-cu129.md). |
| PyTorch            | 2.11.0+cu129                                    | Pulled in by vLLM nightly; matches CUDA 12.9.                                             |
| transformers       | 5.8.0 (≥ 5.5 required)                          | Gemma 4 architecture support landed in 5.5.                                               |
| Model              | `BCCard/gemma-4-31B-it-FP8-Dynamic`             | GSM8K Platinum 0.977 (BF16 baseline 0.976); clean output. See [ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md). |
| Draft head         | `google/gemma-4-31B-it-assistant` (78.8M, ~150 MB BF16) | MTP target with 78% acceptance rate at `num_speculative_tokens=5`.                  |
| KV cache           | `--kv-cache-dtype fp8` (standard FP8 e4m3)      | TurboQuant blocked on Gemma 4 (TRITON_ATTN backend, no kv_cache_dtype). See [ADR-001](docs/adr/001-fp8-kv-over-turboquant.md). |
| Proxy              | FastAPI in separate `.venv-proxy` (no torch); code in [separate repo](https://github.com/H4RUming/gemma4-vllm-proxy) | Logging, sampling defaults, thinking-param normalization. Interface spec in [proxy/README.md](proxy/README.md); rationale in [ADR-005](docs/adr/005-logging-proxy-separation.md). |
| Supervision        | systemd (two units, proxy `Requires=` vLLM)     | Cascading restarts; reboot-safe.                                                          |
| Package manager    | `uv` (Python 3.12)                              | Reproducible installs; handles `--extra-index-url` for nightly wheels cleanly.            |

---

## Architecture

```
                      ┌──────────────────────────────┐
                      │      Clients (HTTP/SSE)      │
                      └──────────────┬───────────────┘
                                     │
                                     ▼  :8000  (0.0.0.0)
                      ┌──────────────────────────────┐
                      │  vllm-proxy.service          │
                      │  FastAPI + httpx             │
                      │  .venv-proxy (no torch)      │
                      │  - thinking normalization    │
                      │  - sampling defaults         │
                      │  - JSONL request logging     │
                      │  - blocked-path enforcement  │
                      └──────────────┬───────────────┘
                                     │ Requires=
                                     ▼  :8001  (127.0.0.1)
                      ┌──────────────────────────────┐
                      │  vllm-gemma4.service         │
                      │  vLLM nightly cu129          │
                      │  .venv                       │
                      │  - Gemma 4 31B FP8-Dynamic   │
                      │  - MTP draft (num_spec=5)    │
                      │  - FP8 KV cache, 256K ctx    │
                      └──────────────────────────────┘
```

---

## Repository layout

```
gemma4-vllm-stack/
├── README.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   ├── adr/
│   │   ├── 001-fp8-kv-over-turboquant.md
│   │   ├── 002-vllm-nightly-cu129.md
│   │   ├── 003-mtp-num-spec-tokens-5.md
│   │   ├── 004-bccard-fp8-over-nvfp4.md
│   │   ├── 005-logging-proxy-separation.md
│   │   ├── 006-llmcompressor-venv-isolation.md
│   │   └── 007-max-num-batched-tokens-32768.md
│   └── investigations/
│       └── 2026-05-23-prefill-investigation.md
├── systemd/
│   ├── vllm-gemma4.service
│   ├── vllm-proxy.service
│   ├── run_vllm.sh
│   └── run_proxy.sh
├── proxy/
│   └── README.md          # interface spec only — code in separate repo
├── benchmarks/
│   ├── README.md
│   ├── mtp-tuning/
│   │   └── README.md
│   └── concurrency/
│       └── README.md
└── configs/
    └── .env.example
```

---

## Quick reproduce (from clean Ubuntu 24.04)

```bash
# 1. NVIDIA driver 595.x + persistence daemon
sudo apt install -y nvidia-driver-595 nvidia-utils-595
sudo systemctl enable --now nvidia-persistenced

# 2. CUDA Toolkit 12.9 (do NOT add to PATH — systemd units inject)
sudo apt install -y cuda-toolkit-12-9

# 3. Python 3.12 + uv
sudo apt install -y python3.12 python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Three venvs (.venv, .venv-proxy, .venv-quant)
uv venv .venv --python 3.12
uv venv .venv-proxy --python 3.12

# 5. Install vLLM nightly cu129 into .venv (see docs/deployment.md §6)
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm

# 6. Install systemd units and enable
sudo cp systemd/vllm-gemma4.service systemd/vllm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-gemma4.service vllm-proxy.service
```

Full procedure in [docs/deployment.md](docs/deployment.md).

---

## Key decisions (TL;DR)

1. **FP8 KV cache, not TurboQuant.** TurboQuant is blocked on Gemma 4 by two independent issues: heterogeneous head dimensions (256 local, 512 global) force the `TRITON_ATTN` backend which doesn't implement `kv_cache_dtype`, and vLLM PR #40534's `use_bidirectional_attention='vision'` propagates `use_mm_prefix=True` on full-attention layers. Standard FP8 e4m3 gives ~32% decode ITL improvement at no measurable accuracy cost. Current pool: 495,400 tokens / ~1.89x at 256K context (driven by the chunk-size choice in [ADR-007](docs/adr/007-max-num-batched-tokens-32768.md)). See [ADR-001](docs/adr/001-fp8-kv-over-turboquant.md).

2. **vLLM nightly cu129, not PyPI stable.** The PyPI wheel links `libcudart.so.13` and the system has CUDA 12.9 — `ldd` confirms the mismatch. Nightly cu129 wheels from `https://wheels.vllm.ai/nightly/cu129` are the supported route until cu13 wheels are practical. Required env: `TORCH_CUDA_ARCH_LIST=12.0`, `CUDA_ARCHITECTURES=120`, `CUDA_HOME=/usr/local/cuda-12.9`. See [ADR-002](docs/adr/002-vllm-nightly-cu129.md).

3. **`num_speculative_tokens=5`, not the model-card-recommended 4.** Measured: MTP=4 at 95–98 tok/s vs MTP=5 at 103.2 tok/s. 5% improvement is free; vLLM's >1 warning is generic and not specific to this hardware/draft combination. Initial 5-run acceptance rate 78%; sustained live-traffic acceptance 94–97% (re-measured 2026-05-23). See [ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md).

4. **`BCCard/gemma-4-31B-it-FP8-Dynamic`, not `nvidia/Gemma-4-31B-IT-NVFP4`.** Initial deployment chose NVFP4 due to the llmcompressor FP8 garbage-output bug (vLLM Issue #39407). The bug is patched for FP8-Dynamic specifically in vLLM nightly ≥0.20.2rc1; verified with a clean Korean Fibonacci output and GSM8K Platinum 0.977 vs NVFP4 ~0.94. Weights grew from 17 GB to 33 GB; served-model-name held constant. See [ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md).

5. **Hand-rolled FastAPI proxy in its own venv.** LiteLLM/Langfuse/Helicone all pull in more than ~10-user deployment needs and obscure the request log schema. ~300 lines, owns its log shape, boots in seconds with ~400 MB RAM, no torch. Lives in `.venv-proxy` so vLLM dependency changes can't break it. Code is versioned in a [separate repository](https://github.com/H4RUming/gemma4-vllm-proxy); this repo keeps [proxy/README.md](proxy/README.md) as the interface spec. See [ADR-005](docs/adr/005-logging-proxy-separation.md).

6. **Three venvs (`.venv`, `.venv-proxy`, `.venv-quant`).** `llmcompressor` aggressively pins old torch/transformers and silently downgrades the serving environment if installed in `.venv` — ~90 minutes of recovery the first time. Role-separated venvs prevent the entire class of failure. See [ADR-006](docs/adr/006-llmcompressor-venv-isolation.md).

7. **`--max-num-batched-tokens 32768` (explicit prefill chunk).** Bounds the per-step token budget so the activation envelope is predictable under bursty prefill, at the cost of ~14% KV pool (576,912 → 495,400 tokens). Pool is now a measured value, not a fixed promise. See [ADR-007](docs/adr/007-max-num-batched-tokens-32768.md).

---

## Deployment layout (production host)

The repository tree above shows how files are organized for review. On the production host, the working tree is **flat** — systemd unit files, wrapper scripts, and the proxy clone all live at the top level of `${LLM_OPS_DIR}` (default `/home/haru/Desktop/LLM-OPS`) because that's the `WorkingDirectory` declared in each `*.service` unit and `ExecStart=/home/haru/Desktop/LLM-OPS/run_vllm.sh` resolves there.

```
/home/haru/Desktop/LLM-OPS/        (= WorkingDirectory in both .service units)
├── run_vllm.sh                    (from this repo's systemd/)
├── run_proxy.sh                   (from this repo's systemd/)
├── vllm-gemma4.service            (also installed under /etc/systemd/system/)
├── vllm-proxy.service             (also installed under /etc/systemd/system/)
├── proxy/                         (clone of github.com/H4RUming/gemma4-vllm-proxy)
├── logs/                          (runtime, gitignored)
├── hf-cache/                      (HF_HOME)
├── .venv/                         (vLLM serving)
├── .venv-proxy/                   (FastAPI proxy)
└── .venv-quant/                   (llmcompressor; never installed in .venv)
```

The deploy step from `docs/deployment.md §9` copies `systemd/run_vllm.sh` and `systemd/run_proxy.sh` from this repo to `${LLM_OPS_DIR}/`, and copies the `.service` files to `/etc/systemd/system/`. The flat layout is intentional: it matches the absolute paths baked into the units, which keeps `daemon-reload → restart` self-contained.

---

## Operating notes

- **Health & restart**: `systemctl status vllm-gemma4 vllm-proxy`; `sudo systemctl restart vllm-gemma4` cascades to the proxy via `Requires=`.
- **Logs**: `journalctl -u vllm-gemma4 -f` for serving; per-request JSONL at `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl`.
- **External port**: clients hit the proxy at `0.0.0.0:8000`. vLLM is bound to `127.0.0.1:8001` and never directly exposed.
- **Served model name**: `gemma-4-31b-it` — held constant across the NVFP4 → FP8-Dynamic migration so clients did not change.
- **First start**: `TimeoutStartSec=600` because FlashInfer JIT compile + CUDA graph capture takes 3–5 minutes on first launch after a model or vLLM version change.

---

## Investigations

Dated troubleshooting notes live under [`docs/investigations/`](docs/investigations/). They capture the diagnostic path for a specific incident — what was observed, what was ruled out, what was changed — separate from the ADRs (which record decisions) and the benchmarks (which record measurements).

- [2026-05-23 — prefill performance investigation](docs/investigations/2026-05-23-prefill-investigation.md). Diagnoses the ~90 → ~25 tok/s TPS regression after p50 input tokens grew 11× (to ~37K) on 2026-05-20. Pins the slowdown on prefill, locks in the `--max-num-batched-tokens 32768` setting recorded in [ADR-007](docs/adr/007-max-num-batched-tokens-32768.md), and identifies client-side cache-key stability as the largest remaining win.

---

## Status

Running in production for an internal ~10-user lab; reboot-tested, surviving driver/kernel updates and clean shutdowns.

---

## Notes

AI tools were used to assist with content organization and grammar editing on this page.
