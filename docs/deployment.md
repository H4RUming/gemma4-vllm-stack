# Deployment

End-to-end procedure for bringing up the stack on a clean Ubuntu 24.04 host with one RTX PRO 6000 Blackwell.

---

## 1. Prerequisites

- **Hardware**: RTX PRO 6000 Blackwell (96 GB VRAM, SM120) on PCIe; tested with a single GPU at index 0.
- **Disk**: ~120 GB free for model weights (~33 GB), MTP draft (~150 MB), CUDA toolkit, venvs, and Hugging Face cache headroom.
- **Network**: outbound HTTPS to `huggingface.co`, `wheels.vllm.ai`, `download.pytorch.org`, and `developer.download.nvidia.com`.
- **OS**: Ubuntu 24.04 LTS, kernel 6.8+.

---

## 2. NVIDIA driver (595.x)

```bash
sudo apt update
sudo apt install -y nvidia-driver-595 nvidia-utils-595
sudo systemctl enable --now nvidia-persistenced
nvidia-smi   # confirm: Driver Version: 595.58.03, CUDA Version: 13.2
```

The driver advertises CUDA 13.2 compatibility, but we install the 12.9 toolkit alongside — the driver supports older toolkits and we explicitly do not want vLLM to pick up CUDA 13.

---

## 3. CUDA Toolkit 12.9

```bash
# 시스템 PATH에 절대 추가하지 말 것 — systemd unit이 직접 주입한다.
sudo apt install -y cuda-toolkit-12-9
ls /usr/local/cuda-12.9
```

Do **not** export `CUDA_HOME` or add `/usr/local/cuda-12.9/bin` to a shell rc file. The systemd units own that environment so an interactive shell never accidentally drifts the install.

---

## 4. Python 3.12 + uv

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL   # pick up uv on PATH
uv --version
```

---

## 5. Working directory + three venvs

```bash
mkdir -p ~/Desktop/LLM-OPS
cd ~/Desktop/LLM-OPS

uv venv .venv        --python 3.12   # 서빙용
uv venv .venv-proxy  --python 3.12   # 프록시용
uv venv .venv-quant  --python 3.12   # 양자화용 (별도 격리)
```

Rationale for three venvs: [ADR-006](adr/006-llmcompressor-venv-isolation.md).

---

## 6. Install vLLM nightly cu129 into `.venv`

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm
```

`--index-strategy unsafe-best-match` is required so uv considers the nightly indices for `torch`/`vllm`/`flashinfer` together; the default `first-match` strategy resolves `torch` from PyPI (cu13 wheel) and breaks the install.

Verify:

```bash
.venv/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
# 예: 0.20.2rc1.dev128  2.11.0+cu129

ldd $(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cuda.so'))") \
  | grep cudart
# 기대: libcudart.so.12 => /usr/local/cuda-12.9/lib64/libcudart.so.12 (...)
# 절대 안 됨: libcudart.so.13 => not found
```

If `ldd` shows `libcudart.so.13`, you got the PyPI stable wheel; reinstall with both `--extra-index-url` flags. See [ADR-002](adr/002-vllm-nightly-cu129.md).

---

## 7. Model downloads

```bash
export HF_HOME=~/Desktop/LLM-OPS/hf-cache

# Target (33 GB)
.venv/bin/hf download BCCard/gemma-4-31B-it-FP8-Dynamic

# MTP draft head (~150 MB)
.venv/bin/hf download google/gemma-4-31B-it-assistant
```

If you maintain a private mirror or need `HF_TOKEN`, set it in `configs/.env` before downloading.

---

## 8. Proxy venv setup (`.venv-proxy`)

```bash
uv pip install --python .venv-proxy/bin/python -r proxy/requirements.txt

# Verify torch is NOT present
.venv-proxy/bin/python -c "import torch" 2>&1 | grep -i "no module named 'torch'"
# 기대: 위 grep이 매치되어야 함 — torch가 없는 게 정상.
```

The proxy stays light on purpose. If `import torch` ever succeeds in `.venv-proxy`, something pulled it in transitively — find and pin the culprit before deploying.

---

## 9. Install systemd units

```bash
# 경로 안의 /home/haru/Desktop/LLM-OPS 는 실제 배포 위치에 맞춰 치환할 것.
sudo cp systemd/vllm-gemma4.service /etc/systemd/system/
sudo cp systemd/vllm-proxy.service /etc/systemd/system/
chmod +x systemd/run_vllm.sh systemd/run_proxy.sh
cp systemd/run_vllm.sh systemd/run_proxy.sh ~/Desktop/LLM-OPS/

sudo systemctl daemon-reload
sudo systemctl enable vllm-gemma4.service vllm-proxy.service
sudo systemctl start  vllm-gemma4.service   # proxy will follow via Requires=
```

---

## 10. First start — what to watch in the log

`journalctl -u vllm-gemma4 -f` and confirm these 9 messages appear in roughly this order before the HTTP endpoint binds:

1. `Loaded Gemma4MTPModel for speculative decoding`
2. `Sharing input embedding weights with draft head`
3. `Loading model weights took ... GiB` (≈ 33 GiB)
4. `KV cache size: 853,392 tokens` (or close)
5. `Maximum concurrency for 262,144 tokens per request: 3.26x`
6. `Compiling the model with mode='full_and_piecewise' for FlashInfer SM120`
7. `Capturing CUDA graphs (decode, mixed prefill-decode)`
8. `Application startup complete.`
9. `Uvicorn running on http://127.0.0.1:8001`

First launch after a vLLM or model change spends 3–5 minutes between (6) and (8). Subsequent starts are <60 seconds because FlashInfer caches compiled kernels.

---

## 11. Smoke tests

```bash
# (a) Model list (through proxy)
curl -s http://127.0.0.1:8000/v1/models | jq

# (b) Basic chat
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role":"user","content":"두 단어로 인사해줘."}]
  }' | jq -r '.choices[0].message.content'

# (c) Streaming with thinking ON
curl -N -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [{"role":"user","content":"Fibonacci 10개를 한국어로 설명"}],
    "stream": true,
    "reasoning_effort": "high"
  }'
```

---

## 12. Reboot test

```bash
sudo reboot
# 다시 SSH 후
systemctl status vllm-gemma4 vllm-proxy
journalctl -u vllm-gemma4 -b | grep "Uvicorn running"
```

If both units come back `active (running)` and the smoke tests in §11 pass, the stack is reboot-safe.

---

## Troubleshooting

### `libcudart.so.13 => not found`

The PyPI stable vLLM wheel landed in `.venv`. Reinstall with the nightly cu129 indices:

```bash
uv pip uninstall --python .venv/bin/python vllm torch
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm
```

See [ADR-002](adr/002-vllm-nightly-cu129.md).

### `No kernel image available for SM120`

The build did not pick up Blackwell. Confirm the environment exported by the systemd unit:

```bash
systemctl show vllm-gemma4 -p Environment
# 기대 라인:
# TORCH_CUDA_ARCH_LIST=12.0
# CUDA_ARCHITECTURES=120
# CUDA_HOME=/usr/local/cuda-12.9
```

If any are missing, edit `/etc/systemd/system/vllm-gemma4.service`, then `daemon-reload && restart`.

### First-launch hang at "Capturing CUDA graphs"

This is normal for 3–5 minutes the first time after a vLLM upgrade or model change. `TimeoutStartSec=600` accommodates it. If it exceeds 10 minutes, capture `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5` — utilization should sit at 100% during compile. If it's 0% and memory isn't climbing, kill the unit and re-check that the model files downloaded fully (`hf scan-cache`).

### Output looks like "a a a a a"

You are on an older llmcompressor-quantized FP8 model with the garbage-output bug. Confirm the model on disk:

```bash
ls ~/Desktop/LLM-OPS/hf-cache/models--BCCard--gemma-4-31B-it-FP8-Dynamic
```

If it's `nvidia/Gemma-4-31B-IT-NVFP4` (legacy) or an older FP8 checkpoint, switch to `BCCard/gemma-4-31B-it-FP8-Dynamic` and confirm vLLM is on nightly ≥ `0.20.2rc1.dev128`. The bug is patched for FP8-Dynamic specifically; FP8_BLOCK is not retested. See [ADR-004](adr/004-bccard-fp8-over-nvfp4.md).
