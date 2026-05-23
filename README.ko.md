# vllm-gemma4-31b-fp8

[![English](https://img.shields.io/badge/English-lightgrey?style=flat-square)](README.md) [![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue?style=flat-square)](README.ko.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-lightgrey?style=flat-square)](README.ja.md)

RTX PRO 6000 Blackwell 위에서 Gemma 4 31B를 서빙하기 위한 프로덕션급 vLLM 스택. FP8 KV 캐시, MTP 추측 디코딩, 비동기 FastAPI 로깅 프록시까지 한 세트로 묶었다.

---

## 측정 결과

단일 GPU, 800-token 디코드, 결정적 한국어 프롬프트, 5회 평균.

| 구성                                | 처리량              | 베이스라인 대비 | 비고                                |
| ----------------------------------- | ------------------- | --------------- | ----------------------------------- |
| Baseline (MTP 비활성)               | 42.1 tok/s (σ 0.03) | 1.00x           | 기준값                              |
| MTP=5, 단일 요청                    | 103.2 tok/s (σ ~3)  | **2.45x**       | acceptance rate 78%                 |
| MTP=5, 3 동시 (요청당)              | ~99 tok/s           | 2.35x           | 단일 요청 대비 96% — 거의 선형      |
| MTP=5, 3 동시 (집계)                | **294 tok/s**       | **7.0x**        | wall time 8.2s vs 7.7s 단일 (+6%)   |

MTP=4 (모델 카드 권장값)는 95–98 tok/s로 측정되어, 채택한 MTP=5보다 약 5% 낮았다. 자세한 내용은 [ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md) 참고.

---

## 메모리 프로파일

- 모델 가중치: **약 33 GB** (BCCard FP8-Dynamic, BF16 + F8_E4M3 혼합)
- KV 캐시 풀: **853K 토큰** (`--kv-cache-dtype fp8`)
- 요청당 최대 컨텍스트: **262,144** (256K, `--max-model-len`)
- 256K 컨텍스트에서 보장되는 동시 요청 수: **3.26x**
- BF16 KV 대비 디코드 ITL 개선: 약 32% (vLLM이 공개한 Gemma 4 수치 기준)

---

## 스택

| 계층               | 선택                                            | 이유                                                                                                                  |
| ------------------ | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| GPU                | RTX PRO 6000 Blackwell, 96 GB VRAM (SM120)      | 단일 카드로 31B를 256K 컨텍스트까지 올릴 수 있다.                                                                       |
| 드라이버           | NVIDIA 595.58.03 (CUDA 13.2 호환 표시)          | 안정적인 Blackwell 드라이버. vLLM이 cu129 휠을 배포하므로 CUDA 13 툴킷은 필요 없다.                                     |
| CUDA Toolkit       | 12.9                                            | FlashInfer SM120 JIT 컴파일에 필수.                                                                                   |
| vLLM               | nightly cu129 (≥ 0.20.2rc1.dev128)              | PyPI 안정 휠은 libcudart.so.13에 링크되어 있다. nightly cu129가 설치된 툴킷과 맞는다. [ADR-002](docs/adr/002-vllm-nightly-cu129.md) 참고. |
| PyTorch            | 2.11.0+cu129                                    | vLLM nightly가 끌어온다. CUDA 12.9에 맞춰져 있다.                                                                       |
| transformers       | 5.8.0 (≥ 5.5 필수)                              | Gemma 4 아키텍처 지원이 5.5에서 들어왔다.                                                                              |
| 모델               | `BCCard/gemma-4-31B-it-FP8-Dynamic`             | GSM8K Platinum 0.977 (BF16 baseline 0.976), 출력 깨끗함. [ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md) 참고.        |
| 드래프트 헤드      | `google/gemma-4-31B-it-assistant` (78.8M, 약 150 MB BF16) | `num_speculative_tokens=5`에서 78% acceptance rate로 동작하는 MTP 타깃.                                       |
| KV 캐시            | `--kv-cache-dtype fp8` (표준 FP8 e4m3)          | Gemma 4에서는 TurboQuant가 막혀 있다 (TRITON_ATTN 백엔드, `kv_cache_dtype` 미지원). [ADR-001](docs/adr/001-fp8-kv-over-turboquant.md) 참고. |
| 프록시             | 별도 `.venv-proxy`의 FastAPI (torch 없음). 코드는 [별도 레포](#) <!-- TODO: update proxy repo URL --> | 로깅, 샘플링 기본값, thinking 파라미터 정규화 담당. 인터페이스 명세는 [proxy/README.md](proxy/README.md), 결정 배경은 [ADR-005](docs/adr/005-logging-proxy-separation.md) 참고. |
| 감시(Supervision)  | systemd 2개 unit (proxy가 vLLM을 `Requires=`)   | 재시작이 연쇄로 일어난다. 재부팅에 안전.                                                                               |
| 패키지 매니저      | `uv` (Python 3.12)                              | 재현 가능한 설치. nightly 휠을 위한 `--extra-index-url`을 매끄럽게 처리한다.                                            |

---

## 아키텍처

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

## 저장소 구조

```
gemma4-vllm-stack/
├── README.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   └── adr/
│       ├── 001-fp8-kv-over-turboquant.md
│       ├── 002-vllm-nightly-cu129.md
│       ├── 003-mtp-num-spec-tokens-5.md
│       ├── 004-bccard-fp8-over-nvfp4.md
│       ├── 005-logging-proxy-separation.md
│       └── 006-llmcompressor-venv-isolation.md
├── systemd/
│   ├── vllm-gemma4.service
│   ├── vllm-proxy.service
│   ├── run_vllm.sh
│   └── run_proxy.sh
├── proxy/
│   └── README.md          # 인터페이스 명세만 보관 — 코드는 별도 레포
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

## 빠른 재현 (Ubuntu 24.04 클린 설치 기준)

```bash
# 1. NVIDIA 드라이버 595.x + persistence daemon
sudo apt install -y nvidia-driver-595 nvidia-utils-595
sudo systemctl enable --now nvidia-persistenced

# 2. CUDA Toolkit 12.9 (PATH에는 추가하지 말 것 — systemd unit이 직접 주입한다)
sudo apt install -y cuda-toolkit-12-9

# 3. Python 3.12 + uv
sudo apt install -y python3.12 python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. venv 3개 (.venv, .venv-proxy, .venv-quant)
uv venv .venv --python 3.12
uv venv .venv-proxy --python 3.12

# 5. .venv에 vLLM nightly cu129 설치 (자세한 내용은 docs/deployment.md §6)
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm

# 6. systemd unit 설치 및 활성화
sudo cp systemd/vllm-gemma4.service systemd/vllm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-gemma4.service vllm-proxy.service
```

전체 절차는 [docs/deployment.md](docs/deployment.md) 참고.

---

## 핵심 결정 (요약)

1. **FP8 KV 캐시, TurboQuant 아님.** Gemma 4에서 TurboQuant는 두 개의 독립적인 이슈로 막혀 있다. 첫째, 비대칭 head dimension (256 local, 512 global)이 `TRITON_ATTN` 백엔드를 강제하는데 이 백엔드는 `kv_cache_dtype`을 구현하지 않는다. 둘째, vLLM PR #40534가 추가한 `use_bidirectional_attention='vision'`이 full-attention 레이어에까지 `use_mm_prefix=True`를 전파한다. 표준 FP8 e4m3는 853K 토큰 풀, 256K 컨텍스트에서 3.26x 보장 동시성, 약 32%의 디코드 ITL 개선을 준다. [ADR-001](docs/adr/001-fp8-kv-over-turboquant.md) 참고.

2. **vLLM nightly cu129, PyPI 안정판 아님.** PyPI 휠은 `libcudart.so.13`에 링크되어 있고 시스템에는 CUDA 12.9가 있다 — `ldd`로 불일치가 확인된다. `https://wheels.vllm.ai/nightly/cu129`의 nightly 휠이 cu13 휠이 실용화되기 전까지의 유일한 정공법이다. 필수 환경변수: `TORCH_CUDA_ARCH_LIST=12.0`, `CUDA_ARCHITECTURES=120`, `CUDA_HOME=/usr/local/cuda-12.9`. [ADR-002](docs/adr/002-vllm-nightly-cu129.md) 참고.

3. **`num_speculative_tokens=5`, 모델 카드 권장값 4가 아님.** 측정 결과: MTP=4는 95–98 tok/s, MTP=5는 103.2 tok/s. 5% 개선이 공짜로 들어온다. vLLM의 >1 경고는 일반적인 안내일 뿐 이 하드웨어/드래프트 조합에 특화된 것이 아니다. Acceptance rate 78%. [ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md) 참고.

4. **`BCCard/gemma-4-31B-it-FP8-Dynamic`, `nvidia/Gemma-4-31B-IT-NVFP4` 아님.** 초기 배포에서는 llmcompressor FP8 garbage-output 버그 (vLLM Issue #39407) 때문에 NVFP4를 골랐다. 이 버그는 vLLM nightly ≥ 0.20.2rc1에서 FP8-Dynamic에 한해 패치되었다. 한국어 피보나치 프롬프트로 깨끗한 출력 확인, GSM8K Platinum 0.977 (NVFP4는 약 0.94). 가중치는 17 GB → 33 GB로 늘었지만 served-model-name은 동일하게 유지. [ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md) 참고.

5. **자체 제작한 FastAPI 프록시, 별도 venv.** LiteLLM/Langfuse/Helicone 모두 ~10명 규모 배포가 필요로 하는 것 이상을 끌어오고 요청 로그 스키마를 가린다. 약 300줄, 자체 로그 스키마 소유, 수 초 안에 부팅, 약 400 MB RAM, torch 없음. `.venv-proxy`에 들어가 있으므로 vLLM 의존성 변경이 프록시를 깨뜨릴 수 없다. 코드는 [별도 레포](#)에서 버전 관리하며 <!-- TODO: update proxy repo URL -->, 이 레포에는 인터페이스 명세인 [proxy/README.md](proxy/README.md)만 둔다. [ADR-005](docs/adr/005-logging-proxy-separation.md) 참고.

6. **venv 3개 (`.venv`, `.venv-proxy`, `.venv-quant`).** `llmcompressor`는 오래된 torch/transformers를 강하게 핀하기 때문에 `.venv`에 설치하면 서빙 환경을 조용히 다운그레이드시킨다 — 처음 당했을 때 복구에 약 90분 걸렸다. 역할별 venv 분리는 이 전체 부류의 실패를 차단한다. [ADR-006](docs/adr/006-llmcompressor-venv-isolation.md) 참고.

---

## 운영 노트

- **헬스 / 재시작**: `systemctl status vllm-gemma4 vllm-proxy`. `sudo systemctl restart vllm-gemma4`는 `Requires=` 덕분에 프록시로 연쇄된다.
- **로그**: 서빙은 `journalctl -u vllm-gemma4 -f`. 요청 단위 JSONL은 `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl`.
- **외부 포트**: 클라이언트는 `0.0.0.0:8000`의 프록시를 친다. vLLM은 `127.0.0.1:8001`에만 바인딩되며 외부에 직접 노출되지 않는다.
- **served-model-name**: `gemma-4-31b-it` — NVFP4 → FP8-Dynamic 마이그레이션 동안에도 동일하게 유지해서 클라이언트 변경이 필요 없었다.
- **첫 기동**: `TimeoutStartSec=600`. 모델이나 vLLM 버전을 바꾼 직후 첫 기동에는 FlashInfer JIT 컴파일 + CUDA graph capture에 3~5분이 걸리기 때문이다.

---

## 상태

내부 ~10명 규모 랩에서 프로덕션 운영 중. 재부팅 테스트 완료, 드라이버/커널 업데이트와 정상 종료에 안전.

---

## Notes

이 페이지의 내용 정리에 AI 도구를 사용했다.
