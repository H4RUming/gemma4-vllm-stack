# vllm-gemma4-31b-fp8

[![English](https://img.shields.io/badge/English-lightgrey?style=flat-square)](README.md) [![한국어](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-lightgrey?style=flat-square)](README.ko.md) [![日本語](https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue?style=flat-square)](README.ja.md)

RTX PRO 6000 Blackwell 上で Gemma 4 31B をサービングするためのプロダクショングレードの vLLM スタック。FP8 KV キャッシュ、MTP 投機的デコーディング、非同期 FastAPI ロギングプロキシをひとつにまとめている。

---

## 測定結果

シングル GPU、800-token デコード、決定的な韓国語プロンプト、5 回の平均値。

| 構成                                | スループット         | ベースライン比 | 備考                                  |
| ----------------------------------- | -------------------- | -------------- | ------------------------------------- |
| ベースライン (MTP 無効)             | 42.1 tok/s (σ 0.03)  | 1.00x          | 基準値                                |
| MTP=5、シングルリクエスト           | 103.2 tok/s (σ ~3)   | **2.45x**      | acceptance rate 78%                   |
| MTP=5、3 並列 (リクエストあたり)    | ~99 tok/s            | 2.35x          | シングルの 96% — ほぼ線形              |
| MTP=5、3 並列 (集計)                | **294 tok/s**        | **7.0x**       | wall time 8.2s vs 7.7s シングル (+6%) |

MTP=4 (モデルカード推奨値) は 95–98 tok/s で、採用した MTP=5 より約 5% 低い結果となった。詳細は [ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md) を参照。

---

## メモリプロファイル

- モデル重み: **約 33 GB** (BCCard FP8-Dynamic、BF16 + F8_E4M3 混在)
- KV キャッシュプール: **853K トークン** (`--kv-cache-dtype fp8`)
- リクエストあたりの最大コンテキスト: **262,144** (256K、`--max-model-len`)
- 256K コンテキスト時の保証同時実行数: **3.26x**
- BF16 KV と比較したデコード ITL の改善: 約 32% (vLLM が公開している Gemma 4 の数値に基づく)

---

## スタック構成

| レイヤー           | 選択                                            | 理由                                                                                                                  |
| ------------------ | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| GPU                | RTX PRO 6000 Blackwell、96 GB VRAM (SM120)      | シングルカードで 31B を 256K コンテキストで動かせる。                                                                  |
| ドライバ           | NVIDIA 595.58.03 (CUDA 13.2 互換と表記)         | 安定した Blackwell ドライバ。vLLM が cu129 ホイールを配布しているため CUDA 13 ツールキットは不要。                     |
| CUDA Toolkit       | 12.9                                            | FlashInfer SM120 の JIT コンパイルに必須。                                                                            |
| vLLM               | nightly cu129 (≥ 0.20.2rc1.dev128)              | PyPI 安定版ホイールは libcudart.so.13 にリンクされている。nightly cu129 がインストール済みツールキットと一致する。[ADR-002](docs/adr/002-vllm-nightly-cu129.md) 参照。 |
| PyTorch            | 2.11.0+cu129                                    | vLLM nightly が引いてくる。CUDA 12.9 に合わせてある。                                                                  |
| transformers       | 5.8.0 (≥ 5.5 必須)                              | Gemma 4 アーキテクチャのサポートは 5.5 で入った。                                                                      |
| モデル             | `BCCard/gemma-4-31B-it-FP8-Dynamic`             | GSM8K Platinum 0.977 (BF16 baseline 0.976)、出力が崩れない。[ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md) 参照。    |
| ドラフトヘッド     | `google/gemma-4-31B-it-assistant` (78.8M、約 150 MB BF16) | `num_speculative_tokens=5` で acceptance rate 78% を達成する MTP ターゲット。                                |
| KV キャッシュ      | `--kv-cache-dtype fp8` (標準 FP8 e4m3)          | Gemma 4 では TurboQuant が使えない (TRITON_ATTN バックエンドが `kv_cache_dtype` 未対応)。[ADR-001](docs/adr/001-fp8-kv-over-turboquant.md) 参照。 |
| プロキシ           | 別 venv `.venv-proxy` 上の FastAPI (torch 無し)。コードは[別リポジトリ](#) <!-- TODO: update proxy repo URL --> | ロギング、サンプリングデフォルト、thinking パラメータの正規化を担当。インターフェース仕様は [proxy/README.md](proxy/README.md)、決定の背景は [ADR-005](docs/adr/005-logging-proxy-separation.md) を参照。 |
| プロセス管理       | systemd 2 ユニット (proxy が vLLM を `Requires=`) | 再起動が連鎖する。再起動に耐える。                                                                                    |
| パッケージマネージャ | `uv` (Python 3.12)                              | 再現性のあるインストール。nightly ホイール用の `--extra-index-url` をきれいに扱える。                                  |

---

## アーキテクチャ

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

## リポジトリ構成

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
│   └── README.md          # インターフェース仕様のみ — コードは別リポジトリ
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

## クイック再現手順 (Ubuntu 24.04 クリーンインストールから)

```bash
# 1. NVIDIA ドライバ 595.x + persistence daemon
sudo apt install -y nvidia-driver-595 nvidia-utils-595
sudo systemctl enable --now nvidia-persistenced

# 2. CUDA Toolkit 12.9 (PATH には追加しない — systemd unit が直接注入する)
sudo apt install -y cuda-toolkit-12-9

# 3. Python 3.12 + uv
sudo apt install -y python3.12 python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. venv 3 つ (.venv, .venv-proxy, .venv-quant)
uv venv .venv --python 3.12
uv venv .venv-proxy --python 3.12

# 5. .venv に vLLM nightly cu129 をインストール (詳細は docs/deployment.md §6)
uv pip install --python .venv/bin/python \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu129 \
  --index-strategy unsafe-best-match \
  vllm

# 6. systemd unit のインストールと有効化
sudo cp systemd/vllm-gemma4.service systemd/vllm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-gemma4.service vllm-proxy.service
```

完全な手順は [docs/deployment.md](docs/deployment.md) を参照。

---

## 主要な決定 (要約)

1. **FP8 KV キャッシュ、TurboQuant ではない。** Gemma 4 で TurboQuant は 2 つの独立した問題により使えない。第一に、head dimension が不均一 (256 local、512 global) のため `TRITON_ATTN` バックエンドが選択されるが、このバックエンドは `kv_cache_dtype` を実装していない。第二に、vLLM PR #40534 が追加した `use_bidirectional_attention='vision'` が full-attention レイヤーにまで `use_mm_prefix=True` を伝播する。標準 FP8 e4m3 で 853K トークンプール、256K コンテキストで 3.26x の保証同時実行、約 32% のデコード ITL 改善が得られる。[ADR-001](docs/adr/001-fp8-kv-over-turboquant.md) 参照。

2. **vLLM nightly cu129、PyPI 安定版ではない。** PyPI のホイールは `libcudart.so.13` にリンクされているが、システムには CUDA 12.9 が入っている — `ldd` でその不一致が確認できる。`https://wheels.vllm.ai/nightly/cu129` の nightly ホイールが cu13 ホイールが実用化されるまでの唯一の正攻法である。必須の環境変数: `TORCH_CUDA_ARCH_LIST=12.0`、`CUDA_ARCHITECTURES=120`、`CUDA_HOME=/usr/local/cuda-12.9`。[ADR-002](docs/adr/002-vllm-nightly-cu129.md) 参照。

3. **`num_speculative_tokens=5`、モデルカード推奨の 4 ではない。** 測定結果: MTP=4 は 95–98 tok/s、MTP=5 は 103.2 tok/s。5% の改善が無料で手に入る。vLLM の >1 警告は一般的な注意喚起にすぎず、このハードウェアやドラフト構成に特化したものではない。Acceptance rate 78%。[ADR-003](docs/adr/003-mtp-num-spec-tokens-5.md) 参照。

4. **`BCCard/gemma-4-31B-it-FP8-Dynamic`、`nvidia/Gemma-4-31B-IT-NVFP4` ではない。** 初期デプロイでは llmcompressor の FP8 garbage-output バグ (vLLM Issue #39407) のため NVFP4 を選択していた。このバグは vLLM nightly ≥ 0.20.2rc1 において FP8-Dynamic 限定でパッチされた。韓国語のフィボナッチプロンプトで出力が崩れないことを確認し、GSM8K Platinum 0.977 (NVFP4 は約 0.94)。重みは 17 GB → 33 GB に増えたが、served-model-name は変更しなかった。[ADR-004](docs/adr/004-bccard-fp8-over-nvfp4.md) 参照。

5. **自作 FastAPI プロキシを別 venv で運用。** LiteLLM/Langfuse/Helicone はいずれも ~10 名規模のデプロイに必要なものを超える依存を引き込み、リクエストログのスキーマを覆い隠す。約 300 行、自前のログスキーマ、外部ストレージ無し、数秒で起動、約 400 MB RAM、torch 無し。`.venv-proxy` に配置することで vLLM の依存変更がプロキシを壊さない。コードは[別リポジトリ](#)でバージョン管理しており <!-- TODO: update proxy repo URL -->、このリポジトリにはインターフェース仕様としての [proxy/README.md](proxy/README.md) のみを残している。[ADR-005](docs/adr/005-logging-proxy-separation.md) 参照。

6. **venv を 3 つ (`.venv`、`.venv-proxy`、`.venv-quant`)。** `llmcompressor` は古い torch/transformers を強く pin するため、`.venv` にインストールするとサービング環境を静かにダウングレードしてしまう — 最初に踏んだときは復旧に約 90 分かかった。役割ごとに venv を分けることで、この種の障害そのものを排除している。[ADR-006](docs/adr/006-llmcompressor-venv-isolation.md) 参照。

---

## 運用メモ

- **ヘルス / 再起動**: `systemctl status vllm-gemma4 vllm-proxy`。`sudo systemctl restart vllm-gemma4` は `Requires=` によりプロキシにも連鎖する。
- **ログ**: サービング側は `journalctl -u vllm-gemma4 -f`。リクエスト単位の JSONL は `~/Desktop/LLM-OPS/logs/requests-YYYY-MM-DD.jsonl`。
- **外部ポート**: クライアントは `0.0.0.0:8000` のプロキシにアクセスする。vLLM は `127.0.0.1:8001` にのみバインドされ、外部に直接公開されない。
- **served-model-name**: `gemma-4-31b-it` — NVFP4 → FP8-Dynamic のマイグレーション中も変えずに固定し、クライアント側の変更を不要にした。
- **初回起動**: `TimeoutStartSec=600`。モデルや vLLM のバージョンを変えた直後の初回起動は、FlashInfer の JIT コンパイル + CUDA graph capture に 3〜5 分かかるため。

---

## ステータス

社内 ~10 名規模のラボで本番運用中。再起動テスト済みで、ドライバ/カーネルの更新や通常のシャットダウンに耐える。

---

## Notes

このページの作成にあたり、内容整理・文法チェック・全般的な執筆において AI ツールの支援を受けています。筆者本人の日本語能力は JLPT N3 程度であり、文法の誤りや不正確な記述が含まれる可能性があります。あらかじめご了承ください。
