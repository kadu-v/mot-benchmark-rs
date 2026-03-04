# MOT Benchmark for Rust Trackers

MOT17データセットを使用して、Rust実装のトラッカー（ByteTracker, BoostTracker, OC-SORT）をベンチマークするためのツールです。

## 概要

このプロジェクトは以下のパイプラインでMOTベンチマークを実行します：

```
MOT17画像 → YOLOX検出 → det.txt → Rust Tracker → 追跡結果.txt → TrackEval評価
```

## クイックスタート（結果の再現）

以下の最短手順でベンチマーク結果を再現できます（詳細は後述）。

```bash
git clone --recursive https://github.com/<username>/mot-benchmark-rs.git
cd mot-benchmark-rs

uv sync

mkdir -p datasets
wget https://motchallenge.net/data/MOT17.zip -O datasets/MOT17.zip
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_x_mot17.pth.tar
uv run python scripts/prepare_dataset.py

./scripts/run_benchmark.sh
```

※ Rust/uv の導入や個別実行は「環境構築」「ベンチマーク実行」を参照してください。

## ディレクトリ構成

```
mot-benchmark-rs/
├── src/
│   └── mot_benchmark.rs      # Rustトラッカーベンチマーク
├── scripts/
│   ├── prepare_dataset.py    # データセット準備
│   ├── run_yolox_detection.py # YOLOX検出実行
│   ├── run_python_bytetracker.py  # Python ByteTracker (比較用)
│   ├── run_python_boosttracker.py # Python BoostTracker (比較用)
│   ├── evaluate.py           # TrackEval評価
│   ├── interpolation.py      # DTI (Disconnected Track Interpolation)
│   └── run_benchmark.sh      # 一括実行スクリプト
├── trackers/                  # 外部トラッカー (git clone)
│   ├── ByteTrack/            # 公式ByteTrack
│   ├── BoostTrack/           # 公式BoostTrack
│   └── FastTracker/          # FastTracker (TrackEval含む)
├── datasets/
│   └── MOT17.zip             # MOT17データセット
├── benchmark/
│   └── data/                 # ベンチマーク用データ (自動生成)
├── bytetrack_s_mot17.pth.tar # YOLOXモデル
├── Cargo.toml
└── pyproject.toml
```

## 環境構築

### 必要条件

- Python 3.11+
- Rust 1.75+
- [uv](https://github.com/astral-sh/uv) (Python パッケージマネージャー)

### ツールのインストール

```bash
# uv (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust (未インストールの場合)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python依存関係
uv sync
```

### jamtrack-rs の準備

Rustトラッカーライブラリ [jamtrack-rs](https://github.com/kadu-v/jamtrack-rs) の v0.3.1 tag を使用します。

### 外部トラッカーの初期化

```bash
# submoduleとして含まれているため、--recursive でcloneした場合は不要
# 個別に初期化する場合:
git submodule update --init --recursive
```

### 動作確認環境

- macOS 14.0+ (Apple Silicon M1/M2/M3)
- Python 3.11
- Rust 1.75+
- YOLOX-X detector (`bytetrack_x_mot17.pth.tar`)

### YOLOX-S モデル (軽量版) 追加で使う場合

```bash
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_s_mot17.pth.tar
```

## ベンチマーク実行

### 一括実行

```bash
./scripts/run_benchmark.sh
```

### 個別実行

#### Step 1: YOLOX検出

```bash
# YOLOX-X (デフォルト、高精度)
uv run python scripts/run_yolox_detection.py

# YOLOX-S (軽量版)
uv run python scripts/run_yolox_detection.py \
    --exp-file trackers/FastTracker/exps/example/mot/yolox_s_mix_det.py \
    --checkpoint bytetrack_s_mot17.pth.tar
```

出力: `benchmark/data/gt/mot_challenge/MOT17-train/*/det/det.txt`

#### Step 2: Rustトラッカー実行

```bash
# ByteTracker (チューニングあり)
cargo run --release --bin mot_benchmark -- \
    --data-dir ./benchmark/data \
    --output-dir ./benchmark/data \
    --benchmark MOT17 \
    --split train \
    --tracker ByteTrackerTuned

# BoostTracker
cargo run --release --bin mot_benchmark -- \
    --data-dir ./benchmark/data \
    --output-dir ./benchmark/data \
    --benchmark MOT17 \
    --split train \
    --tracker BoostTrack

# BoostTracker (ECC有効)
cargo run --release --bin mot_benchmark -- \
    --data-dir ./benchmark/data \
    --output-dir ./benchmark/data \
    --benchmark MOT17 \
    --split train \
    --tracker BoostTrackECC

# BoostTrack++
cargo run --release --bin mot_benchmark -- \
    --data-dir ./benchmark/data \
    --output-dir ./benchmark/data \
    --benchmark MOT17 \
    --split train \
    --tracker BoostTrackPlusPlus
```

利用可能なトラッカー:
- `ByteTracker` - 固定パラメータ
- `ByteTrackerTuned` - シーケンス固有パラメータ
- `BoostTrack` - 基本BoostTrack
- `BoostTrackECC` - ECC付きBoostTrack（カメラモーション補正）
- `BoostTrackPlus` - BoostTrack+
- `BoostTrackPlusPlus` - BoostTrack++
- `OCSortTracker` - OC-SORT（Observation-Centric SORT）

#### Step 3: TrackEval評価

```bash
# 単一トラッカー評価
uv run python scripts/evaluate.py --trackers ByteTrackerTuned

# 複数トラッカー比較
uv run python scripts/evaluate.py --trackers ByteTrackerTuned BoostTrack BoostTrackECC BoostTrackPlusPlus
```

## Python公式実装との比較

### Python ByteTracker実行

```bash
uv run python scripts/run_python_bytetracker.py --tuned
```

### Python BoostTracker実行

```bash
# BoostTrack
uv run python scripts/run_python_boosttracker.py --mode boost

# BoostTrack++
uv run python scripts/run_python_boosttracker.py --mode boost++
```

### 比較評価

```bash
uv run python scripts/evaluate.py \
    --trackers ByteTrackerTuned OfficialByteTrackerTuned \
               BoostTrack OfficialBoostTrack \
               BoostTrackPlusPlus OfficialBoostTrackPlusPlus
```

## ベンチマーク結果 (MOT17-train)

### YOLOX-X 検出器使用 (推奨)

| Tracker | HOTA | MOTA | IDF1 | IDSW |
|---------|------|------|------|------|
| **OfficialBoostTrack++ECC (Python)** | **69.71** | 79.92 | **79.82** | **287** |
| OfficialBoostTrackECC (Python) | 69.28 | 79.17 | 79.10 | 308 |
| **ByteTrackerTuned (Rust)** | 68.55 | **80.95** | 78.27 | 450 |
| BoostTrackECC (Rust) | 68.38 | 79.11 | 77.71 | 349 |
| ByteTracker (Rust) | 68.35 | 80.97 | 77.89 | 454 |
| OfficialByteTrackerTuned (Python) | 67.92 | 80.90 | 77.47 | 453 |
| OfficialBoostTrack++ (Python) | 67.87 | 78.89 | 76.91 | 515 |
| OCSortTracker (Rust) | 67.73 | 78.55 | 76.67 | 484 |
| OfficialBoostTrack (Python) | 67.30 | 78.26 | 76.00 | 520 |
| OfficialByteTracker (Python) | 67.82 | 80.92 | 77.29 | 458 |
| BoostTrack++ (Rust) | 66.02 | 78.86 | 74.29 | 558 |
| BoostTrack (Rust) | 66.03 | 78.24 | 74.13 | 536 |
| BoostTrackPlus (Rust) | 65.93 | 78.57 | 74.11 | 560 |

> **Note**:
> - ECC版はカメラモーション補正により、特にHOTA/IDF1/IDSWで大きな改善
> - このベンチマークには Rust `BoostTrackECC`（ECC有効）を含む
> - このリポジトリの Python `OfficialBoostTrack*` は `use_embedding=False`（Re-ID無効）で計測
> - MOTAはコアアルゴリズムで決まるため、Rust版とPython版でほぼ同等

### YOLOX-S 検出器使用 (軽量版)

| Tracker | HOTA | MOTA | IDF1 |
|---------|------|------|------|
| ByteTrackerTuned (Rust) | 58.88 | 70.36 | 68.57 |
| OfficialByteTracker (Python) | 59.91 | 70.99 | 70.23 |
| BoostTrack (Rust) | 56.89 | 64.79 | 65.59 |
| OfficialBoostTrack (Python) | 58.40 | 64.80 | 67.97 |

## 評価メトリクス

- **MOTA** (Multiple Object Tracking Accuracy): 追跡精度の総合指標
- **HOTA** (Higher Order Tracking Accuracy): 検出とアソシエーションの統合評価
- **IDF1**: ID保持率の調和平均
- **IDSW**: IDスイッチ回数

## Rust vs Python 実装の違い

### 実装済み機能

| 機能 | ByteTracker | BoostTracker | OC-SORT |
|------|:-----------:|:------------:|:-------:|
| Kalman Filter | ✅ | ✅ | ✅ |
| IoU Association | ✅ | ✅ | ✅ |
| 2段階マッチング (BYTE) | ✅ | - | optional |
| VDC（速度方向一貫性） | - | - | ✅ |
| OCR（Observation-Centric再関連付け） | - | - | ✅ |
| Online Smoothing（KF凍結/解凍） | - | - | ✅ |
| DLO Boost | - | ✅ | - |
| DUO Boost | - | ✅ | - |
| Mahalanobis距離 | - | ✅ | - |
| Shape Similarity | - | ✅ | - |
| Soft BIoU | - | ✅ | - |
| ECC（カメラモーション補正） | - | ✅ | - |

### 未実装機能 (BoostTracker)

| 機能 | 説明 | 影響 |
|------|------|------|
| **Embedding** | Re-ID特徴量 (CNN) | HOTA/IDF1の低下 |

このベンチマーク設定ではEmbeddingを使っていないため、HOTA/IDF1にはPython版との差が少し残ります。MOTAはコアアルゴリズムで決まるため、ほぼ同等の値です。

## トラブルシューティング

### NumPy互換性エラー

ByteTrackで `np.float` エラーが発生する場合:

```bash
# trackers/ByteTrack/yolox/tracker/*.py で
# np.float → np.float64 に置換
sed -i '' 's/np\.float\b/np.float64/g' trackers/ByteTrack/yolox/tracker/*.py
```

### デバイス選択

```bash
# Apple Silicon (MPS) - デフォルト
uv run python scripts/run_yolox_detection.py --device mps

# NVIDIA GPU (CUDA)
uv run python scripts/run_yolox_detection.py --device cuda

# CPU
uv run python scripts/run_yolox_detection.py --device cpu
```

## ライセンス

MIT License

## 参考文献

- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [MOTChallenge](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
