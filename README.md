# MOT Benchmark for Rust Trackers

MOT17データセットを使用して、Rust実装のトラッカー（ByteTracker, BoostTracker）をベンチマークするためのツールです。

## クイックスタート（結果の再現）

以下の手順でベンチマーク結果を再現できます。

### 1. リポジトリのクローン

```bash
git clone --recursive https://github.com/<username>/mot-benchmark-rs.git
cd mot-benchmark-rs
```

### 2. 依存関係のインストール

```bash
# Python依存関係
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Rust (未インストールの場合)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 3. データセットとモデルの準備

```bash
# MOT17データセットのダウンロード
mkdir -p datasets
cd datasets
wget https://motchallenge.net/data/MOT17.zip
cd ..

# YOLOXモデルのダウンロード
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_x_mot17.pth.tar

# データセット展開
uv run python scripts/prepare_dataset.py
```

### 4. ベンチマーク実行

```bash
# Step 1: YOLOX検出
uv run python scripts/run_yolox_detection.py

# Step 2: Rustトラッカー実行
cargo run --release --bin mot_benchmark -- --tracker all

# Step 3: Python公式トラッカー実行 (比較用)
uv run python scripts/run_python_bytetracker.py
uv run python scripts/run_python_bytetracker.py --tuned
uv run python scripts/run_python_boosttracker.py --mode boost
uv run python scripts/run_python_boosttracker.py --mode boost++
uv run python scripts/run_python_boosttracker.py --mode boost --use-ecc
uv run python scripts/run_python_boosttracker.py --mode boost++ --use-ecc

# Step 4: 評価
uv run python scripts/evaluate.py --trackers \
    ByteTracker ByteTrackerTuned \
    BoostTrack BoostTrackPlusPlus \
    OfficialByteTracker OfficialByteTrackerTuned \
    OfficialBoostTrack OfficialBoostTrackPlusPlus \
    OfficialBoostTrackECC OfficialBoostTrackPlusPlusECC
```

### 5. jamtrack-rsの準備

Rustトラッカーライブラリ [jamtrack-rs](https://github.com/<username>/jamtrack-rs) が必要です:

```bash
# mot-benchmark-rs と同じ親ディレクトリにクローン
cd ..
git clone https://github.com/<username>/jamtrack-rs.git
cd mot-benchmark-rs
```

### 動作確認環境

- macOS 14.0+ (Apple Silicon M1/M2/M3)
- Python 3.11
- Rust 1.75+
- YOLOX-X detector (`bytetrack_x_mot17.pth.tar`)

## 概要

このプロジェクトは以下のパイプラインでMOTベンチマークを実行します：

```
MOT17画像 → YOLOX検出 → det.txt → Rust Tracker → 追跡結果.txt → TrackEval評価
```

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

### 1. Python依存関係のインストール

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python依存関係のインストール
uv sync
```

### 2. 外部トラッカーの初期化

```bash
# submoduleとして含まれているため、--recursive でcloneした場合は不要
# 個別に初期化する場合:
git submodule update --init --recursive
```

### 3. YOLOXモデルのダウンロード

```bash
# YOLOX-X モデル (推奨、高精度)
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_x_mot17.pth.tar

# YOLOX-S モデル (軽量版)
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_s_mot17.pth.tar
```

### 4. MOT17データセットの準備

```bash
# datasets/ フォルダにMOT17.zipを配置
mkdir -p datasets
# MOT17.zip を datasets/ にダウンロード

# データセット展開 & TrackEval用ディレクトリ構造作成
uv run python scripts/prepare_dataset.py
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
- `BoostTrackPlus` - BoostTrack+
- `BoostTrackPlusPlus` - BoostTrack++

#### Step 3: TrackEval評価

```bash
# 単一トラッカー評価
uv run python scripts/evaluate.py --trackers ByteTrackerTuned

# 複数トラッカー比較
uv run python scripts/evaluate.py --trackers ByteTrackerTuned BoostTrack BoostTrackPlusPlus
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
| OfficialByteTrackerTuned (Python) | 67.92 | 80.90 | 77.47 | 453 |
| OfficialBoostTrack++ (Python) | 67.87 | 78.89 | 76.91 | 515 |
| OfficialBoostTrack (Python) | 67.30 | 78.26 | 76.00 | 520 |
| BoostTrack (Rust) | 66.18 | 78.20 | 74.27 | 539 |
| BoostTrack++ (Rust) | 66.11 | 78.81 | 74.21 | 570 |
| OfficialByteTracker (Python) | 59.73 | 70.31 | 69.94 | 483 |
| ByteTracker (Rust) | 58.98 | 69.91 | 68.68 | 503 |
| BoostTrackPlus (Rust) | 56.62 | 66.47 | 65.40 | 774 |

> **Note**:
> - ECC版はカメラモーション補正により、特にHOTA/IDF1/IDSWで大きな改善
> - Rust版BoostTrackはECC（カメラモーション補正）とEmbedding（Re-ID特徴量）が未実装のため、HOTA/IDF1が若干低い
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

| 機能 | ByteTracker | BoostTracker |
|------|:-----------:|:------------:|
| Kalman Filter | ✅ | ✅ |
| IoU Association | ✅ | ✅ |
| 2段階マッチング | ✅ | - |
| DLO Boost | - | ✅ |
| DUO Boost | - | ✅ |
| Mahalanobis距離 | - | ✅ |
| Shape Similarity | - | ✅ |
| Soft BIoU | - | ✅ |

### 未実装機能 (BoostTracker)

| 機能 | 説明 | 影響 |
|------|------|------|
| **ECC** | カメラモーション補正 | 動くカメラでのIDSW増加 |
| **Embedding** | Re-ID特徴量 (CNN) | HOTA/IDF1の低下 |

これらの機能が未実装のため、Rust版BoostTrackのHOTA/IDF1は公式Python版より若干低くなります。MOTAはコアアルゴリズムで決まるため、ほぼ同等の値を達成しています。

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
- [MOTChallenge](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
