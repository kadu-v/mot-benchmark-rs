# MOT Benchmark for Rust Trackers

A benchmarking tool for Rust trackers (ByteTracker, BoostTracker, OC-SORT) using the MOT17 dataset.

## Overview

This project runs the MOT benchmark with the following pipeline:

```
MOT17 images → YOLOX detections → det.txt → Rust tracker → tracking results.txt → TrackEval
```

## Quickstart (Reproduce Results)

Use the shortest path below to reproduce the benchmark (details later).

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

See "Environment Setup" and "Benchmark Execution" for installing Rust/uv or running steps individually.

## Directory Structure

```
mot-benchmark-rs/
├── src/
│   └── mot_benchmark.rs      # Rust tracker benchmark
├── scripts/
│   ├── prepare_dataset.py    # Dataset preparation
│   ├── run_yolox_detection.py # YOLOX detection
│   ├── run_python_bytetracker.py  # Python ByteTracker (for comparison)
│   ├── run_python_boosttracker.py # Python BoostTracker (for comparison)
│   ├── evaluate.py           # TrackEval evaluation
│   ├── interpolation.py      # DTI (Disconnected Track Interpolation)
│   └── run_benchmark.sh      # Batch runner
├── trackers/                  # External trackers (git clone)
│   ├── ByteTrack/            # Official ByteTrack
│   ├── BoostTrack/           # Official BoostTrack
│   └── FastTracker/          # FastTracker (includes TrackEval)
├── datasets/
│   └── MOT17.zip             # MOT17 dataset
├── benchmark/
│   └── data/                 # Benchmark data (generated)
├── bytetrack_s_mot17.pth.tar # YOLOX model
├── Cargo.toml
└── pyproject.toml
```

## Environment Setup

### Requirements

- Python 3.11+
- Rust 1.75+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Tool Installation

```bash
# uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python dependencies
uv sync
```

### jamtrack-rs

Use the Rust tracker library [jamtrack-rs](https://github.com/kadu-v/jamtrack-rs) tag v0.3.1.

### Initialize External Trackers

```bash
# Not required if cloned with --recursive
# If you need to initialize manually:
git submodule update --init --recursive
```

### Verified Environment

- macOS 14.0+ (Apple Silicon M1/M2/M3)
- Python 3.11
- Rust 1.75+
- YOLOX-X detector (`bytetrack_x_mot17.pth.tar`)

### Optional: YOLOX-S Model (Lightweight)

```bash
wget https://github.com/ifzhang/ByteTrack/releases/download/v0.1.0/bytetrack_s_mot17.pth.tar
```

## Benchmark Execution

### Batch Run

```bash
./scripts/run_benchmark.sh
```

### Run Each Step

#### Step 1: YOLOX Detection

```bash
# YOLOX-X (default, higher accuracy)
uv run python scripts/run_yolox_detection.py

# YOLOX-S (lightweight)
uv run python scripts/run_yolox_detection.py \
    --exp-file trackers/FastTracker/exps/example/mot/yolox_s_mix_det.py \
    --checkpoint bytetrack_s_mot17.pth.tar
```

Output: `benchmark/data/gt/mot_challenge/MOT17-train/*/det/det.txt`

#### Step 2: Run Rust Trackers

```bash
# ByteTracker (tuned)
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

# BoostTracker (ECC enabled)
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

Available trackers:
- `ByteTracker` - fixed parameters
- `ByteTrackerTuned` - sequence-specific parameters
- `BoostTrack` - base BoostTrack
- `BoostTrackECC` - BoostTrack with ECC (camera motion compensation)
- `BoostTrackPlus` - BoostTrack+
- `BoostTrackPlusPlus` - BoostTrack++
- `OCSortTracker` - OC-SORT (Observation-Centric SORT)

#### Step 3: TrackEval Evaluation

```bash
# Single tracker
uv run python scripts/evaluate.py --trackers ByteTrackerTuned

# Compare multiple trackers
uv run python scripts/evaluate.py --trackers ByteTrackerTuned BoostTrack BoostTrackECC BoostTrackPlusPlus
```

## Comparison with Official Python Implementations

### Python ByteTracker

```bash
uv run python scripts/run_python_bytetracker.py --tuned
```

### Python BoostTracker

```bash
# BoostTrack
uv run python scripts/run_python_boosttracker.py --mode boost

# BoostTrack++
uv run python scripts/run_python_boosttracker.py --mode boost++
```

### Comparative Evaluation

```bash
uv run python scripts/evaluate.py \
    --trackers ByteTrackerTuned OfficialByteTrackerTuned \
               BoostTrack OfficialBoostTrack \
               BoostTrackPlusPlus OfficialBoostTrackPlusPlus
```

## Benchmark Results (MOT17-train)

### Using YOLOX-X (recommended)

| Tracker | HOTA | MOTA | IDF1 | IDSW |
|---------|------|------|------|------|
| **OfficialBoostTrack++ECC (Python)** | **69.71** | 79.92 | **79.82** | **287** |
| OfficialBoostTrackECC (Python) | 69.28 | 79.17 | 79.10 | 308 |
| **ByteTrackerTuned (Rust)** | 68.55 | **80.95** | 78.27 | 450 |
| BoostTrack++ECC (Rust) | 68.35 | 79.80 | 77.98 | 318 |
| BoostTrackECC (Rust) | 68.39 | 79.06 | 77.94 | 344 |
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
> - ECC improves HOTA/IDF1/IDSW significantly via camera motion compensation.
> - This benchmark includes Rust ECC variants (`BoostTrackECC`, `BoostTrackPlusPlusECC`).
> - Python `OfficialBoostTrack*` values in this repo are measured with `use_embedding=False` for fair non-ReID comparison.
> - MOTA is mainly determined by the core algorithm, so Rust and Python are roughly comparable.

### Using YOLOX-S (lightweight)

| Tracker | HOTA | MOTA | IDF1 |
|---------|------|------|------|
| ByteTrackerTuned (Rust) | 58.88 | 70.36 | 68.57 |
| OfficialByteTracker (Python) | 59.91 | 70.99 | 70.23 |
| BoostTrack (Rust) | 56.89 | 64.79 | 65.59 |
| OfficialBoostTrack (Python) | 58.40 | 64.80 | 67.97 |

## Metrics

- **MOTA** (Multiple Object Tracking Accuracy): overall tracking accuracy
- **HOTA** (Higher Order Tracking Accuracy): joint detection and association
- **IDF1**: harmonic mean of ID precision and recall
- **IDSW**: number of ID switches

## Rust vs Python Implementation Differences

### Implemented Features

| Feature | ByteTracker | BoostTracker | OC-SORT |
|------|:-----------:|:------------:|:-------:|
| Kalman Filter | ✅ | ✅ | ✅ |
| IoU Association | ✅ | ✅ | ✅ |
| Two-stage matching (BYTE) | ✅ | - | optional |
| VDC (Velocity Direction Consistency) | - | - | ✅ |
| OCR (Observation-Centric Re-association) | - | - | ✅ |
| Online Smoothing (KF freeze/unfreeze) | - | - | ✅ |
| DLO Boost | - | ✅ | - |
| DUO Boost | - | ✅ | - |
| Mahalanobis Distance | - | ✅ | - |
| Shape Similarity | - | ✅ | - |
| Soft BIoU | - | ✅ | - |
| ECC (camera motion compensation) | - | ✅ | - |

### Missing Features (BoostTracker)

| Feature | Description | Impact |
|------|------|------|
| **Embedding** | Re-ID features (CNN) | lower HOTA/IDF1 |

Because embedding is not used in this benchmark setup, there is still a small HOTA/IDF1 gap vs Python. MOTA is mostly determined by the core algorithm, so the values are roughly comparable.

## Troubleshooting

### NumPy Compatibility Error

If ByteTrack raises a `np.float` error:

```bash
# Replace in trackers/ByteTrack/yolox/tracker/*.py
# np.float → np.float64
sed -i '' 's/np\.float\b/np.float64/g' trackers/ByteTrack/yolox/tracker/*.py
```

### Device Selection

```bash
# Apple Silicon (MPS) - default
uv run python scripts/run_yolox_detection.py --device mps

# NVIDIA GPU (CUDA)
uv run python scripts/run_yolox_detection.py --device cuda

# CPU
uv run python scripts/run_yolox_detection.py --device cpu
```

## License

MIT License

## References

- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [MOTChallenge](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
