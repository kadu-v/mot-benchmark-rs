#!/bin/bash
set -e

# Configuration
BENCHMARK="MOT17"
SPLIT="train"
TRACKER="ByteTracker"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== MOT17 Benchmark Pipeline ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Prepare dataset
echo "[1/4] Preparing dataset..."
uv run scripts/prepare_dataset.py
echo ""

# Step 2: Run YOLOX detection
echo "[2/4] Running YOLOX detection..."
uv run scripts/run_yolox_detection.py \
    --exp-file FastTracker/exps/example/mot/yolox_s_mix_det.py \
    --checkpoint bytetrack_s_mot17.pth.tar \
    --dataset-dir datasets/MOT17/train \
    --output-dir benchmark/data/gt/mot_challenge/MOT17-train \
    --device cpu
echo ""

# Step 3: Run Rust ByteTracker
echo "[3/4] Running Rust ByteTracker..."
cargo run --release --bin mot_benchmark -- \
    --data-dir ./benchmark/data \
    --output-dir ./benchmark/data \
    --benchmark $BENCHMARK \
    --split $SPLIT \
    --tracker $TRACKER
echo ""

# Step 4: Run interpolation (DTI)
echo "[4/5] Running interpolation (DTI)..."
uv run scripts/interpolation.py \
    --input-dir benchmark/data/trackers/mot_challenge/MOT17-train/$TRACKER/data \
    --n-min 25 \
    --n-dti 20
echo ""

# Step 5: Run TrackEval
echo "[5/5] Running TrackEval..."
uv run scripts/evaluate.py \
    --gt-folder benchmark/data/gt/mot_challenge \
    --trackers-folder benchmark/data/trackers/mot_challenge \
    --benchmark $BENCHMARK \
    --split $SPLIT \
    --trackers $TRACKER

echo ""
echo "=== Benchmark Complete ==="
