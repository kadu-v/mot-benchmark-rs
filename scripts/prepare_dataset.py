#!/usr/bin/env python3
"""
Prepare MOT17 dataset for benchmark.
Extracts MOT17.zip and creates TrackEval-compatible directory structure.
"""
import os
import shutil
import zipfile
from pathlib import Path


def prepare_mot17_dataset(
    zip_path: str = "datasets/MOT17.zip",
    extract_dir: str = "datasets/MOT17",
    benchmark_dir: str = "benchmark/data",
):
    """Extract and prepare MOT17 dataset."""

    # 1. Extract ZIP if not already extracted
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("datasets/")
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_dir}")

    # 2. Create TrackEval directory structure
    gt_dir = Path(benchmark_dir) / "gt" / "mot_challenge" / "MOT17-train"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create seqmaps directory
    seqmaps_dir = Path(benchmark_dir) / "gt" / "mot_challenge" / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)

    # 4. Process each sequence
    sequences = []
    train_dir = Path(extract_dir) / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    for seq_dir in sorted(train_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name
        # Only process FRCNN detector sequences (we'll use YOLOX detections)
        if "-FRCNN" not in seq_name:
            continue

        # Create new sequence name with YOLOX suffix
        new_seq_name = seq_name.replace("-FRCNN", "-YOLOX")
        sequences.append(new_seq_name)

        target_seq_dir = gt_dir / new_seq_name
        target_seq_dir.mkdir(parents=True, exist_ok=True)

        # Copy seqinfo.ini (update name field)
        seqinfo_src = seq_dir / "seqinfo.ini"
        seqinfo_dst = target_seq_dir / "seqinfo.ini"
        if seqinfo_src.exists():
            with open(seqinfo_src, "r") as f:
                content = f.read()
            content = content.replace(seq_name, new_seq_name)
            with open(seqinfo_dst, "w") as f:
                f.write(content)

        # Copy ground truth
        gt_src = seq_dir / "gt" / "gt.txt"
        gt_dst_dir = target_seq_dir / "gt"
        gt_dst_dir.mkdir(exist_ok=True)
        if gt_src.exists():
            shutil.copy(gt_src, gt_dst_dir / "gt.txt")

        # Create det directory (will be populated by YOLOX)
        det_dir = target_seq_dir / "det"
        det_dir.mkdir(exist_ok=True)

        print(f"  Prepared: {new_seq_name}")

    # 5. Write seqmap file
    seqmap_file = seqmaps_dir / "MOT17-train.txt"
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")

    print(f"\nPrepared {len(sequences)} sequences")
    print(f"Seqmap written to: {seqmap_file}")
    print(f"\nDirectory structure created at: {gt_dir}")


if __name__ == "__main__":
    prepare_mot17_dataset()
