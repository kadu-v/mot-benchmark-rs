#!/usr/bin/env python3
"""
Run Python ByteTracker on detection results for comparison with Rust implementation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
# Use official ByteTrack repository
BYTETRACK_DIR = PROJECT_DIR / "trackers" / "ByteTrack"
sys.path.insert(0, str(BYTETRACK_DIR))

from yolox.tracker.byte_tracker import BYTETracker


class Args:
    """Arguments for BYTETracker."""
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.9, min_box_area=100):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.mot20 = False


def get_seq_params(seq_name):
    """Get sequence-specific parameters (like official ByteTrack mot_evaluator.py)."""
    # Extract base sequence name (e.g., "MOT17-05" from "MOT17-05-YOLOX")
    parts = seq_name.split('-')
    base_name = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else seq_name

    # Sequence-specific track_buffer
    if base_name in ['MOT17-05', 'MOT17-06']:
        track_buffer = 14
    elif base_name in ['MOT17-13', 'MOT17-14']:
        track_buffer = 25
    else:
        track_buffer = 30

    # Sequence-specific track_thresh
    if base_name == 'MOT17-01':
        track_thresh = 0.65
    elif base_name == 'MOT17-06':
        track_thresh = 0.65
    elif base_name == 'MOT17-12':
        track_thresh = 0.7
    elif base_name == 'MOT17-14':
        track_thresh = 0.67
    else:
        track_thresh = 0.6

    return track_thresh, track_buffer


def filter_track(tlwh, min_box_area=100):
    """Filter track like official ByteTrack (mot_evaluator.py line 197-198)."""
    w, h = tlwh[2], tlwh[3]
    area = w * h
    # vertical means width/height > 1.6 (wider than tall - actually horizontal)
    vertical = w / h > 1.6 if h > 0 else True
    return area > min_box_area and not vertical


def load_detections(det_file):
    """Load detections from MOT format file."""
    detections = {}
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])

            if frame not in detections:
                detections[frame] = []
            # Convert to x1, y1, x2, y2, conf format
            detections[frame].append([x, y, x + w, y + h, conf])

    return detections


def load_seqinfo(seq_dir):
    """Load sequence info from seqinfo.ini."""
    seqinfo_path = seq_dir / "seqinfo.ini"
    info = {}
    with open(seqinfo_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                info[key.strip()] = value.strip()
    return info


def run_tracker_on_sequence(seq_dir, output_file, args, use_tuned_params=False):
    """Run Python ByteTracker on a sequence."""
    # Load sequence info
    seqinfo = load_seqinfo(seq_dir)
    frame_rate = int(seqinfo.get('frameRate', 30))
    seq_length = int(seqinfo.get('seqLength', 0))
    img_width = int(seqinfo.get('imWidth', 1920))
    img_height = int(seqinfo.get('imHeight', 1080))

    # Load detections
    det_file = seq_dir / "det" / "det.txt"
    detections = load_detections(det_file)

    # Get sequence-specific parameters if tuned mode
    seq_name = seq_dir.name
    if use_tuned_params:
        track_thresh, track_buffer = get_seq_params(seq_name)
        tracker_args = Args(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=args.match_thresh,
            min_box_area=args.min_box_area,
        )
    else:
        tracker_args = args

    # Initialize tracker
    tracker = BYTETracker(tracker_args, frame_rate=frame_rate)

    results = []

    for frame_id in tqdm(range(1, seq_length + 1), desc=seq_dir.name):
        dets = detections.get(frame_id, [])

        if len(dets) > 0:
            dets_array = np.array(dets, dtype=np.float32)
            # BYTETracker expects [x1, y1, x2, y2, score] format
            online_targets = tracker.update(dets_array, [img_height, img_width], [img_height, img_width])
        else:
            online_targets = tracker.update(np.empty((0, 5), dtype=np.float32), [img_height, img_width], [img_height, img_width])

        for track in online_targets:
            tlwh = track.tlwh
            track_id = track.track_id
            score = track.score

            # Filter like official ByteTrack (min_box_area and aspect ratio)
            if not filter_track(tlwh, args.min_box_area):
                continue

            # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
            results.append(f"{frame_id},{track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.6f},-1,-1,-1\n")

    # Write results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(results)

    print(f"  Saved {len(results)} tracks to {output_file}")


def main():
    parser = argparse.ArgumentParser("Run Python ByteTracker")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmark/data/gt/mot_challenge/MOT17-train",
        help="Input directory with sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/data/trackers/mot_challenge/MOT17-train/PythonByteTracker/data",
        help="Output directory for tracking results",
    )
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100, help="Minimum box area to keep")
    parser.add_argument("--tuned", action="store_true", help="Use sequence-specific parameters")
    args = parser.parse_args()

    input_dir = PROJECT_DIR / args.input_dir
    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_args = Args(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_box_area=args.min_box_area,
    )

    print(f"=== Python ByteTracker ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"track_thresh: {args.track_thresh}, match_thresh: {args.match_thresh}")
    print(f"min_box_area: {args.min_box_area}, tuned: {args.tuned}")
    print()

    for seq_dir in sorted(input_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "det" / "det.txt").exists():
            continue

        output_file = output_dir / f"{seq_dir.name}.txt"
        print(f"Processing: {seq_dir.name}")
        run_tracker_on_sequence(seq_dir, output_file, tracker_args, use_tuned_params=args.tuned)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
