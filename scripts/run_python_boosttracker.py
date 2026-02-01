#!/usr/bin/env python3
"""
Run official Python BoostTracker on detection results for comparison with Rust implementation.
"""
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
# Use official BoostTrack repository
BOOSTTRACK_DIR = PROJECT_DIR / "trackers" / "BoostTrack"
sys.path.insert(0, str(BOOSTTRACK_DIR))

# Mock torchreid and other optional dependencies before importing BoostTrack
sys.modules['torchreid'] = MagicMock()
sys.modules['external'] = MagicMock()
sys.modules['external.adaptors'] = MagicMock()
sys.modules['external.adaptors.fastreid_adaptor'] = MagicMock()


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


def filter_track(tlwh, min_box_area=10, aspect_ratio_thresh=1.6):
    """Filter track like official BoostTrack."""
    w, h = tlwh[2], tlwh[3]
    area = w * h
    aspect_ratio = w / h if h > 0 else 0
    return area > min_box_area and aspect_ratio <= aspect_ratio_thresh


def get_image_path(seq_dir, frame_id, seqinfo):
    """Get image path for a frame."""
    img_dir = seqinfo.get('imDir', 'img1')
    img_ext = seqinfo.get('imExt', '.jpg')
    seq_name = seq_dir.name

    # For YOLOX sequences, look for the original FRCNN sequence images
    original_seq_name = seq_name.replace('-YOLOX', '-FRCNN')

    # Check various possible locations
    possible_paths = [
        seq_dir / img_dir / f"{frame_id:06d}{img_ext}",
        PROJECT_DIR / "datasets" / "MOT17" / "train" / original_seq_name / img_dir / f"{frame_id:06d}{img_ext}",
        PROJECT_DIR / "datasets" / "MOT17" / "train" / seq_name / img_dir / f"{frame_id:06d}{img_ext}",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def run_tracker_on_sequence(seq_dir, output_file, tracker_mode='boost', use_ecc=False):
    """Run Python BoostTracker on a sequence."""
    # Import here to ensure settings are applied
    import default_settings
    from tracker.boost_track import BoostTrack, KalmanBoxTracker

    # Load sequence info
    seqinfo = load_seqinfo(seq_dir)
    seq_length = int(seqinfo.get('seqLength', 0))
    img_width = int(seqinfo.get('imWidth', 1920))
    img_height = int(seqinfo.get('imHeight', 1080))
    seq_name = seq_dir.name

    # Load detections
    det_file = seq_dir / "det" / "det.txt"
    detections = load_detections(det_file)

    # Reset track ID counter
    KalmanBoxTracker.count = 0

    # Configure tracker mode
    if tracker_mode == 'boost':
        # BoostTrack basic: use_rich_s=False, use_sb=False, use_vt=False
        default_settings.BoostTrackPlusPlusSettings.values['use_rich_s'] = False
        default_settings.BoostTrackPlusPlusSettings.values['use_sb'] = False
        default_settings.BoostTrackPlusPlusSettings.values['use_vt'] = False
    elif tracker_mode == 'boost+':
        # BoostTrack+: use_rich_s=True, use_sb=False, use_vt=False
        default_settings.BoostTrackPlusPlusSettings.values['use_rich_s'] = True
        default_settings.BoostTrackPlusPlusSettings.values['use_sb'] = False
        default_settings.BoostTrackPlusPlusSettings.values['use_vt'] = False
    elif tracker_mode == 'boost++':
        # BoostTrack++: use_rich_s=True, use_sb=True, use_vt=True
        default_settings.BoostTrackPlusPlusSettings.values['use_rich_s'] = True
        default_settings.BoostTrackPlusPlusSettings.values['use_sb'] = True
        default_settings.BoostTrackPlusPlusSettings.values['use_vt'] = True

    # Initialize tracker with video_name for ECC cache
    video_name = seq_name if use_ecc else None
    tracker = BoostTrack(video_name=video_name)

    results = []

    # Create dummy tensors for img_tensor
    dummy_tensor = np.zeros((1, 3, img_height, img_width))

    for frame_id in tqdm(range(1, seq_length + 1), desc=seq_dir.name):
        dets = detections.get(frame_id, [])

        if len(dets) > 0:
            dets_array = np.array(dets, dtype=np.float32)
        else:
            dets_array = np.empty((0, 5), dtype=np.float32)

        # Load image if ECC is enabled
        if use_ecc:
            img_path = get_image_path(seq_dir, frame_id, seqinfo)
            if img_path is not None and img_path.exists():
                img_numpy = cv2.imread(str(img_path))
            else:
                # Fallback to dummy image
                img_numpy = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        else:
            img_numpy = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Update tracker
        online_targets = tracker.update(dets_array, dummy_tensor, img_numpy, seq_name)

        if online_targets is not None and len(online_targets) > 0:
            for target in online_targets:
                x1, y1, x2, y2, track_id = target[:5]
                conf = target[5] if len(target) > 5 else 1.0

                # Convert to tlwh format
                w = x2 - x1
                h = y2 - y1

                # Filter like official BoostTrack
                if not filter_track([x1, y1, w, h]):
                    continue

                # MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
                results.append(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")

    # Save ECC cache if enabled
    if use_ecc:
        tracker.dump_cache()

    # Write results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(results)

    print(f"  Saved {len(results)} tracks to {output_file}")


def main():
    parser = argparse.ArgumentParser("Run Python BoostTracker")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmark/data/gt/mot_challenge/MOT17-train",
        help="Input directory with sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/data/trackers/mot_challenge/MOT17-train/OfficialBoostTrack/data",
        help="Output directory for tracking results",
    )
    parser.add_argument("--mode", type=str, default="boost", choices=["boost", "boost+", "boost++"],
                        help="Tracker mode: boost, boost+, or boost++")
    parser.add_argument("--use-ecc", action="store_true",
                        help="Enable ECC camera motion compensation")
    args = parser.parse_args()

    # Configure settings BEFORE importing BoostTrack modules
    import default_settings
    default_settings.GeneralSettings.values['use_ecc'] = args.use_ecc
    default_settings.GeneralSettings.values['use_embedding'] = False

    input_dir = PROJECT_DIR / args.input_dir
    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ecc_status = "with ECC" if args.use_ecc else "without ECC"
    print(f"=== Python BoostTracker ({args.mode}) {ecc_status} ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Create cache directory for ECC
    if args.use_ecc:
        (BOOSTTRACK_DIR / "cache").mkdir(exist_ok=True)

    for seq_dir in sorted(input_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        if not (seq_dir / "det" / "det.txt").exists():
            continue

        output_file = output_dir / f"{seq_dir.name}.txt"
        print(f"Processing: {seq_dir.name}")
        run_tracker_on_sequence(seq_dir, output_file, tracker_mode=args.mode, use_ecc=args.use_ecc)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
