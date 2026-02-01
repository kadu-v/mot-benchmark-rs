#!/usr/bin/env python3
"""
Disconnected Track Interpolation (DTI) for MOT results.
Interpolates missing frames in tracking results to improve MOTA.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results(filename, results):
    """Write results in MOT format."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    with open(filename, "w") as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(
                frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=score
            )
            f.write(line)


def dti(txt_path, save_path, n_min=25, n_dti=20):
    """
    Disconnected Track Interpolation.

    Args:
        txt_path: Input tracking results directory
        save_path: Output directory for interpolated results
        n_min: Minimum track length to consider for interpolation
        n_dti: Maximum gap to interpolate (frames)
    """
    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))

    for seq_txt in seq_txts:
        seq_name = os.path.basename(seq_txt)
        print(f"  Processing: {seq_name}")

        try:
            seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        except Exception as e:
            print(f"    Warning: Could not load {seq_txt}: {e}")
            continue

        if seq_data.size == 0:
            print(f"    Warning: Empty file {seq_txt}")
            continue

        if seq_data.ndim == 1:
            seq_data = seq_data.reshape(1, -1)

        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)

        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet

            if tracklet.shape[0] == 0:
                continue

            n_frame = tracklet.shape[0]

            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}

                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]

                    # Disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]

                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (
                                right_bbox - left_bbox
                            ) / (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox

                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n, frame_key in enumerate(frames_dti.keys()):
                        data_dti[n, 0] = frame_key
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[frame_key]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))

            seq_results = np.vstack((seq_results, tracklet_dti))

        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results(save_seq_txt, seq_results)

        print(f"    Interpolated {seq_name}: {len(seq_data)} -> {len(seq_results)} detections")


def main():
    parser = argparse.ArgumentParser("Disconnected Track Interpolation")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmark/data/trackers/mot_challenge/MOT17-train/ByteTracker/data",
        help="Input tracking results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite input)",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=25,
        help="Minimum track length for interpolation",
    )
    parser.add_argument(
        "--n-dti",
        type=int,
        default=20,
        help="Maximum gap to interpolate (frames)",
    )
    args = parser.parse_args()

    input_dir = str(PROJECT_DIR / args.input_dir)

    if args.output_dir is None:
        output_dir = input_dir
    else:
        output_dir = str(PROJECT_DIR / args.output_dir)

    mkdir_if_missing(output_dir)

    print(f"=== Disconnected Track Interpolation ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"n_min: {args.n_min}, n_dti: {args.n_dti}")
    print()

    dti(input_dir, output_dir, args.n_min, args.n_dti)

    print("\n=== Interpolation Complete ===")


if __name__ == "__main__":
    main()
