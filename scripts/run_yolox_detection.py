#!/usr/bin/env python3
"""
Run YOLOX detection on MOT17 sequences and save in MOT format.

Output format (det.txt):
    frame_id, -1, bb_left, bb_top, bb_width, bb_height, confidence, -1, -1, -1
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add FastTracker to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
FASTTRACKER_DIR = PROJECT_DIR / "trackers" / "FastTracker"
sys.path.insert(0, str(FASTTRACKER_DIR))

from yolox.exp import get_exp
from yolox.utils import postprocess


def preproc(image, input_size, mean, std):
    """Preprocess image for YOLOX inference."""
    # Resize while maintaining aspect ratio
    img_h, img_w = image.shape[:2]
    target_h, target_w = input_size

    # Calculate scale
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Normalize
    padded = padded.astype(np.float32)
    padded = padded[:, :, ::-1]  # BGR to RGB
    padded /= 255.0
    padded = (padded - np.array(mean)) / np.array(std)

    # HWC to CHW
    padded = padded.transpose(2, 0, 1)
    padded = np.ascontiguousarray(padded)

    return padded, scale


class YOLOXDetector:
    """YOLOX detector for MOT17 benchmark."""

    def __init__(
        self,
        exp_file: str,
        checkpoint: str,
        device: str = "cuda",
        fp16: bool = True,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.7,
    ):
        # Device selection: cuda > mps > cpu
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device in ("cuda", "mps"):
            print(f"Warning: {device} not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # FP16 only supported on CUDA
        self.fp16 = fp16 and self.device.type == "cuda"
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load experiment config
        self.exp = get_exp(exp_file, None)
        self.exp.test_conf = conf_thresh
        self.exp.nmsthre = nms_thresh
        self.test_size = self.exp.test_size  # (608, 1088)
        self.num_classes = self.exp.num_classes

        # Build and load model
        self.model = self.exp.get_model().to(self.device)
        self.model.eval()

        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

        if self.fp16:
            self.model = self.model.half()

        print(f"Model loaded on {self.device}, FP16: {self.fp16}")
        print(f"Test size: {self.test_size}, Classes: {self.num_classes}")

        # Preprocessing params
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Run detection on a single image.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            detections: (N, 5) array of [x1, y1, x2, y2, conf]
        """
        height, width = img.shape[:2]

        # Preprocess
        img_proc, scale = preproc(img, self.test_size, self.rgb_means, self.std)
        img_tensor = torch.from_numpy(img_proc).float().unsqueeze(0).to(self.device)

        if self.fp16:
            img_tensor = img_tensor.half()

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(
                outputs, self.num_classes, self.conf_thresh, self.nms_thresh
            )

        if outputs[0] is None:
            return np.empty((0, 5), dtype=np.float32)

        output = outputs[0].cpu().numpy()

        # Scale back to original image size
        # Output format: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        boxes = output[:, :4] / scale
        scores = output[:, 4] * output[:, 5]  # obj_conf * class_conf

        # Clip to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

        return np.column_stack([boxes, scores])


def run_detection_on_sequence(
    detector: YOLOXDetector,
    seq_dir: Path,
    output_file: Path,
):
    """Run detection on a sequence and save in MOT format."""

    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        print(f"  Warning: Image directory not found: {img_dir}")
        return

    # Get sorted image list
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if not img_files:
        print(f"  Warning: No images found in {img_dir}")
        return

    results = []

    for frame_id, img_path in enumerate(tqdm(img_files, desc=seq_dir.name), start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detections = detector.detect(img)

        for det in detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1

            # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
            results.append(
                f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n"
            )

    # Write results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.writelines(results)

    print(f"  Saved {len(results)} detections to {output_file}")


def main():
    parser = argparse.ArgumentParser("Run YOLOX detection on MOT17")
    parser.add_argument(
        "--exp-file",
        type=str,
        default="trackers/FastTracker/exps/example/mot/yolox_x_mix_det.py",
        help="YOLOX experiment file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bytetrack_x_mot17.pth.tar",
        help="YOLOX checkpoint",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/MOT17/train",
        help="MOT17 train directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/data/gt/mot_challenge/MOT17-train",
        help="Output directory for detections",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device (cuda, mps, or cpu)"
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.01,
        help="Detection confidence threshold",
    )
    args = parser.parse_args()

    # Convert to absolute paths
    exp_file = str(PROJECT_DIR / args.exp_file)
    checkpoint = str(PROJECT_DIR / args.checkpoint)

    # Initialize detector
    detector = YOLOXDetector(
        exp_file=exp_file,
        checkpoint=checkpoint,
        device=args.device,
        conf_thresh=args.conf_thresh,
    )

    # Process each sequence
    dataset_dir = PROJECT_DIR / args.dataset_dir
    output_dir = PROJECT_DIR / args.output_dir

    print(f"\nProcessing sequences in: {dataset_dir}")

    for seq_dir in sorted(dataset_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name
        # Only process FRCNN sequences (we use same images, different detector)
        if "-FRCNN" not in seq_name:
            continue

        # Output sequence name with YOLOX
        new_seq_name = seq_name.replace("-FRCNN", "-YOLOX")
        output_file = output_dir / new_seq_name / "det" / "det.txt"

        print(f"\nProcessing: {seq_name} -> {new_seq_name}")
        run_detection_on_sequence(detector, seq_dir, output_file)

    print("\nDetection complete!")


if __name__ == "__main__":
    main()
