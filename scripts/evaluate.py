#!/usr/bin/env python3
"""
Run TrackEval on benchmark results.
Evaluates tracking results using HOTA, CLEAR, and Identity metrics.
"""
import argparse
from pathlib import Path

import trackeval

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent


def run_evaluation(
    gt_folder: str = "benchmark/data/gt/mot_challenge",
    trackers_folder: str = "benchmark/data/trackers/mot_challenge",
    benchmark: str = "MOT17",
    split: str = "train",
    trackers: list = None,
):
    """Run TrackEval on tracking results."""

    # Convert to absolute paths
    gt_folder = str(PROJECT_DIR / gt_folder)
    trackers_folder = str(PROJECT_DIR / trackers_folder)

    print("=== TrackEval Evaluation ===")
    print(f"GT folder: {gt_folder}")
    print(f"Trackers folder: {trackers_folder}")
    print(f"Benchmark: {benchmark}-{split}")
    print(f"Trackers: {trackers if trackers else 'all'}")
    print()

    # Evaluation config
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": True,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": True,
    })

    # Dataset config
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": gt_folder,
        "TRACKERS_FOLDER": trackers_folder,
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": trackers,
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": True,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None,
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL": False,
    })

    # Metrics config
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

    # Run evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        metrics_list.append(metric(metrics_config))

    evaluator.evaluate(dataset_list, metrics_list)

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run TrackEval")
    parser.add_argument(
        "--gt-folder",
        type=str,
        default="benchmark/data/gt/mot_challenge",
        help="Ground truth folder",
    )
    parser.add_argument(
        "--trackers-folder",
        type=str,
        default="benchmark/data/trackers/mot_challenge",
        help="Trackers results folder",
    )
    parser.add_argument(
        "--benchmark", type=str, default="MOT17", help="Benchmark name"
    )
    parser.add_argument("--split", type=str, default="train", help="Split to evaluate")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=None,
        help="Trackers to evaluate (default: all)",
    )
    args = parser.parse_args()

    run_evaluation(
        gt_folder=args.gt_folder,
        trackers_folder=args.trackers_folder,
        benchmark=args.benchmark,
        split=args.split,
        trackers=args.trackers,
    )
