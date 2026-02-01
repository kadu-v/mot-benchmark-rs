//! MOTChallenge Benchmark Tool
//!
//! This tool runs tracking on MOT17/MOT20 sequences using public detections
//! and outputs results in MOTChallenge format for evaluation with TrackEval.
//!
//! Usage:
//!   cargo run --release --example mot_benchmark -- \
//!     --data-dir ./scripts/benchmark/data \
//!     --output-dir ./scripts/benchmark/output \
//!     --benchmark MOT17 \
//!     --tracker ByteTracker

use indicatif::{ProgressBar, ProgressStyle};
use jamtrack_rs::{
    boost_tracker::BoostTracker, byte_tracker::ByteTracker, object::Object, rect::Rect,
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

/// Detection from MOTChallenge det.txt format
#[derive(Debug)]
struct MotDetection {
    frame: u32,
    _id: i32,      // Usually -1 for detections
    bb_left: f32,
    bb_top: f32,
    bb_width: f32,
    bb_height: f32,
    conf: f32,
}

/// Tracking result in MOTChallenge format
#[derive(Debug)]
struct MotResult {
    frame: u32,
    track_id: usize,
    bb_left: f32,
    bb_top: f32,
    bb_width: f32,
    bb_height: f32,
    conf: f32,
}

/// Sequence info from seqinfo.ini
#[derive(Debug)]
struct SeqInfo {
    name: String,
    seq_length: u32,
    frame_rate: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrackerType {
    ByteTracker,
    ByteTrackerTuned,
    BoostTrack,
    BoostTrackPlus,
    BoostTrackPlusPlus,
}

impl TrackerType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bytetracker" | "byte" => Some(Self::ByteTracker),
            "bytetrackertuned" | "bytetuned" => Some(Self::ByteTrackerTuned),
            "boosttrack" | "boost" => Some(Self::BoostTrack),
            "boosttrackplus" | "boost+" | "boosttrack+" => Some(Self::BoostTrackPlus),
            "boosttrackplusplus" | "boost++" | "boosttrack++" => Some(Self::BoostTrackPlusPlus),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::ByteTracker => "ByteTracker",
            Self::ByteTrackerTuned => "ByteTrackerTuned",
            Self::BoostTrack => "BoostTrack",
            Self::BoostTrackPlus => "BoostTrackPlus",
            Self::BoostTrackPlusPlus => "BoostTrackPlusPlus",
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::ByteTracker,
            Self::ByteTrackerTuned,
            Self::BoostTrack,
            Self::BoostTrackPlus,
            Self::BoostTrackPlusPlus,
        ]
    }
}

struct Config {
    data_dir: PathBuf,
    output_dir: PathBuf,
    benchmark: String,
    trackers: Vec<TrackerType>,
    split: String,
    use_gt_as_det: bool, // Use ground truth as detection (for testing)
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;

    println!("=== MOTChallenge Benchmark ===");
    println!("Data directory: {}", config.data_dir.display());
    println!("Output directory: {}", config.output_dir.display());
    println!("Benchmark: {}-{}", config.benchmark, config.split);
    println!(
        "Trackers: {:?}",
        config.trackers.iter().map(|t| t.name()).collect::<Vec<_>>()
    );
    if config.use_gt_as_det {
        println!("Detection source: Ground Truth (--use-gt-as-det)");
    } else {
        println!("Detection source: Public detection (det/det.txt)");
    }
    println!();

    // Find sequences
    let gt_dir = config
        .data_dir
        .join("gt/mot_challenge")
        .join(format!("{}-{}", config.benchmark, config.split));
    let sequences = find_sequences(&gt_dir, config.use_gt_as_det)?;

    if sequences.is_empty() {
        return Err(format!("No sequences found in {}", gt_dir.display()).into());
    }

    println!("Found {} sequences:", sequences.len());
    for seq in &sequences {
        println!("  - {}", seq.file_name().unwrap_or_default().to_string_lossy());
    }
    println!();

    // Run each tracker
    for tracker_type in &config.trackers {
        println!("Running {}...", tracker_type.name());

        let tracker_output_dir = config
            .output_dir
            .join("trackers/mot_challenge")
            .join(format!("{}-{}", config.benchmark, config.split))
            .join(tracker_type.name())
            .join("data");
        fs::create_dir_all(&tracker_output_dir)?;

        let progress = ProgressBar::new(sequences.len() as u64);
        let style = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
        )?
        .progress_chars("=>-");
        progress.set_style(style);

        for seq_dir in &sequences {
            let seq_name = seq_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            progress.set_message(seq_name.clone());

            let results = run_tracker_on_sequence(seq_dir, *tracker_type, config.use_gt_as_det)?;

            // Write results
            let output_file = tracker_output_dir.join(format!("{}.txt", seq_name));
            write_mot_results(&output_file, &results)?;

            progress.inc(1);
        }

        progress.finish_with_message("done");
        println!("  Results saved to: {}", tracker_output_dir.display());
    }

    println!();
    println!("=== Benchmark Complete ===");
    println!("Run TrackEval to evaluate:");
    println!(
        "  python scripts/run_mot_challenge.py --BENCHMARK {} --SPLIT_TO_EVAL {} --TRACKERS_TO_EVAL {}",
        config.benchmark,
        config.split,
        config.trackers.iter().map(|t| t.name()).collect::<Vec<_>>().join(" ")
    );

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        std::process::exit(0);
    }

    let mut data_dir = PathBuf::from("scripts/benchmark/data");
    let mut output_dir = PathBuf::from("scripts/benchmark/output");
    let mut benchmark = String::from("MOT17");
    let mut trackers: Vec<TrackerType> = Vec::new();
    let mut split = String::from("train");
    let mut use_gt_as_det = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                data_dir = PathBuf::from(&args[i]);
            }
            "--output-dir" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--benchmark" => {
                i += 1;
                benchmark = args[i].clone();
            }
            "--tracker" => {
                i += 1;
                if args[i].to_lowercase() == "all" {
                    trackers = TrackerType::all();
                } else {
                    if let Some(t) = TrackerType::from_str(&args[i]) {
                        trackers.push(t);
                    } else {
                        return Err(format!("Unknown tracker: {}", args[i]).into());
                    }
                }
            }
            "--split" => {
                i += 1;
                split = args[i].clone();
            }
            "--use-gt-as-det" => {
                use_gt_as_det = true;
            }
            _ => {}
        }
        i += 1;
    }

    if trackers.is_empty() {
        trackers = TrackerType::all();
    }

    Ok(Config {
        data_dir,
        output_dir,
        benchmark,
        trackers,
        split,
        use_gt_as_det,
    })
}

fn print_usage() {
    println!(
        r#"MOTChallenge Benchmark Tool

Usage:
  cargo run --release --example mot_benchmark -- [OPTIONS]

Options:
  --data-dir <PATH>      Data directory containing gt/mot_challenge/ (default: scripts/benchmark/data)
  --output-dir <PATH>    Output directory for tracking results (default: scripts/benchmark/output)
  --benchmark <NAME>     Benchmark name: MOT17, MOT20 (default: MOT17)
  --tracker <NAME>       Tracker to run: ByteTracker, BoostTrack, BoostTrackPlus, BoostTrackPlusPlus, all (default: all)
  --split <NAME>         Split to evaluate: train, test (default: train)
  --use-gt-as-det        Use ground truth as detection (for testing when det.txt is not available)
  -h, --help             Show this help message

Examples:
  # Run all trackers on MOT17-train
  cargo run --release --example mot_benchmark

  # Run only ByteTracker on MOT20-train
  cargo run --release --example mot_benchmark -- --benchmark MOT20 --tracker ByteTracker

  # Run BoostTrack++ on MOT17-train with custom paths
  cargo run --release --example mot_benchmark -- \
    --data-dir ./data \
    --output-dir ./results \
    --tracker BoostTrack++

  # Use ground truth as detection (for testing)
  cargo run --release --example mot_benchmark -- --use-gt-as-det
"#
    );
}

fn find_sequences(gt_dir: &Path, use_gt_as_det: bool) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if !gt_dir.exists() {
        return Err(format!("Ground truth directory not found: {}", gt_dir.display()).into());
    }

    let mut sequences = Vec::new();
    for entry in fs::read_dir(gt_dir)? {
        let path = entry?.path();
        if path.is_dir() {
            let seq_info = path.join("seqinfo.ini");
            if !seq_info.exists() {
                continue;
            }

            // Check for detection source
            if use_gt_as_det {
                // Use gt/gt.txt as detection
                let gt_file = path.join("gt/gt.txt");
                if gt_file.exists() {
                    sequences.push(path);
                }
            } else {
                // Use det/det.txt as detection
                let det_file = path.join("det/det.txt");
                if det_file.exists() {
                    sequences.push(path);
                }
            }
        }
    }
    sequences.sort();
    Ok(sequences)
}

fn parse_seqinfo(path: &Path) -> Result<SeqInfo, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut name = String::new();
    let mut seq_length = 0u32;
    let mut frame_rate = 30u32;

    for line in content.lines() {
        let line = line.trim();
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "name" => name = value.trim().to_string(),
                "seqLength" => seq_length = value.trim().parse().unwrap_or(0),
                "frameRate" => frame_rate = value.trim().parse().unwrap_or(30),
                _ => {}
            }
        }
    }

    Ok(SeqInfo {
        name,
        seq_length,
        frame_rate,
    })
}

fn load_detections(det_file: &Path) -> Result<Vec<MotDetection>, Box<dyn Error>> {
    let file = File::open(det_file)?;
    let reader = BufReader::new(file);
    let mut detections = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 7 {
            continue;
        }

        let frame: u32 = parts[0].trim().parse()?;
        let id: i32 = parts[1].trim().parse()?;
        let bb_left: f32 = parts[2].trim().parse()?;
        let bb_top: f32 = parts[3].trim().parse()?;
        let bb_width: f32 = parts[4].trim().parse()?;
        let bb_height: f32 = parts[5].trim().parse()?;
        let conf: f32 = parts[6].trim().parse()?;

        // Skip invalid detections
        if bb_width <= 0.0 || bb_height <= 0.0 || conf < 0.0 {
            continue;
        }

        detections.push(MotDetection {
            frame,
            _id: id,
            bb_left,
            bb_top,
            bb_width,
            bb_height,
            conf,
        });
    }

    Ok(detections)
}

/// Get sequence-specific ByteTracker parameters (like official ByteTrack)
fn get_bytetracker_params(seq_name: &str) -> (usize, f32, f32) {
    // Extract base sequence name (e.g., "MOT17-05" from "MOT17-05-YOLOX")
    let base_name = seq_name.split('-').take(2).collect::<Vec<_>>().join("-");

    // Sequence-specific track_buffer (from official ByteTrack mot_evaluator.py)
    let track_buffer = if base_name == "MOT17-05" || base_name == "MOT17-06" {
        14
    } else if base_name == "MOT17-13" || base_name == "MOT17-14" {
        25
    } else {
        30
    };

    // Sequence-specific track_thresh (from official ByteTrack mot_evaluator.py)
    let track_thresh = if base_name == "MOT17-01" {
        0.65
    } else if base_name == "MOT17-06" {
        0.65
    } else if base_name == "MOT17-12" {
        0.7
    } else if base_name == "MOT17-14" {
        0.67
    } else {
        0.6
    };

    // high_thresh = track_thresh + 0.1 (like official ByteTrack)
    let high_thresh = track_thresh + 0.1;

    (track_buffer, track_thresh, high_thresh)
}

/// Filter tracking results (like official ByteTrack)
fn filter_track_result(width: f32, height: f32, min_box_area: f32) -> bool {
    let area = width * height;
    let aspect_ratio = width / height;

    // Filter out small boxes and vertical boxes (aspect ratio > 1.6)
    area > min_box_area && aspect_ratio <= 1.6
}

fn run_tracker_on_sequence(
    seq_dir: &Path,
    tracker_type: TrackerType,
    use_gt_as_det: bool,
) -> Result<Vec<MotResult>, Box<dyn Error>> {
    let seq_info = parse_seqinfo(&seq_dir.join("seqinfo.ini"))?;
    let seq_name = seq_dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let det_file = if use_gt_as_det {
        seq_dir.join("gt/gt.txt")
    } else {
        seq_dir.join("det/det.txt")
    };
    let detections = load_detections(&det_file)?;

    // Group detections by frame
    let max_frame = detections.iter().map(|d| d.frame).max().unwrap_or(0);
    let mut frame_detections: Vec<Vec<&MotDetection>> = vec![Vec::new(); max_frame as usize + 1];
    for det in &detections {
        if (det.frame as usize) < frame_detections.len() {
            frame_detections[det.frame as usize].push(det);
        }
    }

    let mut results = Vec::new();
    let min_box_area = 100.0; // Same as official ByteTrack

    match tracker_type {
        TrackerType::ByteTracker => {
            // Fixed parameters (for fair comparison)
            let mut tracker = ByteTracker::new(
                seq_info.frame_rate as usize,
                30,   // track_buffer
                0.6,  // track_thresh
                0.7,  // high_thresh
                0.9,  // match_thresh
            );

            for frame in 1..=seq_info.seq_length {
                let dets = &frame_detections.get(frame as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let objects = dets_to_objects(dets);
                if let Ok(tracks) = tracker.update(&objects) {
                    for track in tracks {
                        let width = track.get_width();
                        let height = track.get_height();

                        // Apply filtering like official ByteTrack
                        if filter_track_result(width, height, min_box_area) {
                            results.push(MotResult {
                                frame,
                                track_id: track.get_track_id().unwrap_or(0),
                                bb_left: track.get_x(),
                                bb_top: track.get_y(),
                                bb_width: width,
                                bb_height: height,
                                conf: track.get_prob(),
                            });
                        }
                    }
                }
            }
        }
        TrackerType::ByteTrackerTuned => {
            // Get sequence-specific parameters (like official ByteTrack)
            let (track_buffer, track_thresh, high_thresh) = get_bytetracker_params(&seq_name);
            let mut tracker = ByteTracker::new(
                seq_info.frame_rate as usize,
                track_buffer,
                track_thresh,
                high_thresh,
                0.9, // match_thresh
            );

            for frame in 1..=seq_info.seq_length {
                let dets = &frame_detections.get(frame as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let objects = dets_to_objects(dets);
                if let Ok(tracks) = tracker.update(&objects) {
                    for track in tracks {
                        let width = track.get_width();
                        let height = track.get_height();

                        // Apply filtering like official ByteTrack
                        if filter_track_result(width, height, min_box_area) {
                            results.push(MotResult {
                                frame,
                                track_id: track.get_track_id().unwrap_or(0),
                                bb_left: track.get_x(),
                                bb_top: track.get_y(),
                                bb_width: width,
                                bb_height: height,
                                conf: track.get_prob(),
                            });
                        }
                    }
                }
            }
        }
        TrackerType::BoostTrack => {
            let boost_min_box_area = 10.0; // Official BoostTrack uses 10
            let mut tracker = BoostTracker::new(0.6, 0.3, 30, 3);
            for frame in 1..=seq_info.seq_length {
                let dets = &frame_detections.get(frame as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let objects = dets_to_objects(dets);
                let tracks = tracker.update(&objects)?;
                for track in tracks {
                    let width = track.get_width();
                    let height = track.get_height();
                    if filter_track_result(width, height, boost_min_box_area) {
                        results.push(MotResult {
                            frame,
                            track_id: track.get_track_id().unwrap_or(0),
                            bb_left: track.get_x(),
                            bb_top: track.get_y(),
                            bb_width: width,
                            bb_height: height,
                            conf: track.get_prob(),
                        });
                    }
                }
            }
        }
        TrackerType::BoostTrackPlus => {
            let boost_min_box_area = 10.0; // Official BoostTrack uses 10
            let mut tracker = BoostTracker::new(0.6, 0.3, 30, 3).with_boost_plus();
            for frame in 1..=seq_info.seq_length {
                let dets = &frame_detections.get(frame as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let objects = dets_to_objects(dets);
                let tracks = tracker.update(&objects)?;
                for track in tracks {
                    let width = track.get_width();
                    let height = track.get_height();
                    if filter_track_result(width, height, boost_min_box_area) {
                        results.push(MotResult {
                            frame,
                            track_id: track.get_track_id().unwrap_or(0),
                            bb_left: track.get_x(),
                            bb_top: track.get_y(),
                            bb_width: width,
                            bb_height: height,
                            conf: track.get_prob(),
                        });
                    }
                }
            }
        }
        TrackerType::BoostTrackPlusPlus => {
            let boost_min_box_area = 10.0; // Official BoostTrack uses 10
            let mut tracker = BoostTracker::new(0.6, 0.3, 30, 3).with_boost_plus_plus();
            for frame in 1..=seq_info.seq_length {
                let dets = &frame_detections.get(frame as usize).map(|v| v.as_slice()).unwrap_or(&[]);
                let objects = dets_to_objects(dets);
                let tracks = tracker.update(&objects)?;
                for track in tracks {
                    let width = track.get_width();
                    let height = track.get_height();
                    if filter_track_result(width, height, boost_min_box_area) {
                        results.push(MotResult {
                            frame,
                            track_id: track.get_track_id().unwrap_or(0),
                            bb_left: track.get_x(),
                            bb_top: track.get_y(),
                            bb_width: width,
                            bb_height: height,
                            conf: track.get_prob(),
                        });
                    }
                }
            }
        }
    }

    Ok(results)
}

fn dets_to_objects(dets: &[&MotDetection]) -> Vec<Object> {
    dets.iter()
        .map(|d| {
            let rect = Rect::new(d.bb_left, d.bb_top, d.bb_width, d.bb_height);
            Object::new(rect, d.conf, None)
        })
        .collect()
}

fn write_mot_results(path: &Path, results: &[MotResult]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    for r in results {
        // MOTChallenge format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
        writeln!(
            file,
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.4},-1,-1,-1",
            r.frame, r.track_id, r.bb_left, r.bb_top, r.bb_width, r.bb_height, r.conf
        )?;
    }

    Ok(())
}
