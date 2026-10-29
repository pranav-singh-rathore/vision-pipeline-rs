//! Tracking + anonymisation stage, Rust port of `python/pipeline.py`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use opencv::core::{self, Mat};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, VideoWriter};
use serde_json::json;

use vision_pipeline::blur::blur_boxes;
use vision_pipeline::detections;
use vision_pipeline::sort::{Sort, Track};

#[derive(Parser)]
#[command(about = "SORT tracking + box blurring over a video and a fixed detection file")]
struct Args {
    /// Input video. Pass an empty string to run the tracker alone on the detections.
    #[arg(long, default_value = "data/vtest.avi")]
    video: String,
    #[arg(long, default_value = "data/detections.csv")]
    dets: String,
    #[arg(long, default_value = "results/tracks_rust.csv")]
    tracks_out: String,
    #[arg(long, default_value = "results/timing_rust.json")]
    timing_out: String,
    /// Optional output video with the blurred boxes.
    #[arg(long, default_value = "")]
    video_out: String,
    #[arg(long, default_value_t = 31)]
    ksize: i32,
    #[arg(long, default_value_t = 5)]
    runs: usize,
}

#[derive(Default)]
struct RunTiming {
    frames: usize,
    decode_ms: f64,
    track_ms: f64,
    blur_ms: f64,
    crc: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    core::set_num_threads(1)?;
    let dets = detections::load(&args.dets)?;

    let mut runs = Vec::with_capacity(args.runs);
    for run in 0..args.runs {
        let record = run == 0;
        let timing = if args.video.is_empty() {
            track_only(&dets, &args, record)?
        } else {
            full_pipeline(&dets, &args, record)?
        };
        runs.push(timing);
    }

    let result = json!({
        "impl": "rust",
        "opencv": core::CV_VERSION,
        "runs": runs.iter().map(|r| json!({
            "frames": r.frames,
            "decode_ms": r.decode_ms,
            "track_ms": r.track_ms,
            "blur_ms": r.blur_ms,
            "frame_crc32": format!("{:08x}", r.crc),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(&args.timing_out, serde_json::to_string_pretty(&result)?)?;

    let best = runs
        .iter()
        .min_by(|a, b| a.track_ms.total_cmp(&b.track_ms))
        .unwrap();
    if args.video.is_empty() {
        println!(
            "rust: {} frames | track {:.4} ms/frame",
            best.frames, best.track_ms
        );
    } else {
        println!(
            "rust: {} frames | track {:.4} ms/frame | blur {:.3} ms/frame | decode {:.3} ms/frame",
            best.frames, best.track_ms, best.blur_ms, best.decode_ms
        );
    }
    Ok(())
}

/// Tracker alone, no video: used for the scaling benchmark.
fn track_only(
    dets: &[Vec<vision_pipeline::sort::Detection>],
    args: &Args,
    record: bool,
) -> Result<RunTiming> {
    let mut sort = Sort::new(1, 3, 0.3);
    let mut tracks: Vec<Track> = Vec::new();
    let mut rows = TrackRows::new(record);
    let mut timing = RunTiming::default();
    for (frame, frame_dets) in dets.iter().enumerate().skip(1) {
        let t0 = Instant::now();
        sort.update(frame_dets, &mut tracks);
        timing.track_ms += t0.elapsed().as_secs_f64() * 1e3;
        rows.push(frame, &tracks);
        timing.frames += 1;
    }
    rows.write(&args.tracks_out)?;
    timing.track_ms /= timing.frames as f64;
    Ok(timing)
}

fn full_pipeline(
    dets: &[Vec<vision_pipeline::sort::Detection>],
    args: &Args,
    record: bool,
) -> Result<RunTiming> {
    let mut cap = VideoCapture::from_file(&args.video, videoio::CAP_ANY)?;
    anyhow::ensure!(cap.is_opened()?, "cannot open {}", args.video);

    let mut sort = Sort::new(1, 3, 0.3);
    let mut tracks: Vec<Track> = Vec::new();
    let mut rows = TrackRows::new(record);
    let mut frame = Mat::default();
    let mut scratch = Mat::default();
    let mut writer: Option<VideoWriter> = None;
    let mut crc = crc32fast::Hasher::new();
    let mut timing = RunTiming::default();
    let empty = Vec::new();

    loop {
        let t0 = Instant::now();
        let ok = cap.read(&mut frame)?;
        let t1 = Instant::now();
        if !ok || frame.empty() {
            break;
        }
        timing.frames += 1;
        let n = timing.frames;
        sort.update(dets.get(n).unwrap_or(&empty), &mut tracks);
        let t2 = Instant::now();
        blur_boxes(&mut frame, &tracks, args.ksize, &mut scratch)?;
        let t3 = Instant::now();
        timing.decode_ms += (t1 - t0).as_secs_f64() * 1e3;
        timing.track_ms += (t2 - t1).as_secs_f64() * 1e3;
        timing.blur_ms += (t3 - t2).as_secs_f64() * 1e3;

        if record {
            rows.push(n, &tracks);
            crc.update(frame.data_bytes()?);
            if !args.video_out.is_empty() {
                if writer.is_none() {
                    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
                    writer = Some(VideoWriter::new(
                        &args.video_out,
                        fourcc,
                        10.0,
                        frame.size()?,
                        true,
                    )?);
                }
                writer.as_mut().unwrap().write(&frame)?;
            }
        }
    }

    rows.write(&args.tracks_out)?;
    let n = timing.frames as f64;
    timing.decode_ms /= n;
    timing.track_ms /= n;
    timing.blur_ms /= n;
    timing.crc = crc.finalize();
    Ok(timing)
}

/// Buffers track rows for the first run so I/O stays out of the timed loop.
struct TrackRows {
    enabled: bool,
    rows: Vec<(usize, Track)>,
}

impl TrackRows {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            rows: Vec::new(),
        }
    }

    fn push(&mut self, frame: usize, tracks: &[Track]) {
        if self.enabled {
            self.rows.extend(tracks.iter().map(|t| (frame, *t)));
        }
    }

    fn write(&self, path: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let mut f = BufWriter::new(File::create(path).with_context(|| format!("creating {path}"))?);
        writeln!(f, "frame,id,x1,y1,x2,y2")?;
        for (frame, t) in &self.rows {
            let b = t.bbox;
            writeln!(
                f,
                "{frame},{},{:.4},{:.4},{:.4},{:.4}",
                t.id, b[0], b[1], b[2], b[3]
            )?;
        }
        Ok(())
    }
}
