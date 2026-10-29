//! Loads the shared detection CSV (frame,x1,y1,x2,y2,score; 1-based frames).

use std::fs;

use anyhow::{Context, Result};

use crate::sort::Detection;

/// Detections indexed by frame (index 0 is unused so `frames[n]` is frame n).
pub fn load(path: &str) -> Result<Vec<Vec<Detection>>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let mut frames: Vec<Vec<Detection>> = vec![Vec::new()];
    for (lineno, line) in text.lines().enumerate().skip(1) {
        if line.is_empty() {
            continue;
        }
        let f: Vec<f64> = line
            .split(',')
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("{path}:{}", lineno + 1))?;
        anyhow::ensure!(f.len() == 6, "{path}:{}: expected 6 columns", lineno + 1);
        let frame = f[0] as usize;
        if frames.len() <= frame {
            frames.resize_with(frame + 1, Vec::new);
        }
        frames[frame].push(Detection {
            bbox: [f[1], f[2], f[3], f[4]],
            score: f[5],
        });
    }
    Ok(frames)
}
