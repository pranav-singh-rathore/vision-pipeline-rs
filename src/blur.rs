//! Anonymisation step: Gaussian-blur every tracked box in place.

use opencv::core::{self, Mat, Rect, Size};
use opencv::imgproc;
use opencv::prelude::*;

use crate::sort::Track;

/// Blurs each box inside `frame`. `scratch` is reused between calls to avoid allocating.
///
/// `BORDER_ISOLATED` keeps the kernel from reading pixels outside the box, which is
/// what the Python version does implicitly (a numpy slice has no parent image).
pub fn blur_boxes(
    frame: &mut Mat,
    tracks: &[Track],
    ksize: i32,
    scratch: &mut Mat,
) -> opencv::Result<()> {
    let (w, h) = (frame.cols(), frame.rows());
    for t in tracks {
        // Truncate toward zero like Python's int(), then clamp to the frame.
        let x1 = (t.bbox[0] as i32).max(0);
        let y1 = (t.bbox[1] as i32).max(0);
        let x2 = (t.bbox[2] as i32).min(w);
        let y2 = (t.bbox[3] as i32).min(h);
        if x2 - x1 < 2 || y2 - y1 < 2 {
            continue;
        }
        let rect = Rect::new(x1, y1, x2 - x1, y2 - y1);
        let roi = Mat::roi(frame, rect)?;
        imgproc::gaussian_blur(
            &roi,
            scratch,
            Size::new(ksize, ksize),
            0.0,
            0.0,
            core::BORDER_DEFAULT | core::BORDER_ISOLATED,
        )?;
        drop(roi);
        let mut roi = Mat::roi_mut(frame, rect)?;
        scratch.copy_to(&mut roi)?;
    }
    Ok(())
}
