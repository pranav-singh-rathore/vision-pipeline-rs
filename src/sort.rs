//! SORT tracker (Bewley et al., 2016), ported from `python/sort.py`.
//!
//! Same model matrices, same association rules and same track lifecycle, so the
//! output matches the Python reference box for box. Fixed-size nalgebra matrices
//! keep the whole Kalman step on the stack.

use nalgebra::{SMatrix, SVector};

use crate::hungarian::linear_sum_assignment;

type Vec4 = SVector<f64, 4>;
type Vec7 = SVector<f64, 7>;
type Mat4 = SMatrix<f64, 4, 4>;
type Mat7 = SMatrix<f64, 7, 7>;
type Mat47 = SMatrix<f64, 4, 7>;

/// Axis-aligned box as [x1, y1, x2, y2].
pub type BBox = [f64; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    pub bbox: BBox,
    pub id: u64,
}

struct Model {
    f: Mat7,
    h: Mat47,
    r: Mat4,
    q: Mat7,
    p0: Mat7,
}

impl Model {
    fn new() -> Self {
        let mut f = Mat7::identity();
        f[(0, 4)] = 1.0;
        f[(1, 5)] = 1.0;
        f[(2, 6)] = 1.0;

        let h = Mat47::identity();

        let mut r = Mat4::identity();
        r[(2, 2)] = 10.0;
        r[(3, 3)] = 10.0;

        let mut q = Mat7::identity();
        q[(4, 4)] = 0.01;
        q[(5, 5)] = 0.01;
        q[(6, 6)] = 0.0001;

        let mut p0 = Mat7::identity() * 10.0;
        for i in 4..7 {
            p0[(i, i)] = 10_000.0; // high uncertainty on the unobserved velocities
        }
        Self { f, h, r, q, p0 }
    }
}

fn bbox_to_z(b: &BBox) -> Vec4 {
    let w = b[2] - b[0];
    let h = b[3] - b[1];
    Vec4::new(b[0] + w / 2.0, b[1] + h / 2.0, w * h, w / h)
}

fn x_to_bbox(x: &Vec7) -> BBox {
    let w = (x[2] * x[3]).sqrt();
    let h = x[2] / w;
    [
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]
}

pub fn iou(a: &BBox, b: &BBox) -> f64 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

struct KalmanBoxTracker {
    x: Vec7,
    p: Mat7,
    id: u64,
    time_since_update: u32,
    hit_streak: u32,
}

impl KalmanBoxTracker {
    fn new(bbox: &BBox, id: u64, model: &Model) -> Self {
        let mut x = Vec7::zeros();
        x.fixed_rows_mut::<4>(0).copy_from(&bbox_to_z(bbox));
        Self {
            x,
            p: model.p0,
            id,
            time_since_update: 0,
            hit_streak: 0,
        }
    }

    fn predict(&mut self, m: &Model) -> BBox {
        if self.x[6] + self.x[2] <= 0.0 {
            self.x[6] = 0.0;
        }
        self.x = m.f * self.x;
        self.p = m.f * self.p * m.f.transpose() + m.q;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        x_to_bbox(&self.x)
    }

    fn update(&mut self, bbox: &BBox, m: &Model) {
        self.time_since_update = 0;
        self.hit_streak += 1;
        let y = bbox_to_z(bbox) - m.h * self.x;
        let pht = self.p * m.h.transpose();
        let s = m.h * pht + m.r;
        let s_inv = s
            .try_inverse()
            .expect("innovation covariance is positive definite");
        let k = pht * s_inv;
        self.x += k * y;
        // Joseph form, as in the Python reference.
        let ikh = Mat7::identity() - k * m.h;
        self.p = ikh * self.p * ikh.transpose() + k * m.r * k.transpose();
    }

    fn state(&self) -> BBox {
        x_to_bbox(&self.x)
    }
}

pub struct Sort {
    max_age: u32,
    min_hits: u32,
    iou_threshold: f64,
    trackers: Vec<KalmanBoxTracker>,
    frame_count: u32,
    next_id: u64,
    model: Model,
    // Scratch buffers reused across frames.
    predicted: Vec<BBox>,
    iou_buf: Vec<f64>,
}

impl Sort {
    pub fn new(max_age: u32, min_hits: u32, iou_threshold: f64) -> Self {
        Self {
            max_age,
            min_hits,
            iou_threshold,
            trackers: Vec::new(),
            frame_count: 0,
            next_id: 0,
            model: Model::new(),
            predicted: Vec::new(),
            iou_buf: Vec::new(),
        }
    }

    /// Advance one frame. Output order matches the Python reference (newest track first).
    pub fn update(&mut self, dets: &[Detection], out: &mut Vec<Track>) {
        self.frame_count += 1;
        out.clear();

        self.predicted.clear();
        let model = &self.model;
        let predicted = &mut self.predicted;
        self.trackers.retain_mut(|t| {
            let b = t.predict(model);
            if b.iter().any(|v| v.is_nan()) {
                return false;
            }
            predicted.push(b);
            true
        });

        let (matches, unmatched_dets) = self.associate(dets);
        for &(d, t) in &matches {
            self.trackers[t].update(&dets[d].bbox, &self.model);
        }
        for d in unmatched_dets {
            self.trackers.push(KalmanBoxTracker::new(
                &dets[d].bbox,
                self.next_id,
                &self.model,
            ));
            self.next_id += 1;
        }

        for t in self.trackers.iter().rev() {
            if t.time_since_update < 1
                && (t.hit_streak >= self.min_hits || self.frame_count <= self.min_hits)
            {
                out.push(Track {
                    bbox: t.state(),
                    id: t.id + 1,
                });
            }
        }
        let max_age = self.max_age;
        self.trackers.retain(|t| t.time_since_update <= max_age);
    }

    fn associate(&mut self, dets: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>) {
        let nd = dets.len();
        let nt = self.predicted.len();
        if nt == 0 {
            return (Vec::new(), (0..nd).collect());
        }

        self.iou_buf.clear();
        for d in dets {
            for t in &self.predicted {
                self.iou_buf.push(iou(&d.bbox, t));
            }
        }
        let iou_m = &self.iou_buf;

        let mut candidates: Vec<(usize, usize)> = Vec::new();
        if nd > 0 {
            // Fast path: if every det and track has at most one overlap above threshold,
            // that set is already the optimal assignment.
            let mut row_hits = vec![0u32; nd];
            let mut col_hits = vec![0u32; nt];
            for d in 0..nd {
                for t in 0..nt {
                    if iou_m[d * nt + t] > self.iou_threshold {
                        row_hits[d] += 1;
                        col_hits[t] += 1;
                    }
                }
            }
            let max_row = row_hits.iter().copied().max().unwrap_or(0);
            let max_col = col_hits.iter().copied().max().unwrap_or(0);
            if max_row == 1 && max_col == 1 {
                for d in 0..nd {
                    for t in 0..nt {
                        if iou_m[d * nt + t] > self.iou_threshold {
                            candidates.push((d, t));
                        }
                    }
                }
            } else {
                let neg: Vec<f64> = iou_m.iter().map(|v| -v).collect();
                candidates = linear_sum_assignment(&neg, nd, nt);
            }
        }

        // Spawn new tracks in detection order. Rows the solver leaves unassigned on a
        // tie differ between solvers, so sorting keeps IDs independent of tie-breaking.
        let mut det_matched = vec![false; nd];
        let mut matches = Vec::with_capacity(candidates.len());
        for (d, t) in candidates {
            if iou_m[d * nt + t] >= self.iou_threshold {
                det_matched[d] = true;
                matches.push((d, t));
            }
        }
        let unmatched: Vec<usize> = (0..nd).filter(|&d| !det_matched[d]).collect();
        (matches, unmatched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f64, y1: f64, x2: f64, y2: f64) -> Detection {
        Detection {
            bbox: [x1, y1, x2, y2],
            score: 1.0,
        }
    }

    #[test]
    fn iou_basics() {
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]), 1.0);
        assert_eq!(iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]), 0.0);
        assert!((iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn keeps_id_for_moving_box() {
        let mut sort = Sort::new(1, 3, 0.3);
        let mut out = Vec::new();
        for i in 0..10 {
            let x = i as f64 * 2.0;
            sort.update(&[det(x, 0.0, x + 20.0, 40.0)], &mut out);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].id, 1);
        }
    }

    #[test]
    fn drops_track_after_max_age() {
        let mut sort = Sort::new(1, 1, 0.3);
        let mut out = Vec::new();
        sort.update(&[det(0.0, 0.0, 20.0, 40.0)], &mut out);
        sort.update(&[], &mut out);
        sort.update(&[], &mut out);
        // Track 1 is gone after two missed frames, so the box comes back as track 2
        // (shown once it has a hit; a brand-new track has no hit yet).
        sort.update(&[det(0.0, 0.0, 20.0, 40.0)], &mut out);
        assert!(out.is_empty());
        sort.update(&[det(0.0, 0.0, 20.0, 40.0)], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 2);
    }
}
