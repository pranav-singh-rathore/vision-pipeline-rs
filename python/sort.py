"""SORT tracker (Bewley et al., 2016): constant-velocity Kalman filter + IoU matching.

Self-contained numpy implementation so the Rust port can mirror it line by line.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    """IoU between every det and every track box, both as [x1, y1, x2, y2]."""
    d = dets[:, None, :4]
    t = trks[None, :, :4]
    xx1 = np.maximum(d[..., 0], t[..., 0])
    yy1 = np.maximum(d[..., 1], t[..., 1])
    xx2 = np.minimum(d[..., 2], t[..., 2])
    yy2 = np.minimum(d[..., 3], t[..., 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    return inter / (area_d + area_t - inter)


def bbox_to_z(b: np.ndarray) -> np.ndarray:
    w = b[2] - b[0]
    h = b[3] - b[1]
    return np.array([b[0] + w / 2.0, b[1] + h / 2.0, w * h, w / h])


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0])


# Model matrices, shared by every track. State: [u, v, s, r, du, dv, ds].
F = np.eye(7)
F[0, 4] = F[1, 5] = F[2, 6] = 1.0
H = np.eye(4, 7)
R = np.eye(4)
R[2:, 2:] *= 10.0
Q = np.eye(7)
Q[-1, -1] *= 0.01
Q[4:, 4:] *= 0.01
P0 = np.eye(7)
P0[4:, 4:] *= 1000.0  # high uncertainty on the unobserved velocities
P0 *= 10.0
I7 = np.eye(7)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = bbox_to_z(bbox)
        self.P = P0.copy()
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_bbox(self.x)

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        y = bbox_to_z(bbox) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form keeps P symmetric and positive definite.
        IKH = I7 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def state(self) -> np.ndarray:
        return x_to_bbox(self.x)


def associate(dets: np.ndarray, trks: np.ndarray, iou_threshold: float):
    if len(trks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty(0, dtype=int)

    iou = iou_batch(dets, trks)
    if min(iou.shape) > 0:
        a = (iou > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched = np.stack(np.where(a), axis=1)
        else:
            rows, cols = linear_sum_assignment(-iou)
            matched = np.stack([rows, cols], axis=1)
    else:
        matched = np.empty((0, 2), dtype=int)

    matches = [[d, t] for d, t in matched if iou[d, t] >= iou_threshold]
    matched_dets = {d for d, _ in matches}
    matched_trks = {t for _, t in matches}
    # Spawn new tracks in detection order. Which row the solver leaves unassigned on a
    # tie is implementation-defined, so sorting keeps IDs independent of tie-breaking.
    unmatched_dets = [d for d in range(len(dets)) if d not in matched_dets]
    unmatched_trks = [t for t in range(len(trks)) if t not in matched_trks]
    matches = np.array(matches, dtype=int).reshape(-1, 2)
    return matches, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


class Sort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """dets: Nx5 [x1, y1, x2, y2, score]. Returns Mx5 [x1, y1, x2, y2, id]."""
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for i, trk in enumerate(self.trackers):
            trks[i] = trk.predict()
            if np.any(np.isnan(trks[i])):
                to_del.append(i)
        trks = np.delete(trks, to_del, axis=0)
        for i in reversed(to_del):
            self.trackers.pop(i)

        matches, unmatched_dets, _ = associate(dets, trks, self.iou_threshold)
        for d, t in matches:
            self.trackers[t].update(dets[d, :4])
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4]))

        out = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                out.append(np.append(trk.state(), trk.id + 1))
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.array(out) if out else np.empty((0, 5))
