"""Tracking + anonymisation stage: SORT over fixed detections, then blur every tracked box.

Times decode, track and blur separately per frame and writes the tracks so the
Rust port can be checked for identical output.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sort import KalmanBoxTracker, Sort


def load_detections(path: str) -> dict[int, np.ndarray]:
    per_frame: dict[int, list] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            per_frame[int(row["frame"])].append(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), float(row["score"])]
            )
    return {k: np.array(v) for k, v in per_frame.items()}


def blur_boxes(frame: np.ndarray, boxes: np.ndarray, ksize: int) -> None:
    h, w = frame.shape[:2]
    for x1, y1, x2, y2, _ in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)


def write_tracks(path: str, rows: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
        w.writerows([[r[0], r[1], *(f"{v:.4f}" for v in r[2:])] for r in rows])


def track_only(dets: dict[int, np.ndarray], args: argparse.Namespace) -> None:
    """Tracker alone, no video: used for the scaling benchmark."""
    n_frames = max(dets) if dets else 0
    empty = np.empty((0, 5))
    runs = []
    for run in range(args.runs):
        KalmanBoxTracker.count = 0
        tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
        rows = []
        t_track = 0.0
        for frame_idx in range(1, n_frames + 1):
            t0 = time.perf_counter()
            tracks = tracker.update(dets.get(frame_idx, empty))
            t_track += time.perf_counter() - t0
            if run == 0:
                rows.extend([frame_idx, int(tid), x1, y1, x2, y2] for x1, y1, x2, y2, tid in tracks)
        if run == 0:
            write_tracks(args.tracks_out, rows)
        runs.append({"frames": n_frames, "track_ms": t_track * 1e3 / n_frames})
    result = {"impl": "python", "opencv": cv2.__version__, "numpy": np.__version__, "runs": runs}
    Path(args.timing_out).write_text(json.dumps(result, indent=2))
    best = min(r["track_ms"] for r in runs)
    print(f"python: {n_frames} frames | track {best:.3f} ms/frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/vtest.avi")
    ap.add_argument("--dets", default="data/detections.csv")
    ap.add_argument("--tracks-out", default="results/tracks_python.csv")
    ap.add_argument("--timing-out", default="results/timing_python.json")
    ap.add_argument("--video-out", default="")
    ap.add_argument("--ksize", type=int, default=31)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    dets = load_detections(args.dets)
    empty = np.empty((0, 5))
    runs = []

    if not args.video:
        track_only(dets, args)
        return

    for run in range(args.runs):
        KalmanBoxTracker.count = 0
        tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
        cap = cv2.VideoCapture(args.video)
        writer = None
        track_rows = []
        t_decode = t_track = t_blur = 0.0
        frame_idx = 0
        crc = 0
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            frame_idx += 1
            tracks = tracker.update(dets.get(frame_idx, empty))
            t2 = time.perf_counter()
            blur_boxes(frame, tracks, args.ksize)
            t3 = time.perf_counter()
            t_decode += t1 - t0
            t_track += t2 - t1
            t_blur += t3 - t2

            if run == 0:
                crc = zlib.crc32(frame.tobytes(), crc)
                for x1, y1, x2, y2, tid in tracks:
                    track_rows.append([frame_idx, int(tid), x1, y1, x2, y2])
                if args.video_out:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.video_out, fourcc, 10, (frame.shape[1], frame.shape[0]))
                    writer.write(frame)

        if writer is not None:
            writer.release()
        if run == 0:
            write_tracks(args.tracks_out, track_rows)
        runs.append(
            {
                "frames": frame_idx,
                "decode_ms": t_decode * 1e3 / frame_idx,
                "track_ms": t_track * 1e3 / frame_idx,
                "blur_ms": t_blur * 1e3 / frame_idx,
                "frame_crc32": f"{crc:08x}",
            }
        )

    result = {"impl": "python", "opencv": cv2.__version__, "numpy": np.__version__, "runs": runs}
    Path(args.timing_out).write_text(json.dumps(result, indent=2))
    best = min(runs, key=lambda r: r["track_ms"])
    print(
        f"python: {best['frames']} frames | track {best['track_ms']:.3f} ms/frame | "
        f"blur {best['blur_ms']:.3f} ms/frame | decode {best['decode_ms']:.3f} ms/frame"
    )


if __name__ == "__main__":
    main()
