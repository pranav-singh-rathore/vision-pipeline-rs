"""Deterministic crowded-scene detections for the tracker scaling benchmark.

N objects walk across a 1920x1080 frame at constant velocity with box jitter,
10% missed detections and a few false positives per frame. Objects that leave
the frame respawn at a random edge, so the scene stays at about N objects.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

W, H = 1920, 1080


def spawn(rng: np.random.Generator, n: int) -> np.ndarray:
    # columns: cx, cy, vx, vy, w, h
    obj = np.empty((n, 6))
    obj[:, 0] = rng.uniform(0, W, n)
    obj[:, 1] = rng.uniform(0, H, n)
    obj[:, 2] = rng.uniform(-6, 6, n)
    obj[:, 3] = rng.uniform(-3, 3, n)
    obj[:, 4] = rng.uniform(30, 120, n)
    obj[:, 5] = obj[:, 4] * rng.uniform(1.0, 2.5, n)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, required=True)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    obj = spawn(rng, args.objects)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "x1", "y1", "x2", "y2", "score"])
        for frame in range(1, args.frames + 1):
            obj[:, 0:2] += obj[:, 2:4]
            gone = (obj[:, 0] < -50) | (obj[:, 0] > W + 50) | (obj[:, 1] < -50) | (obj[:, 1] > H + 50)
            obj[gone] = spawn(rng, int(gone.sum()))
            seen = rng.random(args.objects) > 0.1
            jitter = rng.normal(0, 2.0, (args.objects, 4))
            for (cx, cy, _, _, bw, bh), j in zip(obj[seen], jitter[seen]):
                w.writerow([frame, *(f"{v:.2f}" for v in (cx - bw / 2 + j[0], cy - bh / 2 + j[1], cx + bw / 2 + j[2], cy + bh / 2 + j[3])), "0.9"])
            for _ in range(rng.poisson(2)):
                x, y = rng.uniform(0, W - 80), rng.uniform(0, H - 160)
                w.writerow([frame, f"{x:.2f}", f"{y:.2f}", f"{x + 60:.2f}", f"{y + 140:.2f}", "0.4"])


if __name__ == "__main__":
    main()
