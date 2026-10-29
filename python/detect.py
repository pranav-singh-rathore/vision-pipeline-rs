"""Generate a fixed detection file so both pipelines track the exact same boxes.

Runs OpenCV's HOG people detector once over the video and writes
MOT-style rows: frame,x1,y1,x2,y2,score (frame is 1-based).
"""

from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/vtest.avi")
    ap.add_argument("--out", default="data/detections.csv")
    ap.add_argument("--min-score", type=float, default=0.3)
    args = ap.parse_args()

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(args.video)
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        if len(rects):
            boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in rects]
            scores = np.asarray(weights).ravel().tolist()
            keep = cv2.dnn.NMSBoxes(boxes, scores, args.min_score, 0.4)
            for i in np.asarray(keep).ravel():
                x, y, w, h = boxes[i]
                rows.append([frame_idx, x, y, x + w, y + h, round(scores[i], 4)])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "score"])
        w.writerows(rows)
    print(f"{frame_idx} frames, {len(rows)} detections -> {args.out}")


if __name__ == "__main__":
    main()
