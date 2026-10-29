#!/usr/bin/env bash
# Reproduce every number in results/RESULTS.md.
# Needs: Rust, OpenCV 4 dev libs + clang (for the opencv crate), and Python with
# the SAME OpenCV build (e.g. Ubuntu's python3-opencv) plus numpy and scipy.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
RUNS=${RUNS:-5}

mkdir -p data results
if [ ! -f data/vtest.avi ]; then
  curl -sSL -o data/vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
fi
if [ ! -f data/detections.csv ]; then
  "$PY" python/detect.py
fi

cargo build --release
BIN=target/release/vision-pipeline

echo "== video: tracking + blur =="
"$PY" python/pipeline.py --runs "$RUNS"
"$BIN" --runs "$RUNS"
cmp results/tracks_python.csv results/tracks_rust.csv && echo "tracks: identical"

echo "== tracker scaling =="
for n in 10 50 100 200 400; do
  "$PY" bench/make_synthetic.py --objects "$n" --out "data/synth_$n.csv"
  "$PY" python/pipeline.py --video "" --dets "data/synth_$n.csv" --runs "$RUNS" \
    --tracks-out "results/synth_${n}_py.csv" --timing-out "results/synth_${n}_py.json"
  "$BIN" --video "" --dets "data/synth_$n.csv" --runs "$RUNS" \
    --tracks-out "results/synth_${n}_rs.csv" --timing-out "results/synth_${n}_rs.json"
  cmp "results/synth_${n}_py.csv" "results/synth_${n}_rs.csv" && echo "N=$n tracks: identical"
done

"$PY" bench/report.py
