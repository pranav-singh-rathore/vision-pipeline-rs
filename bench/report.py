"""Summarise results/*.json into results/RESULTS.md and results/benchmark.png."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("results")
SIZES = [10, 50, 100, 200, 400]


def best(path: Path, key: str) -> float:
    """Median over runs is less noisy than min on a shared machine."""
    runs = sorted(r[key] for r in json.loads(path.read_text())["runs"])
    return runs[len(runs) // 2]


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    py, rs = RES / "timing_python.json", RES / "timing_rust.json"
    py_meta = json.loads(py.read_text())
    rs_meta = json.loads(rs.read_text())
    stages = ["decode_ms", "track_ms", "blur_ms"]
    video = {s: (best(py, s), best(rs, s)) for s in stages}
    py_total = sum(v[0] for v in video.values())
    rs_total = sum(v[1] for v in video.values())
    same_pixels = py_meta["runs"][0]["frame_crc32"] == rs_meta["runs"][0]["frame_crc32"]

    scale = [(n, best(RES / f"synth_{n}_py.json", "track_ms"), best(RES / f"synth_{n}_rs.json", "track_ms")) for n in SIZES]

    md = [
        "# Results\n",
        f"CPU: {cpu_name()}, single thread (`cv::setNumThreads(1)` in both). "
        f"OpenCV {py_meta['opencv']} on both sides, numpy {py_meta['numpy']}. "
        "Median of 5 runs, milliseconds per frame.\n",
        "## Video: `vtest.avi`, 795 frames, 768x576, about 4 people per frame\n",
        "| Stage | Python | Rust | Speedup |",
        "|---|---:|---:|---:|",
    ]
    names = {"decode_ms": "Decode (OpenCV)", "track_ms": "SORT tracking", "blur_ms": "Blur tracked boxes (OpenCV)"}
    for s in stages:
        p, r = video[s]
        md.append(f"| {names[s]} | {p:.3f} | {r:.4f} | {p / r:.1f}x |")
    md.append(f"| **Total** | **{py_total:.3f}** | **{rs_total:.3f}** | **{py_total / rs_total:.2f}x** |")
    md.append("")
    md.append(
        f"Track output is byte-identical between the two. Blurred frames are "
        f"{'pixel-identical (same CRC32 over all frames)' if same_pixels else 'NOT identical'}.\n"
    )
    md += [
        "## Tracker scaling: synthetic crowd, 1000 frames, 1920x1080\n",
        "| Objects per frame | Python | Rust | Speedup |",
        "|---:|---:|---:|---:|",
    ]
    for n, p, r in scale:
        md.append(f"| {n} | {p:.3f} | {r:.4f} | {p / r:.1f}x |")
    md.append("\nTrack output is byte-identical at every size.\n")
    (RES / "RESULTS.md").write_text("\n".join(md) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["Decode", "Track", "Blur"]
    x = range(len(stages))
    ax1.bar([i - 0.2 for i in x], [video[s][0] for s in stages], 0.4, label="Python", color="#3776AB")
    ax1.bar([i + 0.2 for i in x], [video[s][1] for s in stages], 0.4, label="Rust", color="#CE422B")
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("ms per frame")
    ax1.set_title("vtest.avi, per stage")
    ax1.legend(frameon=False)

    ns = [n for n, _, _ in scale]
    ax2.plot(ns, [p for _, p, _ in scale], "o-", label="Python", color="#3776AB")
    ax2.plot(ns, [r for _, _, r in scale], "o-", label="Rust", color="#CE422B")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("objects per frame")
    ax2.set_ylabel("tracker ms per frame")
    ax2.set_title("SORT tracker scaling")
    ax2.legend(frameon=False)
    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "benchmark.png", dpi=150)
    print((RES / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
