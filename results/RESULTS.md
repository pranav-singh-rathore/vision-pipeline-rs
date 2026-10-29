# Results

CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, single thread (`cv::setNumThreads(1)` in both). OpenCV 4.6.0 on both sides, numpy 1.26.4. Median of 5 runs, milliseconds per frame.

## Video: `vtest.avi`, 795 frames, 768x576, about 4 people per frame

| Stage | Python | Rust | Speedup |
|---|---:|---:|---:|
| Decode (OpenCV) | 0.815 | 0.7886 | 1.0x |
| SORT tracking | 0.407 | 0.0109 | 37.4x |
| Blur tracked boxes (OpenCV) | 2.589 | 2.6541 | 1.0x |
| **Total** | **3.811** | **3.454** | **1.10x** |

Track output is byte-identical between the two. Blurred frames are pixel-identical (same CRC32 over all frames).

## Tracker scaling: synthetic crowd, 1000 frames, 1920x1080

| Objects per frame | Python | Rust | Speedup |
|---:|---:|---:|---:|
| 10 | 0.583 | 0.0083 | 70.0x |
| 50 | 1.903 | 0.0560 | 34.0x |
| 100 | 3.795 | 0.1679 | 22.6x |
| 200 | 8.293 | 0.5296 | 15.7x |
| 400 | 20.526 | 1.8074 | 11.4x |

Track output is byte-identical at every size.

