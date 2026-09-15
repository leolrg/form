# Additional FORM-paper dataset benchmarks

The final CUDA hybrid is **2.12x faster than original FORM on Hilti 2022 basement_2** and **1.83x on MCD tuhh_day_04** in complete-sequence comparisons. Both are about **1.51x faster than the optimized CPU implementation**. These extend validation beyond Newer College stairs; they do not establish a universal 3x speedup.

## Scope

- All 740 Hilti scans and all 1,879 MCD scans are processed in each run. The first 20 are excluded from timing, leaving 720 and 1,859 timed scans respectively.
- Two sequential repeats per backend, reversing backend order on the second repeat: 12 verified full-sequence runs total. No competing conversion or benchmark process runs during timing.
- NVIDIA A100 80 GB PCIe, AMD EPYC Milan VM, 32 CPU threads, FP64 summaries. These are not desktop GeForce results.
- Current workload: point/plane caps 3/50, inherited feature spacing 5, recent-scan cap 10. Actual retained pose counts can exceed that cap because of keyscans.
- Range limits follow evalio: Hilti 0.5–120 m, MCD 0.1–120 m, identical across all backends. The earlier N21 replay used 0.1–50 m. No enlarged-feature/window experiments were run in this campaign.
- Input conversion uses evalio 0.6.1, including Hilti row reordering/padding and MCD scan-end to scan-start timestamp adjustment. XYZ is stored as float32 without deskew. Ground truth is transformed into the LiDAR frame with normalized quaternions.
- Total latency includes extraction, preparation, matching, optimization, marginalization and maintenance. Sensor normalization, file loading and output serialization are excluded. Optimization latency is semi_ms + full_ms.
- Original FORM is the existing reference backend; optimized CPU is summary-resident; CUDA hybrid is cuda-matching with dimension threshold 240.

## Total estimator time

All latencies are means of the two per-run means, in milliseconds per scan. Speedups divide those means.

| Sequence | Original CPU | Optimized CPU | CUDA hybrid | Original / CUDA | Optimized CPU / CUDA |
|---|---:|---:|---:|---:|---:|
| Hilti 2022 basement_2 | 70.66 | 50.41 | 33.29 | 2.12x | 1.51x |
| MCD tuhh_day_04 | 94.22 | 77.65 | 51.47 | 1.83x | 1.51x |

Mean per-repeat p95 latency:

| Sequence | Original CPU | Optimized CPU | CUDA hybrid |
|---|---:|---:|---:|
| Hilti 2022 basement_2 | 128.87 | 93.67 | 59.52 |
| MCD tuhh_day_04 | 130.79 | 110.40 | 73.72 |

## Optimization time

| Sequence | Original CPU | Optimized CPU | CUDA hybrid | Original / CUDA |
|---|---:|---:|---:|---:|
| Hilti 2022 basement_2 | 35.00 | 18.49 | 15.33 | 2.28x |
| MCD tuhh_day_04 | 33.03 | 20.55 | 16.04 | 2.06x |

## Stage breakdown and interpretation

Mean milliseconds per scan:

| Sequence | Backend | Extraction | Preparation | Matching incl. QR/raw data | Optimization |
|---|---|---:|---:|---:|---:|
| Hilti 2022 basement_2 | reference | 8.50 | 3.53 | 22.38 | 35.00 |
| Hilti 2022 basement_2 | summary-resident | 8.03 | 3.46 | 19.51 | 18.49 |
| Hilti 2022 basement_2 | cuda-matching | 8.03 | 3.34 | 5.93 | 15.33 |
| MCD tuhh_day_04 | reference | 6.94 | 19.39 | 28.87 | 33.03 |
| MCD tuhh_day_04 | summary-resident | 6.54 | 19.20 | 25.83 | 20.55 |
| MCD tuhh_day_04 | cuda-matching | 6.04 | 17.48 | 8.30 | 16.04 |

Both sequences perform zero GPU linear solves: Hilti retains at most 39 poses and MCD at most 30, below the 240-variable crossover. CUDA still performs matching, grouping and QR construction. Evaluation/assembly/solving run through the direct CPU matrix path. These results demonstrate acceleration without GPU linear solving.

MCD preparation remains a large fraction of the accelerated total. Preparation includes CPU world-map construction and matcher snapshot work; this campaign does not profile their individual contributions. Faster matching and optimization therefore translate into a smaller total speedup. Stage variation outside changed code is not attributed solely to CUDA.

## Correctness

All eight accelerated/reference comparisons preserve every per-scan feature, correspondence, factor, pose, rematch and LM-iteration count. Maximum position disagreement across those comparisons is 1.354e-10 m; maximum rotation disagreement is 3.076e-12 rad.

All 1 m/30 m RTE and coverage gates pass for every repeat. This means acceleration retains the reference quality; it does not assert zero absolute odometry error. Mean translation RTE below averages the per-repeat means for the reference (candidates match to displayed precision).

| Sequence | Matched poses / output scans | RTE 1 m | RTE 30 m |
|---|---:|---:|---:|
| Hilti 2022 basement_2 | 689/740 | 0.089091 m | 0.588624 m |
| MCD tuhh_day_04 | 1863/1879 | 0.101879 m | 0.521165 m |

## Reproducibility

- Benchmark adapter/runner commit: `6a2789a`. No estimator or CUDA kernel changes were made for these runs.
- Frozen binary: `benchmarks/results/additional-backend/form-replay`.
- Binary SHA256: `dc5d89ab5276e6b0de6f5d063dec4a8db1f88c96d18fb10902eb04eb90fa7958`.
- Prepared inputs and calibrated ground truth: `benchmarks/results/additional-input/`.
- Source provenance, file sizes, checksums and loader checks: `benchmarks/results/additional-datasets/`.
- Full command lines, build/environment metadata, timing CSVs, trajectories and per-repeat metrics: `benchmarks/results/additional-hilti-current/` and `benchmarks/results/additional-mcd-current/`.
- [Combined machine-readable report](../benchmarks/results/additional-current/final.json) and [generated detailed tables](../benchmarks/results/additional-current/final.md).
- [Preparation and execution instructions](../benchmarks/README.md#hilti-2022-and-multi-campus-inputs).

The 32 Python benchmark tests pass. The rebuilt replay rejects invalid sensor ranges. Both new datasets pass 50-scan reference/CUDA smoke checks; those concurrent-preparation runs are excluded from all reported performance measurements. Beginning/middle/end stored replay samples match conversion hashes and timestamps. Full suite reporting verifies every run output, quality comparison, and workload trace.
