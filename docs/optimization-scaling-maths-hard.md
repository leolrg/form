# FORM optimization scaling: maths_hard

All 18 maths_hard runs are complete and validated. This completes the 72-run
four-sequence evaluation. Each entry below averages two runs; latency excludes
the first 20 of 2,440 scans. Counts average all scans. Every run matches all
2,438 available ground-truth poses.

## Optimization performance

| Workload | Original CPU ms | CPU summaries ms | CUDA ms | CUDA latency change vs CPU summaries |
|---|---:|---:|---:|---:|
| Current | 78.85 | 51.78 | 53.14 | 2.63% higher |
| More features | 126.64 | 63.22 | 63.72 | 0.80% higher |
| Larger window | 113.92 | 76.24 | 66.89 | 12.27% lower |

CUDA loses both current and dense timing pairs. It wins both larger-window
pairs, reducing optimization latency by 12.70% and 11.84%. Larger-window CUDA
combines GPU summary preparation and selective FP64 GPU solving at system
dimension 240 or above. Each repeat makes 75,712 GPU solves with zero numerical
fallbacks. Current and feature CUDA runs retain CPU solving; the window comparison
does not isolate the solver contribution.

## Realized scaling and accuracy

Counts and mean translational relative trajectory error (RTE) average both CUDA
repeats. Other backends agree to the displayed accuracy precision.

| Workload | Active poses | Correspondences | CUDA optimization / current | RTE at 1 m (m) | RTE at 30 m (m) |
|---|---:|---:|---:|---:|---:|
| Current | 33.67 | 205,906 | 1.000 | 0.088381 | 0.804187 |
| More features | 33.61 | 518,633 | 1.199 | 0.085509 | 0.916191 |
| Larger window | 41.74 | 340,797 | 1.259 | 0.087143 | 0.919957 |

Denser selection realizes 2.52x correspondences for 19.9% more CUDA optimization
time. The larger window realizes 1.24x poses and 1.66x correspondences for 25.9%
more optimization time. These are complete-estimator workloads with changed
selection and retention, not isolated complexity measurements.

More features reduce mean 1 m error by 3.25% but increase mean 30 m error by
13.93%. The larger window reduces short-distance error by 1.40% and increases
long-distance error by 14.40%. Neither establishes a general accuracy improvement.

## Complete scan processing

| Workload | Original CPU ms | CPU summaries ms | CUDA ms | CUDA p95 ms | CUDA p99 ms | CUDA scans above 100 ms |
|---|---:|---:|---:|---:|---:|---:|
| Current | 214.70 | 177.83 | 165.24 | 262.69 | 300.27 | 87.54% |
| More features | 399.92 | 320.29 | 281.53 | 473.20 | 561.16 | 99.71% |
| Larger window | 249.07 | 202.40 | 177.12 | 264.46 | 316.52 | 98.78% |

Percentiles and deadline fractions average per-run statistics. Every CUDA
configuration exceeds the 100 ms mean scan budget. Matching remains expensive,
especially with denser selection. Matching time differs across backends despite
unchanged matching code; its cause remains unproven. The entire complete-scan
reduction cannot be attributed to CUDA optimization.

## Validation and reproduction

Both accelerated backends pass all nine predeclared translation-RTE/coverage
checks in both repeats at every workload. All recorded pose, feature,
correspondence, factor, rematch, and LM-iteration counts match each paired
reference. Maximum accelerated/reference position disagreement is below
7.55e-11 m for current settings, 5.33e-11 m for more features, and 6.04e-11 m
for the larger window. Repetitions also preserve all workload counts.

Hardware is an A100 80 GB PCIe GPU with 32 CPU threads on an EPYC Milan VM,
using FP64 and the frozen Release executable. Desktop performance is unmeasured.
Settings are current caps 3/50, inherited spacing, recent10; more features caps
6/100, spacing2, recent10; larger window caps 3/50, inherited spacing, recent40.
Normal and curvature neighborhoods are unchanged.

See the [consolidated results](optimization-acceleration-results.md) and
[verified full report](../benchmarks/results/density-scaling-t32/final.md) for
commands, provenance, latency distributions, memory, quality gates, and traces.
