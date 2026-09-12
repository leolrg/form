# FORM acceleration: final tuning and scaling results

The planned tuning and four-sequence evaluation are complete: 72 full replays,
three backends, three workloads, and two repetitions with reversed backend order.
All 432 predeclared paired trajectory-quality checks pass. Exact cached factor
summaries deliver most of the optimization speedup. CUDA adds a consistent gain
for the larger pose window, but has no general advantage over CPU summaries at
the current workload.

These measurements use an NVIDIA A100 80 GB PCIe GPU and 32 CPU threads on an
AMD EPYC Milan VM, with CUDA 13 and FP64 arithmetic. Performance on the intended
desktop GPU remains unmeasured. The author's historical approximately 12 Hz result
is not a measurement on this host or an acceptance threshold for this checkout.

A subsequent opt-in resident graph-batching experiment is documented separately
in [broader CUDA results](batched-cuda-results.md). Its prefix screening is not
part of the 72 full replays summarized here.

## Current workload

Mean optimization latency in milliseconds, averaging two runs per entry and
excluding the first 20 scans of each run:

| Sequence | Original CPU | CPU summaries | CUDA |
|---|---:|---:|---:|
| stairs | 53.79 | 28.36 | 29.09 |
| quad_easy | 42.42 | 24.77 | 25.50 |
| quad_hard | 70.85 | 44.54 | 43.58 |
| maths_hard | 78.85 | 51.78 | 53.14 |

CPU summaries reduce optimization time by approximately 34–47% versus the
original CPU, depending on sequence. Across all scans, weighted optimization
means are 63.22 ms original, 39.10 ms CPU summaries, and 39.61 ms CUDA. The
weighted CUDA mean is 1.31% higher than CPU summaries. These estimator aggregates
include the repeatability qualifications below; they are not fixed-input kernel
measurements.

CUDA loses both current-setting pairs on quad_easy and maths_hard, and wins one
but loses the other on stairs. Its small quad_hard gain includes a changed-work
first pair. CPU summaries are therefore the supported starting choice for current
settings on this host. The original reference remains the software default;
accelerated backends are explicitly selectable.

## Larger workloads

Feature selection and window size were varied separately. Current settings use
point/plane caps 3/50 per sector, inherited suppression spacing 5, and recent10.
The feature workload uses caps 6/100, spacing2, recent10. The window workload
uses caps 3/50, inherited spacing, recent40. Curvature and normal neighborhoods
remain unchanged. Increasing caps alone did not increase planar selection, so
the suppression-spacing control was necessary to realize the denser workload.

All three backends were measured at every enlarged setting. The table below
compares their scaling directly: each cell gives current → enlarged mean
optimization latency in milliseconds, followed by the enlarged/current ratio
for that same backend. Lower ratios mean less latency growth; absolute timings
establish which backend is faster. Entries average two full replays and exclude
the first 20 scans of each replay.

| Sequence | Workload | Original CPU: current → enlarged | CPU summaries: current → enlarged | CUDA: current → enlarged |
|---|---|---:|---:|---:|
| stairs | More features | 53.79 → 109.71 (2.04×) | 28.36 → 34.89 (1.23×) | 29.09 → 32.77 (1.13×) |
| quad_easy | More features | 42.42 → 78.30 (1.85×) | 24.77 → 32.66 (1.32×) | 25.50 → 31.60 (1.24×) |
| quad_hard | More features | 70.85 → 130.31 (1.84×) | 44.54 → 55.68 (1.25×) | 43.58 → 55.57 (1.28×) |
| maths_hard | More features | 78.85 → 126.64 (1.61×) | 51.78 → 63.22 (1.22×) | 53.14 → 63.72 (1.20×) |
| stairs | Larger window | 53.79 → 96.42 (1.79×) | 28.36 → 51.05 (1.80×) | 29.09 → 42.05 (1.45×) |
| quad_easy | Larger window | 42.42 → 103.83 (2.45×) | 24.77 → 63.61 (2.57×) | 25.50 → 59.47 (2.33×) |
| quad_hard | Larger window | 70.85 → 116.69 (1.65×) | 44.54 → 75.87 (1.70×) | 43.58 → 64.21 (1.47×) |
| maths_hard | Larger window | 78.85 → 113.92 (1.44×) | 51.78 → 76.24 (1.47×) | 53.14 → 66.89 (1.26×) |

With more features, original CPU optimization grows 1.61–2.04×, CPU summaries
1.22–1.32×, and CUDA 1.13–1.28×. CPU factor compression already supplies much of
the improved scaling. With larger windows, the respective ranges are
1.44–2.45×, 1.47–2.57×, and 1.26–2.33×. CUDA has less relative latency growth
than both CPU backends on every larger-window sequence and is faster in absolute
optimization time at those enlarged settings.

These are complete-estimator comparisons with matching selection settings.
Actual estimator work differs in some quad_easy feature and quad_hard current
pairs, so those ratios are not controlled fixed-factor kernel scaling results;
see the repeatability qualifications below.

The following ratios use measured CUDA workload counts and mean optimization
times against current CUDA on the same sequence. Counts include all scans.
The last column compares CUDA with CPU summaries at the same enlarged settings.

| Sequence | Workload | Correspondences × | Active poses × | CUDA optimization × | CUDA latency reduction vs CPU summaries |
|---|---|---:|---:|---:|---:|
| stairs | More features | 2.596 | 0.965 | 1.126 | 6.09% |
| quad_easy | More features | 2.757 | 1.068 | 1.239 | 3.25% |
| quad_hard | More features | 2.897 | 1.048 | 1.275 | 0.18% |
| maths_hard | More features | 2.519 | 0.998 | 1.199 | -0.80% |
| stairs | Larger window | 1.696 | 1.471 | 1.446 | 17.62% |
| quad_easy | Larger window | 2.789 | 1.853 | 2.332 | 6.50% |
| quad_hard | Larger window | 1.660 | 1.341 | 1.473 | 15.38% |
| maths_hard | Larger window | 1.655 | 1.240 | 1.259 | 12.27% |

Denser selection handles 2.52–2.90x correspondences for 13–28% more CUDA
optimization time. Much of this favorable scaling comes from factor compression:
the additional CUDA advantage is small, inconsistent, or absent on some sequences.
Quad_easy's second dense pair also changes estimator work, as described below.

Larger-window CUDA wins both timing pairs on every sequence. This path combines
GPU summary preparation with selective FP64 cuSolver solving at system dimension
240 or above. Current and feature CUDA runs retain CPU solving. The comparison
therefore does not isolate the solver contribution. GPU solve counts per repeat
are 22,793 on stairs, 51,622 on quad_easy, 58,515 on quad_hard, and 75,712 on
maths_hard, with zero numerical fallbacks in every run.

## Accuracy and complete-scan limits

Changes below compare each enlarged CUDA workload with current CUDA. Values are
percentage changes in mean translational relative trajectory error (RTE);
negative values indicate lower error.

| Sequence | Features: 1 m RTE | Features: 30 m RTE | Window: 1 m RTE | Window: 30 m RTE |
|---|---:|---:|---:|---:|
| stairs | +1.38% | +46.13% | -25.12% | +30.19% |
| quad_easy | -2.79% | -0.01% | -3.81% | +3.95% |
| quad_hard | -1.34% | -0.14% | -4.67% | +0.93% |
| maths_hard | -3.25% | +13.93% | -1.40% | +14.40% |

More points do not establish an accuracy or robustness improvement. The larger
window improves short-distance mean error on every sequence but worsens
long-distance mean error on every sequence. These changed-workload comparisons
are separate from the passing equivalent-settings backend quality checks.

Mean complete-scan latency in milliseconds, with the same per-backend
current → enlarged comparison and enlarged/current ratios:

| Sequence | Workload | Original CPU: current → enlarged | CPU summaries: current → enlarged | CUDA: current → enlarged |
|---|---|---:|---:|---:|
| stairs | More features | 159.15 → 487.05 (3.06×) | 129.06 → 390.72 (3.03×) | 112.11 → 336.95 (3.01×) |
| quad_easy | More features | 148.24 → 294.28 (1.99×) | 124.52 → 233.06 (1.87×) | 110.95 → 197.98 (1.78×) |
| quad_hard | More features | 190.20 → 395.87 (2.08×) | 157.42 → 305.20 (1.94×) | 136.82 → 263.29 (1.92×) |
| maths_hard | More features | 214.70 → 399.92 (1.86×) | 177.83 → 320.29 (1.80×) | 165.24 → 281.53 (1.70×) |
| stairs | Larger window | 159.15 → 198.85 (1.25×) | 129.06 → 149.02 (1.15×) | 112.11 → 118.45 (1.06×) |
| quad_easy | Larger window | 148.24 → 224.30 (1.51×) | 124.52 → 177.86 (1.43×) | 110.95 → 164.85 (1.49×) |
| quad_hard | Larger window | 190.20 → 243.38 (1.28×) | 157.42 → 197.86 (1.26×) | 136.82 → 165.05 (1.21×) |
| maths_hard | Larger window | 214.70 → 249.07 (1.16×) | 177.83 → 202.40 (1.14×) | 165.24 → 177.12 (1.07×) |

Every measured CUDA workload exceeds the 100 ms mean budget for 10 Hz LiDAR on
this host. Matching becomes particularly expensive with denser features. Matching
time also differs across backends despite unchanged matching code; its cause is
unproven. The entire scan-time reduction must not be attributed to GPU computation.
Median/p95/p99, throughput, deadline misses, stage timings, and memory are in the
[full report](../benchmarks/results/density-scaling-t32/final.md) and its JSON
companion. GPU memory is sampled and may miss short peaks.

## Why the CUDA gain is limited, and what tuning achieved

Each immutable correspondence set is compressed into a small exact QR summary.
Repeated residual, Jacobian, and Gauss–Newton evaluations then use that summary
on the CPU. Thus large stored correspondence totals are not the amount of data
rebuilt on every iteration. CUDA accelerates summary construction while paying
packing, transfer, launch, and synchronization costs; the remaining small systems
often favor the CPU.

The retained CUDA changes use reusable pinned staging, compact seven-value input
rows, and a measured 64-row QR layout. GPU dense solving is selective: captured
systems favored CPU at dimensions 108 and 174, and GPU at 258. Cached CUDA Graphs,
reciprocal hoisting, direct raw Hessian/cost evaluation, and GPU evaluation of
compressed roots were evaluated and not retained because their measured totals
did not improve the selected path. Numerical correctness alone was insufficient.

This completes the documented bounded tuning search; it does not prove global
optimality. Further CUDA gains would require evidence for a broader change such
as keeping more of an optimization iteration resident on the device. That has
not been demonstrated here. For complete-scan performance, matching remains an
important separate bottleneck. Desktop GPU performance requires measurement
before choosing a deployment backend or transferring the A100 crossover point.

## Validation and reproducibility

All 72 runs finish with the expected scan counts. All 48 accelerated/reference
pairs pass the unchanged nine checks: translation mean, median, RMSE, and maximum
RTE at 1 m and 30 m, plus coverage. The bound is reference plus the greater of 5%
or 0.01 m; coverage may not decrease. Rotation is reported without a separate
predeclared gate. Matching ground-truth counts are 1,190/1,988/1,724/2,438 for
stairs/quad_easy/quad_hard/maths_hard.

43 of the 48 paired traces preserve every recorded workload and iteration count.
The five exceptions are quad_easy dense repeat 2 (both accelerated backends),
and quad_hard current CUDA repeats 1/2 and CPU-summary repeat 2. Original-CPU
repeats also follow different trajectories in those workloads. The maximum
original-repeat position differences are 5.26 cm and 12.52 cm, respectively.
All quality gates pass, but affected timing pairs are complete-estimator
comparisons with changed work. The first internal branching cause is unproven.
See the sequence reports for the trace investigations.

Final verification reran all 38 C++ tests successfully. Saved CPU-only evidence
has 23 passes and two GPU skips; all 27 benchmark Python tests passed. Recorded
CUDA memcheck/synccheck/racecheck tests and the dense replay memcheck reported
zero errors or hazards. Frozen binary, source archive, and build-metadata hashes
match; implementation sources are unchanged since freezing. The final exporter
verifies output hashes, counts, timestamps, and experiment fingerprints; all
432 quality checks were independently recomputed from the saved analyses.

The frozen executable and source snapshot are in
`benchmarks/results/density-backend/`. Reproduce in a fresh output directory:

```bash
/home/ubuntu/.local/share/uv/tools/evalio/bin/python benchmarks/run_suite.py \
  --binary benchmarks/results/density-backend/form-replay \
  --output benchmarks/results/reproduction-t32 \
  --configs features window current --backends reference summary cuda \
  --threads 32 --repeats 2 --sequence-first \
  --cuda-solve-min-dimension 240 --cuda-solve-configs window
```

Dataset preparation, timing boundaries, and analysis conventions are in the
[benchmark instructions](../benchmarks/README.md). See the
[general goals](optimization-acceleration-goals.md),
[implementation description](optimization-acceleration-design.md), and
[tuning history](cuda-tuning-results.md). Sequence reports:
[stairs](optimization-scaling-stairs.md),
[quad_easy](optimization-scaling-quad-easy.md),
[quad_hard](optimization-scaling-quad-hard.md), and
[maths_hard](optimization-scaling-maths-hard.md).
