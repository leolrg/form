# GPU normal construction: results and placement rationale

Follow-up: [integrated GPU boundary results](integrated-gpu-boundaries-results.md)
now test masks/selection and map/materialization inside the full estimator.
That report supersedes the placement questions left open here; the measurements
below remain the earlier experiment.

Measured on 2026-09-30 in `/home/ubuntu/form`, branch
`codex/optimization-acceleration`, with uncommitted changes over
`1255543137f67ac54aea5434b9ef6093f1b9fe02`.

The corrected GPU normal implementation reduces extraction time while preserving
full-sequence quality on stairs. The initial accuracy regression was caused by
changing the numerical eigensolver path. It is not evidence that normals belong
on CPU. The current-density whole-estimator improvement is small and varies
between repeats; the denser-feature experiment shows a larger, consistent gain.

## Implementation and numerical correction

The opt-in `cuda-normals` backend extends `cuda-extraction` without changing
matching or optimizer settings. It reuses the uploaded scan and exact adjacent-row
GPU search, gathers neighborhoods, constructs covariance, and computes the smallest
eigenvector on GPU. One thread processes each selected planar candidate, with 128
threads per block and reusable buffers. Two neighborhood traversals count points
and then accumulate covariance, avoiding variable-size temporary allocation.
One download returns completed normals and validity flags. A sequential host pass
packs the public feature objects. CPU selection and masks remain.

The original query-centered formula is retained: for `m` gathered points,
`a_j = (p_j - query) / T(m)` and `C = sum(a_j * a_j^T)`, in input scalar type `T`.
Division occurs before multiplication; this is not a newly centroid-centered
covariance. Radius comparisons retain the caller's four-coordinate reduction
policy. CUDA FMA contraction remains disabled.

The first working eigensolver used Eigen's generic tridiagonalization on bounded
dynamic-size storage. It passed normal-direction and eigen-residual tests, but
both full-sequence runs failed the existing 30 m translation quality gate:

| Measurement | Existing hybrid | Generic GPU solver | Corrected GPU solver |
| --- | ---: | ---: | ---: |
| Mean 30 m relative translation error, m | 0.236248 | 0.284471 | 0.236248 |
| Maximum position difference from hybrid, m | 0 | 0.302089 | 8.91e-9 |
| Full-sequence quality gate | Control | Fail, both repeats | Pass, both repeats |

The generic version selected the same number of features, but small numerical
differences propagated through discrete correspondence, rematch and window
decisions. A position difference appeared at scan 248, a planar correspondence
count changed at scan 250, rematch counts changed at scan 252, and pose-window
counts changed at scan 344. Small local normal errors alone were therefore an
insufficient acceptance criterion.

The correction changes only the eigensolver computation relative to that generic
version: a local, attributed MPL-2.0 adapter preserves Eigen 3.4's original
specialized 3x3 scaling/tridiagonalization and device-compatible iterative QR.
Covariance construction and sequential feature packing are unchanged. This
restored every recorded workload count and reduced trajectory differences to
about 9 nm. The installed Eigen specialized reduction lacks a device annotation,
which is why a local adapter was needed. CPU SIMD covariance accumulation can
still differ from the GPU's scalar accumulation; bitwise equivalence across
arbitrary inputs, compilers and Eigen versions is not claimed.

An earlier packing experiment also exposed avoidable host overhead: using TBB's
concurrent vector to copy already-computed normals kept normal completion costly.
The final path uses a sequential reserved vector. GPU output follows selected
index order; the CPU path retains TBB insertion order. This can matter for ties
on other inputs, though corrected replay counts agree here.

## Paired latency measurements

Hardware: NVIDIA A100 80 GB PCIe, AMD EPYC-Milan VM exposing 32 CPUs, 32 TBB
threads, CUDA 13.0, Release build targeting `sm_80`. Input:
`/home/ubuntu/datasets/form-input/stairs.formpc`, with its ground truth and metadata.
Each comparison has two serial repeats with backend order reversed. The first
20 scans are excluded from timings. No builds, sanitizers or other benchmark
runs overlapped the timed runs. Both backends use GPU solve threshold 240.
Times include estimator work, transfers and synchronization, and exclude input
file loading and output serialization. These are latency reductions, not FPS
speedup percentages.

| Workload | Timed scans/run | Extraction ms, hybrid → GPU normals | Reduction | Estimator ms, hybrid → GPU normals | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current, full 1190 scans | 1170 | 7.706 → 6.397 | 17.0% | 42.830 → 41.688 | 2.7% |
| Denser features, first 250 scans | 230 | 10.392 → 7.467 | 28.1% | 50.870 → 45.652 | 10.3% |
| Larger window, first 250 scans | 230 | 7.009 → 5.984 | 14.6% | 50.334 → 49.112 | 2.4% |

Values are the mean of the two per-run means. Current uses point/plane caps 3/50
and recent window 10; dense uses 6/100, spacing 2 and window 10; window uses
3/50 and recent window 40. Compare within each row: the prefixes and full
sequence have different workload distributions.

| Workload | Estimator repeat 1, ms | Estimator repeat 2, ms | Extraction p95, ms | Estimator p95, ms |
| --- | ---: | ---: | ---: | ---: |
| Current | 42.282 → 42.585 | 43.379 → 40.790 | 10.344 → 9.370 | 71.789 → 70.478 |
| Dense | 50.704 → 46.211 | 51.035 → 45.093 | 13.306 → 9.833 | 67.881 → 63.730 |
| Window | 50.664 → 49.768 | 50.003 → 48.456 | 9.883 → 9.384 | 66.343 → 65.705 |

Each p95 value is the mean of two run p95s, not a pooled p95. Extraction improved
in both repeats of every workload. Current whole-estimator latency regressed
slightly in one repeat, so the 2.7% mean reduction is not a robust universal
speedup claim. Two repeats on one GPU and sequence do not establish a general
crossover or confidence interval.

Separate instrumented 250-scan runs isolate the normal-completion host scope:

| Workload | Normal-stage mean ms, hybrid → GPU normals | Normal-stage p95 ms |
| --- | ---: | ---: |
| Current | 1.745 → 0.793 | 2.313 → 1.406 |
| Dense | 3.674 → 1.801 | 4.406 → 3.426 |

This scope includes adjacent-row search, transfers, synchronization, normal
computation and initial output construction. The CPU path's final copy from its
concurrent vector falls in the subsequent output scope; whole extraction above
includes both. These are not isolated kernel times. Profiled timings are not
mixed into the clean comparison table; zero normal-stage fields in unprofiled
artifacts mean instrumentation was disabled.

## Quality and verification

Both corrected full runs pass the existing 1 m/30 m translation and coverage
gate against both the paired hybrid control and the saved original CPU reference.
For each mean/median/RMSE/max translation metric, the gate permits
`candidate <= reference + max(5% of reference, 0.01 m)`; matched-pose coverage
must not decrease. Rotation is reported without a predeclared acceptance gate.
All 1190 poses are covered. Every per-scan pose-window, planar/point feature,
rematch, LM iteration, factor and planar/point correspondence count agrees with
the hybrid control.

Dense and window prefixes also preserve all those counts, with maximum position
differences below `3.5e-10 m`. They contain no 30 m evaluation segments, so their
complete quality gate is **missing**, not passed. Full dense/window quality and
other sequences remain untested.

Validation completed:

- CUDA C++: 118 tests passed, including scalar-Eigen and parallel-selection tests.
- CPU-only C++: 49 passed; eight CUDA-dependent tests skipped.
- Benchmark Python: 10 passed. Python extension built and flag roundtrip passed.
- Compute Sanitizer: 12 relevant tests each under memcheck and racecheck;
  zero errors and zero race hazards.
- Tests cover float/double CPU-reference directions and eigen residuals, scales,
  translations, degeneracy, empty/missing rows, strict radius boundaries,
  changing shapes/types, invalid arguments and block-tail sizes through 257.

## What this supports in the paper

| Stage | Evidence and defensible interpretation |
| --- | --- |
| Normal construction | CPU retention was an implementation/compatibility boundary. The corrected GPU version is faster for extraction here; there is no measured performance reason from this experiment to retain CPU normals. |
| Feature selection and masks | Independent scanlines are parallelized with TBB in our implementation. Ordered suppression within rows is a real dependency, but does not prove CPU is faster. A GPU placement comparison has not been performed. |
| Feature/map materialization | CPU objects explain integration boundaries; they do not establish optimal placement. A more GPU-resident front end may remove copies, but that benefit remains to be measured. |
| Small optimizer systems | Existing forced-placement measurements support CPU for the current small systems: complete semi/full calls cost 0.50/0.71 ms on CPU versus 0.73/0.86 ms on GPU. Larger-window calls favor GPU. This is a measured workload-dependent placement rationale. |
| Host control | LM orchestration and input/output remain host responsibilities. Describe that boundary explicitly without asserting that every such operation was proven faster on CPU. |

The solver evidence is detailed in
[the diagnostics report](detailed-diagnostics-results.md#optimizer-where-cpu-and-gpu-time-go).
This experiment supports moving normals to GPU and investigating selection/masks
as the next residency boundary. A fully GPU-resident data path is a coherent
design target, but the current implementation remains hybrid. The paper should
distinguish measured placement decisions from stages that simply have not been
ported. The rejected numerical variant is not an ablation supporting CPU normals.

## Reproduction and retained artifacts

Use `--backend cuda-normals`, or set C++ extraction parameters `use_cuda=true`
and `use_cuda_normals=true`. Python estimator parameters expose
`use_cuda_extraction` and `use_cuda_normals`. Defaults and the `cuda-extraction`
control remain unchanged. See [benchmark instructions](../benchmarks/README.md).

```bash
/home/ubuntu/.local/share/uv/tools/evalio/bin/python benchmarks/run_suite.py \
  --binary build-accel/form-replay --output benchmarks/results/new-normal-comparison \
  --sequences stairs --configs current --threads 32 --repeats 2 \
  --backends reference cuda-extraction cuda-normals --cuda-solve-min-dimension 240
```

Use fresh output directories. For scaling, use `--configs features window
--limit 250`; for separate profiles, use `--configs current features --limit 250
--profile`. Frozen local artifacts live under ignored `benchmarks/results/`:

- `cuda-normals-corrected-full`, `cuda-normals-corrected-scaling`, and
  `cuda-normals-corrected-profile`: corrected paired campaigns, CSV trajectories,
  metrics, logs and provenance manifests.
- `cuda-normals-specialized-build`: frozen binary, source diff, local Eigen
  adapter, kernel resource report, `analyze.py`, `analysis.json`, and
  `quality-vs-original.json`. Binary SHA256:
  `39d4d28dc18d49fbe9c62ffd98d84da693147b83cc47e70a0be72813b973ecae`.
- `cuda-normals-full`, `cuda-normals-scaling`, `cuda-normals-profile-current`,
  `cuda-normals-profile-dense`, and `cuda-normals-final-build`: retained failed
  generic-solver experiment. That binary SHA256 is
  `42b1fbeb5c78ef68c3245dccfb15584e766df998179cff934d63401478820b99`.

`analyze.py` revalidates suite artifacts and applies the existing quality gate
to the paired `cuda-extraction` control. The normal suite report expects a
`reference` backend; the corrected two-backend campaigns intentionally use this
additional analysis for hybrid comparisons. The separate original-reference
comparison uses saved full CPU runs for quality only, not fresh performance.
Raw benchmark artifacts are local and are not included in the tracked report.
