# N21 replay data and analysis

The [final tuning and scaling report](../docs/optimization-acceleration-results.md)
covers the completed 72-run evaluation. Verified measurements and quality checks
are in `results/density-scaling-t32/final.json` and `final.md`.

Use the installed evalio environment:

```bash
PYTHON=/home/ubuntu/.local/share/uv/tools/evalio/bin/python
$PYTHON benchmarks/prepare_n21.py
$PYTHON -m unittest discover -s benchmarks -p 'test_*.py'
```

The converter reads `/home/ubuntu/datasets/newer_college_2021/` and writes
`/home/ubuntu/datasets/form-input/<sequence>.formpc`, `.gt.tum`, and `.json` for
stairs, quad_easy, quad_hard, and maths_hard. `--sequences` selects a subset;
`--limit` is for development inputs only. Existing results are never overwritten.
A failed conversion can leave a `.formpc.partial`; inspect and remove that
incomplete output before rerunning. Conversion requires space for all requested
outputs plus a 2 GiB reserve and holds only a scan and bag reader buffers in memory.

The binary layout is little endian, without struct alignment or a frame-count
header:

| Bytes | Contents |
|---|---|
| 8 | ASCII `FORMPC01` |
| 4 + 4 | uint32 rows=128 and columns=1024 |
| repeated 8 + 4 | int64 Unix scan-start timestamp in nanoseconds; uint32 point count |
| repeated count × 16 | float32 x, y, z, zero-padding for every point |

EOF ends the frame list. Every point, including zero/invalid points, retains its
original row-major position. No range filtering, deskew, reordering, or IMU
measurements are applied during conversion. First, middle, and last scans in each
sequence are checked against evalio's N21 conversion for exact XYZ, order, and
nanosecond timestamp agreement. Metadata records those checks, source sizes,
evalio version, dimensions, counts, and calibration matrices.

Ground truth is exported as standard whitespace-separated TUM with decimal
seconds and quaternion x y z w. It expresses `world_T_lidar`, computed as
`world_T_gt * inverse(imu_T_gt) * imu_T_lidar` using evalio's N21 extrinsics. Source ground-truth and extrinsic quaternions
are normalized before composing SE(3) transforms: stairs contains source
quaternions with norms as low as 0.993232. Metadata records the largest deviation.
The ground-truth world frame remains the dataset's original world frame.
Replay pose output must also be `world_T_lidar` at the immediate scan-output stage.

```bash
$PYTHON benchmarks/analyze_n21.py \
  --timings /path/to/run.csv \
  --trajectory /path/to/run.tum \
  --ground-truth /home/ubuntu/datasets/form-input/stairs.gt.tum \
  --output /path/to/summary.json
```

Analysis reports all-scan and steady-state timing summaries (the latter excludes
the first 20 rows by default), median/p95/p99, throughput, and the fraction above
100 ms. Optimization is `semi_ms + full_ms`; marginalization stays separate.
`factor_*_cpu_ms` timers sum per-call elapsed durations that can overlap across
threads; they are not additive wall stages. Estimator timing excludes replay
loading and output serialization. Realized workload counts are summarized too.

Trajectory association uses globally closest unique timestamp pairs within
50 ms; it does not interpolate through missing ground truth. Coverage and the
largest accepted timestamp difference are reported. ATE uses first-matched-pose
SE(3) alignment with no fitted scale or best-fit trajectory alignment. RTE uses
evalio distance windows of 1 m and 30 m; translation is metres and rotation is
degrees. The summaries include means and RMSE, and no-window results are null
with zero samples. Ground-truth coverage gaps remain visible in coverage counts.

## Repeated benchmark suites

`run_suite.py` runs one subprocess at a time. The default matrix contains 48 runs:
all four complete sequences, reference and summary backends, two repetitions,
and these independent workloads. By default all four sequences finish the current workload
before enlarged workloads begin. Use `--sequence-first` to finish current, feature,
and window settings on each sequence before moving to the next. Backend order reverses on even-numbered repeats
to reduce order bias:

| Workload | Point cap/sector | Plane cap/sector | Suppression spacing | Recent scans |
|---|---:|---:|---:|---:|
| current | 3 | 50 | inherited (5) | 10 |
| features | 6 | 100 | 2 | 10 |
| window | 3 | 50 | inherited (5) | 40 |

Inspect the exact matrix before executing it, choosing the thread count established
by the short-segment measurements:

```bash
$PYTHON benchmarks/run_suite.py --threads 4 --plan
$PYTHON benchmarks/run_suite.py --threads 4 \
  --output benchmarks/results/suite-t4
```

The binary defaults to `build-accel/form-replay`. `--binary`, `--input-dir`,
`--sequences`, `--configs`, `--backends`, `--threads`, `--repeats`, `--profile`, and
`--limit` select the experiment. Repeats must be at least two. `--limit` marks the
report as a development comparison, and short inputs may lack 30 m RTE windows.
The default leaves detailed factor profiling disabled to avoid its overhead.
For example, a bounded two-backend check is:

```bash
$PYTHON benchmarks/run_suite.py --sequences stairs --configs current \
  --backends reference summary --threads 4 --repeats 2 --limit 250 \
  --output benchmarks/results/development-t4
```

Each run retains the exact subprocess argument array, binary SHA-256, git revision,
tracked dirty-diff hash, untracked-source hashes, CPU/GPU/CUDA information, affinity,
selected execution environment variables, `/usr/bin/time` peak RSS and process
wall/user/system time, logs, raw timing CSV, raw TUM trajectory, and analysis.
Peak RSS is in KiB. The saved source metadata describes the checkout when the suite
started; build provenance still requires building the executable from that checkout.
Process wall time includes input/output and startup, while estimator timing retains
the separate boundaries described above. Dataset metadata and GT hashes are recorded;
large scan inputs are checked against their declared size, not rehashed in every run.

A lock prevents two runners from writing the same suite. Resume uses the same command
and output directory; it accepts only completed runs with identical experiment/build
fingerprints, exact expected CSV/TUM counts from input metadata, matching timestamps,
and unchanged output hashes. Existing running, partial, failed, or incompatible runs
are never restarted or overwritten. Inspect those artifacts and select a fresh output
directory for a deliberate rerun. The runner also stops if the binary changes during
the suite. Failed subprocesses are recorded explicitly and other planned runs continue.

`report.json` retains run statuses, mean/min/max/sample-standard-deviation/CV across
repeats for timing summaries and memory, and paired speedup variability against the
reference at the same workload. `report.md` gives a compact comparison table.
Both reports refresh after every run, showing pending/missing/failed results explicitly.
To rebuild reports without replay:

```bash
$PYTHON benchmarks/run_suite.py --aggregate-only \
  --output benchmarks/results/suite-t4
```

Before any run, the suite records this acceptance rule for equivalent-work trajectory
comparisons: each candidate's translation mean, median, RMSE, and maximum at both 1 m and 30 m must be
at most the paired reference value plus `max(5% of reference, 0.01 m)`. Associated
pose coverage must not decrease. Every repeat must pass; unavailable windows or
missing runs cannot pass. Rotation errors are reported by analysis but have no
separate predeclared acceptance threshold. These gates assess trajectory degradation;
optimization-level numerical agreement requires the separate correctness tests.

To include the CUDA candidate, pass `--backends reference summary cuda`; the full
matrix then contains 72 runs and compares both candidates with the reference:

```bash
$PYTHON benchmarks/run_suite.py --threads 32 \
  --backends reference summary cuda --output benchmarks/results/full-t32
```

GPU process memory is sampled with `nvidia-smi` once per second by default. Only
the replay process tree under `/usr/bin/time` is counted, so unrelated GPU jobs
are excluded. Run records retain the sampled peak in MiB, requested interval,
query count, observed-process count, and query failures; group reports aggregate
those sampled peaks across repeats. This includes CUDA context and allocator
reservations, can miss short allocations, and the polling itself can perturb
timing. Null means no target GPU process was observed, rather than a measured
zero. Set `--gpu-sample-interval SECONDS` to change the interval or use zero to
disable sampling consistently across a comparison.

## Capture real CUDA QR inputs

Use a separate diagnostic replay to record every host QR batch submitted during
one scan. Capture is available only in a CUDA build with `--backend cuda`.
`--capture-scan` is zero-based and defaults to 200, so the replay must process at
least 201 scans. The target directory must be new or empty. Serialization affects
the captured scan's latency; exclude this diagnostic run from performance results.

```bash
build-accel/form-replay \
  --input /home/ubuntu/datasets/form-input/stairs.formpc \
  --output /tmp/stairs-qr-capture --backend cuda --threads 32 --limit 201 \
  --capture-qr /tmp/stairs-qr-input --capture-scan 200
$PYTHON benchmarks/inspect_qr_capture.py /tmp/stairs-qr-input
build-accel/form-qr-benchmark --input-dir /tmp/stairs-qr-input --repeat 100 \
  > /tmp/stairs-qr-pinned.csv
build-accel/form-qr-benchmark --input-dir /tmp/stairs-qr-input --repeat 100 \
  --pageable > /tmp/stairs-qr-pageable.csv
```

Files `batch-000000.formqr`, etc. preserve call order and all matrix values.
`manifest.json` records the scan index, exact scan stamp, filenames, and ragged
matrix shapes. An interrupted capture leaves `manifest.json.partial`; the
microbenchmark refuses to replay that incomplete directory. The independent
Python inspector checks shapes against the manifest and emits SHA-256 hashes.

Each batch uses this little-endian layout without alignment padding:

| Bytes | Contents |
|---|---|
| 8 | ASCII `FORMQR01` |
| 4 | uint32 matrix count, nonzero and at most 100,000 |
| repeated 8 + 4 | uint64 rows, uint32 columns (7 or 13) |
| repeated rows × columns × 8 | IEEE754 float64 values in column-major order |

Zero-row matrices are preserved. Rows cannot exceed signed 32-bit capacity;
readers check remaining file size before allocating matrices and reject truncation
or trailing bytes. C++ readers/writers require a little-endian IEEE754 host.

The captured-input microbenchmark retains one reusable GPU instance across all
batches, warms each batch, then alternates CPU/GPU timing order between repeats.
GPU timing covers the completed `compute` call, including packing, required
allocations, transfers, kernels, synchronization, and returned-matrix construction.
CPU timing includes Eigen Householder QR and equally sized zero-padded R outputs.
Input loading and untimed correctness checks are excluded. Both paths must satisfy
`||RᵀR − AᵀA||F / max(1, ||AᵀA||F) <= 1e-9` for every captured matrix.

CSV contains per-batch and aggregate mean/p50/p95/p99 microseconds, CPU/GPU speed
ratio, shapes/counts, host-buffer mode, Gram errors, and checksum. The aggregate
sums the per-batch time at each repeat index; it excludes gaps and input loading
between calls. `--pageable` selects the same binary's pageable diagnostic path;
the default uses pinned host buffers. Without `--input-dir`, the original nine
synthetic workloads and five-column output remain available.

Use `form-qr-tiles CAPTURE_DIR 30` for a randomized, interleaved comparison of
eight tile/block configurations in the same process. The ordinary microbenchmark
also accepts `--first-rows`, `--reduction-rows`, and `--block-threads`; production
defaults are 64, 64, and 32. Input-file microbenchmarks replay expanded matrices,
whereas production packs compact correspondences directly. Use estimator replay
to measure that additional saving.

The initial tuning decisions and limitations are recorded in
[`docs/cuda-tuning-results.md`](../docs/cuda-tuning-results.md). The rejected CUDA
Graph candidate is retained as `experiments/cuda_qr_graphs.patch`; apply it only
to an isolated experiment checkout, rebuild, and set `FORM_CUDA_QR_GRAPHS=1`.
Its graph-instantiation costs are included in replay timing. The standalone direct
Hessian experiment can be built with `compile_direct_hessian.py`; it is not linked
into production FORM.

After a suite finishes, export its verified per-repeat results with:

```bash
$PYTHON benchmarks/form_report.py --suite benchmarks/results/SUITE \
  --output docs/form-acceleration-results
```

The exporter retains failed/missing quality gates, requires complete validated
sequence coverage before publishing weighted latencies, and checks correspondence
counts and per-scan trajectory agreement against the reference.
Pass multiple directories after `--suite` to combine current and enlarged-workload
suites. Their executable hashes and quality criteria must match. The added scaling
table uses measured correspondence/pose counts, compares CUDA with the optimized
CPU summaries at matching settings, and reports RTE changes relative to current CUDA.

### Selective dense-solver experiment

A CUDA-enabled replay accepts `--cuda-solve-min-dimension 240` to opt into FP64
cuSolver for dense systems with at least 240 scalar unknowns (40 six-DoF poses).
The flag is independent of `--backend`; smaller systems retain CPU Eigen LLT.
It is disabled by default. Numerical GPU rejection falls back to the original
CPU solve; CUDA runtime errors remain visible. The replay CSV records
`cuda_solve_calls` and `cuda_solve_fallbacks` even without `--profile`.

The corresponding C++/Python parameters are `use_cuda_dense_solver` (default
false) and `cuda_solve_min_dimension` (default 240). N240 is the selected benchmark
threshold, not a measured exact crossover or a portable desktop-GPU threshold.
See `cuda_solve_experiment.md` for captured-system evidence. The completed suite
enabled this option only for CUDA window40 and measured lower optimization time
than CPU summaries in both repeats on all four sequences; the comparison includes
GPU summary preparation as well as solving.

The window workload uses recent40 after a recent20 pilot on stairs reduced the
actual mean pose count (27.86→23.53 after warmup), because recent-window size also
affects keyscan promotion. Report actual active poses and correspondences; a
larger setting is not proof of a larger optimization problem. The recent20 pilot
is preserved separately in `results/tuning/window20-stairs-control-profile.*`.

Feature density uses `--feature-spacing 2` as well as caps6/100: doubling caps
alone did not increase planar selection on stairs. `feature_spacing=0` inherits
`neighbor_points` and preserves original behavior. Positive values change only
the two feature-suppression loops; curvature, input-validity masks, and normal
neighborhoods keep the original `neighbor_points`. Spacing1 marks the selected
sample only, spacing2 also marks the adjacent sample on each side. Explicit
spacing may not exceed `neighbor_points`. Final scaling must report retained
counts, not assume feature count doubles when caps double.

### Resident matrix pipeline

The experimental `summary-resident` backend combines CPU QR preparation with a
direct dense CPU evaluator and LM controller. `cuda-resident` uses CUDA QR and
retains frozen terms, assembly, damping, Cholesky and model errors on the device.
Pose retraction and LM decisions remain on CPU. The original backend is unchanged.

`cuda-resident-hybrid` retains CUDA QR preparation and selects the whole matrix
pipeline before assembly: CPU below 240 scalar unknowns, resident CUDA at/above
240. Override this policy with `--cuda-solve-min-dimension N`; it uses the actual
optimized pose count, including unary ablation mode. It is a measured experimental
policy for the test host, not an automatically tuned or portable crossover.
Resident numerical factorization failures follow the LM damping retry policy;
they do not invoke the legacy CPU solve fallback.

```bash
python benchmarks/run_suite.py --binary build-accel/form-replay \
  --output benchmarks/results/my-resident-comparison \
  --sequences stairs --configs current features window \
  --backends reference summary-resident cuda-resident-hybrid \
  --threads 32 --repeats 2 --limit 250 --cuda-solve-min-dimension 240
```

Use a new output directory. Omit `--cuda-solve-configs` for the hybrid: it selects
by dimension for every workload, and the suite rejects that incompatible option.
For the all-CUDA resident experiment, select `cuda-resident` without a selective
solver flag. Repeat with full sequences before claiming the complete quality gate.

C++ callers set `use_resident_optimizer=true`; `use_cuda_summaries` selects the
summary preparation backend and, by default, the resident matrix backend. Setting
`use_cuda_dense_solver=true` instead selects the matrix backend by
`cuda_solve_min_dimension`. Set `use_batch_summaries=true` to match replay's batching
of previous matches before freezing. Resident/batch flags are currently C++/replay
options; they are not exposed in the evalio Python wrapper.

See [resident results](../docs/resident-cuda-results.md) for all-CUDA and hybrid
comparisons, actual correspondence/pose scaling, validation and frozen binaries.

### CUDA matching with direct summaries

`cuda-matching` selects the same optimizer policy as `cuda-resident-hybrid` and
also accelerates correspondence search. It uploads a snapshot of FORM's world
voxel map and query features once per incoming scan, then reuses them through ICP.
A warp cooperatively searches each query's 27 neighbor voxels, retaining the CPU
first-hit rule on distance ties. Accepted matches become compact feature rows on
GPU and feed FP64 QR directly. There is no full correspondence upload for QR.

Accepted query indices are grouped on GPU in stable query order. During ICP the
host receives group counts and QR roots; raw matches and correspondence arrays
are materialized once after the final rematch, or on demand through raw factor
evaluation. World-to-local target preparation also runs on GPU. The world voxel
map is still built on CPU, and the optimizer still uses the hybrid policy above.
The existing CPU empty-feature behavior (retaining previous raw matches) is also
preserved for comparison; this change does not independently repair that behavior.

```bash
python benchmarks/run_suite.py --binary build-accel/form-replay \
  --output benchmarks/results/my-matching-comparison \
  --sequences stairs --configs current features window \
  --backends reference summary-resident cuda-resident-hybrid cuda-matching \
  --threads 32 --repeats 2 --limit 250 --cuda-solve-min-dimension 240
```

Use the same threshold for both CUDA backends; `--cuda-solve-configs` is rejected
for both because they select their matrix pipeline by dimension at every workload.
C++ callers enable `params.matcher.use_cuda` together with CUDA summaries and the
hybrid optimizer settings above. CUDA-disabled builds explicitly reject this path.
This option is currently exposed through C++/replay, not the evalio Python wrapper.

Summary construction moves from `semi_ms` into `match_ms` in this backend. Compare
`match_ms + semi_ms + full_ms`, and include `map_ms` to account for snapshot/upload
cost. Neither `match_ms` nor `optimization_ms` alone isolates the change. Total
processing time includes feature extraction, mapping and marginalization as well.

To interleave a frozen previous matching executable with fresh CPU and CUDA
controls, add `--previous-matching-binary /path/to/frozen/form-replay` and include
`cuda-matching-previous` in `--backends`. The runner invokes its existing
`cuda-matching` option, records both binary hashes, and reverses backend order on
the second repeat. All workload and solver-threshold arguments remain identical.

See [matching residency results](../docs/cuda-matching-residency-results.md) for
implementation details, measured comparisons, and correctness coverage.

### Remaining-stage diagnostics

`form-replay --profile` also emits extraction substage wall timers, a deterministic
approximately 1/64 sample of normal-construction worker durations, graph and
resident-reset/error wall timers, CUDA map preparation timers, and final match
materialization time. Normal sample CPU durations are nested inside the normal
wall stage; do not sum them with wall stages. Detailed counters remain zero when
profiling is disabled. Profiling measurements and the next runtime estimate are
in [the remaining optimization report](../docs/remaining-optimization-profile.md).

## Hilti 2022 and Multi-Campus inputs

The downloaded `hilti_2022/basement_2` and `multi_campus/tuhh_day_04` sequences use
sensor-specific evalio conversion, including Hilti point reordering/padding and
MCD end-to-start timestamp adjustment. Ground truth is exported in the LiDAR
frame with the existing calibrated transform and quaternion normalization.

```bash
PYTHON=/home/ubuntu/.local/share/uv/tools/evalio/bin/python
$PYTHON benchmarks/prepare_evalio.py \
  --data-root benchmarks/results/additional-datasets \
  --output benchmarks/results/additional-input
$PYTHON benchmarks/run_suite.py \
  --binary build-accel/form-replay \
  --input-dir benchmarks/results/additional-input \
  --output benchmarks/results/additional-current \
  --sequences basement_2 tuhh_day_04 --configs current \
  --backends reference summary-resident cuda-matching \
  --threads 32 --repeats 2 --cuda-solve-min-dimension 240
$PYTHON benchmarks/form_report.py \
  --suite benchmarks/results/additional-current \
  --output benchmarks/results/additional-current/final
```

All backends receive the same sensor range limits from input metadata: 0.5–120 m
for Hilti and 0.1–120 m for MCD, matching evalio's dataset settings. Existing N21
metadata without `replay_settings` keeps the replay's original 0.1–50 m defaults.
The preparation adapter writes evalio XYZ coordinates as float32, in normalized
row-major order, without deskewing. It records complete-sequence coverage,
source files, dimensions, sample point hashes, timestamps, and calibration.
Sensor normalization and input-file loading are outside estimator timings.

### GPU-assisted feature extraction

`--backend cuda-extraction` extends `cuda-matching` with CUDA curvature and
batched exact nearest-point searches on adjacent scanlines. The ordered feature
selector and Eigen covariance/eigenvector calculations remain on the CPU. Rows
are selected in parallel, but sectors within a row keep their original order and
suppression rules. The extraction timer includes scan packing, uploads, downloads,
and CPU completion; this is a hybrid extractor, not a fully GPU-resident one.

The controls separate CPU parallelism from GPU offload:

| Backend | Feature extraction | Matching/optimizer |
|---|---|---|
| `reference` | Original CPU | Original FORM CPU |
| `summary-resident` | Original CPU | Optimized CPU |
| `cpu-extraction` | CPU with row-parallel selection | Optimized CPU |
| `cuda-matching` | Original CPU | Existing CUDA matching/hybrid optimizer |
| `cuda-selection` | CPU with row-parallel selection | Existing CUDA matching/hybrid optimizer |
| `cuda-extraction` | CUDA curvature/search plus row-parallel CPU selection and CPU normal completion | Existing CUDA matching/hybrid optimizer |

```bash
python benchmarks/run_suite.py \
  --binary build-accel/form-replay \
  --input-dir /path/to/form-input \
  --output benchmarks/results/my-extraction-comparison \
  --sequences stairs --configs current features window \
  --backends reference cpu-extraction cuda-matching cuda-selection cuda-extraction \
  --threads 32 --repeats 2 --limit 250
```

Omit `--limit` and use `--configs current` for a complete-sequence run. Replays are
sequential and reverse backend order on the second repeat. Use the same range,
feature-spacing, point/plane caps, and pose-window settings for every backend.

For C++, independently set `params.extraction.use_cuda` and
`params.extraction.parallel_selection`; both default to false. In evalio FORMDev,
the corresponding settings are `use_cuda_extraction` and `parallel_selection`.
The standalone Python extraction parameters expose `use_cuda` and
`parallel_selection`. CUDA extraction requires a `FORM_ENABLE_CUDA=ON` build;
CPU-only builds reject an explicit CUDA request. Buffer capacity is reused across
scans and calls sharing an extractor serialize access to its CUDA workspace.

Measured full-sequence timings, matched scaling controls, and validation details
are in [CUDA extraction results](../docs/cuda-extraction-results.md).
