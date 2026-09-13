# CUDA matching and direct summary results

This report records the first CUDA matching implementation at `c4d1064`. The
subsequent [matching residency work](cuda-matching-residency-results.md) adds GPU
grouping, deferred raw data, faster search, and local target preparation.

GPU matching reduces total estimator processing time from **68.39 to 50.05 ms**
at current settings, **216.75 to 102.17 ms** with denser features, and
**93.40 to 72.75 ms** with a larger window, relative to the preceding hybrid
optimizer. These are **1.37× / 2.12× / 1.28× end-to-end speedups** in a paired
250-scan screening experiment. This is additional acceleration beyond the exact
factor summaries and resident/hybrid optimizer already implemented.

## Paired scaling experiment

- A100 80 GB PCIe; 32-vCPU EPYC Milan VM; 32 TBB threads.
- Newer College 2021 `stairs`, first 250 scans, two reversed-order repeats for
  every backend/workload: 24 completed runs. First 20 scans excluded from timing.
- Frozen source `c4d1064`; binary SHA256
  `ba684ea5141a8470034ede38891978a6a8701b51eeff89fae18c60f3c8fe1c8a`.
- Identical extraction, correspondence threshold, search domain, objective,
  FP64 precision, LM policy and stopping criteria across each comparison.
- CPU and GPU controls use identical point/window settings. Both hybrid backends
  use the same explicit 240-scalar-variable CPU/GPU matrix threshold.

All table entries are **mean processing milliseconds per scan**. The optimized
CPU column is `summary-resident`, with TBB CPU matching and direct CPU LM.
The previous hybrid column uses CPU matching with `cuda-resident-hybrid`.

| Workload | Original FORM CPU | Optimized CPU | Previous hybrid | GPU matching | Speedup vs optimized CPU | Speedup vs previous hybrid |
|---|---:|---:|---:|---:|---:|---:|
| Current | 112.89 | 83.52 | 68.39 | **50.05** | 1.67× | 1.37× |
| More points | 349.60 | 252.17 | 216.75 | **102.17** | 2.47× | 2.12× |
| Larger window | 196.82 | 109.82 | 93.40 | **72.75** | 1.51× | 1.28× |

Relative to original FORM, total speedups are **2.26× / 3.42× / 2.71×**.
These include the earlier optimization changes; they are not GPU-kernel-only
speedups. Total time excludes replay loading/serialization but includes feature
extraction, map construction, matching, optimization, marginalization and upkeep.

Summary construction moves from the optimization timer into the matching timer.
The fair subsystem comparison therefore includes both, plus the map snapshot and
upload cost. Comparing either matching or optimization alone misattributes work.

| Map preparation + matching + optimization | Optimized CPU | Previous hybrid | GPU matching |
|---|---:|---:|---:|
| Current | 66.70 | 53.90 | **33.77** |
| More points | 225.62 | 189.14 | **74.86** |
| Larger window | 93.71 | 77.53 | **56.83** |

The new backend's map preparation costs 5.91 / 13.56 / 6.22 ms, versus
4.59 / 10.20 / 5.61 ms for the previous hybrid. These costs are included above.
Other stages also vary between runs; the total difference is not attributed solely
to the search kernel.

The GPU-matching repeat means are 50.31/49.79 ms, 98.78/105.56 ms and
72.11/73.39 ms. Two repeats provide screening evidence, not broad statistical
confidence or performance predictions for a consumer desktop GPU.

| GPU matching latency | Current | More points | Larger window |
|---|---:|---:|---:|
| Mean total (ms) | 50.05 | 102.17 | 72.75 |
| Mean of per-repeat p95 (ms) | 72.20 | 161.56 | 101.58 |
| Scans exceeding 100 ms | 0.22% | 41.09% | 6.74% |

Denser points are substantially faster, but this workload does **not** yet meet
100 ms consistently. The mean is also slightly above that budget.

## Workload and numerical agreement

Actual mean stored correspondences are 168,122 / 493,474 / 520,164. Mean retained
poses are 14.13 / 14.26 / 38.56. Thus denser features realize 2.94× correspondences;
the larger window realizes 3.09× correspondences and 2.73× poses. Every backend
was run on the same enlarged workloads, not compared against a smaller CPU case.

Total-time growth relative to each backend's own current-size run is:

| Enlarged workload | Original CPU | Optimized CPU | Previous hybrid | GPU matching |
|---|---:|---:|---:|---:|
| More points | 3.10× | 3.02× | 3.17× | 2.04× |
| Larger window | 1.74× | 1.31× | 1.37× | 1.45× |

The lower GPU-matching current-size baseline is retained in these growth ratios;
absolute GPU-matching latency remains lower for both enlarged workloads.

All 18 accelerated/reference pilot trace pairs have identical feature counts,
correspondence counts, factors, poses, rematches and LM iteration counts. Across
all controls the maximum position difference is 1.19e-11 m; for the new
`cuda-matching` backend specifically it is **3.99e-12 m**. These figures apply to
this pilot, not to all possible future trajectories.

All available 1 m and coverage checks pass. The 250-scan prefix contains no 30 m
segments, so it does not pass the complete predeclared trajectory quality gate.
The full-stairs follow-up below evaluates the complete quality gate at current
settings. Full four-sequence acceptance, and full-sequence acceptance of the two
enlarged workloads, have not been established.

## Complete stairs sequence

The same frozen binary was then run on all **1,190 scans**, with original CPU
and GPU matching each repeated twice in reversed order: four additional runs.
The first 20 rows are again excluded from latency statistics. This is a separate
experiment from the 250-scan scaling table; its longer trajectory retains larger
pose sets during portions of the sequence.

| Backend | Mean total processing (ms/scan) | Mean of per-repeat p95 (ms) |
|---|---:|---:|
| Original FORM CPU | 166.45 | 300.42 |
| GPU matching | **73.40** | **123.42** |

The full-sequence speedup over original CPU is **2.27×**. The previous hybrid and
optimized CPU were not rerun in this full-sequence follow-up; their paired speedup
comparisons above apply to the pilot.

Both GPU/reference pairs **pass the complete predeclared quality gate**: all four
translation statistics at 1 m and 30 m, plus coverage. Each trajectory has 970
valid 1 m segments and 410 valid 30 m segments, with all 1,190 poses matched to
ground truth. Mean RTE is 0.0347074 m at 1 m and 0.236248 m at 30 m for both
backends to the displayed precision.

Every feature, correspondence, factor, pose, rematch and LM iteration count agrees
throughout both complete runs. Maximum position difference is **1.16e-11 m**;
maximum rotation difference is **1.74e-12 rad**. These measurements support
numerical equivalence on this complete sequence, in addition to the factor and
matching tests. They are not a guarantee for every sequence or hardware platform.

- [Verified full-sequence data](../benchmarks/results/matching-stairs-full/final.json)
  and [report](../benchmarks/results/matching-stairs-full/final.md).
- Reproduce with the command in the benchmark README, selecting only
  `--configs current --backends reference cuda-matching` and omitting `--limit`.

## Implementation and validation

- A snapshot of the CPU world voxel map and query features is uploaded once per
  incoming scan and reused across rematches. Target points/normals are transformed
  back to local coordinates once per snapshot with FORM's original transform.
- One warp owns a query and cooperatively scans its neighbor voxels. Reduction
  orders candidates by distance, neighbor order, then point order, preserving
  FORM's first-hit tie rule. Negative floor and strict distance comparisons match
  the CPU path.
- Host bookkeeping groups accepted query indices by scan pair and retains raw
  matches for the existing map/factor APIs. Coordinates remain on device through
  seven-column feature packing and FP64 QR. Producer events order reads without
  overwriting caller buffers. Host roots populate the existing summary caches.
- Map creation, grouping indices, raw-match materialization, and the existing
  host portions of LM remain on CPU. The full estimator is not device resident.
- The original CPU path remains the default. Empty-feature behavior is preserved,
  including the CPU matcher's existing retention of previous raw matches.

The initial thread-per-query kernel was slower in a smoke test. Nsight Systems
attributed 93.7% of its GPU kernel time to nearest search (2.03 s across 100 scans).
Warp-cooperative scanning addresses that serial dense-voxel loop. The formal
paired results above use the corrected, frozen implementation.

Validation: **74 CUDA-build C++ tests pass; CPU-only has 39 passes and 8 CUDA
skips; 29 Python tests pass**. Nine matcher/integration tests pass Compute
Sanitizer memcheck and racecheck with zero errors/hazards. Coverage includes all
27 neighbors, negative coordinates, thresholds, ties, dense 33/65/257/4097-point
voxels, query tails, map replacement, pose changes, invalid input and raw-factor
cost/Hessian agreement. A read-only spec/code review checked lifetimes, cache
revisions, CPU-only behavior and stream/warp semantics.

Commits: `037b548` adds device-input QR; `5b0aae4` adds matching and estimator
integration; `c4d1064` adds cooperative search. Reproduction flags are in
[the benchmark README](../benchmarks/README.md#cuda-matching-with-direct-summaries).

Artifacts:

- [Verified pilot data](../benchmarks/results/matching-pilot/final.json) and
  [per-backend report](../benchmarks/results/matching-pilot/final.md).
- Frozen binary, source archive, sanitizer/test logs and initial kernel profile:
  `benchmarks/results/matching-backend/`.
