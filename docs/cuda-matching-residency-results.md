# CUDA matching residency

This extends FORM's exact CUDA matching backend with device grouping, deferred raw correspondence construction, compact search coordinates, parallel voxel probes, and GPU preparation of local target coordinates. It preserves the selected points, pose windows, search neighborhoods, strict thresholds, tie rules, FP64 summary objective, and LM controller.

## Complete 1190-scan stairs result

Two reversed repeats at current settings, with the first 20 scans excluded from timing (1170 timed scans per run):

| Backend | Mean total ms/scan | Per-run means | Mean per-repeat p95 |
|---|---:|---:|---:|
| Original FORM CPU |155.76|158.24 / 153.29|272.90|
| Previous CUDA matching |70.80|68.85 / 72.74|118.89|
| Final CUDA matching |50.64|50.46 / 50.82|78.66|

This is **1.40x faster than the previous CUDA matcher and 3.08x faster than original FORM across all 1190 scans**. Source: `benchmarks/results/matching-reuse-stairs-full/final.json`; all six runs are verified. No fresh full-sequence optimized-CPU control was included; optimized CPU comparisons are in the matched 250-scan scaling table below.

Both final CUDA/reference pairs pass the complete 1 m and 30 m trajectory quality gate with full pose coverage (970 short segments and 410 long segments per run). Mean translation RTE is 0.034707355 m over 1 m and 0.236247974 m over 30 m, matching original FORM to displayed precision. Every per-scan feature, factor, correspondence, pose, rematch and LM-iteration count agrees. Maximum final-CUDA position difference is <1.18e-11 m; maximum rotation difference is <1.73e-12 rad.

Full-sequence matching drops 32.01 → 12.44 ms, including QR and raw materialization. Preparation is 4.05 → 4.25 ms and optimization 17.55 → 17.73 ms. Thus preparation changes have a small, mixed measured effect: the full-sequence map timer regresses 0.19 ms, while the overall estimator saves 20.16 ms. The dominant gain comes from rematch grouping and deferred raw work. This report does not claim every individual change speeds every workload.

Full-sequence acceptance here covers stairs at current settings. The other three downloaded sequences and full-length enlarged workloads remain unvalidated by this experiment.

## Final paired scaling results

All numbers below are mean total estimator milliseconds per scan: first 250 stairs scans, first 20 excluded from timing, two reversed repeats, 32 CPU threads. They include extraction, map preparation, matching, optimization, and maintenance; input-file I/O is outside the estimator timer. Each row uses identical workload settings across all four backends.

| Workload | Original FORM CPU | Optimized CPU | Previous CUDA | Final CUDA | Gain over previous CUDA |
|---|---:|---:|---:|---:|---:|
| Current |108.04|73.94|45.62|33.90|1.35x|
| Denser features |327.52|227.42|99.06|62.42|1.59x|
| Larger window |187.96|106.30|74.34|58.31|1.27x|

Final CUDA is 3.19x/5.25x/3.22x faster than the original FORM CPU path, and 2.18x/3.64x/1.82x faster than optimized CPU. Source: `benchmarks/results/matching-reuse-pilot/final.json` (24 verified runs). These are estimator speedups, not optimization-only timings or full-sequence results.

Matching and map-preparation means, previous CUDA → final CUDA:

| Workload | Matching, including QR and final raw data | Map preparation |
|---|---:|---:|
| Current |20.01 → 8.42 ms|4.83 → 4.74 ms|
| Denser features |54.49 → 20.95 ms|12.56 → 10.75 ms|
| Larger window |23.90 → 9.98 ms|6.28 → 5.95 ms|

Preparation gains are modest; the dominant improvement is avoiding repeated bulk match downloads and raw-array construction. Mean per-repeat p95 total times are 44.47/82.01/75.92 ms. The denser-feature fraction over 100 ms falls from 36.52% to 0.43% on this prefix; full-sequence enlarged workloads have not been evaluated.

The current workload averages 168,122 correspondences and 14.13 poses. Denser selection uses point/plane limits 6/100 and spacing 2 instead of 3/50 and spacing 5, producing 493,474 correspondences (2.94x) and 14.26 poses. The larger window raises recent scans 10 → 40, producing 520,164 correspondences (3.09x) and 38.56 poses (2.73x). These are actual mean counts over 250 scans, not merely configured size multipliers.

All 18 candidate/reference trace comparisons preserve every per-scan feature, correspondence, factor, pose, rematch and LM-iteration count. Maximum position difference is <6.14e-12 m. The prefix provides 1 m RTE segments but no 30 m segments, so its complete quality gate is unavailable rather than passed.

## Implementation

- Search reads four padded coordinate arrays instead of striding through 80-byte target records. The 27 voxel hash probes run across warp lanes. Occupied voxels are visited in original order, with exact distance/tie reduction. A 64-thread block won the tested A100 launch sweep over 64/128/256/512 threads.
- Accepted matches are grouped by target scan on device. A stable CUB radix sort preserves query order within each group. Group counts and compact QR roots return during each rematch; the full results stay on device.
- Correspondence owners expose counts immediately and materialize raw arrays on demand. Normal estimator operation materializes the final rematch once, before full optimization and map insertion. Surviving copies retain their data across rematches, resets, and destruction. Concurrent raw reads and retry after a loading error are covered by tests.
- Snapshot preparation converts world targets and normals back to local coordinates on GPU, fused with extraction of compact search coordinates. CPU conversion occurs only when final raw matches are requested. The CPU still builds the world voxel map, preserving its within-voxel insertion order.

The public `CudaMatching::match` remains eager by default; the estimator opts into deferred output. Its constraint map must cover all map scans. Direct access to deferred correspondence arrays requires `ensureRaw()` first. Raw factor evaluation and summary construction do this automatically. Mutation remains serialized.

`match_ms` includes device grouping, QR construction, and final raw materialization. No deferred work is omitted from total timing. `map_ms` includes snapshot packing, upload, and local target preparation.

## Verification and diagnostic measurements

Hardware: NVIDIA A100 80 GB PCIe and 32-vCPU AMD EPYC Milan VM; CUDA 13.0, FP64, matcher compiled with FMA contraction disabled.

87 CUDA C++ tests pass. CPU-only tests: 42 pass, 8 CUDA-dependent skips. 30 Python benchmark tests pass. The 18 matcher/adapter tests before buffer reuse pass Compute Sanitizer memcheck, synccheck, and racecheck with zero errors or hazards. After reuse and the added reset regression, all six adapter tests pass memcheck and racecheck. The current source has 13 matcher tests and six adapter tests.

A seeded 8193-query synthetic search benchmark checks every result index and distance bit pattern against the frozen previous kernel. Selected 64-thread timings improved 2.20x for sparse voxels, 2.11x for 32-point dense voxels, and 4.31x for 128-point dense voxels. Register use was 48/thread, with no shared memory and 62.5% theoretical occupancy for 64-thread blocks. Nsight Compute hardware counters were unavailable (`ERR_NVGPUCTRPERM`), so there is no measured DRAM-bandwidth claim. Hash probes remain indirect accesses; coordinate reads within voxel rows are now contiguous per warp and each coordinate plane is 256-byte aligned. No vectorized-load alignment assumptions are introduced.

The search-only replay diagnostic (first 100 stairs scans; first 20 excluded; two reverse repeats) did not establish a whole-scan improvement: previous 42.07/38.73 ms versus search-only 40.75/41.97 ms. The combined implementation's single 100-scan smoke run averaged 28.57 ms total, 4.16 ms map, and 6.59 ms match. These diagnostics are separate from formal repeated comparisons. The combined smoke had identical per-scan workload/iteration counts and maximum position difference 5.92e-15 m against the previous backend.

The tuned previous matcher profile over 100 scans recorded 201.84 ms nearest-kernel time, 130.76 ms QR and 6.11 ms feature packing. These are accumulated GPU times, not mean estimator times.

## Reproducibility

Implementation commits: `d325e66` deferred raw storage, `d76f359` search layout/probes, `24c6da0` grouping/local transforms, and `8a4af10` estimator integration. Benchmark support `c404345` interleaves the previous frozen matcher with current CPU/CUDA controls, reverses backend order on the second repeat, records both binary hashes, and checks the selected binary before every run.

Initial combined binary: `benchmarks/results/matching-residency-backend/form-replay`, SHA256 `77d7a1f297aeb09d932d53149c05ee5173ac4ca00cb3416139964b0ab140485f`; source archive and validation logs are alongside it. Previous matcher: `benchmarks/results/matching-backend/form-replay` at `c4d1064`, SHA256 `ba684ea5141a8470034ede38891978a6a8701b51eeff89fae18c60f3c8fe1c8a`. Search-only frozen binary/source and self-contained kernel sweep sources live in `benchmarks/results/matching-search-only`.

## Initial combined benchmark, before buffer reuse

`matching-residency-pilot/final.json` records 24 interleaved runs: first 250 stairs scans, first 20 excluded from timing, two reversed repeats, four backends, three matched workloads. Mean total milliseconds per scan:

| Workload | Original FORM | Optimized CPU | Previous CUDA matching | Initial combined CUDA |
|---|---:|---:|---:|---:|
| Current |106.14|73.03|44.99|35.56|
| Denser features |319.53|217.55|105.67|67.52|
| Larger window |191.32|105.06|71.26|57.05|

All 18 candidate/reference trace comparisons retained every workload and iteration count. Maximum position difference across all comparisons was <4.78e-12 m. This short prefix contains no 30 m evaluation segments; it does not establish the complete trajectory quality gate.

Current-workload map preparation regressed from 4.59 to 5.52 ms even though matching fell 19.44 to 8.96 ms. This prompted a follow-up to retain uniquely owned host snapshots, upload staging, and materialized match capacity between scans. Leased batches and snapshots remain immutable. No work is moved between timers to manufacture a preparation improvement.

Buffer reuse and failed-reset recovery are committed in `ae92335`. The updated frozen binary is `benchmarks/results/matching-reuse-backend/form-replay`, SHA256 `ab0ed896cf55d0f0f137b5e7beac76b94a3002017a964c64edf030c66bc6dcac`. All 87 CUDA tests pass; the six adapter tests also pass memcheck and racecheck after this change. A reset failure blocks subsequent matching without mutating retained constraints until a successful reset.

## Final CUDA profile

Nsight Systems on the same first 100 stairs scans records nearest-search time 201.84 → 126.72 ms (1.59x), while QR remains 130.76 → 131.31 ms. All GPU kernels together take 338.71 → 296.00 ms; device grouping adds some work while removing host processing. These are accumulated kernel times, separate from replay latency.

Device-to-host traffic falls 252.39 → 44.47 MB (82.4% less); host-to-device traffic falls 331.13 → 291.37 MB. Small count/error downloads remain, so transfer-call count is not a measure of residency. The optimized path still returns counts and compact roots during each rematch and bulk results once per scan. Artifacts: `matching-reuse-backend/nsys.csv` and `transfer-comparison.json`; baseline trace is `/tmp/form-matching-warp-baseline.sqlite`.

The synthetic 2.1–4.3x search-kernel improvements do not translate directly into full-estimator speedups. The measured gains come primarily from changing where data is grouped and when host raw data is built. On the 250-scan prefix, feature extraction now consumes about 43% of current-workload time and 38% of denser-workload time; larger-window optimization consumes about 45% of its total. Matching has not been proven globally optimal, but the three approved changes are implemented and measured.
