# CUDA matching residency

This extends FORM's exact CUDA matching backend with device grouping, deferred raw correspondence construction, compact search coordinates, parallel voxel probes, and GPU preparation of local target coordinates. It preserves the selected points, pose windows, search neighborhoods, strict thresholds, tie rules, FP64 summary objective, and LM controller.

## Implementation

- Search reads four padded coordinate arrays instead of striding through 80-byte target records. The 27 voxel hash probes run across warp lanes. Occupied voxels are visited in original order, with exact distance/tie reduction. A 64-thread block won the tested A100 launch sweep over 64/128/256/512 threads.
- Accepted matches are grouped by target scan on device. A stable CUB radix sort preserves query order within each group. Group counts and compact QR roots return during each rematch; the full results stay on device.
- Correspondence owners expose counts immediately and materialize raw arrays on demand. Normal estimator operation materializes the final rematch once, before full optimization and map insertion. Surviving copies retain their data across rematches, resets, and destruction. Concurrent raw reads and retry after a loading error are covered by tests.
- Snapshot preparation converts world targets and normals back to local coordinates on GPU, fused with extraction of compact search coordinates. CPU conversion occurs only when final raw matches are requested. The CPU still builds the world voxel map, preserving its within-voxel insertion order.

The public `CudaMatching::match` remains eager by default; the estimator opts into deferred output. Its constraint map must cover all map scans. Direct access to deferred correspondence arrays requires `ensureRaw()` first. Raw factor evaluation and summary construction do this automatically. Mutation remains serialized.

`match_ms` includes device grouping, QR construction, and final raw materialization. No deferred work is omitted from total timing. `map_ms` includes snapshot packing, upload, and local target preparation.

## Verification and diagnostic measurements

Hardware: NVIDIA A100 80 GB PCIe and 32-vCPU AMD EPYC Milan VM; CUDA 13.0, FP64, matcher compiled with FMA contraction disabled.

87 CUDA C++ tests pass. CPU-only tests:42 pass, 8 CUDA-dependent skips. 30 Python benchmark tests pass. All 18 matcher/adapter tests pass Compute Sanitizer memcheck, synccheck, and racecheck with zero errors or hazards.

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

All 18 candidate/reference trace comparisons retained every workload and iteration count. Maximum position difference across all comparisons was 4.78e-12 m. This short prefix contains no 30 m evaluation segments; it does not establish the complete trajectory quality gate.

Current-workload map preparation regressed from 4.59 to 5.52 ms even though matching fell 19.44 to 8.96 ms. This prompted a follow-up to retain uniquely owned host snapshots, upload staging, and materialized match capacity between scans. Leased batches and snapshots remain immutable. No work is moved between timers to manufacture a preparation improvement.

Buffer reuse and failed-reset recovery are committed in `ae92335`. The updated frozen binary is `benchmarks/results/matching-reuse-backend/form-replay`, SHA256 `ab0ed896cf55d0f0f137b5e7beac76b94a3002017a964c64edf030c66bc6dcac`. All87 CUDA tests pass; the6 adapter tests also pass memcheck and racecheck after this change. A reset failure blocks subsequent matching without mutating retained constraints until a successful reset.
