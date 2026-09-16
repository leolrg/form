# CUDA-assisted feature extraction results

The new hybrid extractor reduces full-stairs extraction time from **15.35 to 6.78 ms/scan (2.26x)** and total estimator time from **49.58 to 40.30 ms/scan (1.23x)** relative to the previous CUDA pipeline. Against original FORM in the same experiment, total acceleration is **3.95x** (159.28 / 40.30). These are measured A100 results, not GeForce estimates.

## Scope and controls

- Source revision: `e4a49c7` (implementation `d2b4bb3`, reviewed arithmetic/concurrency fixes `e4a49c7`).
- NVIDIA A100 80 GB PCIe; AMD EPYC Milan VM with 32 CPU threads; CUDA 13.0; existing hybrid optimizer and 240-variable solver threshold unchanged.
- Ten full-sequence runs: five backends, two sequential repeats in reversed order, all 1,190 Newer College 2021 stairs scans; first 20 excluded from timing (1,170 timed scans).
- Twenty scaling runs: the same five backends and two reversed repeats, first 250 stairs scans with denser features or a larger window; first 20 excluded (230 timed scans).
- All workloads use identical settings across CPU/CUDA controls. Current: point/plane caps 3/50, spacing 5, recent-scan cap 10. Denser: 6/100, spacing 2. Larger window: recent-scan cap 40. Sensor range 0.1–50 m.
- No builds, sanitizer runs, conversions, or other benchmarks overlapped these timed runs. Initial CUDA runtime startup is present in raw traces but excluded by the 20-scan warmup.
- Total estimator timing includes extraction, map preparation, matching/QR/raw materialization, optimization, and maintenance. Input-file loading and output serialization remain outside the estimator timer.

## Complete-sequence results

Each entry is the mean of two per-run means, in milliseconds per scan.

| Backend | Extraction | Total | Original CPU / total |
|---|---:|---:|---:|
| Original FORM CPU | 16.80 | 159.28 | 1.00x |
| Optimized CPU + parallel selection | 12.00 | 119.29 | 1.34x |
| Previous CUDA pipeline | 15.35 | 49.58 | 3.21x |
| Previous CUDA + parallel CPU selection | 11.54 | 46.00 | 3.46x |
| CUDA-assisted extraction | 6.78 | 40.30 | 3.95x |

The new total repeats are **40.50 / 40.11 ms**, and extraction repeats are **6.84 / 6.72 ms**. Mean per-repeat total p95 falls from 77.13 ms for the previous CUDA pipeline to 67.78 ms.

The selection-only control separates CPU work from GPU offload: row-parallel CPU selection reduces total time from 49.58 to 46.00 ms; adding CUDA curvature and normal-neighborhood search reduces it to 40.30 ms. Thus the GPU addition beyond the improved CPU selector is **1.14x end-to-end**. The pure CPU control also receives the improved selector; it does not use CUDA matching or optimization.

## Scaling on the same 250-scan prefix

The current-workload row below is computed from scans 0–249 of the full runs (timing scans 20–249), so every row covers the same scan range. These prefix measurements must not be presented as full-sequence enlarged-workload results.

| Workload | Original CPU | Optimized CPU + parallel selection | Previous CUDA | Selection-only CUDA control | New CUDA extraction | Previous CUDA / new |
|---|---:|---:|---:|---:|---:|---:|
| Current | 110.65 | 75.31 | 33.09 | 28.92 | 25.54 | 1.30x |
| Denser features | 338.32 | 237.31 | 63.45 | 60.51 | 48.75 | 1.30x |
| Larger window | 188.47 | 105.68 | 59.37 | 56.76 | 50.19 | 1.18x |

| Workload | Previous CUDA extraction ms | New extraction ms | Mean active poses | Mean graph correspondences |
|---|---:|---:|---:|---:|
| Current | 14.06 | 6.74 | 14.60 | 170,469 |
| Denser features | 23.85 | 9.95 | 14.74 | 503,056 |
| Larger window | 14.39 | 7.42 | 41.00 | 549,980 |

## Implementation

- A reusable CUDA workspace uploads the organized scan and the CPU validity mask once per extraction. CUDA computes curvature with the original double accumulation order and returns scores for selection.
- Each scanline is selected on a CPU worker. Sectors within a row retain their original sort order, greedy selection, suppression dependencies, and quota behavior. Byte masks prevent races between neighboring rows that would share packed `vector<bool>` words. Selected indices are concatenated in original row order.
- For planar candidates, CUDA performs exact full-row nearest searches on the preceding and following scanlines. One warp processes each query/adjacent-row pair. Distances retain input float/double precision and CPU Eigen reduction order; first-index ties are preserved.
- The CPU gathers the short neighborhoods and runs the original Eigen covariance/eigenvector calculation. This retains normal components rather than substituting a new approximate eigensolver.
- Curvature and normal-search buffers are reused across scans. Concurrent calls sharing an extractor serialize CUDA-workspace access. CPU-only builds reject an explicit CUDA request.
- This is **hybrid GPU-assisted extraction**, not full GPU residency. Validity masks, ordered selection, final normal construction, and output feature objects remain on CPU; outputs still feed the existing map/matcher adapter.

## Final detailed profile

One separate full-sequence profiled replay measured 40.16 ms total and 6.90 ms extraction. It is diagnostic, not included in the paired timing means above. The profile preserves all workload/iteration counts, with maximum position difference below 8.6e-13 m relative to the unprofiled new backend.

| Extraction component | Profiled ms/scan |
|---|---:|
| CPU validity mask | 0.54 |
| Scan packing, upload, CUDA curvature, download, curvature records | 1.88 |
| Parallel CPU sector sorting/planar selection | 1.05 |
| CPU point mask | 0.71 |
| CPU point selection | 0.41 |
| CUDA adjacent-row search plus CPU neighborhood/covariance/eigenvectors | 1.92 |
| Feature output packing | 0.19 |
| Other initialization/cleanup | 0.20 |
| **Whole extraction** | **6.90** |

The initial offload-only 100-scan diagnostic (before row-parallel selection) reduced normal construction from 6.14 to 1.74 ms while planar selection remained about 4.85 ms. That motivated the independent-row CPU change. The timings above include all extraction transfers; kernel-only timing is not used as the speedup claim.

## Correctness and verification

- All **30 runs** completed with verified output counts and timestamps. Across all **24 accelerated/reference comparisons**, every per-scan feature, pose, correspondence, factor, rematch and LM-iteration count agrees.
- Full sequence: maximum translation difference **1.179e-11 m**, maximum rotation difference **1.770e-12 rad**. Both repeats of every backend pass the complete 1 m / 30 m trajectory quality and coverage gates.
- Full-sequence RTE matches original FORM to displayed precision: 1 m mean translation RTE 0.034707 m; 30 m 0.236248 m. All 1,190 poses match ground truth.
- Enlarged prefixes: maximum translation difference **6.535e-12 m**, rotation difference **6.694e-13 rad**. The short prefix has no 30 m evaluation segments, so its complete quality gate is unavailable, not passed. Counts and trajectories still agree with the matched reference.
- **105 C++ tests pass** in the CUDA build: 94 main tests, two isolated parallel/concurrent tests, nine scalar-Eigen parity tests. **45 CPU-only tests pass**, with eight CUDA-dependent skips. **32 Python benchmark tests pass**.
- The six new CUDA/helper integration tests pass Compute Sanitizer memcheck, racecheck, and synccheck with zero errors/hazards. Python/evalio bindings build and the extraction options round-trip.
- Independent review identified reduction-order portability and a single-threaded-test gap. Fixes add an Eigen-derived reduction policy, adversarial rounding-sensitive ties, a scalar-Eigen executable, actual TBB worker observation, and concurrent calls using different scans and scan IDs.

## Reproduce

Use `--backend cuda-extraction`; `cuda-selection` isolates the CPU-selector change and `cpu-extraction` is the improved all-CPU control. C++ settings are `params.extraction.use_cuda` and `params.extraction.parallel_selection`; evalio settings are `use_cuda_extraction` and `parallel_selection`. Both options remain off by default. See [benchmark instructions](../benchmarks/README.md#gpu-assisted-feature-extraction).

- Frozen executable: `benchmarks/results/extraction-reviewed-backend/form-replay`.
- SHA256: `db7bd70d8563b904979fda012939aa9b6192d842379e640a66e2316b629f0fe4`.
- Full controls: `benchmarks/results/extraction-stairs-full/final.json` and `.md`.
- Scaling controls: `benchmarks/results/extraction-scaling/final.json` and `.md`.
- Raw CSV/TUM, commands, settings, build/environment fingerprints, and quality comparisons are in those suite directories.
- Summary and verification: `benchmarks/results/extraction-reviewed-backend/summary.json`, `current-prefix.json`, `verification.json`, and test/sanitizer logs.
- Detailed profile: `benchmarks/results/extraction-final-profile/analysis.json` and `stairs.csv`.

Generated benchmark artifacts and datasets are ignored by Git. The implementation, tests, benchmark integration, and this report are committed. Newer College stairs is the full-sequence validation for this change; Hilti/MCD have not been rebenchmarked with GPU extraction.
