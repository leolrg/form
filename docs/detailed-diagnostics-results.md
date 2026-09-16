# Detailed FORM performance diagnostics

This campaign measures the existing implementation before further acceleration. It changes instrumentation and analysis, not FORM's matching, objective, feature selection, LM decisions, or backend threshold. The central finding is that the remaining cost is distributed across CPU optimization, matching orchestration/materialization, extraction, and map preparation. Kernel-only timings are insufficient to choose the next change.

## Scope and reproducibility

- Hardware: NVIDIA A100 80 GB PCIe, AMD EPYC Milan VM, 32 CPU threads, CUDA 13.0. Results are for this machine, not estimates for GeForce GPUs.
- Dataset: Newer College 2021 **stairs**. Full runs process all 1,190 scans; timing excludes scans 0–19, leaving 1,170. Scaling and forced-backend experiments process the same first 250 scans, timing the same 230 scans. Enlarged-workload results are not full-sequence results.
- Current settings: point/plane caps 3/50, feature spacing 5, recent-scan cap 10. Denser features: 6/100, spacing 2. Larger window: recent-scan cap 40 with current feature settings. The active pose count also includes retained keyscans, so it can exceed the recent-scan cap.
- Three controls: `reference` is original FORM's CPU algorithm; `cpu-extraction` includes our summaries, CPU optimizer, and parallel CPU feature selection; `cuda-extraction` is our current hybrid CUDA pipeline. Every comparison uses identical input/settings across these controls.
- The 51 main runs comprise nine clean full runs (three repeats/backend), 18 clean scaling runs (two repeats/backend/workload), four detailed full profiles, 12 detailed scaling profiles, and eight forced CPU/GPU optimizer profiles. Six additional full runs calibrate instrumentation, in old/disabled/enabled/enabled/disabled/old order. Two separate Nsight Systems traces cover full stairs and the larger-window prefix.
- Runs were sequential. No builds, conversions, sanitizer runs, or other benchmarks overlapped timing. This is a VM experiment without fixed-clock control or a CPU affinity tuning study; small timing differences should be treated cautiously.
- Build: Release `-O3 -DNDEBUG`, CUDA architecture 80; no `-march=native`, GTSAM native-architecture optimization disabled, system Eigen and TBB enabled. This is the portable build used by the project, not an exhaustive study of the fastest CPU compiler/ISA configuration.
- The estimator timer includes extraction, map preparation, matching and QR construction, final raw materialization, optimization, and maintenance. Replay input loading and output serialization are outside that timer. This is estimator latency, not total dataset-reader process throughput.
- C++ instrumentation frozen at `a8158cb`; script corrections at `841df28`. Frozen executable SHA256: `2113784cb20af5f0cc312156ee8227e2a9ee186d3d7b582ddb94005a9f5979ac`.
- Raw artifacts: `benchmarks/results/detailed-diagnostics-20260916/`, including commands, settings, binary/build fingerprints, CSV/TUM outputs, per-optimizer CSVs, resource measurements, profiles and Nsight SQLite exports. Generated artifacts and datasets are ignored by Git; this report and analysis code are versioned.

## How to read the timers

The seven top-level estimator stages are disjoint. Their difference from `total_ms` measures work between timers, return/destruction, and instrumentation boundaries. Substage counters are **inclusive host wall times**, not additional independent stages. Parent and child counters must not be added together.

Each optimizer record covers an entire `ConstraintManager::optimize` call, including summary/graph setup and cleanup, and records semi/full phase, system dimension, chosen resident CPU/GPU backend, LM decisions and timings. CPU-selected optimization can still use CUDA during graph preparation; the label identifies the resident optimizer, not every operation in the call.

GPU linearization launches asynchronously. Its host duration is not the duration of its device work; a subsequent solve or error evaluation can pay the wait. Nsight attributes kernels and transfers to their **issuing API and nearest NVTX scope**, not to the later waiting scope. API time and device time overlap. Device-active time is an interval union, avoiding double-counting concurrent activity.

Important nested-counter details:

- Batch reset/topology work occurs under both graph preparation and resident reset. Aggregate batch time cannot be subtracted wholesale from resident reset.
- Batch-root preparation includes GPU reset/allocation/upload/synchronization. It is not pure CPU packing.
- CUDA configuration “upload” includes allocation, workspace queries, synchronization and cleanup. The separate host scope excludes initial synchronization/first solver creation.
- QR execution waits can include upstream sort/packing. Matching row/output/materialization counters have several callers and worker threads; their inclusive totals are not an additive matching budget.
- LM “rejected” means not accepted. It does not by itself imply that another trial was taken. Decision counts are checked against solve attempts.

## Clean full-sequence timing

All values are milliseconds per scan. The mean is the mean of three per-run means; p95/p99 are averages of the three per-run percentiles, not pooled percentiles.

| Backend | Run means | Mean ± repeat SD | Median¹ | p95¹ | p99¹ | Original / backend |
| --- | --- | --- | --- | --- | --- | --- |
| Original CPU | 182.08 / 177.31 / 179.44 | 179.61 ± 2.39 | 167.14 | 318.29 | 371.07 | 1.00× |
| Improved CPU | 131.58 / 131.67 / 129.76 | 131.00 ± 1.08 | 119.27 | 252.41 | 292.05 | 1.37× |
| Current CUDA | 43.63 / 42.74 / 41.13 | 42.50 ± 1.27 | 40.75 | 70.06 | 78.56 | 4.23× |

¹ Average per-run statistic. The new speedup ratio describes this campaign, **not a new algorithmic improvement** over the previous extraction report. Hardware/runtime variation and instrumentation are examined separately below.

| Stage | Original CPU | Improved CPU | Current CUDA |
| --- | --- | --- | --- |
| Extraction | 17.29 | 12.01 | 7.49 |
| Map preparation | 3.78 | 3.39 | 4.03 |
| Matching, summaries, final materialization | 101.36 | 89.50 | 12.30 |
| Semi-linearized optimization | 36.76 | 22.61 | 15.71 |
| Full optimization | 18.06 | 2.11 | 2.08 |
| Marginalization | 0.87 | 0.46 | 0.53 |
| Maintenance | 0.74 | 0.37 | 0.20 |

CUDA scans exceeding 100 ms: 0.00%. Mean unassigned top-level timer residual: 0.160 ms; average per-run residual p99: 1.443 ms.

Process peak host RSS ranges across the three full runs are 439.8–441.2 MiB (original CPU), 329.9–334.9 MiB (improved CPU), and 451.4–461.0 MiB (CUDA). These are process peaks, not incremental algorithm allocations. GPU memory polling was disabled during timing to avoid introducing another periodic workload; this campaign does not report GPU peak memory.

## Scaling with matched CPU controls

Every row uses scans 20–249 of independent 250-scan replays, two repeats. A dense-feature result is compared with a CPU run using those same dense features, and likewise for the larger window.

| Workload | Original CPU | Improved CPU | CUDA | Original / CUDA | Mean poses | Graph correspondences |
| --- | --- | --- | --- | --- | --- | --- |
| current | 120.31 | 80.46 | 25.86 | 4.65× | 14.60 | 170,469 |
| features | 353.50 | 240.26 | 49.58 | 7.13× | 14.74 | 503,056 |
| window | 199.27 | 112.63 | 50.31 | 3.96× | 41.00 | 549,980 |

CUDA repeat means (current / denser / larger-window): 26.38 / 25.33; 51.05 / 48.11; 50.56 / 50.07 ms.

## Optimizer: where CPU and GPU time go

| Inclusive component | Full stairs ms/scan | Larger-window prefix ms/scan |
| --- | --- | --- |
| Summary + graph + values preparation | 1.95 | 3.85 |
| Resident reset/configuration | 3.38 | 10.64 |
| Linearization/assembly | 3.79 | 1.09 |
| Solve including waits | 5.38 | 5.94 |
| Error evaluation | 1.85 | 2.53 |
| Pose retraction | 0.46 | 0.58 |
| CPU Cholesky (inside solve) | 4.68 | 0.40 |
| CPU frozen matrix setup (inside reset) | 1.68 | 0.04 |
| Factor classification/copying (inside reset) | 0.95 | 1.52 |
| CUDA configuration host (inside reset) | 0.14 | 3.99 |
| CUDA configuration upload etc. (inside reset) | 0.10 | 2.76 |

Full-sequence CPU solve time is 5.22 ms/scan, versus 0.15 ms in GPU-selected solve calls. CPU Cholesky alone takes 4.68 ms. Thus faster GPU Cholesky alone misses most of the current workload's solve cost. These naturally selected groups have different dimensions; their times are not an apples-to-apples CPU/GPU speed comparison.

There are 18.836 CPU and 0.488 GPU solve attempts per scan, averaging 19.187 accepted and 0.137 not-accepted steps, with zero solve failures. Each rematch invokes semi-linearized optimization, followed by one full optimization per scan. The default GPU threshold remains 240 scalar pose variables (40 poses).

The following experiment forces the resident optimizer to CPU (threshold 6000) or GPU (threshold 1) while retaining CUDA extraction/matching. Invocation pairing requires identical scan, call index, semi/full phase and dimension. Values are pooled mean **milliseconds per complete call**, including preparation and cleanup, over two repeats.

| Workload/phase | Paired calls | CPU complete | GPU complete | CPU/GPU complete | CPU/GPU solve only | CPU setup | GPU setup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current/semi | 4486 | 0.50 | 0.73 | 0.69× | 0.49× | 0.11 | 0.23 |
| current/full | 460 | 0.71 | 0.86 | 0.83× | 0.43× | 0.14 | 0.35 |
| window/semi | 4686 | 2.73 | 2.20 | 1.24× | 2.22× | 0.72 | 1.29 |
| window/full | 460 | 4.94 | 3.45 | 1.44× | 1.90× | 1.10 | 2.31 |

Ratios below 1 favor CPU. All paired LM acceptance, rejection and solve-failure counts agree. Per-repeat and per-dimension results are retained in `optimizer-policy.json`; sparsely sampled dimensions are descriptive, not a validated universal crossover.

Per-repeat complete-call CPU/GPU ratios: current/semi 0.71/0.66; current/full 0.88/0.79; window/semi 1.24/1.24; window/full 1.49/1.38.

## Extraction, map and matching host budgets

| Component | Full stairs ms/scan | Dense prefix ms/scan | Window prefix ms/scan |
| --- | --- | --- | --- |
| Validity mask | 0.56 | 0.55 | 0.55 |
| Curvature preparation and processing | 2.32 | 1.81 | 2.00 |
| Planar selection | 0.96 | 1.12 | 0.94 |
| Point mask | 0.69 | 0.70 | 0.69 |
| Point selection | 0.41 | 0.42 | 0.40 |
| Normals: GPU search + CPU neighborhood/eigensystem | 1.62 | 3.69 | 1.62 |
| Feature output packing | 0.20 | 0.73 | 0.25 |
| Whole extraction stage | 6.98 | 9.70 | 6.76 |
| CPU world-map construction | 1.80 | 5.92 | 3.00 |
| Matcher snapshot preparation | 2.10 | 5.38 | 2.51 |
| Final raw-match materialization (inside matching) | 3.00 | 7.89 | 3.17 |

These host budgets include waits and data structure work. See the device trace below before interpreting a long host scope as a slow kernel. Curvature and normal scopes include CPU packing/postprocessing, not only GPU arithmetic.

## Calibration and correctness

| Calibration binary/mode | Run means ms/scan | Mean |
| --- | --- | --- |
| pre-instrumentation | 41.46 / 40.32 | 40.89 |
| diagnostic-disabled | 41.62 / 40.90 | 41.26 |
| diagnostic-enabled | 41.74 / 41.18 | 41.46 |

The old binary is extraction implementation `e4a49c7`, SHA256 `db7bd70d8563b904979fda012939aa9b6192d842379e640a66e2316b629f0fe4`. Enabled versus disabled measures the combined existing `--profile` counters and new diagnostics; NVTX tracing is off. Two repeats do not justify a precise instrumentation-overhead correction.

All 57 campaign/calibration runs pass artifact, timestamp, scan-count, per-scan workload and trajectory-parity checks against the matching reference. Maximum position difference is 1.203e-11 m; maximum rotation difference is 1.757e-12 rad. Quality-gate statuses: {'pass': 19, 'missing': 38, 'fail': 0}. All complete sequences pass the ground-truth quality/coverage gate. The 250-scan prefixes have no 30 m evaluation segments; their full quality gate is unavailable, not passed.

CSV/TUM hashes are checked against completion-time records. Profiled optimizer tables are mandatory and nonempty; their hashes are snapshotted **after campaign completion**, then checked on every reanalysis. Per-call solve decisions, invocation counts and per-scan counter sums are checked independently. This post-campaign manifest does not retroactively prove completion-time integrity.

The calibration mean differences are approximately **+0.9% disabled versus old** and **+0.5% enabled versus disabled**. Individual repeat variation is of similar or greater size. These are observed differences, not isolated causal overhead estimates. The two detailed full CUDA profiles average 41.31 ms; the clean campaign averages 42.50 ms. No claim that profiling makes the algorithm faster is warranted.

## CUDA timeline: kernels, transfers, launches and waits

These are separate **traced** replays, excluding the same first 20 scans. They are not included in clean speedup means. MB means decimal megabytes.

| Trace measurement, per scan | Full stairs | Larger-window prefix |
| --- | ---: | ---: |
| Traced estimator wall time, ms | 43.96 | 53.56 |
| Union of device kernel/copy/memset activity, ms | 7.24 | 13.72 |
| Device-active fraction of traced scan wall time | 16.46% | 25.61% |
| Kernel-active union, ms | 6.10 | 10.32 |
| Copy-active union, ms | 1.057 | 3.295 |
| Nearest-neighbor kernels, ms | 3.328 | 1.834 |
| QR kernels, ms | 1.849 | 1.404 |
| Curvature kernel, ms | 0.0072 | 0.0072 |
| Adjacent-row nearest kernels, ms | 0.126 | 0.126 |
| CUDA solve-scope kernels, ms | 0.112 | 4.157 |
| H2D transferred, MB | 6.845 | 31.686 |
| D2H transferred, MB | 2.403 | 2.709 |
| Kernel launches | 238.37 | 494.77 |
| Memcpy calls | 145.49 | 407.14 |
| Explicit stream synchronizations | 73.61 | 190.29 |
| Host launch API time, ms | 1.58 | 3.02 |
| Host memcpy API time, ms | 6.20 | 9.69 |
| Host explicit synchronization API time, ms | 2.31 | 7.61 |

The device-active fraction is **not SM occupancy, bandwidth utilization, or uninstrumented GPU utilization**. It measures whether any captured device activity is running during the traced estimator ranges. It shows substantial periods of host-side work, but does not establish that all such work can be parallelized or offloaded profitably.

Both traces have **zero unmatched device events and zero device activity spilling outside its issuing scan**. Their per-scan workload counts match the references, with maximum position differences 1.178e-11 m (full) and 2.686e-12 m (window).

Important findings:

1. **The full-stairs nearest kernel is meaningful, but not the dominant end-to-end target.** Halving its 3.33 ms costs saves at most about 1.66 ms if it all lies on the critical path. Halving QR's 1.85 ms saves at most another 0.92 ms. These are conditional arithmetic budgets, not measured gains. The two workloads have different rematch histories, so the larger window can have less nearest/QR kernel time despite more optimization variables.
2. **The feature kernels are already short.** Curvature plus adjacent-row nearest arithmetic totals about 0.133 ms, whereas whole extraction takes about 7 ms in the host profiles. Curvature preparation transfers 3.277 MB/scan and spends 0.437 ms on device copies. Further extraction acceleration should target masks, packing/transfers, ordered CPU selection and CPU normal construction; faster curvature arithmetic alone barely affects total latency.
3. **Larger windows repeatedly upload optimizer state.** Resident configuration uploads 17.706 MB/scan and batch reset another 7.669 MB/scan: about 25.38 MB, or 80% of the trace's H2D bytes. Their actual copy times total about 2.01 ms. The host reset budget is much larger (10.64 ms in the separate profile), including reconstruction and bookkeeping. Caching only buffers is insufficient; reusable contents and topology must be retained with correct invalidation.
4. **Host memcpy time is not PCIe transfer time.** Full-stairs host memcpy APIs occupy 6.20 ms, while actual copies occupy 1.06 ms. The classify/wait scope accounts for about 4.03 ms of CUDA API time, despite only 0.045 ms of copies. API calls can wait for earlier kernels or perform host staging. These are overlapping explanations of the same latency, not independent savings to add.
5. **Device allocation is not a leading steady-state problem here.** `cudaMalloc` plus `cudaFree` host API time averages approximately 0.025 ms/scan on full stairs and 0.048 ms on the larger window. This does not measure CPU container allocation; it argues against starting with a CUDA allocator rewrite.

CPU hardware counters are unavailable (`perf_event_paranoid=4`). Nsight Compute also fails with `ERR_NVGPUCTRPERM`. Nsight Systems CUDA/NVTX tracing works. Consequently this campaign does **not** establish cache miss rates, achieved bandwidth, register occupancy, warp divergence, or an instruction-level kernel bottleneck. No host security settings were changed.

## Tails and workload dependence

On the full clean CUDA runs, average per-run Spearman correlations of total latency are **0.875 with rematches**, **0.780 with active poses**, **0.765 with planar graph correspondences**, and **0.632 with LM linearizations**. These correlated workload variables do not establish independent causation. Selected planar feature count alone has little correlation (-0.087).

For example, the slowest scan of clean CUDA repeat 1 is scan 936: **87.66 ms**, 37 active poses, 30 rematches and 33 LM linearizations. Matching takes 25.59 ms and semi-linearized optimization 43.22 ms. The clean full runs have no measured estimator scan above 100 ms; this is not a hard real-time guarantee and excludes file I/O.

Full stairs averages 27.86 poses and 13.91 rematches per scan. The current 250-scan prefix averages 14.60 poses and 9.75 rematches. This explains why its 25.86 ms mean is not representative of the full sequence's 42.50 ms mean. Per-scan slow records and timer residual distributions are retained in `analysis.json`.

## Recommended next work and realistic budgets

The measurements support further meaningful acceleration, but not a claim that another kernel tweak will produce another multi-fold end-to-end gain. Recommended order:

1. **Reuse optimizer preparation across rematches.** Preserve the frozen CPU information matrix, factor classification, mappings and compatible graph/batch data; cache distinct semi/full configurations with explicit invalidation for topology, anchors and contents. The measured reset budgets are 3.38 ms on full stairs and 10.64 ms on the larger window, before graph preparation. A tentative saving target is **1.5–2.5 ms full / 4–7 ms larger-window**, not the entire reset budget. These are engineering targets, not measured savings.
2. **Improve the repeated small CPU optimization and then retune backend selection.** CPU Cholesky costs 4.68 ms/scan, summary linearization/assembly about 3.79 ms including packing/auxiliary work. Benchmark better CPU factorization/build choices and reduced assembly/reconstruction against GPU execution with cached setup. The forced experiment rejects “always GPU” for the small prefix. It does not determine the best threshold for every intermediate dimension, scene, phase or machine. A tentative full-sequence target is **1–2 ms** saved in these iteration costs, separate from reset.
3. **Reduce matching/materialization and map representation work.** Final raw materialization costs 3.00 ms full / 7.89 ms dense. World-map plus snapshot preparation costs 3.90 ms full / 11.30 ms dense. Eliminate duplicate reconstruction/packing and investigate persistent map/device representations while preserving matching order, tie rules and downstream raw-data requirements. A tentative combined full-sequence saving target is **1–2 ms**. This requires architectural work, not just coalescing the nearest kernel.
4. **Finish extraction's host-side work only where profiles justify it.** Masks, staging, ordered selection and normal construction offer more than the already tiny feature kernels. A tentative full-sequence saving target is **0.5–1 ms** without changing feature semantics. GPU sorting/selection must preserve dependencies and cannot be assumed automatically faster.

The listed saving targets sum to **4–7.5 ms/scan**. Together these suggest a **working target around 35 ms/scan**, with a **35–38.5 ms planning range**, versus the current clean 42.50 ms: approximately **1.10–1.21× additional acceleration**. This is a forecast conditional on successful implementation and parity checks, not a promised result or a lower bound. It would correspond to approximately 4.67–5.13× the original CPU mean measured in this campaign. Rebenchmark matched controls rather than mixing that forecast with older campaign timings. The larger-window case may benefit more from setup reuse, but its measured 10.64 ms reset time is a ceiling on that component's removable cost, not a predicted saving.

No further acceleration is implemented by this diagnostic task. Newer College stairs is the detailed profiling case; other sequences, the enlarged full sequences, alternative GPUs, CPU-native builds and hardware-counter analysis remain outside the demonstrated scope.

## Verification and reproduction

Before freezing the instrumented executable, **108 CUDA-build C++ tests passed** (97 main, two parallel, nine scalar-Eigen), and **48 CPU-only tests passed with eight CUDA skips**. Matching instrumented/uninstrumented smoke replays preserved workload counts and poses. After the final analysis changes, **37 Python benchmark/analysis tests passed**. Independent review checked instrumentation/accounting and the artifact/quality validation; its identified gaps were corrected.

From the repository root, use the evalio Python environment:

```bash
python benchmarks/run_diagnostics.py --binary <frozen-form-replay> --output <new-directory>
python benchmarks/run_diagnostic_calibration.py --campaign <new-directory> --previous <old-binary> --previous-revision <old-revision>
python benchmarks/analyze_diagnostics.py <new-directory>
python benchmarks/analyze_optimizer_policy.py <new-directory>
```

For each traced workload, separately run `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` with the replay's matching settings and `--trace`; export SQLite with `nsys export --type=sqlite`; then run `benchmarks/analyze_cuda_trace.py <sqlite> --output <json>`. Exact captured commands are recorded in `traces/full.trace.json` and `traces/window.trace.json`. See [benchmark instructions](../benchmarks/README.md#detailed-diagnostic-campaign) for timer hierarchy and controls.
