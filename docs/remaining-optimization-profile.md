# Remaining optimization profile and runtime estimate

The next engineering target is **about 35 ms per scan on full stairs**, with a planning range of **32–40 ms** on this A100/EPYC machine, preserving FORM's workload and quality criteria. This is a forecast, not a measured acceleration or a lower bound. The established current result remains **50.64 ms**, versus **155.76 ms for original FORM**. A 35 ms result would be 1.45x faster than current CUDA and 4.45x faster than original FORM.

The strongest opportunities are extraction and resident optimizer setup reuse. Matching still has material host work, but another round of nearest-kernel tuning alone is unlikely to produce a comparable total gain.

## Measurements

Added optional counters under the existing `--profile` flag, without changing the algorithm. Ran two complete 1190-scan stairs profiles and one 250-scan profile each for denser features and the larger window. Timing excludes the first 20 scans. Settings match the previous paired CPU/CUDA benchmarks: 32 threads; current point/plane limits 3/50, recent scans 10, spacing 5; denser 6/100 with spacing 2; larger window recent scans 40. Both CUDA paths use the existing dimension threshold of 240. Hardware is NVIDIA A100 80 GB PCIe, AMD EPYC Milan 32-vCPU VM, CUDA 13.0.

Diagnostic total means were 51.47/52.82 ms for full stairs, 64.29 ms for denser features, and 59.74 ms for the larger window. These are about 3.0%, 3.0%, and 2.5% above the respective earlier unprofiled means. Instrumentation and run-to-run variation are not separable here; the estimates below use the established unprofiled totals, not these diagnostic totals.

### Extraction

Milliseconds per scan. Full stairs is the mean of two complete runs; enlarged workloads cover the first 250 scans only.

| Extraction component | Full stairs | Denser features | Larger window |
|---|---:|---:|---:|
| Initial validity mask | 0.58 | 0.54 | 0.55 |
| Curvature construction | 2.06 | 1.85 | 1.94 |
| Sector sorting and planar selection | 4.85 | 5.11 | 4.75 |
| Point mask | 0.70 | 0.69 | 0.69 |
| Point selection | 0.36 | 0.37 | 0.35 |
| Parallel normal construction | 5.90 | 13.59 | 5.45 |
| Final output packing | 0.43 | 1.30 | 0.42 |
| Whole extraction stage | 15.36 | 24.21 | 14.61 |

The whole stage also includes prediction/constraint initialization and scope cleanup, so it slightly exceeds the substage sum.

Normal construction searches every valid point of both adjacent scan lines for each selected plane, then gathers local neighbors and forms a covariance/eigensystem. A deterministic approximately 1/64 sample of selected indices attributes **89.6% of sampled normal time to neighbor gathering**, versus 10.4% to covariance/eigenvector work on full stairs. Denser/window proportions are 89.5%/90.1%. These are sampled worker CPU durations, not additive wall-time stages, and may include scheduling effects. Sampling averages 266 normals per scan at current settings and 719 with denser features.

Concrete opportunities:

- Parallelize the row/sector sorting work while preserving the existing curvature order and suppression dependencies. Different sectors on one row can affect each other's masks; naive independent sector selection would change FORM. Row-level CPU parallelism is a useful candidate alongside a GPU approach.
- Batch the exact adjacent-line nearest searches on GPU, retaining distance and first-index tie semantics. Dense feature selection makes this substantially more attractive. Fuse or batch neighbor gathering and normal construction if transfer/packing overhead permits.
- Parallelize validation and curvature and reduce temporary allocations. Exact selection parity requires attention to float rounding, equal-curvature ordering, and normal orientation; replacing the neighborhood search with an approximation is outside this forecast.

A working extraction budget is **7 ms on full stairs**, down from the unprofiled 15.17 ms. This assumes improvements to both sorting and normals; accelerating the eigensolve alone cannot achieve it.

### Optimization

These wall counters cover all semi-linearized rematches and the full optimization per scan. They are distinct phases; worker factor counters are deliberately excluded.

| Optimization component | Full stairs | Denser features | Larger window |
|---|---:|---:|---:|
| Graph construction | 1.75 | 0.61 | 3.47 |
| Resident workspace reset/configuration | 3.43 | 0.72 | **11.68** |
| Linearization/assembly | 3.48 | 1.48 | 1.09 |
| Solve, including its synchronization | 5.49 | 1.33 | 5.98 |
| Nonlinear error evaluation | 1.89 | 0.99 | 2.62 |
| Summary preparation | 0.34 | 0.09 | 0.53 |
| Other optimizer/controller work | 1.33 | 0.55 | 1.53 |
| Whole optimization | 17.71 | 5.75 | 26.89 |

At current full-sequence settings most solves take the small-system CPU path: only 0.488 GPU solve calls per scan, versus 18.252 on the enlarged-window prefix. A GPU-only Cholesky improvement would therefore miss much of the current-setting cost.

`ResidentOptimizer::reset` still reconstructs frozen-system arrays, anchors and batch configuration. The GPU `configureResident` path rebuilds constant matrices and mappings and uploads them again; the CPU path reconstructs the frozen information matrix. Reusing buffers currently does not mean reusing all of this content. Separate cached fast/full configurations and explicit invalidation on topology, anchor or frozen-matrix changes are the leading next optimization. A same-size check alone would be incorrect.

Graph construction also prepares the frozen graph on the first rematch, while error/linearization repeatedly pack pose and anchor displacement data. Avoiding redundant construction and packing is worth exploring after setup reuse. The working full-sequence optimization budget is **13.5 ms**, down from 17.73 ms. Larger windows offer more setup savings; no faster linear solver is assumed necessary to reach the central forecast.

### Matching and map preparation

Final raw materialization still costs **3.97 ms on full stairs**, **8.41 ms with denser features**, and **3.99 ms with the larger window**, already included in matching. This path downloads the final results, reconstructs local target records, groups rows and copies data into both factor arrays and map-insertion output. Direct output packing and avoiding duplicate intermediate representations are the next candidates. Raw data cannot simply be discarded: retained factors, fallback paths and map insertion still use it.

Full-stairs map preparation splits into 1.95 ms for the CPU world map and 2.98 ms for matcher snapshot preparation in the diagnostic runs. Denser features increase these to 6.48/6.07 ms. GPU map construction or persistent map storage could help, but must preserve voxel membership and insertion/tie ordering. This is a less certain target than extraction or optimizer setup.

Working budgets are **10 ms matching** and **3.5 ms preparation** on full stairs. They assume modest improvements and do not count raw-materialization savings twice.

## CUDA trace

A fresh Nsight Systems trace covers all 250 scans of the enlarged-window workload, including warmup. It records:

- 116,561 kernels, totaling **2.417 s** of GPU kernel duration (9.67 ms per scan when divided by all 250 scans).
- **6.815 GB host-to-device** and **359.4 MB device-to-host** transfers, taking 1.157 s and 0.114 s respectively on the device timeline.
- 44,178 stream synchronizations, totaling 2.172 s of host API duration.
- 0.682 s of kernel-launch API duration.
- Kernel totals: nearest search 0.452 s; matching QR 0.357 s; frozen-system displacement work 0.386 s; the dominant factorization kernel 0.546 s.

API durations, device kernels and copies overlap and must not be added to estimate removable latency. Transfer totals cover the entire estimator and are not individually attributed to reset. Together with the reset timings and source inspection, they support investigating repeated setup uploads before further solver micro-optimization. Nsight trace replay timings are not performance-control results.

CPU sampling was unavailable because `perf_event_paranoid=4`; no machine security setting was changed. CUDA hardware-counter profiling was already unavailable. These conclusions use wall timers, sampled explicit normal timers, CUDA activity tracing, and source inspection rather than hardware-counter attribution.

## Forecast and limits

An explicit central budget for **current settings, full stairs**, in total estimator milliseconds:

| Stage | Established current | Working target |
|---|---:|---:|
| Extraction | 15.17 | 7.0 |
| Map preparation | 4.25 | 3.5 |
| Matching, including final raw arrays | 12.44 | 10.0 |
| Optimization | 17.73 | 13.5 |
| Marginalization, maintenance and remaining overhead | 1.05 | 1.05 |
| **Total** | **50.64** | **35.05** |

The **32–40 ms planning range** allows for uncertainty in those improvements and integration overhead; it is not a confidence interval. It corresponds to 1.27–1.58x over current CUDA and 3.89–4.87x over original FORM. An additional 2x would require roughly 25 ms per scan; this profiling does not yet justify promising that.

For the enlarged 250-scan workloads, tentative targets are **35–45 ms for denser features** (currently 62.42 ms) and **35–45 ms for the larger window** (currently 58.31 ms). These are less certain single-profile forecasts, not full-length predictions. They assume extraction/materialization gains for denser features and substantial reset reuse for larger windows. Every implementation should retain paired CPU controls at identical workloads; existing optimized CPU means for these prefixes are 227.42 and 106.30 ms respectively.

Recommended order: extraction, resident setup reuse, direct final correspondence packing, then reassess map preparation and solver kernels from the new profile. These savings interact, so each milestone needs a fresh total-time measurement. No feature-count, correspondence-count, window-size or iteration-count reduction is included in the forecast.

## Validation and artifacts

All four diagnostic runs preserve every per-scan workload and iteration count against the frozen current CUDA backend. Maximum position difference is <2.68e-13 m; maximum quaternion-coordinate norm difference is <1.85e-14, accounting for quaternion sign. This checks instrumentation parity; it is not a new four-sequence quality campaign. 87 CUDA C++ tests and 30 benchmark Python tests pass. A replay with profiling disabled confirms all added counters stay zero.

Artifacts are in `benchmarks/results/remaining-profile/`: per-run commands and binary hashes in `*.run.json`, timings/trajectories, `analysis.json`, reproducible `analyze.py`, `instrumentation.patch`, test logs, `provenance.json`, and `window-cuda-trace.nsys-rep/.sqlite`, `window-nsys.csv`, `cuda-trace-summary.json`. The prior unprofiled controls and CPU comparisons remain documented in `docs/cuda-matching-residency-results.md`.
