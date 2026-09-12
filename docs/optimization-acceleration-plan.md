# FORM Optimization Acceleration Implementation Plan

**Goal:** Satisfy `optimization-acceleration-goals.md` with a measured, numerically validated acceleration and reproducible comparisons on all four N21 sequences.

**Architecture:** Keep the existing estimator as the reference and introduce selectable optimization backends. Profile before choosing the final backend. Evaluate reducing the mathematical work for fixed correspondences as well as moving repeated work to CUDA; keep the original estimation objective and LM behavior for equivalent-work comparisons.

**Tech stack:** C++17, Eigen, GTSAM 4.2-compatible API, TBB, CUDA, evalio/rosbags, Python benchmark analysis.

## Execution

- [x] Establish a Release build with compatible dependencies; record exact versions and hardware. Work on branch `codex/optimization-acceleration` in the current checkout.
- [x] Add a reproducible scan-input benchmark harness and optional stage profiling. Keep data loading and output serialization outside scan timing; report warmup separately. Capture scan-pair optimization inputs for independent validation when needed.
- [x] Repair the obsolete factor-test target with tests of the current factors; verify residual Jacobians against numerical derivatives. Add reference comparisons before enabling any accelerated implementation.
- [x] Profile a representative N21 segment. Split optimization, residual/Jacobian formation, Hessian construction, solving, trial cost, and marginalization. Record actual factor/residual/pose counts.
- [x] Implement and compare exact condensed CPU and CUDA alternatives where supported by the profile. Preserve priors, frozen linearizations, noise scaling, pose tangent conventions, and factor subsets used in marginalization. Explicitly select the reference or accelerated backend.
- [x] Validate factors and optimizer results, including empty/small/mixed batches, nontrivial rotations, weak geometry, and changing correspondences. Predeclare double-precision scaled tolerances before candidate results: factor quadratic coefficients at 1e-9 relative plus 1e-8 absolute; isolated nondegenerate pose solves at 1e-6 tangent norm. Investigate deviations rather than loosening thresholds after seeing results.
- [x] Run short sequential comparisons before full runs. Predeclare trajectory acceptance: no new tracking failures and no more than 5% or 1 cm (whichever is larger) degradation in each reported translational RTE metric against the same-configuration baseline. Report trajectory divergence separately; this quality tolerance does not replace numerical equivalence tests.
- [x] Run all four complete sequences for reference and fastest validated backend, including repeated timing runs. Record mean/median/p95/p99, throughput, >100 ms fraction, RTE at 1 m and 30 m, evaluated ground-truth coverage, and memory consumption.
- [x] Independently evaluate feature-count and pose-window increases against matching CPU configurations. Record realized workloads and accuracy/latency tradeoffs.
- [x] Review code and benchmark methodology, run final checks, and document commands, evidence, limitations, remaining bottlenecks, and the fastest validated configuration. Audit every goal-spec deliverable before marking the goal complete.

## Files and responsibilities

- `form/feature/`: reference factor implementation and exact accelerated factor representations.
- `form/optimization/`: backend selection, optimizer integration, and optimization profiling.
- `form/form.*`: optional scan-level stage counters and benchmark access.
- `tests/`: numerical derivative, quadratic-form, optimizer, and CUDA correctness checks.
- `benchmarks/`: reproducible input conversion, C++ scan replay, factor timing, and result analysis.
- `docs/`: experiment records, final results, build/reproduction instructions.

The implementation choices will be refined using profiles and correctness evidence; the objective and complete evaluation scope remain unchanged.

## Historical evidence checkpoint (superseded by final evaluation)

- CUDA Release build: 21/21 CTest cases passed after integration and objective-counter changes (`/tmp/form-tests-objectives.log`). CPU-only build: 16 applicable cases passed, two CUDA manager cases skipped (`/tmp/form-cpu-tests.log`).
- CUDA memcheck: zero errors; racecheck: zero errors/warnings across seven CUDA and manager cases (`/tmp/form-cuda-memcheck.log`, `/tmp/form-cuda-racecheck.log`).
- Sixteen Python converter, analyzer, and runner tests passed. The Python extension now uses compatible evalio 0.6.1/nanobind 2.13.0 pins and rejects incompatible ABIs with ImportError. Actual evalio pipeline runs on 20 stairs scans match standalone replay within 2.6e-14 in pose-matrix entries for all three backends (`/tmp/form-evalio-integration-t32/integration.json`).
- Isolated 250-scan stairs, 32 threads, first 20 excluded: reference optimization 36.49 ms / total 105.78 ms; serial CPU-summary preparation 19.65 / 76.38 ms; CUDA preparation 16.70 / 74.21 ms. These are exploratory single-run figures, not final acceptance results.
- Corresponding maximum translation differences from the selectable reference were below 2.4e-12 m; the selectable reference differed from the earlier raw baseline by at most 4.0e-14 m.
- Parallel CPU preparation subsequently measured 15.62 ms optimization and CUDA 17.90 ms with detailed profiling. Both remain candidates for complete, repeated comparisons; single-run stage and total timings do not establish the winner.
- Final matrix planned: all four complete sequences × current/doubled-features/recent40 × reference/summary/CUDA × two repeats = 72 sequential runs. Record objective values, actual correspondence counts, LM iterations, latency tails, coverage, RTE, CPU RSS, and sampled GPU memory. Alternate backend order between repeats.

## Completion evidence

All 72 frozen-suite runs are verified, all 432 paired quality checks pass, and
all 38 final C++ tests pass. Both enlargement axes were evaluated independently,
including actual workload counts and accuracy tradeoffs. Implementation sources
match the frozen snapshot. The final report documents the supported backend
choices, numerical repeat variability, measured real-time limits, and the absence
of desktop GPU measurements. See `docs/optimization-acceleration-results.md`,
`benchmarks/results/density-scaling-t32/final.json`, and
`benchmarks/results/final-verification/audit.json`.
