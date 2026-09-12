# FORM CUDA tuning and scaling plan

**Goal:** Tune the CUDA implementation systematically at the original workload, then measure performance and odometry quality as selected features and pose-window size increase.

**Architecture:** Preserve the raw reference and exact CPU summaries. Start with measured overheads in batched FP64 QR, then compare an alternative fused residual/Jacobian/Hessian path if QR remains the limiting algorithm. Keep the existing objective, noise weights, pose tangent conventions, ICP/LM stopping conditions, and marginalization semantics. Changes in precision or objective must be separate experiments.

**Tech stack:** CUDA 13, Nsight Systems/Compute, C++17, Eigen, GTSAM 4.2, TBB, evalio 0.6.1.

The user explicitly reprioritized tuning before scaling. The old 72-run sweep was stopped after eight complete runs; its results and pre-tuning binaries are retained under `benchmarks/results/`. Cancelled partial runs do not count as measurements. Remaining old-matrix runs are superseded by the final tuned-backend evaluation below.

## 1. Establish tuning evidence

- [x] Preserve pre-tuning replay/QR binaries and CUDA sources.
- [x] Stop the old sweep and exclude cancelled partial outputs.
- [x] Profile CUDA API, transfers, GPU kernels, and host preparation separately on both synthetic ragged batches and captured real correspondence batches.
- [x] Retain fixed captured inputs so each candidate does identical work. Record batch sizes, cache misses, transfer bytes, launch counts, and initial/final objectives.

## 2. Tune preparation and QR

- [x] Benchmark reusable pinned host buffers and capacity growth against the current pageable packing path; retain only measured gains.
- [x] Eliminate expanded host feature matrices / redundant packing by constructing features in the first GPU pass from raw correspondence arrays.
- [x] Measure block configurations and QR reduction layouts; inspect registers, occupancy, tail underutilization, and launch count.
- [x] Evaluate fewer launches / CUDA Graph replay for repeated shapes where trace evidence warrants it.
- [x] Validate each candidate against reference quadratic forms, costs, pose updates, and CUDA sanitizers before timing the estimator.

## 3. Evaluate algorithmic alternatives

- [x] Prototype fused GPU residual/Jacobian evaluation and blockwise augmented-Hessian reduction for changing correspondence sets; retain world-frame point Jacobians and FP64 error accumulation.
- [x] Compare preparation + repeated evaluation costs with QR over observed iteration/reuse counts. Integrate only if the measured total and correctness justify it; avoid per-factor GPU synchronization.
- [x] Profile the remaining solve/assembly overhead after factor acceleration. Test further changes only when the measured contribution supports a useful optimization gain.
- [x] Compare FP64 CPU/cuSolver on actual damped systems from 18-, 29-, and 43-pose windows. The full mature-window profile measured 7.17 ms solving per scan, warranting this follow-up. Retain only a verified total-latency improvement with equivalent steps.

## 4. Select and validate the tuned backend

- [x] Record every material candidate, timing/correctness results, and reasons for retaining or rejecting it. Practical stopping criterion: the measured major overhead categories above have been addressed or tested, and remaining candidates have no repeatable benefit or fail correctness. This is a bounded engineering search, not a claim of global optimality.
- [x] Run repeated isolated original-workload replays on all four complete sequences. Explain optimization and end-to-end differences separately, including the unexplained matching-time difference seen in the first backend comparison.
- [x] Preserve the predeclared numerical and trajectory tolerances. Report convergence, realized work, latency distributions/deadlines, memory, and CPU/GPU timing variability.

## 5. Scale only after tuning

- [x] Sweep selected-feature limits at fixed window, recording actual correspondences rather than configured limits alone.
- [x] Sweep active-window limits at fixed feature selection, recording actual poses/factors.
- [x] Compare the tuned CUDA path with matching CPU settings, and with original-workload latency/quality. Report both speed scaling and RTE/robustness tradeoffs; do not infer a quality gain from more points alone.
- [x] Publish reproducible configurations, compact results, operating range, remaining bottlenecks, and final verification evidence.

## Selected-backend checkpoint

Pinned staging, compact seven-value correspondence input, and 64-row QR with
32-thread blocks are selected. Reciprocal hoisting, cached CUDA Graphs, and direct
GPU Hessian/cost evaluation were tested and not retained. See
`docs/cuda-tuning-results.md` for measurements and limits. Selected CUDA passed
23 CTest cases and nine CUDA/manager cases under each of memcheck, racecheck,
and synccheck. Short 250-scan repeats retained all reference workload counts and
matched poses within 2.4e-12 m / 5.8e-13 radians.

The first bounded preparation search is complete. Run current and expanded
settings per sequence to obtain an early complete scaling comparison, then finish
all-four-sequence validation before final claims. Assembly/solve remain CPU operations;
profile a mature full window before accepting the stopping point, then revisit
that decision if enlarged windows make solving materially dominant. The early
250-scan profile does not represent the larger retained windows later in a run.

## Selective dense-solver experiment

Captured full symmetric systems show a GPU crossover between N174 and N258.
Test an optional reusable FP64 cuSolver backend, preserving the CPU path below
N240 and CPU fallback for unsuccessful factorization. The GPU may use the lower
triangle of the same symmetric matrix; the CPU reference stays Eigen Upper.

- [x] Implement/test reusable cuSolver buffers, exact augmented-system packing,
  status handling, growth/shrink reuse, SPD and indefinite cases.
- [x] Add explicit opt-in replay/parameter controls; preserve defaults and the
  frozen current-suite binary. Test full-window DenseLM dispatch.
- [x] Compare isolated full stairs replays with/without selective solving,
  including trajectory, RTE, convergence, workload counts, and total latency.
- [x] Retain only if estimator measurements justify it; freeze final binary and
  finish current and expanded-workload benchmarks with provenance.

Current-workload selective-solve control showed no total optimization gain, so
keep the default CPU solve. Proceed to larger-window profiling with recent20 at
fixed features to check its different cost balance. Run final paired benchmarks
in sequence-first order: current/features/window on stairs, then the remaining
three sequences. This scheduling change accelerates delivery of a full scaling
comparison without reducing sequence coverage, repeats, or accuracy checks.

The recent20 pilot completed and reduced actual poses from 27.86 to 23.53. The
recent40 pilot increased actual poses to 41.01 and correspondences by 1.70x; use
recent40 for the
final larger-window axis. This changes one configuration dimension while keeping
feature limits and keyscan parameters fixed. Selective solving is being checked
on this larger measured shape; primary current-workload solver remains CPU.

Final suite launched: 72 full replays, all four sequences, configurations in
features/window/current order per sequence, reference/summary/CUDA backends,
two reversed-order repetitions, 32 threads. Selective N240 solving is enabled
only for CUDA window40 based on the full pilot pair; the 4.2% optimization gain
is provisional until repeatability checks. Current and feature cases retain
CPU solving. Authoritative suite: benchmarks/results/final-scaling-t32/.

The cap-only feature pilot did not enlarge planar selection, so its three
completed runs were archived and the planned suite superseded. Added
`feature_spacing` (zero inherits `neighbor_points`) to decouple selection
suppression from curvature and normal neighborhoods. The feature workload is
now caps 6/100, spacing 2, recent 10. Real scans 20–99 confirm 2.92x stored
correspondences at the same 12 poses. All 38 C++ tests pass; CPU-only checks
pass 23 tests with two skips. Python replay validation follows before freezing
the density-capable executable and launching the final 72 runs.

Compressed-root GPU Hessian/cost evaluation was checked separately: correct but
slower than CPU cached summaries, including with resident roots. Keep CPU
summary evaluation.

The corrected final suite is active at `benchmarks/results/density-scaling-t32/`
using the frozen binary and source snapshot in `density-backend/`. All 38 C++
tests, CPU-only applicable tests, 27 benchmark Python tests, density-specific
Python/C++ replay, and dense CUDA memcheck passed. Leave the executable fixed
during the suite. Next work is validation and reporting of completed runs, not
further speculative implementation changes.

## Final completion evidence

All 72 frozen-suite replays are complete and validated. All 432 predeclared
paired quality checks pass, and all 38 final C++ tests pass. Larger-window CUDA
reduces optimization latency in both repeats on all four sequences; current-size
and dense-workload results are reported without claiming a general CUDA advantage.
Actual workload scaling, RTE tradeoffs, complete-scan limits, repeat variability,
remaining bottlenecks, and reproducible commands are in
`docs/optimization-acceleration-results.md`. Full verified results are in
`benchmarks/results/density-scaling-t32/final.json`; verification logs and source
provenance checks are in `benchmarks/results/final-verification/`. The historical
checkpoints above describe earlier stages and are superseded by this result.
