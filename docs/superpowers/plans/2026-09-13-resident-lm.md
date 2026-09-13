# Resident dense LM implementation plan

**Goal:** Remove repeated CPU matrix assembly and CPU/GPU matrix round trips from FORM optimization.

**Architecture:** Shared graph decomposition, CPU or resident CUDA matrix pipeline,
and a common LM loop retaining GTSAM state/retraction/convergence semantics.

**Tech stack:** C++17, GTSAM 4.2, Eigen, CUDA FP64, cuSolver, existing summary kernels.

- [x] Profile the previous current/window CUDA backend with existing counters.
- [x] Inspect GTSAM frozen-factor shifts, damping, retry and convergence semantics.
- [x] Write independent dense model/cost/step tests before implementing the new
  evaluator; compare with GaussianFactorGraph assembly, including shifted priors.
- [x] Implement graph decomposition and direct CPU evaluator in
  form/optimization/resident_optimizer.hpp/.cpp and expose a testable workspace.
- [x] Extend the existing summary CUDA workspace with resident frozen terms,
  auxiliary contributions, damping, Cholesky, and model-change scalars. Test
  CPU/GPU equality, resets, invalid inputs and numerical failures.
- [x] Add the common LM loop and compare complete optimization/retry behavior
  with DenseLMOptimizer; preserve hooks, thresholds and lambda policies.
- [x] Integrate explicit CPU/GPU resident backends into replay and the manager.
- [x] Run C++/CPU-only tests and Compute Sanitizer, review code and commit.
- [x] Freeze a binary; run matched current/features/window screening with CPU
  baselines and both new paths. Investigate regressions instead of enabling a
  slower default. Commit measured results and CPU/CUDA scaling tables.

- [x] Add measured hybrid follow-up: choose the complete CPU/CUDA matrix pipeline
  before assembly, with threshold-transition and unary-dimension regressions.
- [x] Benchmark the hybrid against original and optimized CPU at every workload;
  document both all-CUDA and hybrid results and their validation limits.
