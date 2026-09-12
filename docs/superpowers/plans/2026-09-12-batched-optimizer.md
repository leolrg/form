# Batched optimizer implementation plan

**Goal:** Measure and, if useful, integrate resident GPU feature-summary evaluation and graph assembly.

**Architecture:** One resident batch stores roots and endpoint indices. A kernel
evaluates each edge's 34 compressed residual rows; a deterministic gather assembles
the global augmented Hessian. Cost evaluation returns a single scalar. CPU code
uses existing FeatureSummary calculations with the same assembly and weights.

**Tech stack:** CUDA C++17, FP64, Eigen, GTSAM 4.2, TBB, GoogleTest.

- [x] Commit the existing implementation/results after running its 38 tests.
- [ ] Add a failing independent matrix/cost comparison for shared pose graphs,
  zero residuals, and different graph sizes in tests/test_BatchSummary.cpp.
- [ ] Implement form/feature/batch_summary.hpp/.cpp with CPU batching and a
  separate CUDA data interface/kernel. Keep snapshot ownership explicit and
  validate endpoint/pose counts before device access.
- [ ] Run the new numerical tests and all existing tests; run Compute Sanitizer
  on the new kernels before making a correctness claim.
- [ ] Add a benchmark of warmed complete calls and setup for CPU versus GPU at
  10/30/40/80 poses and sparse/dense edges. Run interleaved measurements with
  32 CPU threads. Record setup, cost, and linearization+assembly separately.
- [ ] Commit the validated experiment and its measured decision.
- [ ] If it has a useful crossover, wrap it as a single nonlinear factor and
  screen matched original/CPU/CUDA replays at current/features/window settings.
  If not, preserve the isolated experiment and document the remaining costs
  before selecting a different architecture.
