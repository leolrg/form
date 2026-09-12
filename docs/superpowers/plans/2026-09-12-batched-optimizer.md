# Batched optimizer implementation plan

**Goal:** Measure and, if useful, integrate resident GPU feature-summary evaluation and graph assembly.

**Architecture:** One resident batch stores roots and endpoint indices. A kernel
evaluates each edge's 34 compressed residual rows; a deterministic gather assembles
the global augmented Hessian. Cost evaluation returns a single scalar. CPU code
uses existing FeatureSummary calculations with the same assembly and weights.

**Tech stack:** CUDA C++17, FP64, Eigen, GTSAM 4.2, TBB, GoogleTest.

- [x] Commit the existing implementation/results after running its 38 tests.
- [x] Add a failing independent matrix/cost comparison for shared pose graphs,
  zero residuals, and different graph sizes in tests/test_BatchSummary.cpp.
- [x] Implement form/feature/batch_summary.hpp/.cpp with CPU batching and a
  separate CUDA data interface/kernel. Keep snapshot ownership explicit and
  validate endpoint/pose counts before device access.
- [x] Run the new numerical tests and all existing tests; run Compute Sanitizer
  on the new kernels before making a correctness claim.
- [x] Add a benchmark of warmed complete calls and setup for CPU versus GPU at
  10/30/40/80 poses and sparse/dense edges. Run interleaved measurements with
  32 CPU threads. Record setup, cost, and linearization+assembly separately.
- [x] Commit the validated experiment and its measured decision.
- [x] If it has a useful crossover, wrap it as a single nonlinear factor and
  screen matched original/CPU/CUDA replays at current/features/window settings.
  If not, preserve the isolated experiment and document the remaining costs
  before selecting a different architecture.

First screening: dense 40-pose / 780-edge / 128-point graphs measured 307 us GPU
linearization+assembly versus 827 us existing CPU summaries, but 5208 us GPU
setup. Sparse graphs lose. Next prerequisite is reusable workspace and contiguous
CSR construction before estimator integration. Raw screening is in
benchmarks/results/batch-summary/screening.csv (generated, not versioned).


Completed checkpoint: resident storage and deterministic assembly are integrated
as opt-in summary-batch/cuda-batch backends. The paired pilot completed 30 runs
at 250 stairs scans, including all five backends at all three settings. Batched
CUDA improves current and larger-window optimization versus the previous CUDA
path, but does not consistently improve the denser-point case. See
[measured results](../../batched-cuda-results.md) for CPU scaling and limitations.
The prefix has no 30 m RTE segments, so full trajectory acceptance remains
unestablished. Further full-sequence validation is separate work, not a completed
claim of this screening checkpoint.

Read-only code review identified derived FeatureFactor activation being lost
when batching. A regression reproduced the error; exact-type selection preserves
subclasses. Final checks: 47 CUDA-build C++ tests passed, 30 CPU-only tests passed
with four GPU cases skipped, and 27 Python tests passed. CUDA memcheck and
racecheck passed on the new device code before the final host-only review fix.
