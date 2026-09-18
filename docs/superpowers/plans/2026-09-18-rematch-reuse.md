# Rematch reuse implementation plan

**Goal:** Avoid rebuilding unchanged optimizer constants between rematches while retaining exact reset/snapshot semantics.

**Architecture:** Content-checked CPU frozen-matrix reuse, CUDA assembly-layout reuse, and CUDA resident-configuration reuse. Current nonlinear summaries and pose-dependent shifts always refresh. Work in the ongoing feature branch with the prior binary frozen independently.

**Tech stack:** C++17, Eigen/GTSAM, CUDA/cuSolver, existing GoogleTest and Python replay framework.

- [ ] Add CPU/GPU regression tests in `tests/test_ResidentOptimizer.cpp` and `tests/test_BatchSummary.cpp`. Assert existing reconstruction counters do not increase on repeat resets, while fresh-workspace model/error/solve comparisons hold. Exercise equal-size matrix/anchor/key/topology/weight/auxiliary changes and failed-update recovery. Run the new tests against old code to establish failure.
- [ ] Implement exact FrozenSystem equality in `form/optimization/resident_system.hpp`; retain CPU constant matrix in `resident_optimizer.cpp` when `previous_ready && old_keys == new_keys && old_frozen == new_frozen`. Always refresh anchors/auxiliary references and invalidate `linearized`.
- [ ] In `batch_summary_cuda.cu`, retain successfully uploaded CSR snapshots and resident constant snapshots. Always upload roots and invalidate model readiness on reset. Skip assembly CSR uploads on exact equality. Skip resident configuration on `dimension == 6 * poses && frozen == cached_frozen && auxiliary_poses == cached_auxiliary`. Invalidate configuration before fallible replacement and publish cache contents after successful synchronization.
- [ ] Add reuse counters in `diagnostics.hpp`, run targeted tests and independent code review, then fix any issues and commit.
- [ ] Run `cmake --build build-accel --target check -j 4`, CPU-only equivalent, Python benchmark tests, and Compute Sanitizer on the new CUDA cases. Freeze the verified executable/provenance.
- [ ] Run sequential repeated matched baseline/new benchmarks using `run_suite.py` infrastructure: full stairs CPU/CUDA, same-prefix current/dense/window CPU/CUDA, plus separate profiles. Compare workload/pose/LM traces, ground-truth gates and phase timings. Do not overlap builds or other benchmarks with timing.
- [ ] Write `docs/rematch-reuse-results.md`, update benchmark documentation/plan, review measurements, commit and push; verify remote HEAD.
