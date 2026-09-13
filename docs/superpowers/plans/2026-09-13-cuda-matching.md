# CUDA Matching Implementation Plan

> Execute the approved acceleration work continuously. Use subagent-driven-development
> for the independent device-QR extension and review; implement the coupled matcher
> and estimator integration locally. Commit verified milestones.

**Goal:** Accelerate FORM's correspondence matching and construct exact summaries
from its device output without a correspondence upload round trip.

**Architecture:** Cache a flat voxel hash snapshot and query features for a scan.
Search in parallel, preserve CPU search semantics, gather seven-column feature
rows by scan pair, and invoke device-input FP64 QR. Install roots only after host
raw correspondence updates are complete, using the existing cache revision contract.

**Tech Stack:** C++17, CUDA, Eigen, GTSAM, existing replay/quality tools.

- [ ] Extend `form/feature/cuda_qr.hpp/.cu` with `DeviceInput {size_t rows; bool plane;}`
  and `computeDevicePacked(const double*, const vector<DeviceInput>&, void* producer_stream)`.
  The concatenated matrices are column-major with seven stored columns. Producer
  readiness is ordered by CUDA event; external buffers are never overwritten.
  Test empty/ragged/mixed inputs, readiness, reuse, and quadratic-form equivalence.
- [ ] Add `form/optimization/cuda_matcher.hpp/.cu`: flat voxel buckets, point data,
  query data, reusable device buffers, FP64 nearest search, device grouping and
  feature packing. Before implementing, write tests in `tests/test_CudaMatcher.cpp`
  for empty maps, negative/boundary voxels, all neighbors, ties, strict radius,
  more than one query block, map replacement and repeated pose changes.
- [ ] Add host integration `form/optimization/cuda_matching.hpp/.cpp`: snapshot
  both feature maps, preserve point/normal coordinate transforms, materialize raw
  matches/constraints, and install pair roots. Tests compare counts and full raw
  factor cost/Hessian at varied poses. Reuse data only within its scan lifetime.
- [ ] Wire explicit `MatcherParams::use_cuda` and replay `cuda-matching` backend
  into `form/form.cpp`, build definitions and benchmark backend choices. CPU-only
  requests fail explicitly. Keep existing backend behavior intact.
- [ ] Run `cmake --build build-accel -j 8`, `ctest --test-dir build-accel
  --output-on-failure`, CPU-only build tests and benchmark Python tests. Run new
  matcher tests under memcheck/racecheck. Review correctness and code quality.
- [ ] Freeze a committed binary/source and run the paired CPU/hybrid/GPU-matching
  matrix via `benchmarks/run_suite.py` at all three workloads. Inspect per-stage
  timing, iteration/workload agreement and trajectory quality. Profile/tune only
  if evidence identifies a limiting component. Record results and limitations in
  `docs/cuda-matching-results.md`, update benchmark usage, and commit.
