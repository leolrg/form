# CUDA-assisted feature extraction implementation plan

**Goal:** Accelerate FORM extraction without changing selected features, normal neighborhoods, or estimator settings.

**Architecture:** Upload the organized scan once per extraction to a reusable CUDA workspace. Compute scanline curvature and exact adjacent-row nearest indices on device; keep the existing ordered greedy selector and Eigen covariance/eigensolver on CPU. This deliberately preserves Eigen arithmetic and the original sort/suppression semantics. Expose an opt-in extraction setting and a distinct replay backend so current CUDA matching remains a control. GPU extraction is hybrid, not a fully resident pipeline.

**Tech stack:** C++17, CUDA, Eigen, TBB, GTest, existing replay/report tools.

## Execution

- [x] Add tests for float/double feature parity, holes and boundaries, spacing, empty candidate sets, repeated scans, and explicit CPU-only rejection. Confirm missing implementation fails.
- [x] Add `form/feature/cuda_extraction.hpp/.cu`: reusable buffers/stream, exact-order double curvature accumulation, scalar-compatible four-coordinate nearest distances, first-index tie reduction. Validate input dimensions and workspace state.
- [x] Integrate optional curvature and batched normal-neighbor indices in `extraction.hpp/.tpp`. Keep original CPU path unchanged. Include upload/download in extraction timing; synchronize only before CPU consumption. Serialize use of a reusable workspace.
- [x] Add `cuda-extraction` replay/suite backend (existing CUDA matcher/optimizer plus extraction), public Python parameter, and documentation.
- [x] Run extraction and complete CUDA tests, CPU-only build/tests, sanitizer on new kernels. Check same inputs produce identical feature sets and normals.
- [ ] Benchmark sequential reversed repeats against existing CUDA matching and original/optimized CPU at identical settings. Start with short stairs parity; then full stairs and denser/larger-window prefixes. Report actual timing and workload/trajectory differences, including regressions. No speedup claim from kernel time alone.
- [ ] Commit implementation and measured report; retain baseline backend and default behavior.

## Checks and risks

Use `cmake --build build-accel --target form-tests form-replay -j 4`, `build-accel/tests/form-tests`, benchmark Python tests, and Compute Sanitizer. Preserve padded-coordinate distance reduction order and disable CUDA FMA contraction. Preserve strict comparisons, first-index ties, per-sector quota behavior, invalid-mask semantics, CPU eigenvectors, and cross-sector suppression. Do not approximate adjacent-line search or change feature counts. CPU/GPU buffers must survive repeated sizes and exceptions. A new backend must be represented in suite fingerprints. Source and output artifacts are recorded with benchmarks.

## Measured first iteration and scope refinement

The first 100-scan diagnostic reduced normal construction from 6.14 to 1.74 ms,
but left 4.85 ms of sequential planar sorting/selection. Parallelize independent
rows on CPU while retaining serial sectors and byte masks to avoid packed-bit
races. Add separate `cpu-extraction` and `cuda-selection` controls so the final
report distinguishes CPU-selection gains from CUDA offload. No defaults or feature
selection rules change. Final performance runs follow all builds/sanitizers.
