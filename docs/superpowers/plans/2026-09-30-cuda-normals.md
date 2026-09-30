# GPU normal construction experiment

**Goal:** Implement and benchmark the approved GPU neighborhood, covariance and eigensolver stage against `cuda-extraction`.

**Architecture:** Reuse the uploaded organized scan and exact adjacent-row search. A thread per candidate gathers the same own/previous/next-row neighborhoods, constructs the same query-centered covariance, and runs a device-compatible adaptation of Eigen's specialized 3x3 tridiagonalization followed by its iterative QR on fixed-size storage. Return normal plus validity in one download and pack completed features in a sequential host pass; preserve the existing CPU selection and public feature objects. Keep an opt-in `use_cuda_normals` flag and `cuda-normals` replay backend, with unchanged defaults and `cuda-extraction` control.

**Numerics:** For input scalar T (float or double), count neighbors m, compute a_j = T((p_j - query)/T(m)), C = sum_j a_j a_j^T in T, and normalize the smallest-eigenvalue eigenvector in T. This matches the existing query-centered formula, including division before multiplication (not a new centroid covariance). Use Eigen's scaled iterative eigensolver rather than an analytic approximation. Keep FMA contraction disabled. GPU covariance summation can differ from CPU SIMD reduction, so require exact selected positions/validity, unit normals, eigen-residual checks and direction agreement up to sign (float 2e-4, double 1e-10 on well-conditioned inputs). Degenerate eigenspaces require residual checks rather than arbitrary vector equality. Full replay must satisfy the existing ground-truth translation/coverage gate; record trajectory/count differences rather than claiming bitwise equivalence.

**Risks:** Radius comparisons retain the caller's four-coordinate reduction policy. Bounds must hold for first/last rows, partial blocks and missing neighbors. Query centering avoids subtracting large second moments; eigensolver scaling protects its internal iterations. Neighborhood/covariance precision stays in T to preserve the original objective. Two short neighborhood traversals avoid dynamic per-thread allocations and preserve division-before-product; register/local-memory overhead must be assessed through measured latency.

**Execution (in this existing feature branch):**
- [x] Add reference-based float/double tests before implementation; demonstrate missing normal method fails.
- [x] Implement reusable buffers, shared search launch, GPU gathering/covariance/Eigen and one result download.
- [x] Add extractor flag, Python exposure, replay/suite backend and feature parity tests.
- [x] Run CUDA tests, scalar Eigen tests, CPU-only tests, and Compute Sanitizer memcheck/racecheck.
- [x] Run sequential paired reversed-order stairs benchmarks: full current workload, 250-scan dense/window scaling, 20-scan warmup. Separate clean timings from detailed profiles. Freeze binary/source fingerprints; retain raw outputs.
- [x] Report extraction, normal-stage and estimator time, p95, quality/count differences and limitations. Retain the control even if the experiment loses. See `docs/cuda-normals-results.md`.

**Coverage:** Empty/single/partial-block batches; first/last/single rows; missing rows; radius stop and strict thresholds; float/double; changing shape and dtype; insufficient neighbors; invalid arguments; flat/curved/noisy/degenerate and translated neighborhoods. Packed contiguous scans only are supported by this API; strided tensors are inapplicable. Tests use the original CPU extractor and independently assembled CPU Eigen covariance. Timing includes transfers and synchronization, without overlapping builds/tests.

**Measured refinement:** The generic Eigen tridiagonalization experiment failed the full-stairs 30 m quality gate in both repeats. Porting the original specialized 3x3 reduction restored all workload counts and reduced the first full-run maximum position discrepancy from 0.302 m to 8.9e-9 m while keeping the same GPU covariance and sequential output packing. Preserve the failed experiment artifacts; benchmark the specialized variant independently.
