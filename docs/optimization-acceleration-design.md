# Exact feature-factor compression

The reference CPU path remains the default. The optional `summary` and `cuda`
backends compress each fixed scan-pair correspondence set once, then evaluate
its nonlinear cost and Gauss–Newton system from fixed-size data. They retain the
existing correspondence selection, residual definitions, isotropic noise scale,
LM optimizer, default dense GTSAM solve, and marginalization behavior. An
independent experimental option uses CUDA for sufficiently large dense solves,
as described below. Here, **exact**
means algebraically equivalent least squares, subject to floating-point rounding;
it does not mean bitwise-identical trajectories or QR factors.

## Residuals and sufficient statistics

Let `Ti=(Ri,ti)`, `Tj=(Rj,tj)`, `R=RiᵀRj`, and `t=Riᵀ(tj-ti)`.
Pose increments use GTSAM's right-local tangent convention, with rotation first
and translation second. `[p]×` denotes the matrix satisfying `[p]×v=p×v`.

For a point correspondence, the original world-frame residual and Jacobians are

```text
r  = Rj pj + tj - Ri pi - ti
Ji = [ Ri [pi]×, -Ri ]
Jj = [-Rj [pj]×,  Rj ]
```

Every scalar component is linear in the fixed seven-entry feature row
`q=[1, piᵀ, (pj-pi)ᵀ]`. For a compressed row `[w,pᵀ,dᵀ]`, evaluation uses

```text
r  = (Rj-Ri)p + Rj d + w(tj-ti)
Ji = [ Ri [p]×,     -wRi ]
Jj = [-Rj [p+d]×,    wRj ]
```

These are still world-frame Jacobians. Replacing the point residual with
`Riᵀr` would preserve its squared norm, but differentiating that pose-dependent
rotation adds another Jacobian term. The resulting Gauss–Newton Hessian generally
differs away from zero residual; that substitution is not used.

For a plane correspondence with normal `n` stored in scan i, the original scalar
residual is `(Ri n)ᵀ(Rj pj+tj-Ri pi-ti)`. Its thirteen-entry feature row is

```text
z = [vec_row(n pjᵀ)ᵀ, nᵀ, nᵀ(pj-pi)]
r = z [vec_row(R-I); t; 1]
```

`vec_row` lists matrix entries row by row. A compressed row represents an
arbitrary tensor `A`, vector `b`, and scalar `h`; `A` need not have rank one.
With `crossParts(M)=(M12-M21, M20-M02, M01-M10)` using zero-based indices:

```text
r       = <A,R-I> + bᵀt + h
Ji_rot  = crossParts(A Rᵀ) + b×t
Ji_trans= -b
Jj_rot  = crossParts(Aᵀ R)
Jj_trans= Rᵀb
```

The Jacobian expressions above are three-entry column vectors, transposed when
placed into residual Jacobian rows. Normals retain their original magnitudes.

Stacking the fixed rows gives feature matrices `Fpoint` and `Fplane`. Thin QR
writes each as `F=QU`, where `QᵀQ=I`. Every original residual/Jacobian column is
`Fc(poses)`, so replacing it with `Uc(poses)` preserves all column inner products.
Consequently it preserves `rᵀr`, `Jᵀr`, and `JᵀJ` for every pose, including both
pose derivatives. Only zero-padded `7×7` and `13×13` upper factors are retained:
at most 21 point residual components and 13 plane residuals per pair.

`FeatureSummary::augmentedHessian()` returns
`[Ji,Jj,-r]ᵀ[Ji,Jj,-r]`, with the final column containing `-Jᵀr`.
`FeatureFactor` scales this entire matrix by `1/sigma²` before constructing a
GTSAM HessianFactor. Its nonlinear error is `rᵀr/(2 sigma²)`. The fixed-first-pose
wrapper extracts the second pose's block and right-hand side from that same
system. The public `evaluateError()` continues to return the original full
residuals and optional Jacobians.

## Numerical choices and snapshot lifetime

Both backends use FP64 Householder QR. They do not form `FᵀF` before compression,
truncate small singular values, regularize the roots, or normalize normals.
Computing a small cost by subtracting large Gram-matrix quadratic terms can lose
all meaningful digits. Difference features `pj-pi` and `nᵀ(pj-pi)` retain small
residual information before QR; cost is subsequently evaluated as a sum of
squared compressed residuals. The implementation evaluates `R-I` as
`Riᵀ(Rj-Ri)`, keeping exactly identical rotations' difference exactly zero.

Each `PlanePoint` owns a cache associated with a particular `PointPoint` object
and its revision. Plane `push_back()`/`clear()` invalidate that cache; point
`push_back()`/`clear()` increment the point revision. Replacing the point object
also invalidates reuse, even when its revision matches. Direct edits to the
public arrays require calling that object's `invalidateSummary()` afterward.

New factors share a valid immutable summary. Existing factors keep their old
summary after rematching, so rebuild the graph/factors to consume changed
correspondences. Their raw `evaluateError()` still reads the live arrays; callers
must not treat a previously constructed factor's raw evaluation as its snapshot
after changing those arrays. Pose updates alone do not invalidate a summary.
Preparation finishes before factors consume the cache; concurrent mutation of
correspondence arrays is unsupported.

## CPU and CUDA preparation

| Selection | Preparation | Iterative cost, Hessian, and solve |
|---|---|---|
| `reference` | None | Original CPU residual/Jacobian path and GTSAM |
| `summary` | Eigen in-place QR, TBB across pending pairs | Common fixed-size CPU summary path and GTSAM |
| `cuda` | Compact correspondence packing, batched CUDA QR, roots copied back | Common fixed-size CPU summary path and GTSAM |

`ConstraintManager::Params::use_summary` and `use_cuda_summaries` both default to
`false`. Set only `use_summary=true` for CPU summaries. Set
`use_cuda_summaries=true` for CUDA preparation; it takes precedence if both are
set. The replay executable exposes `--backend reference|summary|cuda`.
`FORM_ENABLE_CUDA=ON` compiles CUDA support without changing the runtime default.
Manager `optimize()` prepares pending summaries before graph construction.
Direct factor/graph construction can build a missing summary on the CPU.

`BatchedCudaQr::compute` accepts ragged column-major matrices with exactly 7 or
13 columns. The manager uses `computeCorrespondences` to pack seven FP64 values
per correspondence directly into reusable pinned staging memory. Plane rows store
`[pj, n, n.dot(pj-pi)]`; the first kernel expands their 13-column tensor features
while loading. Point rows keep the same seven difference features.

One warp performs Householder QR on up to 64 rows, holding two rows per lane.
Subsequent levels combine nine 7-row roots or four 13-row roots per warp until one root remains per matrix.
Tail rows are zero padded; all lanes participate in reductions. The implementation
uses 32-thread blocks by default, FP64 arithmetic, and no atomic accumulation.
Constructor arguments retain 32-row and alternative block configurations for
controlled comparisons. Interleaved captured-batch tests selected the defaults;
they should be measured again on other GPU models.

A reusable CUDA stream and device buffers belong to each `BatchedCudaQr` instance;
only one caller may use an instance at a time. `compute()` includes host packing,
host-to-device copies, all QR levels, device-to-host root copies, and stream
synchronization. In `compute`, input matrices and the packed host copy coexist;
`computeCorrespondences` avoids the expanded Eigen matrices. Diagnostic observers
reconstruct expanded matrices only when capture is explicitly enabled. If `I` is the
number of input doubles and `L` the number of doubles in all first-level roots,
the two reusable device data buffers require approximately
`8*(max(I,L)+L)` bytes at requested capacities, plus descriptors for every reduction
level. Host staging/output and device allocations grow by at least 1.5 times their
previous capacity when needed, and retain those capacities until destruction. Entire pending batches
must fit host/device memory; there is no memory-budget batching or out-of-memory
CPU fallback. CUDA failures are reported as exceptions. Matrix row counts must
fit `int`.

The CUDA summary backend accelerates preparation. Iterative summary evaluation,
factor-graph assembly, LM control, and marginalization remain on the CPU; solving
also remains on the CPU unless the independent option below is enabled.
This implementation targets
fixed correspondences and one isotropic sigma per factor; pose-dependent robust
weights or arbitrary anisotropic noise need a separate derivation. Inputs must
be finite and reasonably scaled: the CUDA norm reductions directly sum squares
and do not provide overflow-resistant scaling for extreme magnitudes.

## Optional selective dense solving

`ConstraintManager::Params::use_cuda_dense_solver` defaults to `false` and is
independent of summary preparation. When enabled, DenseLM assembles the complete
active-pose system on the CPU and dispatches systems with dimension at least
`cuda_solve_min_dimension` (default 240) to FP64 cuSolver Cholesky. Older poses
with frozen linear factors are still variables in the semi-linearized solve;
the dispatch dimension includes them. Smaller systems use Eigen on the CPU.
The replay option `--cuda-solve-min-dimension 240` enables this path explicitly.

The solver retains pinned staging, device storage, workspace, a stream, and a
cuSolver handle across calls. It packs the dense matrix and RHS from GTSAM's
augmented system, uses the lower triangle of the same full symmetric matrix,
and returns a completed solution after synchronization. The CPU path uses the
upper triangle. Numerical failure leaves the input and output unchanged and
returns control to CPU solving; CUDA runtime/setup errors throw. Dispatch and
numerical fallback counts are recorded in replay output. The host wrapper is
compiled as ordinary C++ to retain Eigen's CPU vectorization.

This option did not improve overall optimization time in the current-workload
pilot. The final benchmark enabled it only for CUDA recent40; the combined
summary-preparation and selective-solve path reduced optimization latency by
6.5–17.6% versus CPU summaries, winning both repeats on all four sequences.
The option remains explicitly selectable and disabled by default. Desktop GPU
performance and its crossover point are unmeasured. See the
[final results](optimization-acceleration-results.md) and
[tuning history](cuda-tuning-results.md) for the evidence and limitations.

## Controlled larger workloads

Feature caps alone can be inactive because selection suppresses nearby samples.
`FeatureExtractor::Params::feature_spacing=0` retains the original suppression
width by inheriting `neighbor_points`. Positive values change suppression only;
normal estimation, curvature, and validity neighborhoods retain their existing
settings. Explicit spacing must not exceed `neighbor_points`. The larger-feature
workload uses point/plane caps 6/100 and spacing 2, versus caps 3/50 and inherited
spacing for the current workload. Both use recent10.

The larger-window workload uses recent40 with original feature selection.
Changing this setting also affects keyscan retention, so configured window size
is not a direct count of optimized poses. The benchmark records realized poses,
features, correspondences, rematches, and iterations for every scan. Accuracy
and runtime comparisons use matching settings across backends; increased density
or window size alone is not evidence of improved estimation quality.

## Build and validation

Use C++17, CMake 3.24 or newer (header file sets and FetchContent package
arguments), Eigen3, Boost, GTSAM 4.2, TBB, and GTest for tests. CMake obtains `tsl::robin_map` from an installed
package or its pinned FetchContent source. An existing `.vcpkg` checkout is used
as the project's toolchain. GTSAM and FORM must use compatible Eigen headers.

For a GTSAM 4.2 source checkout and system dependencies, the following creates
separate build trees. Set the source path before running:

```bash
FORM_GTSAM_SOURCE=/absolute/path/to/gtsam-4.2
FORM_GTSAM_BUILD=/absolute/path/to/gtsam-4.2-build
cmake -S "$FORM_GTSAM_SOURCE" -B "$FORM_GTSAM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_WITH_TBB=ON -DGTSAM_POSE3_EXPMAP=ON \
  -DGTSAM_ALLOW_DEPRECATED_SINCE_V42=ON
cmake --build "$FORM_GTSAM_BUILD" --parallel 8

cmake -S . -B build/cpu \
  -DCMAKE_BUILD_TYPE=Release -DGTSAM_DIR="$FORM_GTSAM_BUILD" \
  -DFORM_BUILD_TESTS=ON -DFORM_BUILD_BENCHMARKS=ON \
  -DFORM_BUILD_PYTHON=OFF -DFORM_ENABLE_CUDA=OFF
cmake --build build/cpu --parallel 8
ctest --test-dir build/cpu --output-on-failure
```

For an NVIDIA A100, configure a separate build with a compatible installed CUDA
toolkit and `nvcc` on `PATH`. Architecture 80 is specific to that target:

```bash
cmake -S . -B build/cuda \
  -DCMAKE_BUILD_TYPE=Release -DGTSAM_DIR="$FORM_GTSAM_BUILD" \
  -DFORM_BUILD_TESTS=ON -DFORM_BUILD_BENCHMARKS=ON \
  -DFORM_BUILD_PYTHON=OFF -DFORM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda --parallel 8
ctest --test-dir build/cuda --output-on-failure
```

CUDA tests require an available GPU. A CPU-only build rejects a request for CUDA
preparation; it does not silently change backends. For replay data preparation and
analysis, see [the benchmark README](../benchmarks/README.md). Explicit selection:

```bash
build/cpu/form-replay --input /path/to/sequence.formpc --output /path/to/reference --backend reference
build/cpu/form-replay --input /path/to/sequence.formpc --output /path/to/summary --backend summary
build/cuda/form-replay --input /path/to/sequence.formpc --output /path/to/cuda --backend cuda
```

The Python extension pins evalio 0.6.1 and nanobind 2.13.0, whose cross-module
binding ABIs match. An import-time check rejects incompatible evalio ABIs before
registering the inherited pipeline class. The existing editable installation
workflow (`pip install -e .`) uses these build dependencies. To enable CUDA in a
Python build, add `-Ccmake.define.FORM_ENABLE_CUDA=ON` and an appropriate
`-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80` (80 is for the tested A100).

For a CMake-built Python module, the bounded integration check exercises actual
evalio pipeline input, backend parameters, immediate poses, and the ABI guard:

```bash
python benchmarks/check_evalio_integration.py \
  --module-dir build-python/python --replay build-accel/form-replay \
  --input /path/to/stairs.formpc --output /tmp/form-evalio-check \
  --threads 32 --limit 20
```

The checked-in tests compare original factors, summaries, external QR roots, and
manager behavior. Relevant acceptance tolerances are:

- Right-local numerical Jacobians: maximum absolute error below `1e-8`.
- Random summary augmented-Hessian entries: `2e-10*max(1,abs(reference))`;
  squared error: `2e-12*max(1,reference)`.
- Tiny residuals at `1e4` coordinate scale: squared-error relative tolerance
  `2e-12`; coincident nonidentity poses require exactly zero error.
- CUDA ragged-batch roots: Frobenius error in `UᵀU` at most
  `1e-10*(1+norm(FᵀF))`, including repeated columns and buffer reuse.
- Factor integration: augmented information relative tolerance `1e-10`, cost
  absolute tolerance `1e-8`; cache replacement, explicit invalidation, retained
  snapshots, inactive factors, and fixed-pose wrappers are covered.
- Manager integration: pose local-coordinate difference below `2e-7`, graph
  augmented information relative tolerance `1e-9`, and graph cost absolute
  tolerance `2e-7`, through full/fast optimization, rematching, marginalization,
  subsequent scans, and disabled smoothing.
