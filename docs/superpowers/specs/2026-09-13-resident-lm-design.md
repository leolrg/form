# Resident dense LM system

The user approved pursuing a resident architecture for repeated semi-linearized
optimization. Preserve 3b28389 as the previous committed implementation. Continue
committing validated checkpoints and comparing original CPU and optimized CPU
at every point/window setting.

The measured profiled window prefix spends about 7.76 ms linearizing, 5.75 ms
assembling, and 7.55 ms solving, with 5.79 ms summary preparation. Profiling changes
absolute latency; these counters select work, not final speedup claims.

Implement a dense graph evaluator shared by CPU and CUDA implementations. It
extracts exact FeatureFactor summaries, exact Hessian LinearContainerFactor
snapshots, and leaves other factors on their original CPU virtual interfaces.
Use sorted Pose3 keys. Frozen G stays constant; each local-coordinate displacement
d shifts b to b-Gd and f to f+d'Gd-2d'b, matching GTSAM's existing approximation.
Containers with no linearization point retain zero nonlinear error while still
contributing their stored linear model. Derived classes keep their own behavior.

The CUDA evaluator keeps roots, frozen matrices, index maps, assembled normal
matrix, and factorization workspace on one stream. Upload only current poses,
frozen displacement vectors, and the small remaining CPU-factor contributions
per linearization. Damping and Cholesky operate on the resident matrix; return
only the increment, linear model error scalars, and numerical status. Trial cost
uses resident roots/frozen matrices and returns one scalar. CPU uses the same
decomposition, damping and LM control with Eigen solving.

Retain GTSAM's outer convergence checks, Pose3 retraction, lambda policies,
model-fidelity test, and accepted-step bookkeeping. Implement the inner loop
without materializing a GaussianFactorGraph per retry. CPU pose updates remain
initially: replacing their manifold implementation is independent of removing
matrix transfers. Unsupported non-Pose3 values or unsupported parameter modes
must fail explicitly or select the existing path, never silently change math.

Compared alternatives: extra kernel tuning leaves CPU graph assembly in the
critical path; moving manifold operations and all LM control to device adds a
larger correctness surface before the matrix-residency benefit is established.
The chosen boundary removes matrix round trips while retaining tested manifold
and acceptance behavior. No reduced precision or relaxed tolerances.

Validation: compare dense models, shifted frozen terms, trial costs, damped steps,
rejected steps, both damping modes and lambda policies against GTSAM. Check empty
features, nonconsecutive keys, inactive derived factors, graph reset, numerical
failure, and CPU-only build. Sanitize new CUDA code. Then run matched replays for
current/features/window with original CPU, best previous CPU/CUDA, and both new
implementations. Profile separately from timing, include setup/transfers, and
report per-backend scaling, trajectory agreement and unavailable quality metrics.
