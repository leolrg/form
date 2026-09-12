# Resident summary evaluation and graph assembly

The user authorized broader CUDA changes after the initial implementation's
limited gain. Preserve commit 60c3dfb as the measured baseline. Commit validated
checkpoints while developing the next path.

The first extension keeps immutable summary roots and graph connectivity on the
GPU for the lifetime of an optimization graph. Transfer current poses once per
cost/linearization call, evaluate all feature pairs in one batch, and assemble
one dense augmented system on the device. Return one matrix for linearization or
one scalar for cost. Preserve world-frame point Jacobians, stable difference
residuals, FP64 QR summaries, isotropic weights, frozen factors, and GTSAM LM
acceptance/damping behavior. Prior and frozen linear factors remain handled by
GTSAM initially. This is a step toward device-resident optimization, with a
measurable boundary, rather than a claim of a fully resident solver.

Alternatives are a fully custom GPU LM loop (greater opportunity but substantial
acceptance/retraction/frozen-factor correctness surface), or additional QR tuning
(already investigated, limited scope). Start with batched evaluation/assembly to
measure whether removing per-factor CPU dispatch changes the crossover.

Implement an equivalent CPU batch to distinguish batching improvements from GPU
improvements. Test matrices, costs, perturbations, shared endpoints, empty and
rank-deficient summaries against existing independent per-factor calculations.
Use deterministic assembly rather than unordered global atomics. Transfer and
synchronization costs count in timing. Include setup separately and amortized
across iterations; screen several edge counts and pose counts before replay.
Only integrate a backend into estimator replay when correctness passes and the
screening establishes a plausible useful size regime. A failed screening is an
experimental result, not grounds to silently enable a slower path.

Replay validation must compare original CPU, optimized CPU, and candidate CUDA
with identical point/window settings, and report absolute latency and each
backend's growth. Use current, denser points, and larger windows. All trajectory
quality gates remain in force. Synthetic screening alone cannot establish a
FORM or desktop speedup; this host is an A100.
