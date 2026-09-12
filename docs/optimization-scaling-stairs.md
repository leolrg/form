# FORM optimization scaling: stairs

All 18 stairs runs are complete: three backends, three workloads, and two
repetitions in reversed backend order. This is a verified single-sequence
result. All 18 [quad_easy runs](optimization-scaling-quad-easy.md) are also
complete, as are all 18 [quad_hard runs](optimization-scaling-quad-hard.md).
All 18 [maths_hard runs](optimization-scaling-maths-hard.md) are complete too;
the full 72-run evaluation is verified. See the
[consolidated results](optimization-acceleration-results.md).

## Optimization performance

Mean latency in milliseconds, excluding the first 20 of 1,190 scans:

| Workload | Original CPU | CPU summaries | CUDA | CUDA latency change vs CPU summaries |
|---|---:|---:|---:|---:|
| Current | 53.79 | 28.36 | 29.09 | 2.6% higher |
| More features | 109.71 | 34.89 | 32.77 | 6.1% lower |
| Larger window | 96.42 | 51.05 | 42.05 | 17.6% lower |

Exact factor summaries provide most of the acceleration. CUDA has no consistent
advantage over CPU summaries at the current workload: it wins one repeat and
loses the other. It wins both repeats at each enlarged workload. Larger-window
CUDA combines GPU summary preparation with selective FP64 GPU solving at system
dimension 240 or above. Current and more-feature CUDA runs retain CPU solving.
The larger-window comparison therefore measures the combined path, not the
isolated solver contribution. Both window CUDA runs perform 22,793 GPU solves
with zero numerical fallbacks.

## Realized scaling and accuracy

Counts average all scans, including startup. RTE values are mean translational
relative trajectory errors in meters from the CUDA runs; equivalent-backend
errors agree to the displayed precision.

| Workload | Active poses | Stored correspondences | CUDA optimization / current | RTE at 1 m | RTE at 30 m |
|---|---:|---:|---:|---:|---:|
| Current | 27.54 | 267,627 | 1.00 | 0.034707 | 0.236248 |
| More features | 26.57 | 694,713 | 1.13 | 0.035187 | 0.345229 |
| Larger window | 40.50 | 453,778 | 1.45 | 0.025988 | 0.307581 |

The denser selection produces 2.60 times as many correspondences, with slightly
fewer retained poses, for 12.6% more CUDA optimization time. The larger window
produces 1.47 times as many poses and 1.70 times as many correspondences, for
44.6% more CUDA optimization time. These are measured estimator workloads with
changed selection and retention, not isolated algorithmic-complexity tests.

More features do not improve accuracy on stairs: short-distance error is nearly
unchanged and 30 m error increases by 46.1%. The larger window improves 1 m
error by 25.1% but increases 30 m error by 30.2%. These findings should not be
generalized to the other sequences before their runs finish.

## Complete scan processing

| Workload | Original CPU ms | CPU summaries ms | CUDA ms |
|---|---:|---:|---:|
| Current | 159.15 | 129.06 | 112.11 |
| More features | 487.05 | 390.72 | 336.95 |
| Larger window | 198.85 | 149.02 | 118.45 |

Every CUDA workload exceeds the 100 ms mean scan budget on this machine.
Matching dominates the dense workload, so its favorable optimization scaling
does not translate into favorable complete-scan scaling. Matching times also
differ across backends despite unchanged matching code; their cause remains
unproven. The entire observed scan-time reduction must not be attributed to GPU
computation.

## Validation and reproduction

Both accelerated backends pass all predeclared translational RTE and coverage
gates in both repeats at each workload. Every recorded per-scan pose, feature,
correspondence, factor, rematch, and LM-iteration count matches its corresponding
reference. Maximum accelerated/reference position disagreement is below
7.72e-11 m across these runs. Algebraic equivalence is subject to floating-point
rounding; this is not a claim of bitwise equality.

Measurements use an A100 80 GB PCIe GPU and 32 CPU threads on an EPYC Milan VM,
with FP64 arithmetic and a frozen Release executable. Desktop GPU performance
has not been measured. Workloads are current caps 3/50, inherited suppression
spacing, recent10; more features caps 6/100, spacing2, recent10; and larger window
caps 3/50, inherited spacing, recent40. Normal and curvature neighborhoods stay
at their original settings.

Exact commands, binary/data fingerprints, per-run timings, latency tails,
deadline misses, memory samples, trajectory statistics, and quality comparisons
are in [the verified checkpoint](../benchmarks/results/density-scaling-t32/checkpoint.md)
and its JSON companion. The frozen binary, complete source snapshot, and build
metadata are under `benchmarks/results/density-backend/`. The full experiment
history and rejected candidates are in [CUDA tuning results](cuda-tuning-results.md).
