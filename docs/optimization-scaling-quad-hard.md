# FORM optimization scaling: quad_hard

All 18 quad_hard runs are complete. All 72 planned runs across the four
sequences are now verified; see the
[consolidated results](optimization-acceleration-results.md).

## Optimization performance

Mean milliseconds, averaging two repetitions with reversed backend order and
excluding the first 20 of 1,880 scans:

| Workload | Original CPU | CPU summaries | CUDA | CUDA latency change vs CPU summaries |
|---|---:|---:|---:|---:|
| Current | 70.85 | 44.54 | 43.58 | 2.14% lower, changed-work caveat |
| More features | 130.31 | 55.68 | 55.57 | 0.18% lower; effectively tied |
| Larger window | 116.69 | 75.87 | 64.21 | 15.38% lower |

More-feature CUDA loses one repeat and wins the other. Larger-window CUDA wins
both repeats, with 16.00% and 14.75% lower latency. Its combined path uses GPU
summary preparation and selective FP64 GPU solving at system dimension 240 or
above; each run makes 58,515 GPU solves with zero numerical fallbacks. Current
and more-feature CUDA retain CPU solving. This comparison does not isolate the
solver contribution.

## Realized scaling and accuracy

Counts average all scans across both CUDA repetitions. Relative trajectory error
(RTE) averages each repetition's mean translational error. Every run matches
1,724 ground-truth poses.

| Workload | Active poses | Correspondences | CUDA optimization / current | RTE at 1 m (m) | RTE at 30 m (m) |
|---|---:|---:|---:|---:|---:|
| Current | 30.64 | 226,716 | 1.00 | 0.078035 | 0.717465 |
| More features | 32.10 | 656,721 | 1.275 | 0.076987 | 0.716434 |
| Larger window | 41.08 | 376,283 | 1.473 | 0.074390 | 0.724144 |

Denser selection produces 2.90x correspondences for 27.5% more CUDA optimization
time. The larger window produces 1.34x poses and 1.66x correspondences for 47.3%
more optimization time. More features slightly reduce both reported mean errors;
the larger window reduces short-distance error but increases long-distance error.
These are measured estimator workloads, with changed selection and retention.

## Complete scan processing

| Workload | Original CPU ms | CPU summaries ms | CUDA ms |
|---|---:|---:|---:|
| Current | 190.20 | 157.42 | 136.82 |
| More features | 395.87 | 305.20 | 263.29 |
| Larger window | 243.38 | 197.86 | 165.05 |

Every CUDA workload exceeds the 100 ms mean scan budget. Matching remains
expensive and its measured time differs across backends despite unchanged
matching code. Its cause is unproven; the full scan-time reduction cannot be
attributed to GPU computation.

## Validation and repeatability

Both accelerated backends pass all nine predeclared translation-RTE and coverage
checks in both repeats at every workload. More-feature and larger-window runs
preserve every recorded workload and iteration count against their paired
references; maximum position disagreement is below 6.37e-10 m and 5.72e-10 m,
respectively.

Current settings exhibit two trajectories in both CUDA and the original CPU.
Original repeat 2 follows CUDA repeat 1, agreeing in every recorded count with
maximum position disagreement below 5.58e-10 m. Original repeat 1 follows CUDA
repeat 2 and both CPU-summary repeats. The two trajectories differ by up to
0.12515 m and have different rematch, iteration, correspondence, and pose counts.
Extracted feature counts remain identical. The internal branching cause is
unproven, and this variability also occurs without CUDA.

Consequently, the current two-repeat timing average is not repeated equivalent-work
acceleration. The second CPU-summary/CUDA comparison does preserve all recorded
workload counts and shows 2.21% lower CUDA optimization latency. Quality gates
were retained unchanged throughout.

## Reproduction

Measurements use an A100 80 GB PCIe GPU, 32 CPU threads on an EPYC Milan VM,
FP64 arithmetic, and the frozen Release binary in
`benchmarks/results/density-backend/`. Desktop GPU performance is unmeasured.
Workloads use caps 3/50, inherited suppression spacing, recent10 for current;
caps 6/100, spacing2, recent10 for more features; and caps 3/50, inherited spacing,
recent40 for the larger window. Curvature and normal neighborhoods are unchanged.

See the [verified checkpoint](../benchmarks/results/density-scaling-t32/checkpoint.md)
and its JSON companion for commands, fingerprints, quality checks, memory,
latency tails, and traces. Additional current-repeat evidence is in
`benchmarks/results/density-scaling-t32/quad_hard-current-final-agreement.json`.
