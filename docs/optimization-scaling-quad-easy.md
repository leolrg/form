# FORM optimization scaling: quad_easy

All 18 quad_easy runs are complete: three backends, three workloads, and two
repetitions in reversed backend order. Together with stairs, this completes
36 runs. All 18 [quad_hard runs](optimization-scaling-quad-hard.md) are also
complete, as are all 18 [maths_hard runs](optimization-scaling-maths-hard.md).
The full 72-run evaluation is verified; see the
[consolidated results](optimization-acceleration-results.md).

## Optimization performance

Mean latency in milliseconds, averaged across two runs after excluding the
first 20 of 1,991 scans:

| Workload | Original CPU | CPU summaries | CUDA | CUDA latency change vs CPU summaries |
|---|---:|---:|---:|---:|
| Current | 42.42 | 24.77 | 25.50 | 2.95% higher |
| More features | 78.30 | 32.66 | 31.60 | 3.25% lower |
| Larger window | 103.83 | 63.61 | 59.47 | 6.50% lower |

CUDA loses both current-size comparisons and wins both comparisons at each
enlarged workload. Exact cached factor summaries provide most of the gain over
the original. Larger-window CUDA combines GPU summary preparation with
selective FP64 GPU solving for system dimension 240 or above; the other two
CUDA workloads retain CPU solving. Each larger-window CUDA run makes 51,622
GPU solves with zero numerical fallbacks. This comparison does not isolate
the solver contribution.

## Realized scaling and accuracy

Counts and relative trajectory error (RTE) below average both CUDA runs over
all scans, including startup. All runs match 1,988 ground-truth poses.

| Workload | Mean active poses | Mean correspondences | CUDA optimization / current | Mean 1 m RTE m | Mean 30 m RTE m |
|---|---:|---:|---:|---:|---:|
| Current | 22.44 | 121,603 | 1.00 | 0.044630 | 0.422419 |
| More features | 23.96 | 335,307 | 1.24 | 0.043385 | 0.422396 |
| Larger window | 41.57 | 339,204 | 2.33 | 0.042930 | 0.439115 |

Denser selection gives 2.76x correspondences and 1.07x retained poses for 24%
more CUDA optimization time. The larger window gives 1.85x poses and 2.79x
correspondences for 133% more optimization time. These are measured estimator
workloads with changed matching and retention, not isolated complexity tests.

More features slightly improve short-distance mean error, while longer-distance
mean error is essentially unchanged across the two CUDA runs. The larger window
improves short-distance mean error by 3.81% but increases longer-distance mean
error by 3.95%. These results do not establish a general accuracy or robustness
gain from increasing workload.

## Complete scan processing

| Workload | Original CPU ms | CPU summaries ms | CUDA ms |
|---|---:|---:|---:|
| Current | 148.24 | 124.52 | 110.95 |
| More features | 294.28 | 233.06 | 197.98 |
| Larger window | 224.30 | 177.86 | 164.85 |

All CUDA configurations exceed the 100 ms mean scan budget. Matching remains
expensive in the denser workload. Matching time also differs between backends
despite unchanged matching code, with its cause unproven; the entire scan-time
difference cannot be attributed to GPU acceleration. Per-run latency tails,
deadline misses, and memory samples are in the verified checkpoint.

CUDA complete-scan latency statistics below average the two per-run statistics;
percentiles are not pooled across runs.

| Workload | Median ms | p95 ms | p99 ms | Scans exceeding 100 ms |
|---|---:|---:|---:|---:|
| Current | 111.31 | 147.14 | 165.43 | 69.96% |
| More features | 194.65 | 268.99 | 302.35 | 99.87% |
| Larger window | 162.45 | 212.92 | 233.11 | 99.21% |

## Validation and variability

Both accelerated backends pass all nine predeclared translation-RTE/coverage
checks in both repeats at every workload. At current settings and the larger
window, every recorded workload and iteration count matches its paired original
reference. Maximum accelerated/reference position disagreement is below
1.20e-10 m for current settings and 5.41e-11 m for the larger window.

The denser workload has a material repeatability caveat: CUDA repeat 2 and
original-CPU repeat 2 each diverge from their own first trajectories, changing
rematches, iterations, correspondences, and retention. Maximum repeat-2
CUDA/reference position difference is 4.08 cm; CPU-summary/reference difference
is 5.26 cm. All quality gates still pass. The second dense timing comparison
therefore measures complete-estimator behavior with changed work. It is not an
identical-input kernel comparison. The specific internal branching cause is
unproven; the original CPU also exhibits this variability.

## Reproduction

Measurements use an A100 80 GB PCIe GPU and 32 CPU threads on an EPYC Milan VM,
with FP64 arithmetic and the frozen Release binary under
`benchmarks/results/density-backend/`. Desktop GPU performance is unmeasured.
Workloads are current caps 3/50, inherited suppression spacing, recent10;
more features caps 6/100, spacing2, recent10; and larger window caps 3/50,
inherited spacing, recent40. Normal and curvature neighborhoods are unchanged.

Commands, fingerprints, repeated measurements, quality gates, and trace checks
are in [the verified checkpoint](../benchmarks/results/density-scaling-t32/checkpoint.md)
and its JSON companion. Detailed repeat investigation and tuning evidence are
in [CUDA tuning results](cuda-tuning-results.md). Compare with the completed
[stairs results](optimization-scaling-stairs.md) before generalizing across
sequences.
