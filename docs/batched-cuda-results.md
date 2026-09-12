# Broader CUDA experiment: resident graph evaluation and assembly

This extension is opt-in. It keeps immutable QR summary roots and graph
connectivity on the GPU across LM iterations, evaluates all selected binary
feature factors together, and returns one dense augmented system or one cost.
GTSAM retains LM control, priors, frozen-factor semantics, and the existing
CPU/selective-GPU solve choice. Sparse graphs retain per-factor evaluation.

Replay backends `summary-batch` and `cuda-batch` use the same batching selection:
at least 64 feature pairs and at least twice as many pairs as participating poses.
A shared workspace is reused only after older graphs release it, preserving
retained graph snapshots. No default backend has changed.

## Synthetic screening

A100 80 GB PCIe, 32 CPU threads. Fixed synthetic roots, two correspondence
counts per edge, interleaved timing order, 12 measured calls after warmup.
Linearization includes system assembly and completed transfers; cost includes
completed scalar transfer. These are not full-estimator or desktop timings.
Setup measures constructing a new batch; refresh measures replacing its roots
with the same connectivity and reusing allocations. The benchmark also reports
an illustrative refresh + 5 × (linearization + cost), computed from medians;
this is a model, not a measured LM solve.

Selected dense graphs, 128 point and 128 plane correspondences per edge.
All times below are microseconds. CPU summary is the existing GTSAM factor graph;
CPU batch uses the same new deterministic assembly layout as CUDA.

| Poses | Edges | Backend | New batch setup | Reused refresh | Linearization + assembly | Cost |
|---:|---:|---|---:|---:|---:|---:|
| 30 | 435 | original_cpu | 0.00 | 0.00 | 1162.20 | 1715.75 |
| 30 | 435 | summary_cpu | 0.00 | 0.00 | 527.51 | 100.57 |
| 30 | 435 | batch_cpu | 198.01 | 5.83 | 473.71 | 89.57 |
| 30 | 435 | batch_cuda | 1036.65 | 331.73 | 110.40 | 38.79 |
| 40 | 780 | original_cpu | 0.00 | 0.00 | 2051.99 | 3057.07 |
| 40 | 780 | summary_cpu | 0.00 | 0.00 | 804.07 | 198.29 |
| 40 | 780 | batch_cpu | 367.93 | 14.51 | 709.63 | 127.35 |
| 40 | 780 | batch_cuda | 1018.92 | 602.78 | 159.20 | 46.01 |
| 80 | 3160 | original_cpu | 0.00 | 0.00 | 8579.64 | 19879.84 |
| 80 | 3160 | summary_cpu | 0.00 | 0.00 | 3310.84 | 1645.82 |
| 80 | 3160 | batch_cpu | 1691.07 | 81.55 | 5701.07 | 215.78 |
| 80 | 3160 | batch_cuda | 8097.54 | 2350.60 | 613.35 | 120.70 |

A zero setup entry for existing GTSAM graphs means setup was outside their
measurement, not that graph construction is free. Root construction is outside
all four evaluation timings. No claim about more points improving accuracy is
made by these synthetic cases. Raw generated CSVs are in
`benchmarks/results/batch-summary/`; reproduce with `build-accel/form-batch-benchmark`.

## Validation and estimator screening

Before replay: 46 C++ tests pass; CUDA memcheck reports zero errors on 12 batch
and manager cases, racecheck reports zero hazards on four batch cases. The
CPU-only build passes applicable tests; four GPU manager cases skip. All 27
Python benchmark tests pass. Tests include independent GTSAM assembled matrices,
updated poses, shared endpoints, zero residuals, growing/shrinking graph storage,
retained graph snapshots, full/partial optimization, rematching and marginalization.

A paired 250-scan stairs pilot is the next evaluation: current, denser points,
and larger window; original CPU, CPU summaries, existing CUDA, batched CPU,
and batched CUDA; two repetitions with reversed backend order. Its results must
be treated as a prefix screening, separate from the earlier 72 full replays.
