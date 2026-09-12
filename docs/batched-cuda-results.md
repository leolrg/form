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

Before replay: 46 C++ tests passed; CUDA memcheck reports zero errors on 12 batch
and manager cases, racecheck reports zero hazards on four batch cases. The
CPU-only build passes applicable tests; four GPU manager cases skip. All 27
Python benchmark tests pass. Tests include independent GTSAM assembled matrices,
updated poses, shared endpoints, zero residuals, growing/shrinking graph storage,
retained graph snapshots, full/partial optimization, rematching and marginalization.

## Completed paired estimator pilot

All 30 runs completed on the first 250 stairs scans: three settings, five
backends, and two repetitions with reversed backend order. The first 20 scans
are excluded from latency statistics. The frozen replay was built at commit
`e7503ef`; SHA-256
`4bbcd21300298a08db2c7848e7fbf8d529a595ae5f32029d991091b39006bcb4`.
The later reviewed fix preserves subclass behavior and does not change which
concrete feature factors in these replays are eligible for batching.

Mean optimization latency in milliseconds (semi + full optimization, including
batch refresh and completed GPU work):

| Settings | Original CPU | CPU summaries | Batched CPU | Previous CUDA | Batched CUDA |
|---|---:|---:|---:|---:|---:|
| Current | 37.81 | 13.16 | 12.68 | 13.48 | 11.79 |
| More points | 109.21 | 21.54 | 21.40 | 14.95 | 15.18 |
| Larger window | 113.21 | 47.10 | 46.35 | 40.02 | 37.29 |

Latency growth relative to each backend's own current-setting baseline:

| Enlarged settings | Original CPU | CPU summaries | Batched CPU | Previous CUDA | Batched CUDA |
|---|---:|---:|---:|---:|---:|
| More points | 2.888× | 1.637× | 1.688× | 1.110× | 1.287× |
| Larger window | 2.994× | 3.580× | 3.656× | 2.969× | 3.163× |

Current settings are caps 3/50, suppression spacing 5, recent10. More points use
caps 6/100, spacing2, recent10. Larger windows use caps 3/50, spacing5, recent40.
In the batched CUDA runs, mean correspondences increased from 168,122 to 493,474
with more points (2.94×), or to 520,164 with larger windows (3.09×). Mean active
poses were 14.13, 14.26, and 38.56 respectively. All backends matched these work
counts. These prefix sizes differ from the full-sequence averages previously
reported; compare timings within this pilot.

Batched CUDA takes 10.40% less optimization time than existing CPU summaries
and 7.02% less than batched CPU at current settings. At the larger window it
takes 20.84% less than CPU summaries and 19.55% less than batched CPU. Against
the previous CUDA path, reductions are 12.53% and 6.82%, respectively, with a win
in both repeats for each of those settings. With more points, batched CUDA is
1.47% slower than previous CUDA on average and wins only one repeat. Do not
enable it indiscriminately based on these results.

Both CUDA backends use the same optional FP64 GPU solve threshold of 240 for
the window setting, and CPU solves for current/features. Thus the incremental
previous-CUDA/batched-CUDA comparison has a matching solve policy. Compared
with CPU, the window CUDA path combines preparation, evaluation/assembly, and
selective solving; it does not isolate any one kernel.

Mean complete-scan latency in milliseconds:

| Settings | Original CPU | CPU summaries | Batched CPU | Previous CUDA | Batched CUDA |
|---|---:|---:|---:|---:|---:|
| Current | 110.11 | 83.98 | 81.35 | 74.05 | 68.23 |
| More points | 328.42 | 226.04 | 224.30 | 194.22 | 193.50 |
| Larger window | 190.60 | 120.45 | 117.93 | 100.04 | 96.33 |

Complete-scan growth relative to each backend's own current baseline:

| Enlarged settings | Original CPU | CPU summaries | Batched CPU | Previous CUDA | Batched CUDA |
|---|---:|---:|---:|---:|---:|
| More points | 2.983× | 2.692× | 2.757× | 2.623× | 2.836× |
| Larger window | 1.731× | 1.434× | 1.450× | 1.351× | 1.412× |

Matching code is unchanged, but its timings differ between backends. Therefore
complete-scan gains cannot all be attributed to the GPU. Prefix mean latency
below 100 ms is not proof of full-sequence real-time operation.

All 24 accelerated/reference pairs have identical feature, correspondence,
pose, rematch, factor, and iteration counts. Maximum position disagreement is
1.28e-11 m. The available 1 m RTE and coverage checks all pass (120 checks).
The 96 checks involving 30 m RTE are unavailable because the prefix contains
no such trajectory segments. Accordingly the machine-readable report correctly
sets `all_quality_gates_pass` to false. This pilot establishes neither full
trajectory-quality acceptance nor performance on the other three sequences or
a desktop GPU. The original 72 full-run report remains separate.

[Verified machine-readable results](../benchmarks/results/batch-pilot/final.json)
and [full generated tables](../benchmarks/results/batch-pilot/final.md) include
latency tails and all paired results. Reproduce the pilot with:

```bash
/home/ubuntu/.local/share/uv/tools/evalio/bin/python benchmarks/run_suite.py \
  --binary benchmarks/results/batch-backend/form-replay \
  --output benchmarks/results/batch-pilot-new \
  --sequences stairs --configs current features window \
  --backends reference summary cuda summary-batch cuda-batch \
  --threads 32 --repeats 2 --limit 250 \
  --cuda-solve-min-dimension 240 --cuda-solve-configs window
```


Final review added a failing test for inactive FeatureFactor subclasses, then
restricted batching to exact FeatureFactor instances. The final tree passes 47
CUDA-build C++ tests and 30 CPU-only tests (four GPU cases skipped). Subclasses
retain their original virtual behavior, including activation and overridden
cost/linearization. The device kernels are unchanged from the sanitized pilot.

A 100-scan reviewed-code smoke replay matches frozen-pilot workload counts;
maximum position disagreement is 3.36e-14 m.
