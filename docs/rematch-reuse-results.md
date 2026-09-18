# Optimizer reuse between rematching iterations

This change reuses constant optimizer data across repeated resets. Current feature summaries, pose-dependent linearization, error evaluation, damping, and LM decisions still run as before. It does not alter point selection, matching, semi-linearization, the objective, or the default 240-variable CPU/GPU threshold.

The clearest measured end-to-end benefit is on the larger-window workload: **50.36 → 46.21 ms/scan for the CUDA pipeline (8.2% lower latency)**, with both repeats improving. The matched CPU pipeline improves **116.66 → 109.02 ms (6.5%)**. Full-stairs CUDA averages **41.12 → 40.71 ms (1.0%)**, but repeat variation means this is **not a clear full-pipeline speedup**. Its semi-linearized optimization stage improves consistently, **15.24 → 13.99 ms**. Denser features do not show an overall CUDA benefit.

## What changed

- **CPU frozen matrix:** a successful reset retains the assembled constant information matrix. The next reset reuses it only when the sorted pose keys and all frozen factor mappings, matrix entries, and anchor-presence flags agree exactly. Factor contents are compared without a numerical tolerance or pointer-identity shortcut.
- **CUDA assembly mappings:** new summary roots and weights are still uploaded on every reset. Existing CSR offsets and indices remain on device when their actual contents and pose count agree, avoiding redundant uploads.
- **CUDA resident constants:** frozen matrices, constant system terms, frozen-vector mappings, auxiliary-factor mappings and solver configuration survive compatible resets. Reconfiguration is skipped only when dimension and all relevant contents/layouts agree.
- **Mutable data still refreshes:** anchors and auxiliary factor objects are read from the new graph. Anchor displacement, shifted right-hand side/constant terms, nonlinear residuals and LM steps are recomputed at the candidate poses. Changing an anchor does not require rebuilding a constant Hessian; it does require using the new anchor in displacement calculations, which the tests check.
- **Failure handling:** every successful reset invalidates the previous linear model. CUDA updates mark cached data invalid before fallible work; snapshots become reusable only after successful upload/synchronization. A failed update cannot cause a later reset to reuse partially overwritten device data. Existing public argument validation can reject a reset before changing the old state.

This implementation still classifies and snapshots factors on each reset to detect mutations safely. It does not introduce caller-managed generations, assume shared pointers are immutable, or cache the entire graph. It therefore targets expensive reconstruction/uploads while retaining the existing reset interface.

Implementation: `a1d0107`. Benchmark driver and checks: `865cf3f`. Frozen new replay SHA256: `e8a0a7c19fdcc8e9f9ef65927d4fb5247c0de78f03262cc9c04c8fa4dc4d6368`. Baseline is the prior diagnostic binary (runtime source `a8158cb`), SHA256 `2113784cb20af5f0cc312156ee8227e2a9ee186d3d7b582ddb94005a9f5979ac`.

## Verification

Before timing, 111 CUDA-build C++ tests passed (100 main, two parallel extraction, nine scalar-Eigen); 50 CPU-only tests passed with nine CUDA-dependent skips. The three new reuse/invalidation cases also pass Compute Sanitizer memcheck, racecheck and synccheck with zero errors/hazards. The benchmark Python suite passes 39 tests.

The reuse tests first failed against the previous implementation because constants were rebuilt and compatible CUDA configuration was discarded. After implementation they check both skipped reconstruction counters and numerical results. Cases include cloned factors, changed anchors, changed auxiliary objects, new roots/weights, same-size endpoint and pose-key changes, in-place frozen matrix mutation, anchor-presence changes, auxiliary layout changes, empty graphs, stale models, and recovery after invalid updates.

Five 80-scan correctness replays (old/new CPU, old/new CUDA, and new forced-GPU optimization) preserve every per-scan workload count. Maximum position difference versus old CUDA is below 2.5e-14 m. These smoke replays overlapped build/test work and are **not performance measurements**.

## Benchmark method

Hardware and build match the preceding campaign: A100 80 GB PCIe, EPYC Milan VM, 32 CPU threads, Release build, CUDA architecture 80; portable CPU compilation without `-march=native`. Input is all 1,190 scans of Newer College 2021 stairs (128 rows × 1,024 columns). Timing excludes the first 20 scans. Input loading and output serialization are outside the estimator timer.

Thirty runs compare old/new implementations sequentially: two full CPU pairs, three full CUDA pairs; two CPU and CUDA pairs for each denser-feature and larger-window 250-scan workload; and separate old/new CUDA profiles for full stairs and the larger-window prefix. The second clean repeat reverses control order. No builds, sanitizers or other benchmarks overlap these timed runs; GPU-memory polling is disabled. Input SHA256, binary hashes, command lines, environment, CPU affinity, resource data and output hashes are recorded. Resume checks reject changed run conditions or missing/changed optimizer hashes.

Current settings are point/plane caps 3/50, spacing 5, recent-scan cap 10. Dense settings use 6/100 and spacing 2. Larger windows use recent-scan cap 40 with current feature settings. The current 250-scan comparison is extracted from the same prefix of full runs; every scaling row times scans 20–249. Enlarged results are not full-sequence results.

Old/new means quantify the benefit of this reuse change against our previous improved implementation. They are not a fresh original-FORM comparison. Host profile counters are inclusive; parent and child counters must not be added together.

## Clean results

Milliseconds per scan, mean of the per-run means. CPU means the improved all-CPU pipeline; CUDA means the hybrid pipeline, including its CPU-selected optimization for smaller systems.

| Workload | Pipeline | Repeats | Old | New | Saved ms | Old/new speedup | Latency reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean-full/current | cpu-extraction | 2 | 132.236 | 130.738 | 1.497 | 1.011× | 1.13% |
| clean-full/current | cuda-extraction | 3 | 41.121 | 40.711 | 0.410 | 1.010× | 1.00% |
| clean-prefix/current | cpu-extraction | 2 | 81.126 | 77.452 | 3.675 | 1.047× | 4.53% |
| clean-prefix/current | cuda-extraction | 3 | 27.224 | 26.844 | 0.380 | 1.014× | 1.40% |
| clean-scaling/features | cpu-extraction | 2 | 241.429 | 236.254 | 5.175 | 1.022× | 2.14% |
| clean-scaling/features | cuda-extraction | 2 | 50.462 | 51.241 | -0.779 | 0.985× | -1.54% |
| clean-scaling/window | cpu-extraction | 2 | 116.656 | 109.017 | 7.638 | 1.070× | 6.55% |
| clean-scaling/window | cuda-extraction | 2 | 50.362 | 46.209 | 4.152 | 1.090× | 8.24% |

Full-sequence repeats:

| Pipeline | Old repeat means | New repeat means | Old → new p95¹ |
| --- | --- | --- | --- |
| cpu-extraction | 128.328 / 136.143 | 131.504 / 129.973 | 263.100 → 254.608 |
| cuda-extraction | 40.853 / 41.571 / 40.939 | 39.041 / 42.595 / 40.496 | 68.552 → 66.512 |

¹ Average of each run's p95. Three CUDA and two CPU repeats are descriptive measurements, not a confidence interval or hardware-independent speedup guarantee.

Paired old-minus-new total savings (ms), by repeat:

- clean-full/current cpu-extraction: -3.176 / 6.170.
- clean-full/current cuda-extraction: 1.813 / -1.025 / 0.443.
- clean-scaling/features cpu-extraction: 8.753 / 1.598.
- clean-scaling/features cuda-extraction: 0.320 / -1.878.
- clean-scaling/window cpu-extraction: 8.061 / 7.215.
- clean-scaling/window cuda-extraction: 4.867 / 3.437.

Full CUDA stage means from clean runs:

| Stage | Old ms | New ms |
| --- | --- | --- |
| Extraction | 7.160 | 7.323 |
| Map preparation | 3.875 | 4.045 |
| Matching/QR/materialization | 12.004 | 12.381 |
| Semi-linearized optimization | 15.243 | 13.992 |
| Full optimization | 1.943 | 2.064 |
| Marginalization | 0.485 | 0.507 |
| Maintenance | 0.217 | 0.233 |

## Separate diagnostic profiles

One old/new profile pair per workload; these timings are not included in the clean result means. Reuse counters count skipped rebuilds, not skipped pose-dependent computations.

| Component | Full old | Full new | Window old | Window new |
| --- | --- | --- | --- | --- |
| Whole optimizer resets | 3.522 | 2.212 | 11.489 | 6.481 |
| CPU frozen matrix assembly | 1.685 | 0.180 | 0.044 | 0.010 |
| Factor classification/snapshot | 0.966 | 0.988 | 1.512 | 1.486 |
| CUDA constant/layout construction | 0.150 | 0.022 | 4.084 | 0.780 |
| CUDA configuration upload and workspace work | 0.177 | 0.019 | 3.382 | 0.729 |
| Graph construction | 1.651 | 1.576 | 3.311 | 3.401 |

| Mean calls per scan, new build | Full | Window |
| --- | --- | --- |
| CPU resets | 14.513 | 0.765 |
| CPU frozen matrix reuses | 12.564 | 0.600 |
| GPU resets | 0.397 | 10.422 |
| GPU resident configuration reuses | 0.345 | 8.587 |
| GPU assembly-layout upload reuses | 0.279 | 9.152 |

## Replay correctness

All 30 runs complete. Every one of the 15 old/new comparisons preserves per-scan features, poses, graph correspondences, factors, rematches and LM linearization counts. Maximum position difference: 5.829e-13 m; maximum rotation difference: 9.616e-14 rad.

Maximum scaled initial/final objective differences: 3.484e-12 / 9.727e-12. Both are below the predeclared 1e-8 gate.

The two profile comparisons pair 20,017 steady-state optimizer invocations by scan, call, phase and dimension; CPU/GPU choices and LM acceptance/rejection/solve-failure counts agree. All full-sequence comparisons pass ground-truth quality and coverage gates. The short prefixes lack 30 m ground-truth segments, so the complete quality gate is unavailable, not passed; their count, pose and objective comparisons still pass.

Per-run resources, stage means, tails, optimizer records and repeat results are in `benchmarks/results/rematch-reuse-20260918/analysis.json`.

## CUDA transfer attribution

Separate Nsight Systems traces cover the same larger-window scans 20–249. Each trace preserves the workload counts and trajectory of its corresponding clean run. Both have zero unmatched device events and zero attributed device time spilling outside the issuing scan. Values below are per timed scan; MB means 1,000,000 bytes.

| Trace measurement | Old | New |
| --- | --- | --- |
| All host-to-device transfer volume, MB | 31.686 | 13.834 |
| All host-to-device copies | 275.97 | 197.55 |
| Resident configuration upload volume, MB | 17.706 | 2.744 |
| Batch reset upload volume, MB | 7.669 | 4.779 |
| Device-to-host transfer volume, MB | 2.709 | 2.709 |
| CUDA kernel launches | 494.77 | 494.77 |
| Device copy-time union, ms | 4.125 | 1.720 |
| Device kernel-time union, ms | 10.305 | 10.213 |

Host-to-device volume falls **56.3%**, while kernel-launch counts and download volume are unchanged. This supports the intended mechanism: avoiding redundant constant/layout uploads while continuing the same pose-dependent GPU work. Kernel count alone does not establish numerical equivalence; the replay comparisons above provide that check. Device copy/kernel times come from instrumented runs and are not additional clean-run savings to add to the performance table. Device-busy fractions are scheduling coverage, not achieved SM occupancy.

## Interpretation and limits

- **Full stairs:** semi-linearized optimization saves about 1.25 ms/scan in the clean-run mean and improves in all three pairs. Separate profiles show a 1.31 ms reset reduction. However, the total averages differ by only 0.41 ms, with one pair regressing; extraction, matching and other unchanged stages also vary. These measurements support a setup improvement, not a reliable overall 1% speedup claim.
- **Larger window:** both CPU and CUDA improve in both clean pairs. CUDA semi-linearized optimization falls from 22.84 to 18.02 ms, about 21% lower latency. The separate profile records a 5.01 ms reset reduction. More retained poses make rebuilding and uploading constants more expensive, so reuse has more value. The hybrid pipeline uses GPU optimization much more often here: about 10.42 GPU resets/scan versus 0.40 at default settings. CPU total savings also include changes in other measured stages; they cannot all be attributed to constant assembly.
- **Denser features:** CUDA total time averages 50.46 → 51.24 ms, a 1.5% regression with mixed pair results. More correspondences primarily increase matching/summary work; they do not necessarily increase the pose-system dimension. This change does not establish an end-to-end win for that workload.
- **Remaining reset costs:** factor classification and snapshots still run; new roots/weights still upload. Exact content checks have a cost. Reusing full graph structure would need a stronger ownership/invalidation design and separate validation. This implementation makes no claim that every reset cost has been eliminated.

These are measurements on this A100/CPU combination and one sequence, with enlarged workloads limited to its first 250 scans. They do not establish a speedup on a different GPU, a different sequence, or enlarged full-sequence trajectories. Point selection, retained-window settings, rematch counts, LM policy and numerical precision are held equal within every old/new pair.

## Reproduction and artifacts

Use the evalio Python environment and frozen executables:

```bash
/home/ubuntu/.local/share/uv/tools/evalio/bin/python benchmarks/run_reuse_benchmark.py \
  --previous benchmarks/results/detailed-diagnostics-20260916/backend/form-replay \
  --previous-revision a8158cb \
  --binary benchmarks/results/rematch-reuse-20260918/backend/form-replay \
  --revision a1d0107 \
  --output benchmarks/results/rematch-reuse-20260918
/home/ubuntu/.local/share/uv/tools/evalio/bin/python benchmarks/analyze_reuse_benchmark.py \
  benchmarks/results/rematch-reuse-20260918
```

The campaign directory contains run specifications, completion hashes, raw CSV/TUM outputs, optimizer profiles, resources and `analysis.json`. `backend/` contains binary provenance and test/sanitizer logs. `traces/` contains the two Nsight reports, SQLite exports, per-trace analysis and workload/trajectory comparisons. Large binaries, datasets and raw traces remain local ignored artifacts; the benchmark drivers and this report are committed.
