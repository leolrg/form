# Detailed estimator diagnostics plan

**Goal:** Measure the complete existing algorithm before further optimization, including CPU/GPU attribution, setup overhead, copies, waits, and workload scaling.

**Architecture:** Keep existing estimator timers and add optional inclusive host-stage counters, per-optimizer-call records, and separately enabled NVTX ranges. Use CUDA API/activity correlation from Nsight Systems rather than inserting synchronizations. Timed controls, explicit counter profiles, and traced replays are separate populations. Algorithm, solver threshold, features, and iteration rules stay unchanged.

**Execution:** Inline in the current branch; implementation and benchmarking are authorized by the user. Commit instrumentation before freezing the executable, and commit/push the final analysis afterward.

- [x] Add tests for disabled/reset/nested diagnostic counters and accounting; implement header-only diagnostics and optional NVTX ranges.
- [x] Instrument CPU/GPU optimizer phases, graph setup, matching/grouping/QR/materialization, and extraction GPU helpers. Record per-call dimension, semi/full phase, LM attempts/acceptances/failures and stage durations without file I/O inside estimator timing.
- [x] Verify CUDA and CPU-only builds/tests; compare instrumented/uninstrumented scan counts, workloads, trajectories, and counter accounting. Freeze binary and source revision.
- [ ] Run sequential repeated clean controls (original CPU, improved CPU, current CUDA), detailed profiles, and same-prefix density/window scaling. Exclude the same first 20 scans; preserve raw output/provenance and report repeated-run dispersion, median/p95/p99, and slow-scan associations.
- [ ] Collect NVTX/CUDA traces for current and larger-window workloads, split warmup from steady state by scan ranges, correlate device operations to issuing host scopes, and report overlap-aware totals. Probe available CPU/hardware counters without changing host security settings.
- [ ] Measure CPU/GPU threshold alternatives on matched inputs, keeping them diagnostic experiments, and validate output parity.
- [ ] Produce reproducible analysis, explicit timer hierarchy and residuals, instrumentation overhead, limitations, and ranked opportunities; review, commit, and push.
