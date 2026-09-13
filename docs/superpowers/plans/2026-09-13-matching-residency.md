# Matching residency implementation plan

> **For agentic workers:** Use subagent-driven-development for independent implementation/review tasks. Continue under the user's approval of all three changes.

**Goal:** Reduce matching and preparation latency without reducing FORM's workload or changing its optimization problem.

**Architecture:** Compact GPU search, device acceptance/grouping and QR, deferred host correspondence materialization, and device preparation of local target coordinates.

**Tech Stack:** C++17, CUDA FP64, CUB, GTSAM, Eigen, GoogleTest, Nsight Systems.

- [x] Profile frozen current warp matcher over 100 stairs scans. Store `benchmarks/results/matching-next-baseline/nsys.csv`. Observed nearest 201.84 ms, QR 130.76 ms, pack 6.11 ms GPU totals; profile overhead precludes using replay totals as formal benchmarks.
- [x] Optimize `cuda_matcher.cu` coordinate layout and 27 hash probes; add adversarial tie/padding/tail tests in `test_CudaMatcher.cpp`, verify CPU parity and tune launches. Commit verified search milestone.
- [x] Add deferred raw storage contracts in `feature/factor.hpp`, raw evaluation/summary preparation boundaries, and tests. Counts and immutable summary snapshots remain valid before materialization.
- [x] Extend `CudaMatcher` with device grouping by target group, strict threshold acceptance, compact count return, stable query ordering and device QR input packing. Test empty groups, rematches, rejected queries, and raw/device summary equality.
- [x] Integrate deferred matching and finalization in `cuda_matching.cpp`/`form.cpp`; default public raw matching remains compatible. Test raw access and reset lifetimes. Commit verified residency milestone.
- [x] Move eager CPU world-to-local target conversion to GPU snapshot preparation, reuse host/device storage, and defer raw target reconstruction. Test rotation/translation/normal/padding parity and map replacement. Commit verified preparation milestone.
- [x] Build `form-tests` and `form-replay` in CUDA and CPU-only configurations; run CTest and benchmark Python tests. Run matcher memcheck/racecheck. Resolve review findings.
- [ ] Freeze binary and source. Run fresh paired current/features/window CPU and CUDA controls, two reversed repeats on first 250 stairs scans. Compare frozen previous CUDA separately with alternating order and trace agreement.
- [ ] Run complete stairs validation, profile final pipeline, document timings and correctness with stage/coverage labels. Commit final report and completed plan.

Build commands: `cmake --build build-accel --target form-tests form-replay -j 8`, `ctest --test-dir build-accel --output-on-failure -j 8`; repeat for `build-cpu-verify`. Python checks: `/home/ubuntu/.local/share/uv/tools/evalio/bin/python -m unittest discover -s benchmarks -p 'test_*.py'`.

Search milestone `d76f359`: exact result hashes across block sizes 64/128/256/512; chosen64 gave 2.1–4.3x synthetic kernel speedups. Hardware memory counters unavailable (`ERR_NVGPUCTRPERM`). Two reversed first100scan replay repeats showed no reliable total gain (baseline42.07/38.73 ms, search40.75/41.97 ms); this is a diagnostic, not the final comparison. Deferred storage milestone `d325e66`: serial/concurrent raw reads, copies, cancellation, append, and exception retry tested.

Core `24c6da0` and adapter `8a4af10` pass 86 CUDA C++ tests; CPU-only 42 pass with8 CUDA skips; 30 Python tests pass. Matcher/adapter18 tests pass memcheck, synccheck and racecheck with zero errors. Frozen final implementation: `benchmarks/results/matching-residency-backend/form-replay`, SHA256 `77d7a1f297aeb09d932d53149c05ee5173ac4ca00cb3416139964b0ab140485f`.
