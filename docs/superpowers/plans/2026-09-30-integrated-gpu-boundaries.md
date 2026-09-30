# Integrated GPU boundary experiments

**Goal:** Resolve the requested CPU/GPU placement recommendation with integrated
quality and latency evidence on the available A100/stairs workload.

**Design:** Add opt-in GPU-resident masks/selection feeding existing GPU normals,
with original libstdc++ tie ordering. Retain matched CPU/index-stable controls
as a rejected semantics experiment. Separately add
GPU final target gathering/materialization and a GPU voxel-map builder consuming
local features in preserved insertion order. Keep the original hybrid controls.
Do not change defaults or silently substitute a different quality workload.

- [x] Add extraction parity tests and integrated stable-order CPU/GPU controls.
- [x] Implement masks, sector sorting, ordered suppression, device compaction,
  adjacent search and normals without curvature/mask/index round trips.
- [x] Add map/materialization tests and opt-in implementations preserving raw
  ownership and supported API behavior.
- [x] Build and run unit tests and targeted sanitizers before clean timings.
- [x] Run separate staged full-quality pilots. If quality fails, diagnose or
  reject the variant; do not justify CPU placement from an incorrect port.
- [x] Freeze binary and run serial paired reversed-order current/full and dense
  prefix comparisons, keeping changed tie semantics visible and CPU-matched.

The index-stable CPU and GPU selectors failed the full stairs quality gate:
30 m mean translation RTE 0.23625 → 0.43834 m. Implement original-order
introsort on device before accepting extraction timing as a compatible port.
- [x] Report a definite workload-scoped placement recommendation, negative
  results included, with measured limits and outstanding generalization scope.

Results and decision: [integrated GPU boundary results](../../integrated-gpu-boundaries-results.md).
