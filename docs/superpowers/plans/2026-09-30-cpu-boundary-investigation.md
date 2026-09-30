# Remaining CPU boundary investigation

**Goal:** Establish evidence for masks/selection and map/materialization placement,
using the corrected GPU-normal implementation as the baseline.

**Scope:** Diagnostic experiments only; preserve production defaults and previous
uncommitted normal work. No claim of a full GPU port without integrated replay.

**Design:** Reuse the two current/dense instrumented normal campaigns for stage
budgets. Add a standalone CUDA extraction probe using real FORMPC scans. Compare
the original float range/curvature/suppression rules with GPU masks and ordered
greedy selection; separately test index-stable sorting against the CPU's existing
unspecified equal-key order. Time resident and transfer-inclusive scopes explicitly.
Inspect map transforms, voxel insertion ordering, snapshot copies, final match
reconstruction and deferred raw consumers. Prefer preserving numerical/order
semantics over substituting a nominally equivalent algorithm.

- [x] Reproduce current/dense stage budgets from saved verified profiles.
- [x] Build extraction probe with CPU reference and explicit parity checks; verify
  adversarial edge/range/tie cases before accepting timings.
- [x] Measure real-scan CPU/GPU mask and selection scopes; retain commands and data.
- [x] Trace map/raw-data consumers and document required ordering, lifetime and
  representation invariants, distinguishing measured costs from predicted gains.
- [x] Review findings and write a reproducible report with recommendations and
  explicit limits. Preserve failing alternatives as evidence rather than defaults.
