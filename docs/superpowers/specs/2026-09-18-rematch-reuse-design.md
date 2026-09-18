# Reuse optimizer constants between rematches

The user approved accelerating reuse between rematching iterations after the detailed diagnostic campaign. Implement and benchmark on the existing `codex/optimization-acceleration` branch; retain the frozen baseline binary from that campaign. No additional approval is needed for this scoped change.

## Design

Choose exact content-checked reuse of constant data. Alternatives were caller-supplied generations/immutable graph tokens (more invasive ownership contract) or pointer/dimension-only reuse (unsafe for mutable factors and same-size graph changes). Content checks keep existing public reset callers correct, including in-place mutations, and avoid requiring every caller to maintain an invalidation protocol.

1. CPU resident reset still snapshots the current graph, refreshes anchors/auxiliary factor pointers and installs new summary roots. Reuse the assembled frozen information matrix only if the previous reset succeeded, the sorted pose keys match, and every frozen descriptor/matrix/anchor-presence flag matches exactly. Changed anchor values are repacked from the fresh snapshot on every evaluation; they are not baked into the cached constant Hessian.
2. CUDA batch reset always refreshes current feature roots, invalidates the linear model, and reuses assembly CSR uploads only when pose count and actual offsets/indices agree. A same-size reset may retain the independently configured frozen/auxiliary constants.
3. CUDA resident configuration reuses constant matrices and mappings only when dimension, all frozen contents/flags/mappings, and auxiliary pose layouts match. Publish snapshots only after successful configuration. A failed update must not leave an old configuration eligible for reuse.
4. Add optional reuse counters to existing diagnostics. Keep arithmetic, accumulation order, pose updates, feature/matching rules, LM policy and CPU/GPU threshold unchanged.

This first implementation intentionally still classifies/copies factors for validation. It targets expensive reconstruction and upload without assuming that shared factor pointers are immutable. More aggressive immutable-graph ownership can be considered after measuring this result.

## Correctness and measurement

Tests cover updated summaries/weights, changing frozen matrices and anchors, same-size pose-key/layout changes, factor insertion/removal, failed reset/configuration and recovery, stale-model rejection, CPU/GPU parity and unchanged LM outcomes. Cache-hit checks use existing optional diagnostic counters and accompany numerical comparisons to fresh workspaces/original graph evaluation.

Build/test CUDA and CPU-only configurations, run CUDA memory/synchronization checks on new cases, then freeze the binary. Run old/new CPU and CUDA controls sequentially, including all 1190 stairs scans and matching 250-scan current/dense/window workloads. Exclude the same first 20 scans. Keep profiling separate from clean performance runs. Verify counts, timestamps, trajectories, quality where enough segments exist, and LM decisions. Report reset savings and total savings separately, including repeat dispersion and any regressions. Commit implementation along the way and push the final verified result.
