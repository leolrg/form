# Offline model of stable-leaf ordering

This is an operation-count experiment, not a timing benchmark or a proposed
change to production matching. It asks whether placing correspondences with
similar observed stability into the same 64-row QR leaves can reduce leaf and
cached-ancestor recomputation. Probability-based ordering is not claimed as new.

## Policies and accounting

`benchmarks/analyze_stability_order.py` reads the existing binary match audit
through `read_records`. Accepted rows are identified by target scan and target
point index; rejected rows are absent. Distance-only changes do not dirty fixed
feature rows. One scan's feature histories are retained at a time, so memory
scales with the maximum single-scan history rather than the entire sequence.

Five policies use the same 64-row, first-hole slot model:

1. **Query order:** initial rows enter each target group in original query order.
2. **Oracle future changes:** at the first rematch, sort initial arrivals by their
   total future accepted-row identity change count within that snapshot. Ties
   use original query ID. This uses unavailable future information and is only
   an optimistic heuristic for assessing headroom. It is not a mathematical
   optimum or a guaranteed upper bound on attainable savings.
3. **First change flag:** start in query order. At rematch 1, sort the currently
   accepted rows by whether their identity changed from rematch 0, then rebuild
   every current leaf and ancestor. That complete rebuilding cost is included
   in steady-state totals even if the first transition changed no rows. Keep
   this ordering thereafter. This is causal, but sorting, allocation, clearing,
   and transfer costs are not represented by QR counts.
4. **Two-transition persistence:** start in query order. At rematch 2, sort by
   `changed(0,1) + 2 * changed(1,2)`, then rebuild every current leaf and ancestor.
   All earlier work and this complete rebuild are included. Histories ending
   before rematch 2 do not reorder. This tests whether a more recent repeated
   change predicts future instability better than the first transition alone.
5. **Dirty-only partition:** after ordinary first-hole placement, collect the
   live rows only from already-dirty leaves within each target group. Stably
   partition them into currently changed identities first, unchanged identities
   second, preserving their current slot order within each class. Pack them into
   those same dirty leaves in ascending slot order, followed by zero holes.
   Update query-to-slot locations. Repeat on every rematch. Clean leaves never
   move; no extra leaf QR is needed in the current iteration. Gathering,
   partitioning, row movement, and metadata updates remain unmodeled costs.

The three once-sorted policies place stable/low-change rows first. They change
only initial arrival ordering or a single learned rebuild. Dirty-only partition
instead moves rows repeatedly within already-dirty leaves. The model
maps every location and slot back to original query IDs; later arrivals fill
lowest holes in ascending original query order. There is no silent persistent
renumbering that would improve later arrivals using oracle information.

The model reuses `StableBlocks` placement and reports its original counters for
comparison. Use the `actual_*` counters to assess the cached64 implementation:

- `actual_leaf_qr64` includes every dirty allocated leaf, including a leaf that
  becomes completely empty. Its packed row buffer still requires recomputation.
- Allocated leaf extent is persistent highwater. Empty interior and trailing
  leaves continue to participate as zero roots, rather than being optimistically
  compacted out of the tree.
- `actual_cached_merge_qr64` counts dirty ancestors with more than one child.
  Plane roots have fan-in 4; point roots have fan-in 9. A one-child ancestor is
  counted separately in `actual_cached_copy_nodes`.
- `actual_cached_merge_input_roots` counts roots fed to merge QRs. The sum
  `actual_qr64_tasks` counts leaf and merge QR invocations, not FLOPs or elapsed
  time. Different widths and row counts have different costs, so totals are also
  reported separately by feature kind.

The `reorder` totals are an overlapping breakdown of the first-change policy's
rematch-1 work or the persistence policy's rematch-2 work. They are already
included in `steady` and `total`; do not add them a second time. The
`after_rematch2` breakdown contains only rematches strictly after rematch 2 for
every policy. Dirty-only partition has no complete rebuild in `reorder`, but
that does not imply zero rearrangement overhead. Initial preprocessing to discover oracle change counts is
deliberately excluded because this policy cannot be implemented online.

## Why correlation matters

If unstable queries are scattered across leaves, a few changed rows can dirty
nearly every leaf. Grouping those rows can leave the other leaves untouched.
But per-query change frequency alone does not determine the best layout:
queries that change together should share leaves, while queries that alternate
changes can keep a shared leaf dirty on every rematch. Conditional on independent
row changes with probabilities `p_i`, a leaf is dirty with probability
`1 - product(1-p_i)`; actual correspondence changes need not be independent.

The oracle ranks marginal change counts, not joint change events. Its measured
benefit may therefore underestimate a different oracle clustering scheme or
fail to transfer to an online predictor. The first-change flag checks one cheap
predictor, including its rebuilding cost. Later migrations and first-hole
arrivals can gradually dilute either ordering. None of these counts includes
the host work that already limited earlier stable-leaf versions.

## Reproduction and validation

```bash
python benchmarks/analyze_stability_order.py \
  benchmarks/results/certified-rematching/audit-v1/stairs-current-off-r1.matches.bin \
  --warmup 20 --output /tmp/form-stability-order.json
cd benchmarks
python -m unittest test_stability_order.py test_stable_blocks.py
```

The tests check a single changed leaf, no changes, cross-group migration,
deletion of the last row, singleton ancestor copies, query-ID preservation,
ordinary first-hole behavior after reordered initialization, canonical target
scan IDs, query-count consistency, and explicit rematch-1 rebuilding cost.

## Stairs audit result

Run on `audit-v1/stairs-current-off-r1.matches.bin` with warmup 20. The included
data comprise 230 scan snapshots, 460 feature-kind snapshots, and 4,486 feature
matching records: 460 initial and 4,026 subsequent rematches. The largest
retained scan history contains 472,316 query rows across its rematches and feature
kinds. All policies see 2,123,481 changed accepted-row identities during the
subsequent rematches. Nine new tests plus nine existing stable-block tests passed.
The retained JSON is
`benchmarks/results/certified-rematching/audit-v1/stability-order-current.json`.

| Subsequent rematches | Query order | Oracle future changes | First change flag, rebuild included |
|---|---:|---:|---:|
| Leaf QR64 calls | 371,964 | 217,590 | 369,861 |
| Cached ancestor QR64 calls | 191,473 | 134,281 | 190,405 |
| Inputs to ancestor QR | 688,676 | 455,294 | 684,318 |
| Singleton ancestor copies | 7,871 | 10,670 | 8,272 |
| Leaf + ancestor QR calls | 563,437 | 351,871 | 560,266 |
| Reduction in combined QR calls | — | 37.55% | 0.56% |

All three policies have the same initial work: 70,554 leaf QRs and 24,759
ancestor QRs. Including this initial work, their combined QR call counts are
658,750, 447,184, and 655,579 respectively. Oracle ordering reduces this total
by 32.12%; the first-change flag reduces it by only 0.48%.

The learned policy's rematch-1 rebuilding alone comprises 70,571 leaf QRs and
24,766 ancestor QRs (95,337 combined), already included in the table. Sorting
and moving those rows would add unmodeled costs. Oracle ordering reduces leaf
QR calls by 41.50% and ancestor QR calls by 29.87% over subsequent rematches.

| Feature kind, subsequent rematches | Query-order combined QR | Oracle combined QR | First-change combined QR |
|---|---:|---:|---:|
| Planes, 13 columns | 520,432 | 316,480 | 517,242 |
| Points, 7 columns | 43,005 | 35,391 | 43,024 |

Most oracle savings come from plane features. The learned policy slightly
increases point-feature QR calls, so the aggregate does not indicate a uniform
benefit across feature types.

This single trace demonstrates that slot layout can matter even with cached
ancestors. It does **not** support implementing the first-change flag: its small
QR savings leave little room for sorting, rebuilding metadata, host, or device
overhead. The future-change heuristic motivates investigating a stronger causal
predictor only if measured cached-tree performance leaves a relevant bottleneck.
It cannot justify a speedup claim, a general result across scenes, or an optimal
partitioning claim.

## Dirty-only partition invariant

For each target group, the destination slot set equals the union of the
already-dirty 64-row leaves. Partitioning permutes all live query IDs in that
set and fills the remaining slots with holes. Every excluded slot stays
unchanged. Consequently, clean leaves are immutable, the accepted feature-row
multiset remains identical within each group, and each live query still owns
exactly one slot. In real arithmetic this preserves the sum of row outer
products, hence the group Gram matrix. Floating-point QR results can differ
with row ordering; this model does not establish numerical tolerances.

The current dirty leaf set and allocated extent are unchanged, so the current
leaf and cached-ancestor QR counts equal ordinary placement from the same
pre-update state. Future counts may differ because rows now share different
leaves. This is an operation-count opportunity, not a free implementation.

Additional tests cover the weighted two-transition score, short histories,
complete rematch-2 rebuilding, all preceding work, later work separately,
clean-leaf immutability, current dirty-leaf QR equality, hole placement,
query-to-slot ownership, and per-group feature-row multiset preservation across
migration. Fourteen ordering tests plus nine existing stable-block tests pass.

## Causal follow-up on the same stairs capture

The five-policy run uses the same capture, warmup, and 230 snapshots as above.
Its JSON is `benchmarks/results/certified-rematching/audit-v1/stability-order-causal-current.json`. Baseline, oracle, and
first-change counts reproduce the earlier run exactly. All five policies see
the same 2,123,481 subsequent identity changes. The additional phase strictly
after rematch 2 contains 3,106 feature matching records and 506,830 changes.

| QR64 work | Query order | Two-transition persistence | Dirty-only partition | Future-count oracle |
|---|---:|---:|---:|---:|
| Subsequent leaf QRs | 371,964 | 371,032 | 365,028 | 217,590 |
| Subsequent ancestor QRs | 191,473 | 190,050 | 190,203 | 134,281 |
| Subsequent leaf + ancestor QRs | 563,437 | 561,082 | 555,231 | 351,871 |
| Subsequent combined reduction | — | 0.42% | 1.46% | 37.55% |
| All combined QRs, initial work included | 658,750 | 656,395 | 650,544 | 447,184 |
| All combined reduction | — | 0.36% | 1.25% | 32.12% |
| Combined QRs strictly after rematch 2 | 373,936 | 370,092 | 366,217 | 251,551 |
| Reduction strictly after rematch 2 | — | 1.03% | 2.06% | 32.73% |

Persistence's rematch-2 complete rebuild comprises 70,580 leaf QRs and 24,775
ancestor QRs, or 95,355 tasks already included above. Its preceding work is
also included. Its subsequent point-feature work increases to 43,626 tasks
from 43,005; planes fall to 517,456 from 520,432. Even after excluding the
learning transitions for comparison, its 1.03% reduction is modest. The
two-transition score therefore does not materially improve this trace over the
first-change policy, which saved 0.48% overall.

Dirty-only partition gives 512,352 subsequent plane tasks and 42,879 point tasks.
It avoids complete rebuilding and reduces QR work slightly more than either
once-learned policy. The overall saving is still only 8,206 QR calls out of
658,750, before charging any gathering, stable partitioning, row movement, or
location maintenance. This does not support production implementation on the
basis of this trace alone.

These bounded experiments weaken the case for further work on these particular
cheap causal predictors. They do not rule out useful predictors or a layout
that exploits joint change events. The large gap to the unavailable future-count
heuristic indicates that these observed-change policies capture little of its
benefit; it does not establish a speedup available to an online implementation.
