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

Three policies use the same 64-row, first-hole slot model:

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

Both sorted policies place stable/low-change rows first. Only the initial
arrival ordering (or the learned policy's single rebuild) changes. The model
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
rematch-1 work. They are already included in `steady` and `total`; do not add
them a second time. Initial preprocessing to discover oracle change counts is
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
