# Certified rematching and incremental nonlinear factor summaries

Experimental method on `codex/certified-rematching`. Performance evidence and
negative results are tracked separately in `certified-rematching-experiments.md`.
This document describes the correctness target; it does not establish novelty
or predict conference acceptance. See `certified-rematching-prior-art.md` for
the substantial overlap with earlier assignment bounds, cached ICP, sufficient
statistics, and QR maintenance.

## State and validity

One `reset` defines an immutable target-map snapshot and an immutable list of
local query features. FORM changes the current query pose between rematches,
while the target snapshot remains fixed during that loop. A new snapshot
invalidates every search anchor, row assignment, and summary cache.

The reference search is the existing GPU matcher: it visits 27 voxels around
the query voxel in FORM's prescribed order, uses the original four-component
squared-distance arithmetic, and retains the first candidate on a tie. The
certificate preserves this implementation's output for identical snapshot and
query inputs. It does not assert bitwise equality of CPU and GPU arithmetic or
of independently scheduled feature-extraction runs.

Each query has two independent pieces of persistent state:

1. A search anchor: the query position at the last necessary full search, its
   query voxel, winning target index, and a lower-bound construction for every
   competitor. The anchor remains fixed after successful certification.
2. A summary slot: the query's row position inside its current target scan's
   factor, if accepted. This slot can remain unchanged even after a full search,
   provided that search returns the same accepted correspondence.

A certificate concerns the nearest target identity. Acceptance and ownership
are recomputed using the current distance, strict threshold, and target-group
assignment. Stable identity can therefore still cause row insertion/deletion
when the gate changes. Reused matches also receive freshly computed distances
because final map insertion consumes rejected as well as accepted results.

## Search certificate

Require the query voxel to equal its anchor voxel. This preserves the exact
candidate domain and encounter order, including explicitly supplied voxel ranges
whose points are not geometrically inside those voxels. Crossings fall back to
full search. Identical transformed coordinates and truly empty candidate domains
have separate exact shortcuts.

For other cases, let `R` be a lower threshold on every distinct competitor's
computed squared distance at the anchor. The initial implementation obtains it
from the second distinct candidate. The occurrence-bound variant uses the second-smallest
distance over all candidate occurrences. Repeated indices can only lower this
threshold, making it conservative.

Convert `R` into a directed-rounded lower bound `L` on each competitor's real
anchor distance. If `U` bounds query displacement upward, reverse triangle
inequality bounds every competitor's current distance below by `max(0, L-U)`.
Convert that into a lower bound on its **computed** current squared distance.
Certify only when the original-arithmetic winner distance is strictly smaller.
This excludes computed-distance ties. The full binary64 enclosure, including
underflow and overflow, is given in
`form/optimization/certified_rematching.md`. An optional squared predicate uses
directed products and strict polynomial inequalities to avoid square roots and
divisions; extreme scales can make it fall back more often.

There are fused and split GPU schedules. The fused kernel assigns one warp per
query and checks the bound in lane zero. The split schedule uses one thread per
query for certification, then warp-cooperative fallback search. They implement
the same decision. Their launch, memory-access, and fallback costs are measured
independently; skipping more searches is not itself a timing result.

Audit mode additionally runs the unchanged reference kernel on the same device
snapshot, checks winner indices and computed distances exactly, and supplies its
results to the rest of FORM. This avoids conflating matching correctness with
query-order or solver-feedback differences between separate processes.

## Summary rows and updates

The existing accelerated FORM representation uses seven-column point rows and
thirteen-column plane rows:

```text
point: [1, pi, pj-pi]
plane: [vec_row(n pj^T), n, n^T(pj-pi)]
```

Here `pi` and `n` are the matched target's local coordinates/normal, and `pj` is
the query's local coordinate. The rows do not depend on either optimized pose.
For the supported isotropic factors, residuals and both right-local pose
Jacobians are linear functions of these fixed features at every evaluation pose.

Each target scan owns stable row slots grouped into 64-row leaves. A query that
retains the same accepted target keeps its slot. A changed target within the
same scan dirties that leaf. Migration/rejection frees its former slot and dirties
the old leaf; arriving rows fill first-available holes and dirty destination
leaves. Empty slots contribute zero feature rows. Snapshot/group/feature-kind
changes invalidate the state rather than mixing incompatible rows.

The sorted variant obtains incoming rows from a stable grouped radix sort.
The queued variant instead reserves incoming queue segments with warp-aggregated
integer atomics. Removal/enqueue and first-hole filling remain separate kernels
on one stream, so arrivals cannot overwrite rows before departures finish.
It skips global sorting only for ordinary incremental operation; full-QR audits
and raw-row observers retain the original sorted path. Queue reservation order
is nondeterministic. This changes slot/QR ordering, not nearest-neighbor IDs or
the per-group feature-row multiset, and requires separate numerical and timing
validation. It does not add a deterministic-execution guarantee.

A leaf stores a Householder QR root of its feature rows. For feature matrix `F`
and cached root `U`, the invariant in real arithmetic is `U^T U = F^T F`.
An internal node stores the QR root of its children's stacked roots, so its
invariant follows by addition. Rebuilding dirty leaves and their ancestor union
therefore restores the full factor's invariant. Unchanged nodes retain their
roots; empty groups contribute zero roots. Published host summaries own their
matrices, so later device-cache updates do not mutate factors from earlier
optimization calls.

Consequently the supported nonlinear cost, gradient, and both-pose Gauss–Newton
matrix are preserved for every pose in real arithmetic. Floating-point equivalence requires validation on the intended workloads;
different reduction orders can produce different roots and solver iterates.
The QR path assumes finite, reasonably scaled feature rows and does not prevent
all possible overflow. Tests compare objective values and augmented Hessians
at selected poses. Independent summary audit compares normalized Gram error
against a full QR rebuild on identical inputs at a threshold of 1e-10, then
returns the full roots to preserve reference solver feedback. That audit is not
an exhaustive all-pose floating-point error guarantee. No eigenvalue truncation or subtraction of Gram matrices is
needed for these updates.

The flat prototype caches leaves and final group roots, rebuilding all merges
inside an affected group. The tree prototype caches internal ancestors too and
keeps extent/dirty propagation on the device. These are distinct ablations.

## Work and limits

Let `N` be query count, `S` the set of uncertified queries, `Cq` the candidates
visited by query `q`, `D` the dirty leaves, and `A` their dirty ancestor union.
Let `H` count the nodes in the current highwater extents. With fixed feature
width `k` (7 or 13), per-rematch work includes:

```text
O(N) certificate checks
O(sum_{q in S} Cq) fallback distance evaluations
group classification/sorting and stable-slot maintenance
O(H) inspection of leaves/ancestors across all active highwater extents
O(GN/64) clearing of the reserved dirty-leaf buffer
O(64 k^2 |D|) leaf QR
O(64 k^2 |A|) ancestor QR
O(G k^2) compact factor output for G target scans
```

The fused/distinct and split/occurrence variants have different constants in the
first two terms. Initial runner-up acquisition, initialization, kernel launches,
transfers, and synchronizations must be counted. Current slot assignment still
examines queries and uses grouped sorting; total work is not proportional only
to the number of changed correspondences.

The prototype reserves a worst-case slot range per group, giving `O(GN)` backing
storage even though populated high-water extents are much smaller in measured
epochs. Long-lived fragmentation and first-search initialization can erase the
savings. The cached tree reduces QR work when changes are concentrated; uniformly
distributed changes can dirty nearly every leaf. These are measurable failure
regimes, not exceptions to the performance accounting.

The potentially useful contribution is the complete estimator-preserving
procedure and its evaluated operating region. Triangle inequalities, cached
nearest-neighbor search, QR trees, and selective statistics updates are established
ideas. Any publication claim must identify what the complete FORM-specific
procedure adds and demonstrate useful savings against the existing fast pipeline.
