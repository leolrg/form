# Same-cell selective matching certificate

This opt-in research path assumes only the existing immutable `reset` snapshot,
finite binary64 inputs, CUDA binary64 gradual underflow, round-to-nearest ordinary
arithmetic, and the project's `--fmad=false` compilation. It makes no assumption
that an explicitly supplied voxel's points lie geometrically inside that voxel.
The original `nearest` kernel remains the Disabled implementation.

For a query, the transformed binary64 world coordinates are the actual search
inputs. The fourth (padding) coordinate is fixed by `reset`. A cached anchor stores
these world coordinates, query voxel coordinates, selected point index, and the
second distinct point's computed squared distance. Candidate ordering is exactly
(distance, neighbor visit order, point index). Overlapping explicit voxel ranges
can repeat an index; these repetitions are not distinct competitors. All candidates
at a distance greater than or equal to `DBL_MAX` are represented by `DBL_MAX`, as
in the original search. A separate bit records a truly empty candidate domain.

An unchanged query cell means the entire ordered candidate domain is unchanged.
A changed cell always runs full search. An identical transformed query in the same
cell can reuse the index even in a tie, including the original no-winner sentinel.
A genuinely empty domain can reuse the sentinel after any same-cell motion. Other
no-winner cases (for example all squared distances overflowing) fall back.

## Floating-point enclosure

Let S be a candidate's exact real squared distance between the binary64 search
inputs. The ordinary kernel computes

    D = RN(RN(RN(dx*dx) + RN(dz*dz)) + RN(RN(dy*dy) + RN(dw*dw)))

with each difference also rounded to nearest. Write u=2^-53, eta=2^-1074,
E=64*epsilon=2^-46=128u, and A=32*eta. In the absence of overflow,

    (1-E) S - A <= D <= (1+E) S + A.

Here is a conservative justification, including underflow. A subtraction of two
binary64 numbers which is subnormal is exact (the exact difference is a multiple
of eta and fits the subnormal significand); all other finite subtractions have
relative error at most u. The squared rounded difference therefore has relative
factor between (1-u)^3 and (1+u)^3, plus absolute rounding error at most eta/2
for a subnormal product. Each term then passes through two nonnegative additions.
Their relative factors give the bounds (1-u)^5 and (1+u)^5. The four product
absolute errors plus three addition absolute errors, propagated through the two
levels, are less than 4*eta. Since (1+u)^5 < 1+128u and (1-u)^5 > 1-128u,
E and A safely enclose both sides. Cancellation in the coordinate subtraction
is covered by the subtraction model and does not require an absolute-coordinate
heuristic.

Overflow does not defeat the anchor lower bound. If an intermediate subtraction
overflows, S is already vastly larger than DBL_MAX. If the first overflow occurs
in a product or nonnegative addition, the round-to-nearest overflow threshold
is larger than DBL_MAX; the same finite predecessor error factors imply
S >= (DBL_MAX-A)/(1+E). Thus clamping D to DBL_MAX preserves the upper inequality
needed below, even when the ordinary D is infinity. An overflowing current
competitor distance is automatically greater than a finite selected distance.
There are no NaN distances from finite inputs: all squared terms are nonnegative.

Let R be the anchor's minimum clamped computed distance over every point other
than the selected index. With fewer than two distinct selectable points, R is
DBL_MAX; this is still a valid lower threshold for any unselected overflowed
candidate and is vacuous if there are none. For every competitor, an anchor norm
lower bound is

    L = sqrt(max(0, (R-A)/(1+E))).

Compute the displacement norm upper bound U from the anchor's three transformed
coordinates to the new ones; the fourth-coordinate displacement is zero. Each
absolute difference is bounded with the maximum absolute value of directed down
and up subtractions. All squares, additions, and square root in U round upward.
By the reverse triangle inequality, every competitor's current exact norm is at
least max(0,L-U). Therefore its *computed* current squared distance is at least

    B = (1-E) max(0,L-U)^2 - A.

Every operation forming L and B rounds downward, with explicit CUDA directed
intrinsics; U rounds upward. An overflow in U rejects the certificate. Recompute
the cached winner's distance using the original arithmetic, and accept only if
it is finite, less than DBL_MAX, and strictly less than B. This strict comparison
excludes even rounded-distance ties, preserving original visit order. Padding is
included in R and the exact winner distance. The certificate never moves its
anchor after a successful reuse.

## Audit, lifetime, and cost

Audit evaluates this identical predicate, then unconditionally executes the full
ranked search and compares both selected index and recomputed distance exactly.
It additionally launches the untouched original `nearest` kernel on the identical
device snapshot, compares its index and distance against the ranked-search output,
and returns the original oracle results. Either comparison increments the mismatch
count; an original-oracle disagreement invalidates that query's anchor. The
`oracle_searched` statistic counts these independent original-kernel searches.
Successful audit certificates retain the old anchor, simulating actual skipping;
failed certificates refresh it. The `unchanged` statistic compares consecutive
oracle indices, independently of whether certification succeeded. A certificate mismatch
refreshes the anchor; an original-oracle mismatch invalidates it. Certified mode skips all neighbor hash probes only after
a positive certificate. The per-search statistics are stored as one byte per
query and downloaded/aggregated only by an explicit `reuseStats()` call.

A reset or mode change invalidates all anchors before fallible work; failed
searches also invalidate them. Results retain their existing lazy download API.
Grouped search still classifies current distances and rebuilds all QR summaries;
there is no summary reuse in this stage.

Costs include an anchor record per query, one byte of statistics per query,
top-two comparisons and warp shuffles on each full search, and directed arithmetic
for each attempted certificate. Only lane zero evaluates the predicate and reads
the cached point. No speedup follows merely from a high unchanged-index fraction;
benchmarking must account for first-search overhead and inconclusive certificates.

## Search-kernel and competitor-bound ablations

Two independent opt-in controls retain the original fused/distinct implementation
as the default:

- `setSearchKernel(SearchKernel::Fused|Split)`, or
  `FORM_CUDA_MATCH_KERNEL=fused|split`.
- `setOccurrenceBound(false|true)`, or
  `FORM_CUDA_MATCH_TOP2=distinct|occurrences`.

Both setters invalidate existing anchors, even when setting the same value.
Disabled reuse continues to invoke the untouched original search kernel regardless
of these controls. Invalid environment values are rejected.

The split variant evaluates one certificate per thread, with 32 query certificates
per warp. Its prepass writes flags and a current winner result only; it does not
modify any anchor or previous index. The subsequent full-search kernel is ordered
on the same stream. In Certified mode its certified warps return before repeating
the transform or performing neighbor probes. Invalid prepass coordinates retain
the original -2 result. In Audit mode every valid query still executes full search,
checks the prepass candidate, and is independently checked by the untouched oracle.
The first search after invalidation omits the prepass entirely. `split_queries`
counts queries actually passed through this separate certificate kernel.

The occurrence variant keeps the original winner reduction exactly: within each
lane the first strictly smaller computed distance wins; across lanes the minimum
(distance, visit rank) wins. It replaces distinct-index top-two tracking with the
two smallest clamped computed distances over *all candidate occurrences*. For a
new distance d and old local values b,s, the update is

    s' = min(s, max(b,d));    b' = min(b,d).

For two partial pairs (a1,a2) and (b1,b2), the merged second distance is

    min(a2, b2, max(a1,b1)).

Use the pre-merge best values in this expression. DBL_MAX initialization acts as
sentinel padding; infinite and exact-DBL_MAX distances do not create a selectable
winner. All candidate distances are nonnegative and non-NaN for validated finite
search coordinates, so these min/max operations introduce no new arithmetic error.
Templates select these reductions without a per-candidate runtime mode branch.

To prove conservatism with overlapping voxel ranges, let w be the original winner
and j any distinct competitor. There are at least two occurrences no greater than
j's computed distance: one of w and the occurrence of j itself. Thus the second
occurrence distance R is no greater than *every* distinct competitor's clamped
computed distance. The earlier enclosure proof only needs this inequality, not
that R is attained by a distinct competitor. Repeated winner indices can lower R
and reduce certificate yield; they cannot permit a false certificate. Computed
ties lower R to the winner distance and remain subject to the strict rounded-bound
comparison, except for the proven identical-coordinate special case.

The split variant adds a launch and per-query flag traffic, and still launches
masked fallback warps. The occurrence bound saves comparisons, rank storage, and
one rank shuffle per reduction step, at the possible cost of weaker certificates.
Both variants require measured end-to-end comparisons rather than a speedup claim
from certificate yield alone.

## Optional squared certificate

`setSquaredCertificate(true)`, or `FORM_CUDA_MATCH_CERTIFICATE=squared`, selects
an alternative strict predicate without square roots or division. `norm` is the
default and retains the preceding implementation. The setter invalidates anchors,
even when setting the same value; invalid environment values are rejected.
Template specialization selects the predicate in both fused and split kernels.
This is an arithmetic ablation, with no assumed performance benefit.

Let c be the cached winner's newly computed distance, R0 the anchor's stored
competitor threshold, and D an upward-rounded bound on squared displacement as
above. Use directed arithmetic to form

    R = down(max(0, down(R0-A)) * (1-E))
    W = up(up(c+A) * (1+E+epsilon))
    T = down(down(R-D)-W).

Both multiplier constants are exactly representable binary64 numbers. They satisfy

    1-E <= 1/(1+E)
    1+E+epsilon >= 1/(1-E),

because the second reciprocal's remainder after 1+E is E^2/(1-E), smaller than
binary64 epsilon. Therefore R is no greater than any competitor's exact anchor
squared distance, including a clamped overflowed anchor. W is at least
(c+A)/(1-E). Clamping before the lower multiplication preserves the inequality
when R0-A is negative.

Reject nonfinite c, c >= DBL_MAX, nonfinite D, or nonfinite W. Certify only if

    T > 0  and  down(T*T) > up(4 * up(D*W)).

This implies R-D-W > 2*sqrt(D*W), and hence
sqrt(R) > sqrt(D)+sqrt(W). The reverse triangle inequality makes every current
competitor's exact squared distance strictly greater than W. The lower error
enclosure then makes its computed distance strictly greater than c; overflowed
current competitors also exceed c. Thus rounded-distance ties cannot be skipped.
Using a lower R and upper D,W is conservative: the exact criterion
sqrt(R)>sqrt(D)+sqrt(W) is monotone in precisely those directions. The positive
T check is necessary before squaring.

Directed products preserve these implications under gradual underflow. In
particular an underflowed left square can reject an otherwise useful certificate;
an upward-rounded right product cannot silently vanish below its true value.
A downward-rounded positive overflow yields DBL_MAX, still a valid lower bound;
an upward-rounded right overflow yields infinity and rejects. All operands after
the finite checks are nonnegative and finite, so neither product introduces NaN.
There is no rescaling to recover certificate yield at extreme magnitudes. The
norm and squared predicates may therefore make different conservative decisions,
while both must return the exact original search result. Padding is included in
R0 and c and has zero displacement; same-cell, identical-query, empty-domain,
anchor-lifetime, and Audit rules are unchanged.
