# Certified selective rematching: prior art and falsifiable scope

Research note, 2026-09-19. Baseline: `1255543`. This note assesses a hypothesis;
it reports no implementation, benchmark result, novelty determination, or venue
acceptance prediction. Primary papers below were opened and inspected unless
explicitly marked as a follow-up lead. This is a focused search, not an exhaustive
literature review.

The defensible starting point is **established correspondence certificates plus
established square-root maintenance, specialized to FORM's complete rematching
and nonlinear factor contract**. Neither caching, triangle inequalities,
second-nearest bounds, delta statistics, nor hierarchical QR is new. Even the
combination of certified unchanged assignments and updating statistics only for
changed assignments predates this proposal.

## Closest prior art

| Primary source | Verified overlap | What this source does not establish for FORM |
|---|---|---|
| Nüchter, Lingemann, Hertzberg, *Cached k-d tree search for ICP algorithms*, 3DIM 2007, DOI 10.1109/3DIM.2007.15; [author-hosted paper](https://robotik.informatik.uni-wuerzburg.de/telematics/download/3dim2007.pdf), §2.4 and Algorithm 1 | Reuses the previous search leaf; performs local search and ball-within-bounds checks, then backtracks as necessary. The result is exact nearest-neighbor search across ICP iterations. | It does not supply FORM's voxel-domain/tie semantics or maintenance of its 7- and 13-column nonlinear feature roots. An ordinary cached exact-search speedup is already anticipated. |
| Elkan, *Using the Triangle Inequality to Accelerate k-Means*, ICML 2003; [proceedings paper](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf), §§2–3 | Maintains distance bounds to avoid assignment computations while preserving the iterates of the underlying clustering algorithm. | A new application of triangle inequalities is not a new certificate principle. FORM still needs a proof covering its actual candidate domain and acceptance rule. |
| Hamerly, *Making k-means even faster*, SDM 2010; [author paper](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf), §3, Algorithms 1 and 5, No-Delta ablation | Maintains an assigned-center upper bound and one lower bound on every alternative center, described as a second-closest-center bound. Algorithm 1 changes cluster counts/sums only when assignments change. | This is particularly strong overlap with “certify assignments, then update sufficient statistics selectively.” FORM's possible distinction is its nonlinear, two-pose, square-root factor representation and numerical contract. |
| Daniel, Gragg, Kaufman, Stewart, *Reorthogonalization and Stable Algorithms for Updating the Gram–Schmidt QR Factorization*, Mathematics of Computation 30, 1976, pp. 772–795; [paper](https://www.stat.uchicago.edu/~lekheng/courses/31060w14/DGKS.pdf) | Classical algorithms already update QR after rank-one modifications and row/column insertion or deletion, with numerical analysis. | “Incremental QR” alone is not a contribution. A root-only update must be analyzed for its actual retained information; the general full-QR algorithms cannot simply be assumed to justify every proposed downdate. |
| Demmel, Grigori, Hoemmen, Langou, *Communication-optimal parallel and sequential QR and LU factorizations: theory and practice*, 2008; [authors' preprint](https://arxiv.org/abs/0806.2159) | TSQR compresses row blocks into small triangular factors and reduces these factors hierarchically. | A tree of QR summaries is established. Selectively rebuilding dirty branches is a natural maintenance strategy, so novelty cannot rest on the tree itself. |
| Maalouf, Jubran, Feldman, *Fast and Accurate Least-Mean-Squares Solvers*, NeurIPS 2019; [proceedings paper](https://papers.neurips.cc/paper/9040-fast-and-accurate-least-mean-squares-solvers.pdf) | Exact weighted data reduction preserves matrix/least-squares information; the work discusses streaming and distributed extensions. | Exact compression and mergeability are not new. A weighted subset and an orthogonal synthetic-row representation have different implementation and conditioning properties, which must be compared rather than described as a novelty gap. |
| Koide, Oishi, Yokozuka, Banno, *Exact Point Cloud Downsampling for Fast and Accurate Global Trajectory Optimization*, IROS 2023; [author paper](https://staff.aist.go.jp/k.koide/assets/pdf/iros2023.pdf), §III-A, equations (3)–(7), Algorithm 3 | Selects weighted residuals preserving cost, gradient, and Gauss–Newton Hessian at the chosen evaluation point. The selected nonlinear residuals can then be reevaluated. The paper explicitly keeps initial correspondences fixed during optimization. | Its exactness guarantee concerns the quadratic model at the extraction pose, while FORM's existing fixed feature basis preserves those quantities at every pose for its supported residuals. This distinction predates the proposed selective-rematching extension in this repository. |
| Nasser, Jubran, Feldman, *Coresets for Kinematic Data: From Theorems to Real-Time Systems*, 2015 preprint, revised 2017; [paper and metadata](https://arxiv.org/abs/1511.09120), §4, Definition 4 | Streaming matrix summaries and Kabsch coresets retain the optimal rotation under rigid motions of paired point sets; the system reuses coresets over time. | Reusing exact registration summaries under motion is already present. Preserving an optimizer is distinct from preserving the full cost and both-pose Gauss–Newton system at every pose. Changing nearest-neighbor correspondences requires separate treatment. |
| Henzinger and Kale, *Fully-Dynamic Coresets*, 2020; [authors' preprint](https://arxiv.org/abs/2004.14891) | Gives a general method for maintaining approximate coresets under insertions and deletions. | Dynamic summaries themselves are established. The paper does not directly provide FORM's exact correspondence stream or pose-independent square-root factors; do not mistake this scope difference for proof that no registration-specific dynamic method exists. |
| Holz and Behnke, *Sancta Simplicitas – On the efficiency and achievable results of SLAM using ICP-Based Incremental Registration*, ICRA 2010; [author paper](https://www.ais.uni-bonn.de/~holz/papers/holz_2010_icra.pdf), §§I–II | Incrementally registers scans against an accumulated map and maintains the map. | “Incremental ICP” is ambiguous: incremental mapping or pose updates do not, by themselves, mean certified correspondence reuse or partial QR maintenance. |

An additional important lead is Greenspan and Godin, *A Nearest Neighbor Method
for Efficient ICP*, 3DIM 2001, DOI 10.1109/IM.2001.924426. Its original full text
was not retrieved in this search. Greenspan and Yurick's own
[2003 paper](https://www.cvl.iis.u-tokyo.ac.jp/~oishi/Papers/Alignment/Greenspan_ApproximateKDTree_3DIM03.pdf)
describes that earlier work as preprocessing spherical neighborhoods and tracking
them across iterations. Obtain the original before making any “first certified
ICP reuse” claim; this note makes no such claim. The 2003 work itself studies
approximate k-d search, so it is not an exact-equivalence baseline without changing
its search behavior.

## The actual FORM contract

The following observations come from the baseline source, particularly
`form/form.cpp`, `form/mapping/map.tpp`, `form/optimization/matcher.hpp`, and
`docs/optimization-acceleration-design.md`:

- `register_scan` builds a world-map snapshot **once before the rematch loop**.
  Targets do not move inside that epoch. The current query pose changes; the map
  must be reset and certificates invalidated for the next snapshot. Generic
  multi-scan target motion is an extension, not the initial problem.
- The reference matcher visits the query voxel and 26 neighbors in a specified
  order. It compares squared **four-component** point distances, and replaces the
  winner on strict `<`. Equal distances therefore depend on encounter order.
  Voxel keys use xyz coordinates; the certificate's metric must still account for
  the fourth component. Substituting xyz distance needs a separately established
  invariant that the fourth-component differences vanish.
- A match becomes a factor row only when its squared distance is strictly less
  than `max_dist_matching²`. A stable nearest ID does not imply stable acceptance.
  Rejected matches are also useful for final map bookkeeping, so preserving only
  accepted constraints is weaker than preserving the full matching output.
- In `register_scan`, voxel width equals the acceptance distance. This helps
  reason about accepted neighbors, but it does not turn all returned/rejected
  match IDs into unrestricted global nearest-neighbor results.
- Matching currently clears and reconstructs correspondence arrays. Immutable
  summaries belong to a specific correspondence snapshot, with revisions and
  point-object identity participating in validity. A partial update must publish a
  new summary without changing the factors that retain an older snapshot.
- Point and plane summaries already use FP64 QR with difference features. They
  preserve `rᵀr`, `Jᵀr`, and `JᵀJ` for all poses at fixed correspondences, with
  both poses differentiated in the original residual convention. The proposal
  must preserve that existing result, not claim to invent it.

## What a certificate would prove

This section is our derivation of a conservative starting certificate, not a claim
of a new theorem. Let `C` be the exact ordered candidate set at an anchor query
`q0`. Its winner is `a`; let `d1=||q0-a||` and let `d2` be the minimum distance
to any other candidate, in the matcher's four-dimensional Euclidean metric.
Let `δ >= ||q-q0||`. If the candidate set and tie order have remained unchanged,

```text
distance(q,a) <= d1 + δ
distance(q,b) >= d2 - δ     for every b != a
d1 + δ < d2 - δ            implies a remains the unique winner.
```

Recomputing the winner distance can tighten the first bound. This is a standard
triangle-inequality assignment certificate of the type reviewed above. A cached
second-nearest **ID** alone is not sufficient: its new distance is not a lower
bound on every other candidate. Store an anchored bound on all competitors.
Squared distances cannot be substituted directly into the additive bound.

For a fixed snapshot, requiring the same query voxel preserves candidate set and
encounter order. Falling back whenever the query voxel changes is conservative
and simple. More elaborate proofs can cover entering/leaving voxels, but must
bound every newly eligible competitor and preserve membership of the cached
winner. Stable-ID tie handling must agree with the reference order, or ties
must fall back. Near-equality comparisons require conservative error bounds and
fallback; an arbitrary epsilon is not itself a floating-point proof.

Acceptance can be recomputed exactly for a certified winner using the existing
strict squared-distance test. Alternatively, a strict upper bound below the
radius certifies acceptance and a lower bound at or above it certifies rejection,
subject to a documented floating-point treatment. Even a certified unchanged
rejection may require refreshed distance data for final bookkeeping. Empty
candidate sets, a single candidate, duplicates, and non-finite inputs require
explicit policy rather than applying a two-neighbor formula blindly.

The proof anchor is the last query for which the competitor bound was valid.
Repeatedly comparing only against the previous pose while retaining an older
bound is unsound unless bounds are updated cumulatively. Snapshot identity,
feature identity, and threshold changes also belong in cache validity.

## Partial summary maintenance and the narrow hypothesis

Partition each scan-pair feature matrix `F` into stable row blocks. A leaf stores
the QR root for one block. An internal node stores the QR root of its stacked
child roots. In exact arithmetic:

```text
U_leafᵀ U_leaf = F_leafᵀ F_leaf
U_parent = qr_root(stack(U_child_1, ..., U_child_k))
U_parentᵀ U_parent = sum(U_childᵀ U_child)
```

Changing a correspondence, acceptance state, or owning scan pair dirties the
appropriate leaves; rebuild those leaves and their ancestor union. Unchanged
matches with unchanged local feature rows can retain their roots. Changing
queries' poses alone does not dirty the fixed feature rows. Moving a row to a
different scan pair must update both old and new pairs. The identities above
follow from ordinary QR/TSQR composition.

This is a plausible way to avoid subtracting normal-equation contributions.
It does not prove superior speed or numerical behavior to classical QR
updates. Repeated root downdates may be attractive, but rank loss and accumulated
roundoff require careful treatment; a full or affected-leaf rebuild is an
important comparator and fallback. Root equality is not the validation target:
QR sign choices and rank deficiency can produce different valid roots.

The strongest currently supportable **research hypothesis** is:

> On FORM's fixed target-map epochs, conservative certificates can preserve the
> reference correspondence and acceptance stream, while selective rebuilding of
> affected square-root feature summaries preserves the supported nonlinear cost
> and both-pose Gauss–Newton system at every evaluation pose, and the combined
> method can reduce end-to-end smoothing time after including certificate and
> maintenance overhead.

A contribution would need to establish this complete contract, give a stable
maintenance method suited to the 7/13-column features, and demonstrate a useful
region of operating conditions. The generic algorithmic ingredients are old.
The source search does not establish priority for their exact composition, and
the scope may ultimately be a useful engineering result rather than a new
algorithm. Publication suitability cannot be inferred from this note.

## Comparisons and falsification experiments

These are proposed experiments only; none was run for this note.

1. **Matcher oracle at every rematch.** Run the unchanged full matcher on the
   identical snapshot and query pose. Compare winner IDs, owning scan, strict
   gate outcome, final distances, and materialized matches for accepted and
   rejected queries. A single certified false reuse falsifies equivalence.
   Include ties, duplicate points, one/zero candidates, threshold equality and
   its adjacent representable values, positive/negative voxel boundaries, fourth
   coordinate differences, cumulative small motions, and snapshot replacement.
2. **Separate solver feedback from matcher correctness.** First replay identical
   externally recorded poses so floating-point solver divergence cannot mask
   a matching bug. Then run the full closed loop and report pose, iteration,
   termination, correspondence, and trajectory differences. Algebraic factor
   equivalence does not promise bitwise-identical trajectories.
3. **Partial-root oracle.** Feed the same changed correspondence stream into
   full FP64 QR and partial maintenance. At multiple independent pose pairs,
   compare nonlinear cost and the entire augmented Hessian, not only a local
   six-dimensional block or `UᵀU`. Include near-zero residuals at large coordinate
   scale, rank-deficient geometry, repeated rows, complete deletion of a pair,
   acceptance toggles, pair transfers, and long update sequences. Reuse the
   existing design's tolerances as a starting acceptance contract; investigate
   rather than hide any required relaxation.
4. **Factorial ablation.** Compare full rematch/full QR, certified rematch/full
   QR, full rematch/partial QR, and certified rematch/partial QR. Also include
   ordinary persistent buffers and cached search without certificates. Otherwise
   allocator or residency gains can be misattributed to the hypothesis.
5. **Strong established baselines.** Compare an exact cached k-d search inspired
   by Nüchter and a Hamerly-style bound-only scheme. A global k-d search is not
   automatically equivalent to the voxel matcher: either reproduce the candidate
   restriction/tie contract or report it separately as a changed matcher.
   Compare dirty-leaf QR with full TSQR and carefully implemented QR row updates.
   A normal-equation delta baseline is useful for speed/conditioning tradeoffs,
   not as the sole numerical reference.
6. **Registration coreset comparison.** Under compatible residuals and a common
   correspondence stream, compare Koide-style extraction at one pose with
   all-pose feature compression over increasing pose displacement. Report its
   intended local-model guarantee fairly; a mismatch far from extraction does
   not invalidate that paper. Do not compare different GICP/point-plane costs
   as though only compression changed.
7. **End-to-end accounting.** Include initial runner-up acquisition, certificates,
   fallback search, grouping, dirty-leaf rebuilding, ancestor QR, data transfer,
   synchronization, final materialization, optimization, and memory. Report
   wall-time distributions and peak memory for identical sequences/settings;
   do not infer overall speed from skipped distance evaluations alone.
8. **Locate the failure regime.** Sweep query displacement, runner-up margin,
   voxel crossing rate, correspondence churn, dirty-block fraction, leaf size,
   pair sizes, feature density, and window size. Use low-churn and deliberately
   high-churn scenes. If dirty rows spread across almost every leaf, partial
   QR can cost as much as full QR. If runner-up acquisition plus certification
   costs more than saved searches, selective rematching is not beneficial.
9. **Test the scope of the contribution.** If the combined method does not beat
   straightforward cached exact search plus whole-pair summary reuse, or gains
   vanish after honest overhead accounting, the proposed additional machinery
   is not supported. Report that outcome. Before making a novelty claim, complete
   the Greenspan/Godin primary-paper review and forward-citation searches for
   exact ICP correspondence tracking and dynamic registration summaries.

No accuracy improvement is implied: the target is the same estimator with less
work. Any observed accuracy change must be explained by numerical differences,
changed matching semantics, or a deliberately changed algorithm.
