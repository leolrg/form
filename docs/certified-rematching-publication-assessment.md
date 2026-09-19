# Publication assessment of certified rematching

Assessment dated 2026-09-19, against the current method, experiment ledger,
prior-art report, and existing acceleration design. This is a bounded assessment
of possible contribution, not a completed novelty review or a prediction of
acceptance at ACC or another venue. No new experiments or literature search were
performed for this note. Full-sequence/scaled rematching results are pending;
compact QR task scheduling is still a prototype.

## Present judgment

There is a defensible **candidate incremental systems contribution** in making
selective rematching preserve FORM's actual computed matching contract, then
maintaining its nonlinear square-root factors under correspondence changes.
Its value would come from the precise composition, implementation, and measured
operating region. The individual ideas, and even the broad combination of
assignment certificates with selective statistics updates, are established.

As one component of a whole FORM acceleration paper, the search extension can
be useful even if its incremental end-to-end gain remains modest. Current
250-scan screening suggests roughly 5% total savings for selected search-only
configurations and roughly 3% for selected combined configurations, with
substantial variation across earlier trials. These are screening observations,
not settled paper results. The existing evidence does **not** establish an
additional performance benefit from dirty-tree/queue integration over the best
search-only pipeline. The paper should not make that integration a necessary
headline unless broader measurements change this conclusion.

A method can be correct and technically careful yet remain an engineering
specialization of prior ideas. Conversely, a small additional speedup does not
invalidate the broader acceleration work. The relevant question is what each
component contributes beyond the strongest earlier component and comparator.

## What predates this research branch

The baseline for this branch is `1255543`. Repository chronology is separate
from publication chronology: a capability already in that baseline might still
belong to an unpublished whole-system paper, but it cannot be presented as a
new result of the selective-rematching extension.

| Capability | Attribution in the whole-paper story |
|---|---|
| Pose-independent seven-column point and thirteen-column plane feature rows | Existing FORM acceleration system; not introduced by this branch. |
| Preservation of nonlinear cost, gradient, and both-pose Gauss–Newton matrix at every pose for supported fixed correspondences | Existing feature representation and factor evaluation. The new branch preserves this property through updates; it does not derive the original representation. |
| Difference features and FP64 Householder QR instead of subtractive Gram evaluation | Existing numerical design. |
| CPU summaries, immutable factor snapshots, batched CUDA full QR, hierarchical root reduction, compact root transfer | Existing implementation and lifetime contract. A generic QR reduction tree is also classical prior art. |
| Broader acceleration/scaling results in `optimization-acceleration-results.md` | Evidence for the existing acceleration system, not evidence for certified rematching or partial QR. Do not combine different checkpoints' gains multiplicatively. |
| Anchored certificates for the original GPU matcher's ordered candidate domain and computed distances | Branch-specific extension, based on established bound/cached-search principles. |
| Stable correspondence slots, dirty leaf/ancestor maintenance, incoming queues | Branch-specific implementation extending existing roots to changing correspondences; not a new general incremental-QR principle. |
| Same-input matching and full-root audits, variant ablations, failure-region measurements | Branch-specific validation and experimental contribution. |

See [existing design](optimization-acceleration-design.md),
[method](certified-rematching-method.md), and
[experiment ledger](certified-rematching-experiments.md) for the contracts and
checkpoint-specific evidence.

## Why the prior-art distinction must stay narrow

[Hamerly 2010](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf) already combines
assignment bounds with count/sum updates only when assignments change. Thus
“certify stable correspondences and update sufficient statistics selectively”
cannot be the novelty claim. Its retained statistics and problem differ from
FORM, but changing the application does not by itself establish a new principle.

[Cached ICP search](https://robotik.informatik.uni-wuerzburg.de/telematics/download/3dim2007.pdf),
[STCNN neighborhood reuse](https://www.yjschmidt.de/pdf/jschmidt_vmv_2003.pdf), and
[previous-neighbor Delaunay walks](https://avida.cs.wright.edu/publications/pdf/J31.pdf)
already exploit correspondence history with exact search/fallback. The branch's
specific distinction is preserving the reference implementation's restricted
voxel domain, encounter-order ties, four-component metric, strict gate, and
fresh returned distances, with a conservative binary64 enclosure. This is a
stronger implementation contract than merely citing a real-arithmetic gap test.
It is not evidence that earlier exact-search methods are incorrect, nor proof
that this is the first floating-point-safe correspondence certificate.

[TSQR](https://arxiv.org/abs/0806.2159), classical QR updating, and
[iSAM](https://www.cs.cmu.edu/~kaess/pub/Kaess08tro.pdf) establish hierarchical and
incremental square-root maintenance. FORM's retained object is different from
iSAM's linearized global system: it is a local feature root that can evaluate
the supported nonlinear factor at subsequent arbitrary poses. That useful
distinction comes principally from the existing FORM representation; selecting
dirty paths extends its lifetime across changed correspondence sets.

[Koide et al. 2023](https://staff.aist.go.jp/k.koide/assets/pdf/iros2023.pdf)
preserve a quadratic model at an extraction pose. FORM's all-pose property for
its supported factors is a meaningful comparison axis, but it predates this
branch and must be stated with compatible residual/noise assumptions. Existing
registration coresets and compact LiDAR statistics further preclude a broad
“first compact nonlinear registration summary” claim. Full discussion and
remaining literature limits, including unretrieved Greenspan–Godin 2001, are in
the [prior-art report](certified-rematching-prior-art.md).

## A useful claim would compose three different contracts

1. **Search:** for the same immutable target snapshot and transformed query
   inputs, conservative reuse returns the reference kernel's winner and computed
   distance. A successful certificate need not imply the acceptance gate remains
   unchanged. Domain crossings, ambiguous bounds, and invalid state fall back.
   The enclosure assumes the documented binary64 arithmetic, gradual underflow,
   finite inputs, and compilation semantics; this is not an arbitrary-platform
   floating-point guarantee.
2. **Factors:** inserting, deleting, or changing accepted fixed-feature rows
   preserves each group's row multiset. Dirty-root rebuilding restores
   `U^T U = F^T F` in real arithmetic. The pre-existing feature representation
   then supplies all-pose cost and both-pose Gauss–Newton equivalence for supported
   isotropic point/plane factors. Pose-dependent robust weights and general
   anisotropic models require another derivation.
3. **Execution:** queue order and QR reduction order can change floating-point
   roots and subsequent solver poses. Same-input search exactness does not imply
   bitwise equality of independently executed closed-loop trajectories. Published
   summaries must retain snapshot ownership; reset, failure, and mode changes
   must invalidate stale state.

An explicit composition of these contracts is a useful result even though its
ingredients are standard. It should be presented as a proved specialization
with empirical numerical validation, not a universal numerical-equivalence
theorem. The current audits provide strong evidence on tested inputs, including
45,285,082 same-input search checks per reported 250-scan audit and 66,386
full-root checks in relevant summary audits. Audit mode supplies reference
results downstream, so clean runs using actual cached feedback remain necessary.
Neither sampled-pose tests nor small normalized Gram errors alone prove
all-pose floating-point relative error bounds.

## Conditional positive criteria

The branch would strengthen the whole paper if the following hold:

- Full-sequence and scaled-workload comparisons reproduce a useful gain over
  the best accelerated baseline with the same settings. Report distribution
  and repeat variability, initialization, retained memory, transfers, and all
  maintenance overhead; separate optimization latency from whole-scan latency.
- Four-way search/summary ablations identify what helps. To claim an incremental
  summary performance contribution, tree or queue maintenance should add useful
  benefit over the best search-only configuration in an identified workload
  region, rather than merely beating a weaker disabled baseline.
- The paper explains when existing algebra plus conventional maintenance is
  useful: candidate density, query motion, certificate margin, rematch count,
  target-group size, row churn, dirty-leaf fraction, and retained tree extent.
  A reproducible crossover is more informative than a selected best prefix.
- Compatible strong alternatives receive a fair comparison. At minimum separate
  certificate arithmetic/scheduling from warm exact search, and dirty trees from
  full QR, whole-pair reuse, and simpler incremental maintenance. A global 3D
  k-d or Delaunay search is not silently equivalent to FORM's restricted 4D
  matcher; disclose incompatible semantics rather than manufacture a baseline.
- Numerical and lifecycle tests continue to cover rounded ties, extreme search
  scales, acceptance changes, rank loss, migrations, empty groups, and snapshot
  replacement. Validate actual queued/cached solver feedback, not audit output
  alone. Compact task scheduling must satisfy the same checks before being
  included in conclusions.

No single percentage is a publication threshold. A modest gain can be a useful
secondary contribution when repeatable and inexpensive; a large kernel gain
without end-to-end value is not enough to support an acceleration claim.

## Conditional negative criteria and fallback framing

- If full/scaled results do not distinguish the small gains from variation,
  report feasibility and correctness, not established end-to-end acceleration.
- If search-only remains faster than combined, keep certified search as an
  optional component and present partial QR as a bounded negative result or
  omit it from the main performance claim. Search savings cannot be credited to
  the dirty tree simply because both are enabled.
- If simpler warm search or whole-pair reuse obtains the same benefit with less
  state, the extra machinery lacks a demonstrated practical advantage. Its
  correctness contract can still be useful, but should not carry a new general
  algorithm claim.
- If benefits require unavailable future ordering, they are not deployable.
  The [layout model](stability-order-model.md) found 32.12% fewer total QR tasks
  for a future-count heuristic, versus only 0.36% for two-transition persistence
  and 1.25% for dirty-only partition, before rearrangement overhead. These counts
  neither predict time nor support a causal-layout contribution on this trace.
- A result confined to A100 and one prefix should be described that way.
  Full-scene, cross-scene, memory, and hardware scope cannot be inferred from
  sanitizer passes or a high certificate hit rate.

The broader paper can retain the existing all-pose compression and measured
acceleration as its main technical story, while explaining which additional
work reuse succeeds or fails. It should not inflate the new branch to repair
an uncertain novelty argument for the whole paper.

## Suggested claim language

Supportable as a description of the method, with the stated assumptions:

> We specialize anchored correspondence bounds to the ordered voxel matcher's
> computed-distance semantics and maintain FORM's existing pose-independent
> square-root feature factors through selective leaf and ancestor rebuilding.
> The resulting implementation preserves reference matches on identical inputs
> and preserves the supported nonlinear factor in real arithmetic. We evaluate
> numerical agreement and the full cost of maintaining this state.

Add an acceleration sentence only after final measurements, naming the workload,
baseline, hardware, component, and observed uncertainty. For the whole paper,
state the pre-existing all-pose representation as a separate contribution if
publication history permits, rather than attributing it to rematching.

Claims to avoid include “first exact ICP reuse,” “novel triangle-inequality
certificate,” “first certified assignments plus delta statistics,” “new TSQR,”
“work proportional only to changed matches,” “bitwise-identical estimator,”
“all-pose floating-point exactness,” and “partial QR is faster” without its
separate ablation. No literature search so far establishes priority for the
complete composition, and no assessment here establishes venue suitability.
