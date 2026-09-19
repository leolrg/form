# Publication assessment of certified rematching

Assessment dated 2026-09-19, against the current method, experiment ledger,
prior-art report, and existing acceleration design. This is a bounded assessment
of possible contribution, not a completed novelty review or a prediction of
acceptance at ACC or another venue. No new experiments or literature search were
performed for this note. This update incorporates completed V8 full-stairs and
scaled-prefix comparisons, two additional 600-scan scene suites, the quad_hard
confirmation, all three scene audits, and the final full-stairs kernel traces.
The measurement campaign is complete.

## Present judgment

There is a defensible **candidate incremental systems contribution** in making
selective rematching preserve FORM's actual computed matching contract, then
maintaining its nonlinear square-root factors under correspondence changes.
Its value would come from the precise composition, implementation, and measured
operating region. The individual ideas, and even the broad combination of
assignment certificates with selective statistics updates, are established.

As one component of a whole FORM acceleration paper, the search extension can
be useful even if its incremental end-to-end gain remains modest. Completed V8
measurements now support a limited positive result for selective search: mean
total time after the 20-scan warmup falls by 3.71% on full stairs, 5.83% on the dense prefix, and 2.45%
on the larger-window prefix. These are reductions in the reported means, not
confidence bounds or cross-scene guarantees. The completed additional scenes
do not establish a consistent search speedup: search is approximately flat on
quad_hard across four repeats and its mean is higher on maths_hard. Its practical
value is modest and scene-dependent.

Partial QR does **not** add a demonstrated benefit beyond search-only in any
of these three comparisons. Combined is slower than search-only on full stairs
and dense features, and effectively tied on the larger window. This supports a
bounded negative performance finding for partial summaries in the tested
regimes, alongside a modest useful search extension. The paper should not make
dirty-tree/queue integration a performance headline on this evidence. An initial
quad_hard combined-mode win did not consistently replicate in its confirmation;
small cross-scene mean differences sit within substantial run variation. There
is no reliable demonstrated additional partial-summary speedup in this campaign.

A method can be correct and technically careful yet remain an engineering
specialization of prior ideas. Conversely, a small additional speedup does not
invalidate the broader acceleration work. The relevant question is what each
component contributes beyond the strongest earlier component and comparator.

## Completed V8 evidence and outstanding limits

All entries below are mean total milliseconds per scan after warmup. Full
stairs contains 1,190 scans; the density and window comparisons use 250-scan
stairs prefixes. `Off` is the existing accelerated GPU pipeline; `partial`
enables the selected partial-summary path and `combined` also enables selective
search. These incremental comparisons must not use the original CPU as their
denominator.

| Workload | Off | Search-only | Partial-only | Combined |
|---|---:|---:|---:|---:|
| Full stairs | 42.991 | 41.394 | 42.749 | 42.429 |
| Dense features, prefix | 48.721 | 45.881 | 50.230 | 47.010 |
| Larger window, prefix | 51.093 | 49.839 | 50.586 | 49.865 |
| quad_hard, 600 scans, four repeats | 52.984 | 52.944 | 52.428 | 52.508 |
| maths_hard, 600 scans, two repeats | 53.632 | 54.332 | 53.809 | 53.335 |

For full stairs, the means including all 1,190 scans are respectively 43.906,
42.298, 43.686, and 43.316 ms; the search-only advantage therefore also holds
when warmup is included. Against search-only after warmup, combined adds
1.035 ms/scan on full stairs and 1.129 ms on dense features. Its approximately 0.0255 ms
larger-window difference is too small to interpret as a meaningful advantage
for either mode from these means. Partial-only's small reductions in full and
window means are not evidence that it adds value once search is enabled.

The initial two-repeat quad_hard means were 53.623 / 53.907 / 52.724 / 52.166
ms in table order. Its separate two-repeat confirmation gave 52.345 / 51.982 /
52.131 / 52.851 ms. Combined therefore changed from a 1.457 ms advantage over
off to a 0.506 ms disadvantage; its ranking against search-only also reversed.
The four-repeat mean must not hide that failure to consistently replicate.
On maths_hard, combined's 0.297 ms advantage over off is small alongside the
campaign's run variation, while search-only is 0.700 ms slower in the mean.
Neither result establishes a robust additional partial-summary benefit. Workload
counts agree throughout the completed comparisons.

Workload counts agree in both scaled comparisons. Their matched CPU controls
are 228.020 ms improved CPU / 337.647 ms original reference for dense features,
and 119.721 / 203.630 ms for the larger window. These substantial baseline gaps
belong to the broader existing acceleration pipeline; they are not gains from
the new certificates or partial QR.

The full audit reports 308,516,696 same-input search checks with zero mismatches
and 925,510 full-QR checks with maximum normalized Gram error about `2.83e-15`.
The full clean runs using actual cached feedback differ in translation by at
most `6.9e-13 m` in the reported comparison. The latter complements audit mode,
which deliberately returns reference results downstream. Translation agreement
does not alone establish every trajectory metric or universal floating-point
equivalence.

| Audit scene | Checked queries | Checked roots | Maximum normalized Gram error | Certified subsequent queries | Dirty leaves |
|---|---:|---:|---:|---:|---:|
| stairs | 308,516,696 | 925,510 | about `2.83e-15` | 84.74% | 55.36% |
| quad_hard | 154,953,787 | 435,762 | `2.99971e-15` | 78.70% | 67.13% |
| maths_hard | 145,632,500 | 433,566 | `2.99165e-15` | 76.74% | 65.99% |

Across these audits, 609,102,983 queries have zero reported matching mismatches,
and 1,794,838 roots are checked. These are strong tested-input correctness
results. Certificate fractions describe subsequent searches, not all searches;
dirty-leaf fractions describe the audited leaf work, not the fraction of total
processing time. High certification rates on all three scenes alongside mixed
timing demonstrate why skipped searches alone are insufficient evidence of
end-to-end acceleration.

Separate full-stairs traces show nearest-neighbor kernels falling from 3.329 to
1.841 ms/scan with certified search. Fresh QR plus sorting and packing takes
2.286 ms in search-only mode; the corresponding partial-QR tree plus queued
row maintenance and packing takes 2.290 ms. Partial leaves cost 0.515 ms, but
ancestors cost 1.245 ms, before extent/root and row maintenance. The partial
path therefore does not reduce the complete compression kernel budget. These
are attributed GPU durations after warmup, not clean end-to-end timings; host
API durations overlap device time and cannot be added to them. The evidence
supports a limited systems contribution within the whole acceleration paper,
without establishing algorithmic priority or venue acceptance.

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
2. **Factors:** after insertions, deletions, or changes, each group retains the
   same current fixed-feature row multiset as full reconstruction. Dirty-root rebuilding restores
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
theorem. The full V8 audit and actual-feedback checks summarized above extend
the earlier prefix evidence substantially. Audit mode supplies reference results
downstream, so it must remain distinct from clean runs using actual cached
feedback.
Neither sampled-pose tests nor small normalized Gram errors alone prove
all-pose floating-point relative error bounds.

## Conditional positive criteria

The branch would strengthen the whole paper if the following hold:

- Full-sequence and scaled stairs comparisons show a modest search gain over
  the accelerated baseline in the reported means; additional scenes limit its
  breadth. Report the mixed outcome, distribution
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
  alone. Keep the validation tied to the exact compact-scheduling checkpoint
  used for each reported result.

No single percentage is a publication threshold. A modest gain can be a useful
secondary contribution when repeatable and inexpensive; a large kernel gain
without end-to-end value is not enough to support an acceleration claim.

## Conditional negative criteria and fallback framing

- If repetitions or additional scenes do not distinguish small gains from variation,
  report feasibility and correctness, not established end-to-end acceleration.
- Search-only outperforms combined on full stairs and dense features, with an
  effective tie on the larger window. The additional scenes do not establish a
  reliably replicated exception. Keep certified search as an optional component
  and present partial QR as a bounded negative result. Search savings cannot be
  credited to the dirty tree simply because both are enabled.
- If simpler warm search or whole-pair reuse obtains the same benefit with less
  state, the extra machinery lacks a demonstrated practical advantage. Its
  correctness contract can still be useful, but should not carry a new general
  algorithm claim.
- If benefits require unavailable future ordering, they are not deployable.
  The [layout model](stability-order-model.md) found 32.12% fewer total QR tasks
  for a future-count heuristic, versus only 0.36% for two-transition persistence
  and 1.25% for dirty-only partition, before rearrangement overhead. These counts
  neither predict time nor support a causal-layout contribution on this trace.
- Current evidence covers A100, full stairs, scaled stairs prefixes, and
  600-scan quad_hard/maths_hard prefixes. Full/scaled sequences for the other
  scenes, broader memory behavior, and other hardware remain outside the
  established scope. They cannot be inferred from sanitizer passes or a high
  certificate hit rate.

The broader paper can retain the existing all-pose compression and measured
acceleration as its main technical story, while explaining which additional
work reuse succeeds or fails. It should not inflate the new branch to repair
an uncertain novelty argument for the whole paper.

## Practical contribution recommendation

Present selective search as an optional, numerically specified optimization with
modest scene-dependent value. Present dynamic QR maintenance as a correct
extension whose measured maintenance costs prevent a reliable additional speedup
in this campaign. The negative result is informative: low correspondence churn,
high certificate yield, and fewer QR operations do not by themselves produce a
faster estimator.

The strongest branch-specific contribution remains the explicit composition of
computed-order matching certificates with lifetime-safe updates of the existing
nonlinear roots, supported by extensive same-input audits and honest ablations.
This can support the whole-system paper's technical argument and reproducibility.
It does not establish a new general assignment-bound or incremental-statistics
principle beyond Hamerly, cached ICP, and QR prior art. The all-pose feature
representation and much larger baseline acceleration remain separately
attributed to the system that predates this branch.

There is insufficient basis to assert that this extension alone supplies
missing ACC novelty, or guarantees suitability or acceptance at any venue.
Position it as a limited systems/methods component with a measured boundary of
usefulness, rather than making publication depend on an unreplicated partial-QR
gain or a broad novelty claim.

## Suggested claim language

Supportable as a description of the method, with the stated assumptions:

> We specialize anchored correspondence bounds to the ordered voxel matcher's
> computed-distance semantics and maintain FORM's existing pose-independent
> square-root feature factors through selective leaf and ancestor rebuilding.
> The resulting implementation preserves reference matches on identical inputs
> and preserves the supported nonlinear factor in real arithmetic. We evaluate
> numerical agreement and the full cost of maintaining this state.

The completed measurements support the bounded empirical statement:

> On A100 stairs evaluations, selective search reduces mean total processing
> time after a 20-scan warmup by 3.71% for the full sequence and by 2.45–5.83% for the tested larger
> workload prefixes relative to the existing accelerated GPU pipeline. Partial
> summaries do not provide additional benefit beyond search-only in these
> comparisons. On two additional scenes, search-only shows no consistent
> improvement, and an initial combined-mode advantage does not consistently
> replicate. Extensive same-input audits find no matching mismatches; partial
> summaries preserve numerical agreement but provide no reliable additional
> end-to-end speedup in this campaign.

Final wording should retain repeat variability and these additional scene results.
For the whole paper, state the pre-existing all-pose representation as a separate
contribution if publication history permits, rather than attributing it to
rematching.

Claims to avoid include “first exact ICP reuse,” “novel triangle-inequality
certificate,” “first certified assignments plus delta statistics,” “new TSQR,”
“work proportional only to changed matches,” “bitwise-identical estimator,”
“all-pose floating-point exactness,” and “partial QR is faster” without its
separate ablation. No literature search so far establishes priority for the
complete composition, and no assessment here establishes venue suitability.
