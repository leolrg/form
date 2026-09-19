# Certified selective rematching: research results

Status: full-stairs and denser-feature validation are complete; larger-window
and additional-scene comparisons are in progress. Research branch `codex/certified-rematching`, frozen
V8 implementation `1fe130f`. The established acceleration branch remains at
`1255543` and has not been modified.

## Result so far

Selective correspondence search provides a modest measured improvement on full
stairs. Partial QR summaries preserve the estimator but have **not improved on
search-only**. These are incremental comparisons against the already accelerated
FORM pipeline, not speedups against original FORM.

Two sequential clean repetitions, with reversed mode order, process all
**1,190 stairs scans**:

| Method | All 1,190 scans: mean ms/scan | After 20-scan warmup: mean ms/scan | Matching after warmup: ms/scan |
| --- | ---: | ---: | ---: |
| Existing accelerated pipeline |43.906|42.991|12.473|
| Certified search, fresh QR |42.298|41.394|11.037|
| Full search, partial compact QR |43.686|42.749|12.415|
| Certified search + partial compact QR |43.316|42.429|11.446|

Search-only reduces all-scan mean latency by **3.66%** (1.038x throughput),
and the matching-stage mean after warmup by **11.51%**. Its two post-warmup
means are 40.134/42.654 ms against 42.177/43.804 ms for the paired controls.
The combined method is 2.50% slower than search-only after warmup. Repeats show
substantial platform variability, so this is a modest measured gain rather than
a large or universal speedup. Timing covers estimator processing, excluding
replay input loading and output serialization. All-scan means include startup.

Artifacts: `benchmarks/results/certified-rematching/clean-v8-full/`.

## Matched CPU/GPU scaling

First 250 stairs scans, two reversed-order repetitions, means after 20-scan
warmup. Every backend uses the same configuration and input prefix.

| Configuration | Original CPU | Improved CPU | Existing GPU | GPU search-only | GPU partial QR only | GPU combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current |126.66|77.77|26.90|—|—|—|
| Denser features |337.65|228.02|48.72|45.88|50.23|47.01|

All entries are ms/scan. Current CPU controls use frozen V7 with research
options off; the V8 current-prefix variant screen is reported separately in the
ledger because it has substantial repeat variability. Do not form speedups
between candidates and controls from different experiments.

The denser setting requests 6 points/100 planes, feature spacing 2, and recent 10.
It realizes 493,474 average correspondences versus 168,122 at the current setting
(**2.94x**), with 14.26 versus 14.13 average poses. Search-only reduces dense GPU
latency by 5.83%, improving in both repeats. Partial summaries again lose to
search-only. CPU and GPU runs retain identical workload/iteration counts; maximum
translation difference is 8.89e-12 m.

## Correctness and actual saved work

The independent full-stairs audit performs 308,516,696 original-kernel search
comparisons with **zero mismatches**, and 925,510 fresh-QR summary comparisons
with maximum normalized feature-Gram error **2.83e-15**. The audit preserves
reference feedback; separate clean runs exercise cached roots in the optimizer.
All clean full-stairs runs retain identical feature, correspondence, pose,
rematching and LM-iteration counts. Maximum translation difference is 6.90e-13 m.

Subsequent searches retain the same nearest index 96.52% of the time. The
conservative certificate skips 84.74% of subsequent queries. Yet 55.36% of active
64-row summary leaves are dirty: correspondence changes spread through many
blocks. This limits how much QR work can be avoided before maintenance costs.
The compact task plan refreshes on 6,552 of 32,896 calls, uploading 14.7 MB total.

The tests cover ties and strict acceptance gates, voxel transitions, arbitrary
explicit voxel ranges, padding coordinates, target replacement, failures and
cache invalidation, row migration/rejection/reinsertion, rank deficiency,
producer streams, shrinking extents and immutable root snapshots. The build
passes 138 C++ tests; focused memcheck/initcheck/racecheck runs have zero findings.
Matcher initcheck disables API copy checks for pre-existing struct tail padding;
device initialization checks remain enabled.

## Scope of the possible paper contribution

This branch specializes established assignment certificates and QR maintenance
to FORM's actual computed matching contract and nonlinear summary factors.
Triangle inequalities, cached ICP search, selective sufficient-statistic updates,
and QR trees are established prior art. The 7/13-column nonlinear factor
representation also predates this branch within our acceleration system.

The defensible claim is the complete contract-preserving integration, correctness
argument and measured operating region. Search-only could be a supporting
component of a whole-system acceleration paper. Current results do not justify
presenting partial QR updates as an additional performance contribution, or
claiming a new general mathematical compression principle. Conference acceptance
cannot be established by this experiment.

See [publication assessment](certified-rematching-publication-assessment.md),
[prior-art comparison](certified-rematching-prior-art.md),
[method](certified-rematching-method.md), and
[complete experiment ledger](certified-rematching-experiments.md).
