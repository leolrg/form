# Certified selective rematching: research results

Status: completed on the research branch `codex/certified-rematching`, using
frozen V8 implementation `1fe130f`. The established acceleration branch remains
at `1255543` and has not been modified. Hardware: A100 80 GB PCIe, 32-thread
EPYC VM; Release/SM80, FP64, CUDA FMA contraction disabled. These results do
not establish performance on other GPUs.

## Conclusion

Selective correspondence search provides a modest measured improvement on full
stairs. Partial QR summaries preserve the estimator but have **not improved on
search-only in the stairs workloads**. The practical recommendation is to retain certified search as an opt-in
optimization for measured favorable workloads, and keep partial QR as an
experimental ablation. The coupled trick does not supply a convincing new
performance contribution for the paper on this evidence.

These are incremental comparisons against the already accelerated
FORM pipeline, not speedups against original FORM. The initially promising quad_hard result did not reproduce consistently
in a confirmation set.

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
| Larger window |203.63|119.72|51.09|49.84|50.59|49.87|

All entries are ms/scan. Current CPU controls use frozen V7 with research
options off; the V8 current-prefix variant screen is reported separately in the
ledger because it has substantial repeat variability. Do not form speedups
between candidates and controls from different experiments.

The denser setting requests 6 points/100 planes, feature spacing 2, and recent 10.
Its all-scan workload averages are 493,474 retained-graph correspondences versus 168,122 at the current setting
(**2.94x**), with 14.26 versus 14.13 average poses. These correspondence
counts are measured across retained graph factors before marginalization, not
the number of new nearest-neighbor queries. Search-only reduces dense GPU
latency by 5.83%, improving in both repeats. Partial summaries again lose to
search-only. CPU and GPU runs retain identical workload/iteration counts; maximum
translation difference is 8.89e-12 m.

The larger-window setting requests recent40; its all-scan workload averages
are 38.56 poses
(**2.73x** current) and 520,164 retained-graph correspondences (**3.09x**).
Search-only reduces GPU latency by **2.45%**, improving in both repeats.
Combined partial summaries differ from search-only by only approximately 0.0255 ms/scan and
provide no meaningful extra gain. All CPU/GPU workload and iteration counts
agree; maximum translation difference is 2.62e-12 m.

## Cross-scene checks

Both additional tests use the first **600 scans**, not the complete sequences.
Means below exclude the first 20 scans. Quad_hard has four repetitions across
the original and confirmation sets; maths_hard has two.

| Scene | Existing GPU | Search-only | Partial QR only | Combined |
| --- | ---: | ---: | ---: | ---: |
| Quad_hard, four repeats |52.984|52.944|52.428|52.508|
| Maths_hard, two repeats |53.632|54.332|53.809|53.335|

All entries are ms/scan. Quad_hard's first set suggested a 2.72% combined gain,
but the confirmation set made combined 0.97% slower than off. Across all four
repeats, search is flat and the roughly 1% partial/combined differences are
small relative to the between-set variation. Maths_hard likewise provides no
reliable acceleration: search is 1.31% slower in the mean and combined only
0.55% faster. These outcomes limit the method's operating region.

All workload and iteration counts agree. Maximum translation differences are
7.91e-12 m on quad_hard and 1.66e-12 m on maths_hard. No favorable subset has
been substituted for the complete measured segments.

## Correctness and reuse opportunity

Across the full-stairs audit and the two 600-scan audits, the implementation
passes **609,102,983 original-kernel search comparisons with zero mismatches**,
and **1,794,838 fresh-QR summary comparisons** with maximum normalized
feature-Gram error **2.99971e-15**. The audits supply reference results downstream;
separate clean runs exercise the cached roots in the optimizer.

| Audited input | Queries checked | Summaries checked | Subsequent queries certified | Dirty leaves / active leaves |
| --- | ---: | ---: | ---: | ---: |
| Stairs, all 1,190 scans |308,516,696|925,510|84.74%|55.36%|
| Quad_hard, first 600 scans |154,953,787|435,762|78.70%|67.13%|
| Maths_hard, first 600 scans |145,632,500|433,566|76.74%|65.99%|

Certified queries can skip candidate search in non-audit mode; audit still runs
the independent full oracle for every query. All clean comparisons retain
identical feature, correspondence, pose, rematching and LM-iteration counts.
Maximum translation differences remain below 1e-11 m across the measured CPU/GPU
controls and research variants.

On full stairs, subsequent searches retain the same nearest index 96.52% of the
time, yet 55.36% of active 64-row leaves are dirty: a few changes spread through
many blocks. The harder scenes have more dirty leaves. The full-stairs compact
plan refreshes on 6,552 of 32,896 calls, uploading 14.7 MB total.

The tests cover ties and strict acceptance gates, voxel transitions, arbitrary
explicit voxel ranges, padding coordinates, target replacement, failures and
cache invalidation, row migration/rejection/reinsertion, rank deficiency,
producer streams, shrinking extents and immutable root snapshots. The build
passes 138 C++ tests; focused memcheck/initcheck/racecheck runs have zero findings.
Matcher initcheck disables API copy checks for pre-existing struct tail padding;
device initialization checks remain enabled.

## Final trace: where the savings go

Separate full-stairs Nsight traces cover all 1,190 scans, with the table below
aggregating scans 20–1189. All CUDA events during those 1170 scans are attributed
to their issuing ranges; no activity crosses its scan boundary. These are GPU
kernel times, not clean end-to-end times or additive CPU/API durations.

| GPU work | Existing search + fresh QR | Certified search + fresh QR | Certified search + partial QR |
| --- | ---: | ---: | ---: |
| Nearest-neighbor search including certificate |3.329|1.841|1.843|
| QR kernels, including tree extent/root maintenance |1.850|1.859|1.930|
| Sort + pack, or queued stable-row maintenance + pack |0.427|0.427|0.360|
| Compression/group-ordering kernels, previous two rows |2.276|2.286|2.290|

Selective search reduces its targeted kernel time by **44.7%**, saving about
1.49 ms/scan. This explains the measured full-stairs matching-stage reduction,
but targets only part of the roughly 43 ms overall pipeline. The reference
nearest-neighbor and QR kernels together take 5.18 ms/scan.

The partial QR path spends 0.515 ms on leaves, 1.245 ms on ancestors, and 0.171 ms
on extent/root maintenance. Queuing, hole filling and leaf packing add 0.360 ms.
Avoiding the full radix sort offsets some maintenance, but the entire
compression/group-ordering kernel total is essentially unchanged from fresh QR.
Fewer dirty leaves are therefore insufficient to produce an extra kernel-time win.
Host calls and synchronization still exist; their durations overlap device time
and must not be added to this table.

Artifacts: `trace-v8-full/`, with frozen-binary/input provenance, reports, SQLite
exports and attributed JSON summaries.

## Explored variants and stopping decision

The investigation covered eight implementation checkpoints: strict anchored
certificates; fused/split search; distinct/occurrence runner-up bounds; squared
rounding-safe predicates; stable 64-row leaves; cached QR ancestors; bounded
32/128/256-warp launches; incoming-row queues; and compact cached task plans.
Defaults remain unchanged. The ledger retains unsuccessful results.

Offline models also examined block sizes/hole placement and three causal row
ordering policies. The causal policies reduce modeled QR task counts by only
0.36–1.25% before their added movement/sorting costs. A future-information
heuristic shows more headroom, but it is neither implementable online nor a
proved upper bound. The first-hole 64 model has only 235 empty allocated leaves
out of 696,541 visits on the 250-scan prefix, so trimming empty leaf tails is not
a substantial opportunity there.

The measured partial-QR kernel budget is about 1.9 ms/scan, or 2.29 ms including
row ordering and packing. In our judgment, further small launch/metadata changes
are unlikely to address the observed ancestor and maintenance costs enough to
support a large new end-to-end gain; these kernel timings do not bound all
possible host or transfer savings. A different
causal clustering algorithm or different hardware could change the outcome;
we have not proved impossibility for every algorithm or GPU. The tested variants
and operating regions support stopping this coupled approach on the available
machine, rather than presenting its arithmetic savings as a speed contribution.

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
