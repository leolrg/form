# Certified rematching experiment ledger

Research branch: `codex/certified-rematching`, isolated worktree
`/home/ubuntu/form-certified-rematching`. Established acceleration branch remains
unchanged at `1255543`. These are preliminary feasibility observations, not final
speed or novelty claims.

## V3 partial QR checkpoint

The stable 64-row slot implementation keeps unchanged rows in their target scan's
slots, fills first-available holes on migration/insertion, and rebuilds only dirty
leaf roots. Affected groups currently rebuild their root merges; this prototype
does **not** yet cache all internal ancestors. Unchanged groups keep their roots.
All of this is opt-in (`FORM_CUDA_SUMMARY_REUSE=blocks64`).

The default build passes 120 C++ tests (109 main, 2 parallel, 9 scalar); the main
suite also passes with both certified search and partial summaries enabled. Five
incremental tests pass CUDA memcheck, device initcheck and racecheck with zero
errors/hazards. Cases include multiple 256-thread scan chunks, migrations,
rejection, reinsertion, zero/rank-deficient matrices, producer stream ownership,
and comparison of nonlinear cost and the full both-pose augmented Hessian.
Independent read-only reviews found no remaining actionable logic defect.

Frozen V3 SHA256:
`bbe40d949ad75e97065968f2e0113dc76c00a9b61c55e50eae2e28fb7904c526`.
Diagnostic `audit-v3/` replays the first 250 stairs scans. In `audit64` mode,
incremental roots are compared to fresh full QR roots on the identical inputs,
then the full roots are supplied to the optimizer to preserve its original
feedback. Including warmup:

- 45,285,082 original-kernel search comparisons, zero mismatches.
- 66,386 scan-pair summary checks; maximum relative feature-Gram error
  `2.78067e-15` (threshold `1e-10`). Gram error is normalized by `1 + ||G||`.
- 478,796 dirty leaves out of 748,395 active leaves, 63.98%.

These establish correctness evidence and actual partial work, not speed. Clean
four-way off/search-only/summary-only/combined ablations follow separately.

### V3 clean ablation: initial maintenance costs exceed savings

Two reversed-order repetitions on the same current 250-scan prefix:

| Mode | Mean total ms/scan | Mean matching-stage ms/scan |
| --- | --- | --- |
| Both disabled | 27.284 | 8.673 |
| Certified search only | 27.530 | 8.805 |
| Partial leaf summaries only | 27.517 | 9.682 |
| Both enabled | 29.288 | 10.129 |

Artifacts: `clean-v3-current/`. This is another **negative performance result**;
the summary-only total is approximately flat, while its matching stage is slower.
Extraction and other stage variation must not be attributed to the summary method.

A separate partial-summary trace (`trace-v3/blocks.json`) measures these kernels
on scans 20–249:

| Kernel family | ms/scan |
| --- | --- |
| Original nearest search | 1.922 |
| Mark changed rows | 0.052 |
| Fill stable slots | 0.363 |
| Pack dirty leaves | 0.074 |
| Dirty leaf QR | 0.418 |
| Rebuild affected groups' merges | 1.057 |
| Retain group roots | 0.066 |

Thus the partial QR kernels total 1.540 ms, compared to 1.408 ms for the earlier
full-QR trace. Leaf savings alone are insufficient. Partial QR also incurs
2.619 ms of host CUDA API time (inclusive/overlapping device time), versus 1.977 ms
for full QR; slot maintenance adds 1.246 ms of API time under `match_pack`.
Both QR tile kernels use 80 registers without spills, so register occupancy alone
does not explain the difference. V3 includes global work-counter atomics,
per-call descriptor planning/uploads, and rebuilding all ancestor merges in an
affected group. The next prototype caches internal tree nodes and removes that
host planning/transfer path; it must be measured before any gain is claimed.

## Baseline and verification

The clean baseline passed all 108 C++ tests. Frozen executable SHA256:
`cb4fa9d5c9f66e049b7c9666c1be56b66f1cde1591527c8f151cd03d908b6ba2`.
Build uses the established Release configuration, CUDA architecture 80, FP64,
`--fmad=false`, and no native CPU ISA option. Hardware is the A100 80 GB PCIe /
32-vCPU EPYC VM from the existing diagnostic campaign.

The first same-cell certificate prototype passed 18 matcher tests and the
102-test main suite separately in Audit and Certified modes. Two independent
read-only reviews found no blocking correctness defect; numerical contract is
in `form/optimization/certified_rematching.md`. Memcheck on the five new reuse
tests reported zero errors. Default initcheck reported uninitialized tail padding
in downloaded `Result` structs, reproduced in an existing Disabled-mode test;
device initcheck with API-memory checking disabled reported zero errors. This is
not a claim that the default initcheck run is clean.

## V1 diagnostic: current workload, first 250 stairs scans

Artifacts: `benchmarks/results/certified-rematching/audit-v1/`.
Binary SHA256: `6c2d4370ae474babb7e0862228463ba72d14f6649a53606f4414b42862e7c4c7`.
Modes off/audit/certified were replayed sequentially with diagnostic capture and
profiling. These timings include host downloads and serialization and must not
be used as clean performance measurements. Timing/churn warmup excludes the first
20 scans, leaving 230.

This binary contained the binary correspondence capture but missed the subsequently
added statistics hook. Consequently V1 does **not** establish a measured certificate
yield or zero audit mismatches. The experiment runner now checks that both capture
outputs exist. A rebuilt diagnostic is required.

All three runs had identical per-scan feature, rematch, active-pose and final
correspondence counts. Maximum translation differences from off were
`9.33e-14 m` (audit) and `1.12e-13 m` (certified). These are closed-loop consistency
checks, not an independent per-query matching oracle.

Cross-process correspondence streams cannot be compared by query and map indices:
the existing normal-extraction loop appends features through a TBB concurrent
vector, so query ordering and subsequently packed map ordering vary between runs
(`form/feature/extraction.tpp`). For example, scan 1 already has differing IDs
and query-order distances, despite equal counts and essentially identical poses.
Correctness auditing must run the original search on the **same in-memory
snapshot and query poses**; a second full kernel in audit mode supplies that check.
Within a scan's rematching epoch, query and map indices are stable, so V1 remains
valid for churn and block-locality analysis.

### Correspondence and fixed-query-block work

The off capture contains 4,316,680 first-search queries and 37,722,905 subsequent
queries after warmup. Of subsequent queries, 2,123,481 (5.63%) change their accepted
summary row (target identity, target scan, or acceptance); changing distance alone
does not change a pose-independent summary row.

| Fixed query block size | Subsequent accepted rows in dirty blocks | Dirty surviving blocks / full QR 64-row tiles | First-search blocks / full QR tiles |
| --- | --- | --- | --- |
| 32 | 21.84% | 2.82× | 19.29× |
| 64 | 29.64% | 2.30× | 11.23× |
| 128 | 39.03% | 1.72× | 6.04× |
| 256 | 49.62% | 1.20× | 3.14× |

Each block is keyed by (target scan, query-index block). These figures expose
fragmentation: low row churn does not imply less factorization work. Larger blocks
also need multiple 64-row tiles if their actual occupancy exceeds 64, and the
table excludes merge work. This simple layout is therefore unattractive on this
workload. Next evaluate stable per-target-group row slots, keeping unchanged rows
in place and filling vacancies before allocating more blocks.

## V2 independent original-kernel audit

Artifacts: `benchmarks/results/certified-rematching/audit-v2/`.
Binary SHA256: `3d634056247440c9e6b2fa8513dfcc0c84ac7c33c8ffdc4dc15559c3806c4796`.
Audit executes the unchanged original nearest kernel on the same GPU snapshot
and transformed queries, compares index and exact computed distance, and returns
the original results. Disagreeing anchors are invalidated. Both the selective
decision and the new runner-up acquisition path are therefore checked against
the actual old kernel. Statistics explicitly count original-oracle searches.

The 250-scan replay checked **45,285,082 queries including warmup**, with zero
mismatches. On scans 20–249:

| Quantity | Count / fraction |
| --- | --- |
| Queries, including first searches | 42,039,585 |
| Queries after each snapshot's first search | 37,722,905 |
| Certified unchanged | 30,560,029 |
| Certified fraction, including first searches | 72.69% |
| Certified fraction, subsequent searches only | 81.01% |
| Cell-crossing fallbacks | 331,147 / 0.88% of subsequent queries |
| Gap-test fallbacks | 6,831,729 / 18.11% of subsequent queries |
| Consecutive full-search indices unchanged | 35,599,486 / 94.37% of subsequent queries |
| Original-oracle mismatches | 0 |

The 104-test main suite passes in both Audit and Certified modes. Additional tests
cover `nextafter` rounded ties, a runner-up whose anchor distance overflows then
becomes finite, and changing group assignments/exclusion during successful reuse.
Seven new tests pass memcheck and device initcheck. Audit initially copied an ABI
padding byte with aggregate assignment; memberwise stores corrected that new
device read. The pre-existing host-copy padding caveat above still applies.

This supports trying selective search, but does not measure speed. Cross-cell
certificates can recover at most a small fraction of this workload's remaining
searches; improving gap certificates, fallback search cost, or summary maintenance
is a higher-priority experiment. Clean, repeated measurements follow separately.

## V2 clean selective-search comparison and trace

Two sequential repetitions, reversed mode order on repetition 2, use the same
250-scan current workload and exclude scans 0–19. No analysis, build, sanitizers,
or other experiments overlapped clean replay. Artifacts:
`benchmarks/results/certified-rematching/clean-v2-current/`.

| Mode | Total run means (ms/scan) | Mean total | Mean matching stage |
| --- | --- | --- | --- |
| Frozen baseline `1255543` | 27.33 / 26.20 | 26.76 | 8.60 |
| Research binary, reuse disabled | 25.88 / 27.26 | 26.57 | 8.38 |
| Certified search, full summaries | 29.35 / 26.75 | 28.05 | 8.82 |
| Improved CPU, matched settings | 86.50 / 82.06 | 84.28 | 58.54 |

There is **no demonstrated end-to-end acceleration** from this initial search
prototype. It is 5.6% slower by the two-run mean, with appreciable repeat variation.
All modes/repeats preserve per-scan workload counts; maximum translation difference
from the disabled run is `1.02e-13 m` across the eight runs.

Separate Nsight traces (`trace-v2/`, also scans 20–249) show search kernels at
1.920 ms/scan disabled versus 1.668 ms certified: a 0.252 ms or 13.1% reduction.
QR kernel time remains 1.408 versus 1.418 ms. Transfers have identical byte counts
but different measured durations, so subtracting total device busy time would
misattribute transfer variability to the certificate. These traces explain why
the skipped-query count must not be presented as a speedup; they do not supersede
the clean timing result.

The next search variants will separate the cheap certificate into one thread per
query from warp-cooperative fallback search, and reduce runner-up acquisition
overhead while retaining a conservative lower bound. Partial summary maintenance
is an independent ablation, not assumed to rescue the search result.

## Stable per-target-scan slots: offline summary work model

`benchmarks/analyze_stable_blocks.py` replays the V1 off-mode correspondence
capture on the CPU. It changes no production code. Nine tests cover migrations,
rejections, unchanged slots, dirty-hole preference, first-hole allocation,
randomized ownership, retained tree height after deletion, and internal QR work
for leaves larger than 64 rows. The final reports are:

- `benchmarks/results/certified-rematching/audit-v1/stable-blocks-current-v2.json`
- `benchmarks/results/certified-rematching/audit-v1/stable-blocks-first64-current-v2.json`

Both exclude scans 0–19. They cover 460 initial feature-type searches and 4,026
subsequent searches across 230 scans. Initial searches contain 4,316,170 accepted
rows; subsequent searches contain 37,720,241 accepted rows. The latter have
2,123,481 changed summary rows. These accepted-row counts differ slightly from
the query counts above because some queries have no accepted correspondence.

Each target scan initially receives compact slots in query order. A query keeps
its slot while its target scan remains the same, even when its matched point
changes; that change dirties its leaf. Rejection or migration frees the old slot
and dirties its old leaf. Incoming rows fill available slots before allocating
more leaves. The `dirty` policy prefers holes in leaves already requiring an
update; the simpler `first` policy always fills the lowest available slot.
Unchanged leaves retain their previous square-root factors. Empty leaves retain
their indices, and cached tree ancestors are updated when any descendant changes.

The following fractions include initial construction and compare against fully
rebuilding the usual grouped 64-row QR on every search. Smaller is less modeled
work. Merge counts include internal merges needed to construct 128/256-row
virtual leaves from actual 64-row tiles, plus changed ancestors in the cached
tree. Plane trees use fanout four and point trees use fanout nine, matching their
13- and 7-column roots. Counts are not FLOP-weighted or measured runtimes.

| Slots per leaf | Hole policy | Raw rows refactorized / full rows | 64-row tile factorizations / full tiles | Merge input roots / full merge inputs |
| --- | --- | --- | --- | --- |
| 32 | Dirty first | 52.62% | 103.00% | 160.20% |
| 64 | Dirty first | 63.92% | 63.62% | 88.63% |
| 64 | First available | 64.54% | 64.23% | 89.02% |
| 128 | Dirty first | 75.16% | 74.79% | 102.29% |
| 256 | Dirty first | 84.88% | 84.36% | 87.87% |

**Prototype choice: 64-row leaves with first-available slots.** Dirty-first
allocation saves only another 0.61 percentage points of rows and tile work,
which does not justify a more complicated first prototype. Stable slots also
avoid the severe target-scan fragmentation of fixed query-index blocks. At
64 rows, allocated leaf count exceeds active leaf count by only about 0.034%
across this capture, although partially vacant active leaves still exist.

The plausible gain is narrower than the low correspondence churn initially
suggests: this policy still refactorizes about 65% of all rows, and an ideal
cached merge tree still processes about 89% of the baseline's merge inputs.
The model excludes slot maintenance, dirty detection, packing live rows from
holes, cache storage, allocation, synchronization, and kernel launches. A
prototype that rebuilds all affected groups' merge trees will do additional work
beyond the cached-ancestor estimate. These results justify a feasibility prototype;
they establish neither GPU speedup nor end-to-end speedup.

Changing row order preserves the fixed-correspondence objective mathematically,
but changes floating-point QR reduction order. The prototype therefore needs
independent factor/error/Jacobian checks and trajectory comparisons; these work
counts are not a correctness proof for its eventual implementation.

A possible later experiment is to initialize slots by certificate difficulty
or observed correspondence churn, concentrating unstable rows into fewer leaves.
That idea has not been implemented or measured, and its initialization/reordering
cost must be included if pursued.

## V4: split certificates and cheaper runner-up bounds

Frozen binary SHA256 `62d2cacbf96fc31e9fe6b065b33048826f6afd8dd7f51cb669f1d3ad9a760f56`
adds a thread-per-query certificate pass and a conservative second-occurrence
bound. The latter avoids tracking distinct runner-up identities: duplicates can
only weaken the bound. Neither variant changes the reference matcher or flat
incremental QR from V3. All 113 main tests pass under the eight audit/certified,
fused/split, distinct/occurrences combinations. Four targeted matching tests
also pass CUDA memcheck, device initcheck, and racecheck.

The 250-scan `audit-v4` run checks 45,285,082 queries with the unchanged original
GPU kernel on identical inputs, including warmup, with **zero mismatches**.
There are 32,967,529 successful certificates. This is an audit result, not timing.

`clean-v4-current` runs two repeats in reversed mode order, 32 CPU threads,
first 250 stairs scans with the first 20 excluded, no tracing/audit, and no
concurrent benchmark/build. Means in ms/scan:

| Variant | Total | Matching |
| --- | ---: | ---: |
| Off | 27.111 | 8.569 |
| Fused, distinct | 25.827 | 8.029 |
| Fused, occurrences | 26.468 | 7.972 |
| Split, distinct | 27.287 | 8.423 |
| Split, occurrences | 26.375 | 7.979 |
| Split, occurrences + flat incremental QR | 28.583 | 9.479 |

All feature/iteration count comparisons agree; largest translation difference
against the first off run is 1.37e-13 m. Search-only savings are small and noisy:
the existing fused/distinct implementation also looks faster here despite
losing in earlier clean experiments. These two repeats do not establish a
reliable new end-to-end improvement. Flat incremental QR still loses.
Cached internal QR ancestors are the next experimental variant.

The offline stability-order model is now documented in
`stability-order-model.md`. A future-informed initial ordering reduces modeled
whole-run leaf/merge QR calls by 32.12%, but is unavailable online. A causal
first-rematch-change ordering saves only 0.48% including its rebuild. The latter
does not justify a production implementation; these counts are not timings.

## V5: cached internal QR nodes

Revision `370e703`, frozen binary SHA256
`6d89d1835af9e29d898bc1f4f37661d18a00e2718e3fdbb3765e6854c05da6fc`.
Leaves and internal roots persist independently of the ordinary/flat QR scratch
buffers. Device highwater controls active extents; clean leaves and ancestors
retain their roots. Unary ancestors copy their child. Extent status travels with
the compact root download; per-node work counters are downloaded only when
explicitly requested. The matcher no longer downloads highwater before packing.

The full check target passes: 118 main tests, 2 parallel-extraction tests, and
9 scalar-extraction tests. The main suite before the final boundary-test addition
also passed under both `tree64` and `audit-tree64`. Five tree primitive tests
cover unchanged roots, single-leaf paths, reset/failure recovery, delayed producer
ordering, rank deficiency, and shrinking/reactivating extents across multiple
fan-in boundaries. Together with two matcher integration tests they pass CUDA
memcheck/initcheck coverage and combined racecheck with zero reported hazards.
Matcher initcheck uses device-only checking to exclude the previously documented
uninitialized ABI padding in host Result structures. Independent code review
found no blocking defect; a separate host model exercised 36,000 extent/dirty
transitions.

The 250-scan `audit-v5` run enables both the unchanged original search oracle and
independent full QR reconstruction. It uses `--stats-only` to retain diagnostics
without another raw correspondence dump:

- 45,285,082 checked queries; zero search mismatches.
- 66,386 group-root comparisons; maximum normalized Gram error 2.90841e-15.
- 748,395 active-leaf observations, 478,820 dirty leaves (including warmup).
- 40,595,314 queries processed by the split prepass; 4,834 tree-summary calls.

Audit supplies original search results and full-rebuild roots downstream.
Actual cached-root solver feedback is checked separately in the clean runs.
These correctness checks do not establish acceleration. The initial tree launch
caps work at 32 warps per group and launches all capacity-derived ancestor levels;
these are explicit performance variables for the following traces.

Clean V5 means, two reversed-order repeats with the same 250/20-scan protocol:

| Variant | Total ms/scan | Matching ms/scan |
| --- | ---: | ---: |
| Off | 26.988 | 8.498 |
| Fused occurrence-bound search | 25.620 | 7.718 |
| Split occurrence-bound search | 26.178 | 7.877 |
| Cached tree, full search | 27.419 | 9.358 |
| Cached tree + split occurrence-bound search | 26.167 | 8.665 |

All workload counts agree and maximum translation difference is 1.55e-13 m.
See `clean-v5-current/comparison-verified.json`; the comparison tool checks
recorded output hashes and complete run counts before reading results. Small
end-to-end differences still vary between repeats; the cached tree itself does
not yet beat full rebuilding on this workload.

`trace-v5/tree` is a separate 250-scan Nsight trace, excluding warmup. It has zero
unattributed timed device events. Device durations per scan:

| Kernel | ms/scan |
| --- | ---: |
| Original nearest | 1.9221 |
| Update stable row assignments | 0.0527 |
| Fill stable row slots | 0.3550 |
| Pack dirty leaves | 0.0877 |
| Prepare tree extents | 0.0400 |
| Update tree leaves | 0.7124 |
| Update tree ancestors | 1.0635 |
| Finish tree roots | 0.0733 |

Tree QR kernels total 1.8893 ms/scan, versus 1.408 ms for the earlier full QR
trace. Flat partial leaf QR took 0.418 ms; the tree's 0.712 ms despite comparable
leaf work motivates launch tuning. Resource inspection of the frozen V5 binary
reports 90 registers/thread for tree leaves and 92 for ancestors, no spills or
shared memory. The earlier flat QR kernels used 80 registers/thread. At the
current 32-warps-per-group cap, a dominant target group can use only 32 SMs on
this 108-SM GPU; group skew makes the overall grid count an optimistic measure
of parallel useful work. Matching-pack CUDA API time is 0.566 ms and QR API time
2.671 ms; these overlap device execution and must not be added to kernel times.

## V6: squared certificates and launch tuning

Revision `d0c88a2`, frozen binary SHA256
`d5d0833d1ca4eb5b9b09ff1607df55130a70ab73fda6d022b677fae4716af13b`.
The optional squared predicate removes square roots and division using directed
reciprocal enclosures and a strict polynomial inequality. Its proof and extreme
scale fallback behavior are in `form/optimization/certified_rematching.md`.
Tree variants sweep 32/128/256 warps per group and optionally mirror the exact
highwater recurrence on the host to bound launch depth without a download. See
`certified-rematching-launch-tuning.md` for resource estimates and validity.

Validation: 123 main, 2 parallel, and 9 scalar tests pass. The 123-test main suite
also passes all eight audit/certified × fused/split × distinct/occurrences settings
with the squared predicate. Seven QR tests pass memcheck, initcheck, and racecheck;
three squared matcher tests and two incremental matcher tests pass corresponding
memory/race coverage, with device-only matcher initcheck for the documented ABI
padding issue. Review found no blocking issue in the arithmetic or highwater mirror.

The 250-scan `audit-v6` run checks 45,285,082 queries with zero mismatches and
66,386 full-QR comparisons, maximum normalized Gram error 2.16557e-15. Its
32,967,529 successful certificates equal the earlier norm-predicate count on
this replay. All 4,834 summary calls report the bounded tree path. Audit timing
is excluded from performance claims.

Clean V6 (`clean-v6-current`), two reversed-order repeats, same 250/20 protocol:

| Variant | Total ms/scan | Matching ms/scan |
| --- | ---: | ---: |
| Off | 26.863 | 8.496 |
| Tree32 | 28.751 | 9.836 |
| Tree128 | 27.219 | 9.054 |
| Tree256 | 29.446 | 9.753 |
| Bounded tree32 | 28.346 | 9.427 |
| Bounded tree128 | 27.020 | 8.990 |
| Bounded tree256 | 28.518 | 9.442 |
| Fused norm/occurrences | 27.619 | 8.451 |
| Fused squared/occurrences | 27.293 | 8.506 |
| Split squared/occurrences | 25.560 | 7.792 |
| Bounded tree128 + fused squared/occurrences | 26.049 | 8.134 |

All workload counts agree; maximum translation difference is 3.02e-13 m.
Some repeats vary materially (for example tree32 total is 30.706 and 26.796 ms),
so these means select follow-up variants rather than establishing a final gain.
The split squared repeats are 25.535/25.585 ms total and 7.721/7.863 ms matching.
The best tree-only variant still loses in matching; larger grids are not uniformly
better. Final selected variants need broader, repeated validation.

Two separate V6 traces have zero unattributed timed events. The split squared
search uses 1.1180 ms/scan in `nearestReuse` plus 0.0823 ms in `certifyQueries`,
versus about 1.92 ms in original nearest search. This is a 37% kernel-time
reduction for the **combined schedule/bound configuration**, not proof that
eliminating square roots alone accounts for it. Its ordinary full QR is
1.4172 ms/scan.

For bounded tree128, leaf kernels fall to 0.4000 ms and ancestor kernels to
0.9775 ms, versus V5's 0.7124/1.0635 ms. Extent preparation and final-root kernels
add 0.0400/0.0716 ms. Total tree QR remains about 1.489 ms, slightly above full
QR, before stable-row bookkeeping. Sorting consumes 0.2224 ms of kernels and
0.3433 ms of issuing CUDA API time in the V5 trace; these overlap. Queuing only
new rows is a bounded follow-up for that bookkeeping cost. Compact cached task
plans are another follow-up for overlaunching small groups under a global cap.

Only 61 of 4,486 post-warmup V6 audit summary calls have zero dirty leaves,
representing 83,471 of roughly42 million queries. An all-unchanged fast path has
few opportunities in this trace; it is not a substitute for partial updates.

## V7: queue incoming rows

Revision `d195679`, frozen binary SHA256
`ec278c2623d53ff1e0f0093b152a34d6fb7deee8e38cc536a2fe53cdafe7f13a`.
The optional queue path enqueues only accepted queries without a current slot,
then fills first holes in a separate kernel. It removes the ordinary rematch's
radix sort, incoming-row compaction, and group descriptor upload. Full QR audits
and observers keep their original sorting. Queue order can vary, so this is not
a bitwise trajectory-equivalence claim. Sorted rows remain the default.

All 125 main, 2 parallel, and 9 scalar tests pass. Six queued search configurations
pass. Two dedicated tests cover partial warps, mixed groups, no/full arrivals,
migration, rejection, reinsertion, multichunk filling, reset/failure/toggling,
observer order, and full-QR audit. They pass memcheck, device initcheck, and
racecheck with zero findings on SM80. A compile-time scalar-atomic fallback is
provided for pre-SM70 targets; installed CUDA13 rejects compute_60 before source
compilation, so that compatibility path is not verified here.

`audit-v7` checks 45,285,082 searches with zero mismatches and 66,386 fresh-root
comparisons, maximum normalized Gram error 2.55743e-15. All 4,834 summary calls
report queued mode. There are 480,279 dirty leaves out of 748,395 active-leaf
observations, versus 478,820 dirty leaves in V5's sorted audit: approximately0.3%
more dirty leaves, including warmup. Different queue/feature ordering can change
this count across executions. This is a correctness/churn result, not timing.

V7 clean comparison (`clean-v7-current`), same two-repeat protocol:

| Variant | Total ms/scan | Matching ms/scan |
| --- | ---: | ---: |
| Off | 27.595 | 9.025 |
| Split squared search | 27.055 | 8.274 |
| Bounded tree128, sorted rows | 26.767 | 9.013 |
| Bounded tree128 + split squared search, sorted | 26.422 | 8.278 |
| Bounded tree128, queued rows | 27.897 | 9.022 |
| Bounded tree128 + split squared search, queued | 26.951 | 8.110 |

All workload counts agree; maximum translation difference is 2.57e-13 m.
Queuing does not establish a further end-to-end improvement in this screening
run. Search-only total varies from 25.997 to28.113 ms; its matching time varies
from8.022 to8.527 ms. Combined queued matching is7.871/8.349 ms, but total is
26.330/27.572 ms. These fluctuations make small total-time rankings inconclusive.
The stable finding remains that search work falls while partial-summary overhead
largely consumes its own QR savings. Compact task scheduling and broader workload
comparisons remain outstanding.

## Matched CPU controls at the current configuration

Frozen V7, first 250 stairs scans, warmup 20, 32 threads, two reversed-order
repetitions (`scale-v7-current/`):

| Backend | Mean total ms/scan | Mean matching ms/scan |
| --- | ---: | ---: |
| Existing GPU pipeline, research options off |26.901|8.576|
| Improved CPU summaries/extraction |77.774|52.572|
| Original FORM reference |126.656|65.119|

All workload counts agree. Maximum translation differences against GPU off
are 2.89e-13 m for improved CPU and 9.10e-12 m for reference. These are controls
for the broader existing acceleration, **not gains attributable to certified
rematching**. Dense/window and full-sequence comparisons follow separately.

## V8 compact active-node scheduling validation

Revision `1fe130f`, frozen replay SHA256
`b6b87935b02a8221d455d832b9791c3a7ecf87850e5b3e465651838b17259220`.
Per-group tasks remove the rectangular launch across small and large groups;
plans and per-group validation metadata upload only when changed. The same
leaf/ancestor arithmetic is shared with the capped tree. Default behavior is
unchanged; compact mode is opt-in.

Validation completed before measurement:

- 127 main, 2 parallel and 9 scalar C++ tests passed.
- 13 focused tree/matcher tests passed memcheck and racecheck with zero findings.
- 9 QR tests passed default initcheck. Four matcher tests passed initcheck with
  API copy checks disabled for the previously established `Result` tail padding;
  device checks remained enabled.
- Independent source review found no blocking correctness issue.
- Tests cover cached-plan reuse, independent bound uploads, empty/rank-deficient
  inputs, growth/shrink/reactivation, invalid bounds, API switching, producer
  stream ordering, migrations, rejection/reinsertion and queued row lifetimes.

Logs are retained in `validation-v8/`. Full 1,190-scan same-input audit and clean
current/dense/window ablations are the next measurements, not implied by these
unit and sanitizer results.

### V8 full-stairs audit

`audit-v8-full/` processes **all 1,190 scans**, including warmup, with independent
original-search and fresh-summary oracles:

- 308,516,696 queries/oracle comparisons, **zero mismatches**.
- 242,513,013 certified queries (78.61% of all queries, including first searches;
  84.74% of the 286,184,841 subsequent-rematch queries). Subsequent matches
  actually retain their nearest index 96.52% of the time; certification is
  deliberately conservative. Cell fallback is only 0.407% of subsequent queries.
- 925,510 fresh-root comparisons; maximum normalized feature-Gram error
  **2.82910e-15**.
- 2,970,485 dirty leaves out of 5,365,452 active leaves (55.36%).
- 32,896 compact calls; 6,552 task-plan refreshes; 14,715,868 metadata bytes
  uploaded across the complete replay.
- Matching/feature/pose/iteration counts agree with the off replay. Maximum
  translation difference is 5.80e-13 m. Audit substitutes full roots downstream;
  clean runs separately test actual cached-root feedback.
- GPU sampled peaks: off 460 MiB; compact audit 552 MiB, sampled every 0.25 s.
  These include CUDA context/allocator reservations and audit buffers, and can
  miss short allocations. They are not clean production memory deltas.

Full-stairs realized workload averages 27.54 poses and 267,627 point-plus-plane
correspondences per scan, versus 14.13 poses and 168,122 correspondences on the
250-scan prefix. Full-sequence results must therefore be measured directly.
Diagnostic latency is not a performance result.

### V8 current-prefix screening

Two reversed-order clean repetitions, same first 250 stairs scans:

| Mode | Mean total ms/scan | Mean matching ms/scan |
| --- | ---: | ---: |
| Off |27.099|8.796|
| Squared/split search only |27.607|8.503|
| Compact sorted summaries only |28.054|9.463|
| Search + compact sorted summaries |26.870|8.627|
| Compact queued summaries only |26.855|8.737|
| Search + compact queued summaries |27.868|8.619|

Artifacts: `clean-v8-current/`. Counts agree for all runs, with maximum
translation difference 2.75e-13 m. This is **not a demonstrated end-to-end win**.
For example, search-only total time is 26.733/28.481 ms in its two repeats,
while off is 26.820/27.379 ms. The combined queued path is
28.538/27.199 ms. Matching-stage means also vary across repeats.

Queued summary maintenance avoids the sort and has lower summary-only stage
cost than sorted maintenance in this screen. It is selected for the matched
full/scaled four-way tests; that choice is not a claim of reliable total savings.
The combined path must outperform search-only to justify partial summaries.

### V8 full-stairs clean four-way comparison

`clean-v8-full/`, all 1,190 scans, two reversed-order repetitions:

| Mode | All-scan mean total ms | Post-warmup mean total ms | Post-warmup mean matching ms |
| --- | ---: | ---: | ---: |
| Off |43.906|42.991|12.473|
| Squared/split search only |42.298|41.394|11.037|
| Compact queued summaries only |43.686|42.749|12.415|
| Search + compact queued summaries |43.316|42.429|11.446|

Post-warmup excludes scans0–19; correctness includes every scan. Search-only
improves total latency in both repeats (40.134/42.654ms versus
42.177/43.804ms). Combined summaries are2.50% slower than search-only on average.
All workload/iteration counts agree, maximum translation difference6.90e-13m.
Thus the broader sequence supports modest search-only savings, but does not
establish an additional benefit from partial summary maintenance.

### V8 dense-feature scaling with CPU controls

`scale-v8-features/`, same first250 scans, two reversed-order repetitions:

| Backend/mode | Post-warmup total ms | Post-warmup matching ms |
| --- | ---: | ---: |
| Existing GPU/off |48.721|20.464|
| GPU search-only |45.881|17.456|
| GPU partial compact QR only |50.230|21.266|
| GPU combined |47.010|18.279|
| Improved CPU |228.020|170.770|
| Original FORM CPU |337.647|187.215|

The denser setting realizes493,474 average correspondences (2.94x current) and
14.26 average poses. All workload/iteration counts agree. Maximum translation
difference across CPU/GPU runs is8.89e-12m. Search-only improves total latency in
both repeats (45.803/45.959 versus49.152/48.289ms). Its mean gain is5.83%; combined
partial summaries are2.46% slower than search-only. This supports a useful but
modest search optimization and another negative partial-summary result.
