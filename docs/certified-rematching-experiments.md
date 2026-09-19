# Certified rematching experiment ledger

Research branch: `codex/certified-rematching`, isolated worktree
`/home/ubuntu/form-certified-rematching`. Established acceleration branch remains
unchanged at `1255543`. These are preliminary feasibility observations, not final
speed or novelty claims.

## Latest verified checkpoint: V3 partial QR

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
