# CUDA tuning history

Status: the documented bounded tuning search and all 72 scaling runs are
complete and verified. See the [final results](optimization-acceleration-results.md)
for the consolidated conclusions. Earlier cap-only and recent20 pilots are
preserved below as findings, not treated as successful problem-size increases.
Hardware: NVIDIA A100 80 GB PCIe, 32-vCPU AMD EPYC Milan. These are not yet
measurements on the intended desktop GPU.

The final verified report contains 72 of 72 runs. Completed sequence reports
are [stairs](optimization-scaling-stairs.md),
[quad_easy](optimization-scaling-quad-easy.md),
[quad_hard](optimization-scaling-quad-hard.md), and
[maths_hard](optimization-scaling-maths-hard.md). Across all four sequences,
larger-window CUDA optimization latency is 6.5–17.6% lower than CPU summaries.
Denser selection realizes 2.52–2.90x correspondences, but CUDA ranges from
0.8% higher to 6.1% lower optimization latency. Default-size CUDA
has no broad, repeatable advantage over CPU summaries. These conclusions concern
the combined selected backend and include the repeatability qualifications in
the sequence reports. The sections below preserve the chronological experiments;
earlier numbers are not substitutes for the frozen-suite comparisons.

## Why the first CUDA version lagged CPU summaries

Two complete stairs repetitions at the original settings gave the following
mean per-scan optimization latencies, excluding the first 20 scans:

| Backend | Optimization | Complete scan |
|---|---:|---:|
| Original reference | 56.04 ms | 163.43 ms |
| CPU summaries | 29.07 ms | 129.04 ms |
| Initial CUDA summaries | 33.97 ms | 115.04 ms |

All three used point/plane selection limits 3/50 per sector and recent-scan
setting 10. Their realized correspondence, pose, rematch, and iteration counts
matched. Complete-scan CUDA timing was lower partly because matching time was
lower, although matching code was unchanged. That difference is not established
as a causal GPU benefit. All eight predeclared translational RTE gates passed in
both stairs repetitions.

The CPU optimization already compresses each correspondence set into small
exact summaries. CUDA initially only accelerated their construction, incurring
expanded feature construction, packing, transfers, several small QR kernels,
and synchronization whenever ICP changed correspondences. Subsequent residual
and Hessian evaluations use the summaries on the CPU.

A capture at stairs scan 200 contains nine rebuild batches, each with 17 plane
and 17 point matrices. Batch zero has 17,242 plane rows and 1,791 point rows.
That is about 1.89 MB of expanded FP64 input. Stored window correspondence totals
are different from the rows rebuilt in one batch.

An initial real-capture Nsight run measured 78.48 ms of H2D transfers and 16.84 ms
of kernels across 99 calls. Another trace measured 21.88 ms of H2D and 16.60 ms
of kernels. Transfer variability is substantial; CUDA API waits and GPU times
overlap and must not be added together. This motivates interleaved comparisons.

## Candidates evaluated

| Candidate | Evidence | Decision |
|---|---|---|
| Reusable pinned staging buffers | Original real batches took 11.6–11.8 ms in pageable comparisons; pinned measurements varied 5.5–8.4 ms | Retained, with repeatability checked further in estimator runs |
| Shared Householder reciprocal | Kernel trace changed only 24.30 to 23.95 microseconds; aggregate improvement was not repeatable; required a subnormal fallback | Rejected; original division retained |
| Direct compact correspondence packing | Seven FP64 values per row; plane tensors expanded while the first GPU kernel loads them | Retained provisionally after numerical tests and short replay |
| 64-row QR, 32-thread blocks | Same-process interleaved runs reduced aggregate time from 4.210 to 3.510 ms and 4.917 to 4.135 ms | Retained; short estimator validation passed |
| Direct GPU residual/Jacobian/Hessian reduction | Numerical checks pass; resident Hessian evaluation slower than CPU summaries, trial cost about 11 times slower in initial prototype | Experimental only; no production integration |
| Cached CUDA Graph launches | Reversed-order 250-scan repetitions: preparation 3.95 ms with graphs versus 3.80 ms without | Rejected; patch preserved for reproduction |

The tile experiment compared eight configurations in randomized order within
each repetition. Each configuration reused its own warmed buffers. Its aggregate
is the sum of nine captured-batch timings, including packing, transfers, kernels,
synchronization, and output construction. It does not include feature construction
or establish complete-estimator speedup. The 64-row kernel uses 80 registers with
no spills, versus 56 for the 32-row kernel. Batch zero needs four launches instead
of seven. Worst relative Gram errors were below 2e-15.

## Short estimator screening

On the first 250 stairs scans (first 20 excluded), pinned buffers plus compact
packing with the original 32-row QR produced:

| Stage | Earlier profiled CUDA | Compact-packing screening |
|---|---:|---:|
| Summary preparation | 9.728 ms | 4.635 ms |
| Optimization | 17.895 ms | 13.041 ms |
| Complete scan | 74.407 ms | 68.119 ms |

These runs used the same original workload, but are not a fresh paired final
benchmark. They do not establish full-sequence speedups or larger-problem quality.
With the selected 64-row configuration, two reversed-order screening repetitions
averaged 3.797 ms preparation, 12.268 ms optimization, and 68.210 ms complete-scan
time. Relative to the earlier profiled CUDA run, optimization is about 31% lower;
this is still a development comparison, not a full-sequence paired speedup.
The original reference and selected CUDA replays had identical feature,
correspondence, pose, rematch, and iteration counts on all 250 scans. Maximum
position disagreement was below 2.4e-12 m and rotation below 5.8e-13 radians.

The final production path passed 23 CTest cases, including near-zero residuals,
subnormal inputs, ragged boundaries, and manager integration. Compute Sanitizer
memcheck and synccheck reported zero errors, and racecheck zero hazards, over
nine CUDA/manager tests per tool. Both experimental graph and ordinary paths
passed the 23-case suite before the graph code was removed.

Remaining profiled assembly and solve costs average about 1.04 and 1.30 ms per
scan respectively on the first 250 scans. Semi-linearized optimization still
solves the active pose window; older factors are frozen as a linear container.
Only the separate disable-smoothing mode uses a single-pose graph. Assembly and
solving remain on the CPU. Full-sequence and enlarged-window profiling must
determine whether a new solve experiment becomes worthwhile. This stopping point
covers the measured preparation/transfer/launch bottlenecks and the direct-Hessian
alternative; it is not a claim that all possible optimizations are exhausted.

## Evidence and next measurements

A full stairs profile of selected CUDA changes the solve decision. After the
first 20 scans, preparation/linearization/assembly/solve average
5.72/3.93/2.70/5.27 ms per scan. Among 656 scans with at least 30 poses, those
costs are 6.77/4.64/3.44/7.17 ms, out of 33.19 ms optimization. Solving is now
about 22% of optimization in mature windows. The residual 11.17 ms includes graph
construction and optimizer work not separated by these counters. Scan 982 has
43 poses and spends 12.74 ms solving across 19 linearizations.

The preparation-validation suite was paused between stairs and the next sequence
to capture actual damped systems at scans 200, 600, and 982 for an isolated FP64
cuSolver comparison. Its six completed stairs runs are now archived; the broader
sequence-first scaling suite will use the newer frozen binary. Transfer, factorization, triangular solve, status checking, and
output reconstruction will all count toward GPU latency. The CPU baseline must
be compiled with the ordinary C++ compiler: compiling it under NVCC disables
Eigen CPU vectorization and would make this comparison misleading.

The new full-sequence suite has completed its first stairs comparison. In that
repeat, optimization measured 53.323 ms for the original reference, 29.241 ms for
CPU summaries, and 26.693 ms for selected CUDA. Complete-scan means were 159.369,
131.520, and 106.192 ms respectively. The first-repeat CUDA speedup is 2.00 times
over the original and 1.10 times over CPU summaries for optimization. Matching
again accounts for part of the larger total-scan difference; its cause remains
unproven. All eight RTE gates and coverage passed, all realized workload counts
matched, and maximum CUDA/reference position disagreement was below 2.4e-11 m.
Both full stairs repetitions have now finished execution. Their mean optimization
latencies after the first 20 scans are 52.911 ms for the original reference,
28.443 ms for CPU summaries, and 27.177 ms for selected CUDA. Thus CUDA takes
4.5% less optimization time than CPU summaries and is 1.95 times faster than the
original. The second CPU/CUDA pair is essentially tied (27.644/27.661 ms), so the
incremental GPU gain is small and variable. All six completed replay CSVs have been verified. The second reference run's
bookkeeping was recovered after cancelling its paused parent, using the complete
log, time exit status zero, and verified CSV/TUM files. Its unpersisted GPU-memory
samples remain explicitly unavailable. This is a stairs checkpoint,
not a four-sequence aggregate.

At the preparation-only checkpoint, increased-feature and increased-window
experiments had not run. The planned independent settings were point/plane
limits 6/100 with recent setting 10, and original limits 3/50 with recent setting
20. Subsequent window pilots below justify using recent40 for the final axis. These are selection limits rather
than guaranteed correspondence counts; retained keyscans also mean recent setting
10 is not a ten-pose cap. Actual correspondence and pose counts must accompany
scaling results. Increasing correspondence count may improve GPU utilization,
but does not guarantee speedup or better odometry: subsequent summary evaluation
has fixed size for each factor, while larger pose windows also increase assembly
and solving costs.

- Captured inputs and manifest: `benchmarks/results/qr-stairs200/`.
- Interleaved tile results: `benchmarks/results/tuning/tiles-interleaved-r{1,2}.csv`.
- Compact replay: `benchmarks/results/tuning/compact32-stairs250.csv`.
- Original full stairs repetitions: `benchmarks/results/full-t32/`.
- Archived pre-tuning binaries and source: `benchmarks/results/pre-tuning/`.
- Direct Hessian experiment: `benchmarks/results/tuning/direct-*` and
  `benchmarks/compile_direct_hessian.py`.

Run matched reference/CPU-summary/CUDA repetitions on all four complete sequences.
Only afterward increase feature limits and window settings independently, recording
actual correspondences/poses, optimization and total latency, convergence, and RTE.

## Captured dense-solver comparison

The 57 captured damped systems cover N108, N174, and N258 (18, 29, and
43 poses). Thirty alternating CPU/GPU repetitions per system passed the
predeclared solution and backward-error checks. The CPU uses the production
Eigen Upper factorization; CUDA uses Lower on the same full symmetric matrix.
Both use FP64. Host code is compiled with ordinary C++ and Eigen SIMD enabled.

| Dimension | CPU solve | CUDA solve including transfers | CPU/CUDA speedup |
|---|---:|---:|---:|
| 108 | 82.79 us | 182.92 us | 0.453x |
| 174 | 237.93 us | 300.51 us | 0.792x |
| 258 | 627.34 us | 508.58 us | 1.234x |

The largest captured scan saves 2.26 ms across its 19 solves. Maximum relative
solution disagreement is 1.31e-12, and maximum GPU scaled backward error is
5.41e-17. Memcheck self-tests report zero errors. Upper/Upper and Lower/Lower
diagnostics are preserved separately. Transfer variability prevents attributing
cross-run timing differences solely to the triangle choice.

An optional selective solver is being tested with a provisional N240 threshold.
This threshold is an experiment between the observed N174 loss and N258 win,
not a measured exact crossover. Small systems keep the CPU solver; the default
path remains unchanged until isolated estimator timing and trajectory validation
justify adoption. These microbenchmarks alone do not establish estimator speedup.
Evidence: `benchmarks/results/tuning/cuda-solve{.csv,.log,-aggregate.json}` and
`benchmarks/cuda_solve_experiment.md`.

The optional solver is implemented and passed 32 CTest cases. CPU-only tests
passed with the two GPU manager cases skipped; the Python extension rebuilt.
Eight solver/dispatch cases passed Compute Sanitizer memcheck with zero errors.
A 20-scan Python/C++ comparison forced GPU solving at N>=1 solely to exercise
the binding: 382 CUDA solves, zero fallbacks, and maximum pose-matrix difference
below 2.2e-14 across three backends. This is a correctness test, not a performance
configuration. Source/dispatch review found no material defect. The estimator
experiment at N240 is running; no integration speedup is claimed yet.

The first full 1,190-scan selective replay passed every RTE/coverage gate and
preserved all feature, correspondence, pose, rematch, factor, and LM-iteration
counts. It made 571 GPU solves with zero fallbacks. Maximum position disagreement
with the original reference was 2.325e-11 m and rotation 3.37e-12 rad. Across the
30 scans with at least 40 poses, solving averaged 7.227 ms versus 11.749 ms in the
earlier CPU-solve profile. Whole-run optimization nevertheless increased from
26.508 to 30.824 ms, with preparation, assembly, and linearization also higher.
These are separate-run measurements; they do not establish a net solver benefit
or its cause. A new same-binary, solver-disabled control is running before a
retention decision. Full candidate evidence is in
`benchmarks/results/tuning/selective-stairs-profile-analysis.json`.

The same-binary solver-disabled control completed at 29.025 ms optimization and
111.221 ms total, versus 30.824/115.882 ms with selective solving. Among the 30
eligible large-window scans, solve time fell 12.027→7.227 ms, but optimization
was essentially unchanged (42.451→42.583 ms). This pair does not justify enabling
selective solving at the current settings. The option remains experimental and
disabled by default. A recent-setting-20 comparison is now running to measure
actual larger-window behavior before deciding whether this option is useful
there. Evidence: `benchmarks/results/tuning/selective-control-comparison.json`.

## Expanded-window pilot

The complete recent20 stairs pilot does not enlarge the realized problem: after
20 warmup scans, mean active poses fell 27.86→23.53 and stored correspondences
fell approximately 269,789→239,419. Optimization averaged 22.438 ms and complete
scan time 97.84 ms. The keyscan promotion ratio divides by the recent-window
size, so changing that size also changes which older scans are retained. Mean
1m translational RTE changed 0.03471→0.03023 m, while 30m RTE worsened
0.23625→0.38157 m. These are one-run workload comparisons, not matched backend
validation or evidence that a larger window improves accuracy.

The final window workload is therefore set to recent40, with original feature
limits. Its pilot is running to confirm the actual size increase. The original
recent20 pilot remains available; it is not relabeled as recent40.

The full recent40 control confirms a larger problem: after warmup it averages
41.01 poses and 458,505 stored correspondences, versus 27.86 and 269,789 at
recent10. That is 1.47x as many poses and 1.70x as many correspondences.
Optimization increases 29.025→48.315 ms and total scan time 111.221→124.189 ms.
Solving averages 12.391 ms, assembly 5.928 ms, and linearization 6.860 ms.
Mean 1m translational RTE improves 0.03471→0.02599 m, while 30m RTE worsens
0.23625→0.30758 m. These remain single profiled workload comparisons; final
matched CPU/CUDA repetitions and raw-reference validation are pending. The
same recent40 input/settings with selective GPU solving are running next.

The recent40 selective run finished at 46.275 ms optimization and 123.066 ms
complete-scan latency versus 48.315/124.189 ms in its solver-disabled control.
Solving fell 12.391→7.417 ms. It performed 22,793 GPU solves after warmup with no
fallbacks; all realized counts matched, maximum position disagreement was
5.01e-13 m, and all RTE/coverage comparisons passed. This is a single profiled
pair showing a 4.2% optimization reduction; repeated validation remains needed.

The final 72-run suite now uses the frozen `selective-backend/form-replay` binary,
32 CPU threads, two repetitions with reversed backend order, and all four full
sequences. It runs features/window/current in sequence-first order. CUDA uses
selective N240 solving only for the recent40 window workload; current and
feature workloads keep CPU solving. Reference and CPU-summary backends keep
CPU solving everywhere. Exact commands and build/data fingerprints are recorded
in `benchmarks/results/final-scaling-t32/suite.json`. Full-scale results are
pending; the pilot results above are not substituted for that suite.

## Feature-selection correction

The cap-only pilot (6/100, original spacing) finished one reference/summary/CUDA
comparison. The reference retained exactly the same mean planar count as current
settings (17,261 over all scans), while points rose1505→2067 and stored total
correspondences fell267,627→258,734 because retained poses also changed. It is
not a larger correspondence workload. The original spacing loop suppresses four
adjacent samples on each side of a selection; with roughly170samples/sector,
the plane cap50 is already above the attainable count.

The 72-run suite was therefore superseded after these three uninterrupted pilot
runs, preserved under `final-scaling-t32/` with cancellation metadata. Actual
density scaling adds an independent suppression-spacing control, defaulting to
the original neighborhood size, and uses spacing2 with caps6/100 at recent10.
Curvature, validity masking, and normal-estimation neighborhoods stay unchanged.
Default behavior and actual retained-density increase are being verified before
restarting the final matrix.

The real-data density screen confirms the intended increase. On stairs scans
20–99, both configurations retain exactly 12 active poses on average. Spacing 2
with caps 6/100 increases mean planar features from 17,836 to 48,258 (2.71x),
point features from 1,146 to 2,492 (2.18x), and stored correspondences from
171,697 to 501,642 (2.92x). These are workload counts, not final latency or
accuracy results. All 38 C++ tests passed; the CPU-only build passed 23 applicable
tests with two GPU cases skipped. The Python binding was rebuilt, and its
density-specific replay verification is in progress.

A further compressed-factor GPU evaluation experiment is also closed. Nine
captured batches of clean 7x7/13x13 QR roots were correct to below 9e-16 for
Hessian, RHS, and cost, including signed, nonunit, and zero homogeneous point
weights. One Hessian pass cost 0.326 ms on CPU, 0.450 ms on resident GPU roots,
and 0.747 ms including upload. GPU trial cost was 11–12x slower. This measured
rebuilt-pair variant is not integrated. Whole-window fusion is outside its scope.
See `benchmarks/results/tuning/direct-compressed-roots-note.md`.

Density-specific Python/C++ replay verification passed across all three backends
with maximum pose-matrix disagreement below 3e-15. The eight-scan dense CUDA
replay passed Compute Sanitizer memcheck with zero errors. All 27 benchmark
Python tests pass. The new frozen executable, build metadata, and complete source
snapshot are in `benchmarks/results/density-backend/`.

The authoritative final suite is now `benchmarks/results/density-scaling-t32/`:
72 complete-sequence runs, 32 threads, two repetitions, features/window/current
in sequence-first order, and reference/CPU-summary/CUDA backends. Features use
caps 6/100 and spacing 2 at recent10. Window uses original selection at recent40.
Selective GPU solving at N240 is enabled only for the CUDA window workload.
The initial density screen verifies the chosen workload increase; the full
suite will measure its retained counts, speed, and accuracy across all sequences.

## Corrected dense-workload checkpoint

The first complete stairs reference/summary/CUDA comparison in
`density-scaling-t32` has finished. These are single-run means after 20 warmup
scans, using the same frozen executable and enlarged feature settings:

| Backend | Optimization (ms) | Complete scan (ms) | Matching (ms) |
|---|---:|---:|---:|
| Reference | 109.056 | 485.146 | 340.490 |
| CPU summary | 34.775 | 392.609 | 324.171 |
| CUDA summary | 32.936 | 340.289 | 270.818 |

CUDA reduces optimization latency by 5.29% versus CPU summaries and gives a
3.31x optimization speedup versus the reference. Most of the reduction comes
from the exact factor reformulation. Matching code is unchanged: its timing
difference is observed but not established as a causal GPU benefit. Matching
accounts for about 80% of complete-scan latency in this dense CUDA run.

Across all 1,190 scans the reference averages 694,713 stored correspondences
and 26.57 active poses. The previous current-settings pilot averaged about
267,627 correspondences; this preliminary 2.60x comparison spans different
build snapshots. Final scaling ratios require the pending same-suite current
runs. All realized workload counts, including rematches and LM iterations,
match between the three dense backends. Maximum CUDA/reference position
disagreement is 7.94e-12 m; mean translational RTE is 0.035187 m at 1 m and
0.345229 m at 30 m for all three backends to the displayed precision.

The checkpoint verifies 3/72 completed runs. Repeat-order validation and all
four-sequence conclusions remain pending. The suite-wide quality status is
incomplete, even though the completed first-repeat comparisons agree.
Machine-readable evidence and the refreshed report are in
`benchmarks/results/density-scaling-t32/checkpoint.json` and `checkpoint.md`.

The second dense CUDA repeat has also completed, bringing the checkpoint to
4/72 runs. Optimization averages 32.600 ms versus 32.936 ms in the first run
(1.02% lower); complete-scan latency is 333.611 versus 340.289 ms. Mean 1 m and
30 m translational RTE agree with the first repeat to the displayed precision.
The matching second CPU-summary and reference runs remain pending, so this
establishes CUDA repeat variability only, not a repeated backend speedup or a
completed second-repeat quality gate.

The second CPU-summary run subsequently completed at 35.010 ms optimization
and 388.832 ms per scan. Across two repetitions, CPU-summary optimization
averages 34.892 ms and CUDA 32.768 ms: CUDA reduces latency by 6.09%, with gains
in both run orders (5.29% and 6.88%). Second-repeat CPU/CUDA realized counts
match on all 1,190 scans, with maximum position difference 5.12e-12 m.
`stairs-dense-repeat-comparison.json` records both paired timing and trace
comparisons. This is repeat evidence for dense stairs only; the checkpoint is
5/72, the second raw-reference run is pending, and other sequences and final
current/window comparisons remain incomplete.

Both dense stairs reference repeats are now complete. The six-run comparison
passes all predeclared translational RTE and coverage gates for both accelerated
backends. Mean optimization latency is 109.708 ms reference, 34.892 ms CPU
summary, and 32.768 ms CUDA summary. CUDA therefore gives a 3.35x speedup versus
the original and 6.09% lower latency than CPU summaries. Mean complete-scan
latency is 487.048, 390.720, and 336.950 ms respectively; the matching-time
attribution limitation above still applies. Both accelerated backends preserve
all recorded workload and iteration counts in both repeats. Maximum
CUDA/reference position disagreement across either repeat is 2.54e-11 m.

This completes dense stairs validation only (6/72 suite runs). The suite has
advanced to the recent40 window workload. Final feature/window scaling ratios
and the four-sequence operating-range conclusion remain pending.

## Corrected larger-window checkpoint

The first recent40 stairs reference run in `density-scaling-t32` is complete
(7/72 suite runs). After 20 warmup scans, optimization averages 96.693 ms,
complete-scan processing 198.875 ms, and matching 81.746 ms. Across all 1,190
scans, including startup, it averages 40.497 active poses (maximum 42), 808.010
factors, and 453,778 stored correspondences. Mean translational RTE is
0.025988 m at 1 m and 0.307581 m at 30 m. These agree with the earlier window
pilot at the displayed precision, but final comparisons use this frozen suite.

This is the reference baseline only. Matching CPU-summary/CUDA runs, reversed
repetitions, and same-suite current-workload comparisons are still pending;
neither a window speedup nor a quality improvement over current is established
by this run alone.

The first matching CPU-summary and CUDA window runs have now completed (9/72
suite runs). Mean optimization latency is 96.693 ms reference, 52.883 ms CPU
summary, and 41.530 ms CUDA. The combined CUDA preparation/selective-solve path
therefore reduces optimization by 21.47% versus CPU summaries and is 2.33x as
fast as the original in this single comparison. This suite comparison does not
isolate the solver contribution from CUDA preparation; the solver-only pilot
above provides the separate ablation.

CUDA made 22,793 eligible GPU solves and zero numerical fallbacks. All recorded
workload and iteration counts match the reference over 1,190 scans, maximum
position disagreement is 1.34e-11 m, and the first-repeat RTE/coverage checks
pass. Complete-scan means are 198.875 ms reference, 152.976 ms CPU summary,
and 117.163 ms CUDA. Matching means also change (81.746, 80.438, 56.744 ms)
despite unchanged matching code, so attributing the entire scan-time difference
to GPU computation would be unsupported. The CUDA complete-scan mean still
exceeds the 100 ms LiDAR budget. Reversed-order repetitions and comparison with
the same-suite current workload remain pending.

All six recent40 stairs runs are now complete (12/72 suite runs). Both
accelerated backends pass the RTE and coverage gates in both repetitions, and
all recorded workload and iteration counts match their corresponding reference.
Mean optimization latency across repeats is 96.418 ms reference, 51.046 ms CPU
summary, and 42.052 ms CUDA. The combined CUDA path is 2.29x faster than the
original and reduces latency by 17.62% versus CPU summaries. Individual paired
reductions are 21.47% and 13.48%; the repeat spread matters when interpreting
the mean. Both CUDA runs made 22,793 GPU solves with zero numerical fallbacks.
Maximum CUDA/reference position disagreement is 2.90e-11 m across both runs.

Mean complete-scan latency is 198.855, 149.024, and 118.447 ms respectively.
CUDA therefore remains above the 100 ms budget at this window setting. The
unexplained matching-time difference still limits causal attribution of the
end-to-end gain. The suite is now running current settings; same-build scaling
ratios and the other three sequences remain incomplete.

## Initial same-build workload tradeoff

The first current-settings reference run is complete (13/72), enabling a
first-repeat raw-reference comparison across all three stairs workloads with
the same frozen executable. Latencies exclude 20 warmup scans; realized counts
average all 1,190 scans, including startup.

| Workload | Poses | Stored correspondences | Optimization ms | Scan ms | Mean RTE 1 m (m) | Mean RTE 30 m (m) |
|---|---:|---:|---:|---:|---:|---:|
| Current | 27.54 | 267,627 | 53.55 | 158.44 | 0.034707 | 0.236248 |
| More features | 26.57 | 694,713 | 109.06 | 485.15 | 0.035187 | 0.345229 |
| Larger window | 40.50 | 453,778 | 96.69 | 198.88 | 0.025988 | 0.307581 |

The feature workload has 2.60x correspondences but slightly fewer retained
poses (0.965x). The window workload has 1.47x poses and 1.70x correspondences.
Increasing selection density leaves short-distance mean error nearly unchanged
and increases 30 m mean error by 46.1%. The larger window lowers 1 m mean error
by 25.1%, but increases 30 m mean error by 30.2%. This is evidence of a tradeoff
on stairs, not a general benefit or regression across the dataset. Current
repeats and accelerated current-workload timings remain pending; final CUDA
latency scaling and all-four-sequence conclusions are not yet available.
Evidence: `density-scaling-t32/stairs-reference-workload-comparison.json`.

The first current CUDA run is now complete (15/72), averaging 28.644 ms
optimization and 111.436 ms per scan. CPU-summary current averages 30.204 ms
optimization in its first run. Both match reference workload/iteration counts;
maximum CUDA/reference position disagreement is 1.14e-11 m and the first-repeat
RTE/coverage checks pass.

The first-repeat CUDA scaling comparison, on the same binary, is:

| Workload | Correspondences / current | Poses / current | Optimization ms | Optimization / current | Scan ms |
|---|---:|---:|---:|---:|---:|
| Current | 1.00 | 1.00 | 28.644 | 1.00 | 111.436 |
| More features | 2.60 | 0.965 | 32.936 | 1.15 | 340.289 |
| Larger window | 1.70 | 1.47 | 41.530 | 1.45 | 117.163 |

Exact cached summaries make optimization grow much less than correspondence
count in this density experiment. Complete-scan time does not share that
scaling: matching dominates the denser workload. The larger-window path includes
selective GPU solving, whereas current and dense-feature CUDA paths use CPU
solving. These are workload-level comparisons with changed correspondences and
retention, not isolated complexity measurements. Current repetitions remain
pending, so these ratios are provisional. They do not establish a quality gain;
the trajectory tradeoffs in the preceding table still apply. Evidence:
`density-scaling-t32/stairs-cuda-scaling-first-repeat.json`.

The second current CPU-summary/CUDA pair is complete (17/72), and it does not
confirm a current-size CUDA gain. CPU-summary optimization is 30.204/26.510 ms
across repeats, versus CUDA 28.644/29.535 ms. CUDA is 5.16% faster in the first
pair but 11.41% slower in the second. Their two-repeat means are 28.357 ms CPU
summary and 29.090 ms CUDA, making CUDA 2.58% slower on this current workload.
All CPU/CUDA workload and iteration counts still match, with maximum paired
position disagreement below 8.55e-13 m. The correct current-size conclusion is
no demonstrated CUDA advantage over exact CPU summaries, rather than promoting
the favorable first repeat. The dense-feature and larger-window comparisons
retain gains in both pairs. The final raw-reference current repeat is pending.
Evidence: `density-scaling-t32/stairs-current-repeat-comparison.json`.

## Quad_easy dense first comparison

The first reference/CPU-summary/CUDA dense-feature comparison on quad_easy has
completed (21/72 suite runs). All 1,991 scans are processed and all 1,988
available ground-truth poses are matched. Mean optimization latency after 20
warmup scans is 78.585 ms reference, 32.744 ms CPU summary, and 31.638 ms CUDA.
CUDA gives a 2.48x speedup over the original and 3.38% lower optimization latency
than CPU summaries in this single comparison; reversed repetitions are pending.

All recorded workload and LM-iteration counts match the reference. Maximum
CUDA/reference position disagreement is 1.10e-10 m. Mean translational RTE is
0.043366 m at 1 m and 0.423554 m at 30 m for all backends to displayed precision.
Complete-scan means are 294.541, 233.566, and 198.916 ms respectively. Matching
means change from 143.023 to 130.562 to 96.543 ms despite unchanged matching
code; its causal contribution remains unproven. Thus the full scan-time
difference is not evidence of that much direct GPU acceleration.

The same configured dense selection yields about 334,753 stored correspondences
and 23.90 poses on average over all scans here, versus about 694,713 and 26.57
on stairs. Current and larger-window quad_easy runs remain pending, so no
within-sequence density multiplier or accuracy tradeoff is established yet.

All six dense quad_easy runs subsequently completed (24/72), with both
accelerated backends passing every predeclared translational RTE and coverage
gate in both repetitions. Two-repeat optimization means are 78.297 ms original,
32.657 ms CPU summary, and 31.596 ms CUDA: a 3.25% CUDA reduction versus CPU
summaries and 2.48x speedup over the original. Complete-scan means are 294.283,
233.060, and 197.978 ms, respectively.

Unlike stairs, the second dense quad_easy comparison does not preserve identical
per-scan work. CUDA repeat 2 diverges from its first trajectory at row 259; raw
reference repeat 2 also diverges from its first repeat, with count differences
from row 105. CPU-summary repeats agree closely. Maximum repeat-2
CUDA/reference position difference is 4.08 cm and CPU-summary/reference
difference is 5.26 cm, while all quality gates pass unchanged. Report the second
pair as complete-estimator timing with changed rematches, iterations, and
retention, not fixed-input backend timing. The specific internal branching
cause is not proven. Details and evidence are in
`density-scaling-t32/quad_easy-dense-repeat-investigation.md`; the larger-window
quad_easy workload is now running.

## Quad_easy larger-window first comparison

The first reference/CPU-summary/CUDA recent40 comparison is complete (27/72).
Optimization means are 104.826, 64.276, and 59.103 ms respectively. The combined
CUDA preparation/selective-solve path reduces optimization latency by 8.05%
versus CPU summaries and is 1.77x faster than the original in this single pair.
It makes 51,622 GPU solves with zero numerical fallbacks. Reversed repetitions
remain pending; the solver contribution is not isolated by this comparison.

All recorded per-scan workload and iteration counts match the reference, with
maximum CUDA/reference position disagreement 5.37e-11 m. Mean translational RTE
is 0.042930 m at 1 m and 0.439115 m at 30 m; all 1,988 available ground-truth
poses are matched. The original window run averages 41.57 active poses and
339,204 stored correspondences over all 1,991 scans, including startup.

Complete-scan means are 226.414 ms original, 180.571 ms CPU summary, and
164.363 ms CUDA. Matching means also differ (73.008, 65.654, 53.920 ms), so the
existing matching-time attribution limitation applies. CUDA remains above the
100 ms mean scan budget. Same-sequence current-setting scaling is pending.

Both larger-window CPU-summary/CUDA repetitions are now complete (29/72 suite
runs). CPU-summary optimization takes 64.276/62.935 ms and CUDA takes
59.103/59.836 ms: CUDA reduces latency by 8.05% and 4.92%, respectively.
Their two-repeat means are 63.606 ms CPU summary and 59.470 ms CUDA, a 6.50%
reduction. Every recorded workload and iteration count agrees between the
paired backends and between CPU repetitions across all 1,991 scans. Maximum
paired CPU-summary/CUDA position disagreement is below 1.94e-12 m. The second
raw-reference run is still pending, so the final paired reference quality gates
and original-backend mean are not yet available. Evidence:
`density-scaling-t32/quad_easy-window-repeat-comparison.json`.

The second raw-reference window run has now finished, completing all six runs
for this workload (30/72 suite runs). Two-repeat optimization means are
103.834 ms original, 63.606 ms CPU summary, and 59.470 ms CUDA, giving CUDA
1.75x speedup over the original. Complete-scan means are 224.301, 177.857, and
164.849 ms, respectively. Both accelerated backends pass all nine predeclared
translation-RTE/coverage checks in each repeat. All recorded workload and
iteration counts match the paired references; maximum accelerated/reference
position disagreement is below 5.41e-11 m. Reference repetitions also preserve
all counts. Evidence: `density-scaling-t32/quad_easy-window-final-comparison.json`
and the verified checkpoint. Current-setting quad_easy runs are next, so
within-sequence workload scaling and quality tradeoffs remain pending.

## Quad_easy first reference workload comparison

The first current-setting reference run is complete (31/72), enabling an initial
within-sequence comparison of all three workloads. This table uses only the
first original-CPU repeat of each configuration; it does not establish current
CUDA speedup or repeated scaling. Counts average all 1,991 scans and timing
excludes the first 20 scans.

| Workload | Mean poses | Mean correspondences | Optimization ms | Complete scan ms | Mean 1 m RTE m | Mean 30 m RTE m |
|---|---:|---:|---:|---:|---:|---:|
| Current | 22.44 | 121,603 | 42.773 | 148.915 | 0.044630 | 0.422419 |
| More features | 23.90 | 334,753 | 78.585 | 294.541 | 0.043366 | 0.423554 |
| Larger window | 41.57 | 339,204 | 104.826 | 226.414 | 0.042930 | 0.439115 |

Denser selection realizes 2.75x correspondences and 1.07x poses. The larger
window realizes 1.85x poses and 2.79x correspondences. Short-distance mean RTE
decreases by 2.83% and 3.81%, respectively, while longer-distance mean RTE
increases by 0.27% and 3.95%. These small quality differences require the
remaining repeated comparisons before drawing conclusions. The earlier dense
reference/CUDA repeat divergence remains relevant to interpreting variability.
Evidence: `density-scaling-t32/quad_easy-reference-workload-first-comparison.json`.

## Quad_easy first current CUDA comparison and scaling

The first current reference/CPU-summary/CUDA triple is complete (33/72).
Optimization takes 42.773/24.838/25.252 ms, respectively. CUDA is 1.67% slower
than CPU summaries in this pair; reversed repetitions are pending. Both
accelerated backends pass the predeclared translation-RTE/coverage gates.
Every recorded workload and iteration count matches the reference across all
1,991 scans, with maximum CUDA/reference position disagreement 1.15e-10 m.

Using the first CUDA repeat of each workload, optimization grows from 25.252 ms
current to 31.638 ms with denser features and 59.103 ms with the larger window.
That is 1.25x optimization time for 2.75x correspondences in the density case,
and 2.34x time for 1.85x poses / 2.79x correspondences in the window case.
Complete-scan means are 109.758, 198.916, and 164.363 ms, respectively; every
configuration still exceeds the 100 ms mean scan budget. These are provisional
first-repeat scaling ratios, not repeated estimates or isolated complexity
measurements. The small first-reference quality tradeoffs above also hold for
these CUDA runs to displayed precision. The denser second-repeat variability
remains relevant to final interpretation. Evidence:
`density-scaling-t32/quad_easy-cuda-scaling-first-repeat.json`.

Both current CPU-summary/CUDA repeats are now complete (35/72). CPU-summary
optimization takes 24.838/24.709 ms versus CUDA 25.252/25.754 ms. CUDA latency
is 1.67% and 4.23% higher in the paired repeats; the two-repeat means are
24.773 ms CPU summary and 25.503 ms CUDA, a 2.95% CUDA increase. All recorded
workload and iteration counts match in both CPU/CUDA pairs and between CPU
repeats. Maximum paired CPU-summary/CUDA position disagreement is below
2.38e-12 m. Complete-scan means are 124.521 ms CPU summary and 110.954 ms CUDA;
the matching-time attribution limitation still applies, so this difference
does not establish faster GPU optimization. The final current raw-reference
repeat is pending. Evidence:
`density-scaling-t32/quad_easy-current-repeat-comparison.json`.

All 18 quad_easy runs are now complete (36/72 suite runs). The final current
reference repeat preserves workload/iteration counts, and both accelerated
backends pass every predeclared quality check in both repeats. Current original
optimization averages 42.423 ms, versus 24.773 ms CPU summary and 25.503 ms
CUDA. The complete two-repeat performance, scaling, accuracy, and latency-tail
results are in [the quad_easy report](optimization-scaling-quad-easy.md).
Quad_hard and maths_hard remain pending.

## Quad_hard dense reference checkpoint

The first dense-feature original-CPU run is complete (37/72 suite runs), with
all 1,880 scans processed and all 1,724 available ground-truth poses matched.
After excluding 20 warmup scans, mean optimization is 132.001 ms and complete
scan processing is 398.956 ms. Matching averages 200.562 ms. Complete-scan
p95/p99 are 604.326/712.103 ms, with every steady-state scan exceeding 100 ms.

Over all scans, this workload averages 32.10 active poses (maximum 47) and
656,721 stored correspondences. Mean translational RTE is 0.076987 m at 1 m
and 0.716434 m at 30 m. These establish the first matching-settings reference;
accelerated runs, reversed repeats, and same-sequence current/window comparisons
remain pending. No CUDA speedup or benefit from denser selection is established
on quad_hard yet. Evidence:
`density-scaling-t32/quad_hard-features-reference-t32-r1.run.json`.

The first CPU-summary run subsequently completed (38/72). Optimization averages
55.526 ms versus 132.001 ms original, a 2.38x speedup. Complete-scan time falls
from 398.956 to 302.689 ms. All nine predeclared translation-RTE/coverage checks
pass, with all 1,724 available ground-truth poses matched. Every recorded
workload and iteration count matches the original across all 1,880 scans;
maximum position disagreement is below 5.56e-10 m. CUDA and reversed repeats
remain pending. Evidence:
`density-scaling-t32/quad_hard-features-first-cpu-comparison.json`.

The first dense CUDA run is now complete (39/72), averaging 56.118 ms
optimization versus 55.526 ms CPU summary and 132.001 ms original. CUDA is
1.07% slower than CPU summaries in this first pair, despite the larger
correspondence count; no dense-workload CUDA advantage is established on this
sequence. It remains 2.35x faster than the original. All nine quality checks
pass and every recorded workload and iteration count matches the original
across all 1,880 scans. Maximum CUDA/reference position disagreement is below
5.53e-10 m; CUDA/CPU-summary disagreement is below 1.16e-11 m.

Complete-scan means are 398.956 ms original, 302.689 ms CPU summary, and
263.955 ms CUDA. Matching means are 200.562/183.577/142.162 ms with unchanged
matching code; their causal difference remains unproven and must not be
presented as direct GPU optimization acceleration. Reversed repeats are
running. Evidence: `density-scaling-t32/quad_hard-features-first-cuda-comparison.json`.

Both dense CPU-summary/CUDA pairs are complete (41/72). CPU-summary
optimization is 55.526/55.825 ms and CUDA is 56.118/55.030 ms. CUDA loses the
first pair by 1.07% and wins the second by 1.42%; means are 55.675 ms CPU
summary and 55.574 ms CUDA. The 0.18% mean CUDA reduction does not establish a
consistent gain. All recorded workload and iteration counts agree in both
pairs and between CPU repetitions. Maximum paired CPU-summary/CUDA position
disagreement is below 1.22e-10 m. Complete-scan means are 305.201 ms CPU
summary and 263.294 ms CUDA, subject to the unchanged matching-time attribution
limitation. The second original-reference run remains pending. Evidence:
`density-scaling-t32/quad_hard-features-repeat-comparison.json`.

All six dense quad_hard runs are now complete (42/72). Two-repeat means are
130.314 ms original, 55.675 ms CPU summary, and 55.574 ms CUDA for optimization;
complete-scan means are 395.873/305.201/263.294 ms. Both accelerated backends
pass all nine predeclared translation-RTE/coverage checks in both repeats, and
all recorded workload and iteration counts match each paired reference.
Maximum accelerated/reference position disagreement is below 6.37e-10 m.
Original-reference repetitions also preserve all counts. Thus the effective
CPU-summary/CUDA optimization tie is supported by complete matched-work
comparisons on this workload, with roughly 2.34x acceleration over the original.
Same-sequence current and larger-window scaling remain pending; the first
larger-window reference is running. Evidence:
`density-scaling-t32/quad_hard-features-final-comparison.json` and the verified
checkpoint.

## Quad_hard larger-window reference checkpoint

The first recent40 original-CPU run is complete (43/72 suite runs), with all
1,880 scans processed and all 1,724 available ground-truth poses matched.
After 20 warmup scans, mean optimization is 117.034 ms, matching is 85.406 ms,
and complete-scan time is 243.781 ms. Complete-scan p95/p99 are
346.740/376.168 ms, with every steady-state scan exceeding 100 ms.

Over all scans, this workload averages 41.08 active poses (maximum 42) and
376,283 stored correspondences. Mean translational RTE is 0.074390 m at 1 m
and 0.724144 m at 30 m. Accelerated comparisons, reversed repetitions, and
current-setting quad_hard runs remain pending; these values alone establish
neither CUDA speedup nor the benefit of enlarging the window. Evidence:
`density-scaling-t32/quad_hard-window-reference-t32-r1.run.json`.

The first CPU-summary window run is complete (44/72), averaging 76.281 ms
optimization versus 117.034 ms original, a 1.53x speedup. Complete-scan means
are 199.023 ms and 243.781 ms, respectively. All nine predeclared quality
checks pass with 1,724 ground-truth poses matched. Every recorded workload and
iteration count agrees across all 1,880 scans; maximum position disagreement
is below 5.58e-10 m. CUDA preparation/selective solving and reversed repeats
remain pending. Evidence:
`density-scaling-t32/quad_hard-window-first-cpu-comparison.json`.

The first larger-window CUDA run is complete (45/72). Optimization averages
64.075 ms versus 76.281 ms CPU summary and 117.034 ms original: 16.00% lower
latency than CPU summaries and 1.83x speedup over the original in this first
comparison. The combined preparation/selective-solve path records 58,515 GPU
solves with zero numerical fallbacks. Its solver contribution is not isolated
by this comparison. All nine predeclared quality checks pass, and every
recorded workload and iteration count matches the original across all 1,880
scans. Maximum CUDA/reference position disagreement is below 5.56e-10 m.

Complete-scan means are 243.781/199.023/164.111 ms original/CPU-summary/CUDA;
matching means are 85.406/81.427/59.397 ms despite unchanged matching code.
The matching-time attribution limitation remains, and CUDA still exceeds the
100 ms mean complete-scan budget. Reversed repetitions are underway. Evidence:
`density-scaling-t32/quad_hard-window-first-cuda-comparison.json`.

Both larger-window CPU-summary/CUDA pairs are complete (47/72). CPU-summary
optimization is 76.281/75.468 ms and CUDA is 64.075/64.336 ms. CUDA reduces
latency by 16.00% and 14.75%, respectively; means are 75.874 ms CPU summary
and 64.205 ms CUDA, a 15.38% reduction. Every recorded workload and iteration
count matches in both CPU/CUDA pairs and between CPU repetitions. Maximum
paired CPU-summary/CUDA position disagreement is below 2.21e-11 m. Each CUDA
run makes 58,515 GPU solves with zero numerical fallbacks. Complete-scan means
are 197.859 ms CPU summary and 165.054 ms CUDA, retaining the matching-time
attribution limitation. Final paired reference quality validation is pending
the second original run. Evidence:
`density-scaling-t32/quad_hard-window-repeat-comparison.json` and
`density-scaling-t32/quad_hard-window-cuda-repeat-agreement.json`.

All six larger-window quad_hard runs are complete (48/72). Optimization means
are 116.686 ms original, 75.874 ms CPU summary, and 64.205 ms CUDA: CUDA is
1.82x faster than the original and retains the 15.38% reduction versus CPU
summaries. Complete-scan means are 243.381/197.859/165.054 ms. Both accelerated
backends pass all nine quality checks in both repetitions. Every recorded
workload and iteration count matches each paired original reference, and
maximum accelerated/reference position disagreement is below 5.72e-10 m.
Original-reference repetitions also preserve all counts. Current-setting
quad_hard runs are underway; within-sequence scaling and quality tradeoffs
remain pending that baseline. Evidence:
`density-scaling-t32/quad_hard-window-final-comparison.json` and the verified
checkpoint.

## Quad_hard first reference workload comparison

The first current-setting reference run is complete (49/72), enabling a first
within-sequence comparison of all three workloads. This table uses only the
first original-CPU repeat of each configuration. Counts include all 1,880
scans; timing excludes 20 warmup scans. All configurations match 1,724
ground-truth poses.

| Workload | Mean poses | Mean correspondences | Optimization ms | Complete scan ms | Mean 1 m RTE m | Mean 30 m RTE m |
|---|---:|---:|---:|---:|---:|---:|
| Current | 30.67 | 227,254 | 70.557 | 190.735 | 0.077817 | 0.717931 |
| More features | 32.10 | 656,721 | 132.001 | 398.956 | 0.076987 | 0.716434 |
| Larger window | 41.08 | 376,283 | 117.034 | 243.781 | 0.074390 | 0.724144 |

Denser selection realizes 2.89x correspondences and 1.05x retained poses;
short-/long-distance mean RTE decreases by 1.07%/0.21%. The larger window
realizes 1.34x poses and 1.66x correspondences; short-distance mean RTE
decreases by 4.40% while long-distance mean RTE increases by 0.87%. These small
quality changes are provisional first-reference findings. Current accelerated
comparisons and repetitions remain pending, so no CUDA scaling ratios are
established here yet. Evidence:
`density-scaling-t32/quad_hard-reference-workload-first-comparison.json`.

## Quad_hard first current CUDA comparison

The first current reference/CPU-summary/CUDA triple is complete (51/72).
Optimization means are 70.557/44.219/43.304 ms and complete-scan means are
190.735/156.757/137.027 ms, respectively. CPU-summary preserves all reference
workload and iteration counts and passes all quality checks. CUDA also passes
all nine predeclared quality checks, but its trajectory and later workload
diverge. The apparent 2.07% optimization reduction versus CPU summaries is
therefore complete-estimator timing with changed work, not a clean backend
speedup.

CUDA/reference position disagreement grows from 4.53e-10 m at zero-based row
883 to 5.39e-5 m at row 884, while recorded counts still agree at row 884.
The first rematch/LM-count differences occur at row 885; point/plane
correspondence counts first differ at rows 886/887, factor counts at row 1002,
and active poses at row 1014. Extracted feature counts agree throughout.
Maximum position disagreement over the full run is 0.12515 m, with RMS
0.03487 m. Internal ICP/LM decisions are not logged, so these observations do
not establish the first branching cause or a kernel defect.

CUDA mean translational RTE is 0.078254 m at 1 m and 0.716999 m at 30 m,
versus reference 0.077817/0.717931 m. All 1,724 available ground-truth poses
are matched. Reversed repetitions are running; preserve the frozen backend
and predeclared gates. Evidence:
`density-scaling-t32/quad_hard-current-first-cuda-comparison.json` and
`density-scaling-t32/quad_hard-current-first-divergence.json`. Provisional
first-CUDA workload scaling, with this changed-work caveat, is saved in
`density-scaling-t32/quad_hard-cuda-scaling-first-repeat.json`.

The second current CUDA run is complete (52/72). It preserves every recorded
workload and iteration count from the first original reference, with maximum
position disagreement below 5.41e-10 m. Thus the two CUDA repeats follow
different trajectories: their maximum position disagreement is 0.12515 m,
with the same changed-count totals reported for the first CUDA/reference pair.
Optimization is 43.304/43.859 ms (mean 43.582 ms), and complete-scan time is
137.027/136.607 ms. These averages mix different estimator work and are not
fixed-input timing variability. The second CUDA RTE is 0.077817/0.717931 m
at 1 m/30 m, close to the first reference. Second CPU-summary and reference
runs remain pending; the specific internal cause of divergence is unproven.
Evidence: `density-scaling-t32/quad_hard-current-cuda-repeat-agreement.json`.

Both current CPU-summary/CUDA pairs are complete (53/72). CPU-summary
optimization is 44.219/44.851 ms and CUDA is 43.304/43.859 ms. The second pair
preserves every recorded workload and iteration count, with maximum position
disagreement below 1.36e-10 m, and CUDA latency is 2.21% lower. CPU repetitions
also preserve all counts. Two-repeat means are 44.535 ms CPU summary and
43.582 ms CUDA (2.14% lower), but the first pair changes work as documented
above. This average must not be presented as repeated equivalent-work
acceleration. Complete-scan means are 157.417/136.817 ms, subject to the
matching-time attribution limitation. Final reference-paired quality gates are
pending the second original run. Evidence:
`density-scaling-t32/quad_hard-current-repeat-comparison.json`.

## Quad_hard completed validation

All 18 quad_hard runs are now complete and all accelerated/reference quality
gates pass. Current original-CPU optimization averages 70.853 ms, CPU summaries
44.535 ms, and CUDA 43.582 ms. Final reference repeat 2 exhibits the same
trajectory branch as CUDA repeat 1: every recorded workload count agrees and
maximum position disagreement is below 5.58e-10 m. Original repeat 1 instead
follows CUDA repeat 2 and both CPU-summary repeats. The two reference trajectories
differ by up to 0.12515 m, demonstrating repeat variability in the original CPU
estimator as well. Internal branching cause remains unproven.

Consequently, paired current/reference timings include changed work and should
not be interpreted as identical-work acceleration. All nine predeclared quality
checks still pass for each accelerated repeat. Full scaling, latency, accuracy,
and reproducibility details are in the
[quad_hard report](optimization-scaling-quad-hard.md); trace evidence is saved as
`density-scaling-t32/quad_hard-current-final-agreement.json`. The suite is at
54/72 verified runs, with maths_hard in progress.

## Maths_hard first dense reference

The first denser-feature original-CPU run is complete, bringing the suite to
55/72 verified runs. All 2,440 scans finish and all 2,438 available ground-truth
poses are matched. Excluding 20 startup scans, optimization averages 124.659 ms,
complete scan processing 396.214 ms, and matching 193.613 ms. Across all scans,
the mean workload is 33.610 active poses and 518,633 stored correspondences.
Mean translational RTE is 0.085509 m at 1 m and 0.916191 m at 30 m.

The corresponding CPU-summary run is active. Accelerated comparisons, repeated
measurements, and within-sequence scaling remain pending; this first reference
does not establish a CUDA benefit or the effect of denser selection on accuracy.
Evidence: `density-scaling-t32/maths_hard-features-reference-t32-r1.run.json`.

The first matching CPU-summary run is now complete (56/72). Optimization is
63.008 ms versus 124.659 ms for the original, a 1.978x speedup in this first
pair. Complete-scan time is 320.522 ms versus 396.214 ms; matching itself is
178.392 ms versus 193.613 ms despite unchanged matching code, so the entire
scan-time reduction cannot be assigned to optimization acceleration. All nine
quality gates pass, every recorded workload and iteration count agrees, and
maximum position disagreement is below 5.25e-11 m. Evidence:
`density-scaling-t32/maths_hard-features-first-cpu-agreement.json` and the
checkpoint's first paired quality comparison. CUDA repeat 1 is running;
repeatability and CUDA benefit remain pending.

The first dense maths_hard CUDA run is complete (57/72). Its optimization mean
is 63.160 ms versus 63.008 ms for CPU summaries: 0.24% higher, effectively tied
in this first pair. Both are about 1.97–1.98x faster than the original CPU.
Every recorded workload and iteration count matches the reference, all nine
quality checks pass, and maximum position disagreement is below 5.30e-11 m.

CUDA complete-scan time is 278.257 ms, with matching 136.975 ms. The matching
implementation is unchanged, so its reduction from CPU-summary 178.392 ms is
not established as a causal GPU benefit. In particular, the lower complete-scan
time must not be described as faster CUDA optimization when the optimization
measurements are tied. CUDA repeat 2 is running; repeated comparisons remain
pending. Evidence: `density-scaling-t32/maths_hard-features-first-cuda-comparison.json`.

Both dense maths_hard CUDA repetitions are complete (58/72). Optimization is
63.160/64.283 ms (mean 63.721 ms), and complete-scan processing is
278.257/284.797 ms. Every recorded workload and iteration count agrees across
the CUDA repeats, with maximum position disagreement below 1.55e-12 m. The
second CPU-summary run is active; second-reference quality gates and the final
repeated backend comparison remain pending. Evidence:
`density-scaling-t32/maths_hard-features-cuda-repeat-agreement.json`.

Both dense maths_hard CPU-summary/CUDA pairs are complete (59/72). CPU-summary
optimization is 63.008/63.426 ms, versus CUDA 63.160/64.283 ms. CUDA latency is
0.24%/1.35% higher, with two-repeat means 63.217 ms CPU and 63.721 ms CUDA
(0.80% higher). This workload therefore shows no CUDA optimization advantage
over CPU summaries in either repeat. CPU repetitions and the second CPU/CUDA
pair preserve all recorded workload and iteration counts; the latter's maximum
position disagreement is below 2.11e-12 m. Complete-scan means are
320.291/281.527 ms, subject to the unchanged-matching attribution limitation.
The final original-reference repeat is running; its paired quality gates remain
pending. Evidence: `density-scaling-t32/maths_hard-features-repeat-comparison.json`.

## Maths_hard dense workload validated

All six dense maths_hard runs are complete (60/72). Two-repeat optimization
means are 126.637 ms original CPU, 63.217 ms CPU summaries, and 63.721 ms CUDA.
Complete-scan means are 399.922/320.291/281.527 ms. Both accelerated backends
pass all nine predeclared quality checks in both repeats. Every recorded
workload and iteration count matches its paired reference, and maximum
accelerated/reference position disagreement is below 5.33e-11 m. Original
reference repetitions also preserve all counts, with position disagreement
below 2.80e-12 m.

CUDA has no optimization advantage over CPU summaries at this dense workload:
it is slightly slower in both repetitions. More-feature runs are now validated
on all four sequences. The maths_hard larger-window and current workloads remain
pending, so within-sequence scaling and accuracy tradeoffs cannot yet be
concluded. The larger-window reference run is active. Evidence:
`density-scaling-t32/maths_hard-features-final-comparison.json`,
`density-scaling-t32/maths_hard-features-final-agreement.json`, and the verified
checkpoint.

## Maths_hard first larger-window reference

The first larger-window original-CPU run is complete (61/72), covering all
2,440 scans and matching all 2,438 available ground-truth poses. Excluding
20 startup scans, optimization averages 115.338 ms, complete-scan processing
252.855 ms, and matching 91.603 ms. Mean active poses are 41.739 (maximum 43),
with 340,797 stored correspondences across all scans. Mean translational RTE
is 0.087143 m at 1 m and 0.919957 m at 30 m.

The corresponding CPU-summary run is active. Backend speedups, repetitions,
and comparison against the current maths_hard workload remain pending. These
first-reference numbers alone do not establish a CUDA benefit or an accuracy
improvement from increasing the window. Evidence:
`density-scaling-t32/maths_hard-window-reference-t32-r1.run.json`.

The first larger-window CPU-summary comparison is complete (62/72).
Optimization averages 75.523 ms versus original 115.338 ms, a 1.527x speedup
in this first pair. Complete-scan processing averages 201.059 ms versus
252.855 ms, with matching 79.365 ms versus 91.603 ms. Matching code is unchanged;
the entire scan-time difference cannot be attributed to optimization acceleration.
All nine quality checks pass and every recorded workload and iteration count
agrees with the reference. Maximum position disagreement is below 5.69e-11 m.
The first CUDA larger-window run is active; its combined preparation/solver
benefit and repeatability remain pending. Evidence:
`density-scaling-t32/maths_hard-window-first-cpu-comparison.json`.

The first larger-window CUDA run is complete (63/72). Optimization averages
65.931 ms versus CPU-summary 75.523 ms, a 12.70% latency reduction in this
first pair. This measures GPU summary preparation plus selective FP64 GPU
solving, not the isolated solver contribution. The run performs 75,712 GPU
solves with zero numerical fallbacks. All nine quality gates pass; every
recorded workload and iteration count matches the original reference, with
maximum position disagreement below 5.72e-11 m.

CUDA complete-scan processing averages 174.315 ms, including matching
61.202 ms. The unchanged-matching attribution limitation still applies to the
complete-scan comparison. CUDA repeat 2 is running; repeated speedup remains
pending. Evidence: `density-scaling-t32/maths_hard-window-first-cuda-comparison.json`.

Both larger-window maths_hard CUDA runs are complete (64/72). Optimization
is 65.931/67.851 ms (mean 66.891 ms), and complete-scan processing is
174.315/179.920 ms. Each run performs 75,712 GPU solves with zero numerical
fallbacks. Every recorded workload and iteration count agrees across CUDA
repetitions; maximum position disagreement is below 5.64e-12 m. The second
CPU-summary run is active. The final repeated speed comparison and second
reference-paired quality gates remain pending. Evidence:
`density-scaling-t32/maths_hard-window-cuda-repeat-agreement.json`.

Both larger-window maths_hard CPU-summary/CUDA pairs are complete (65/72).
CPU-summary optimization is 75.523/76.967 ms versus CUDA 65.931/67.851 ms:
CUDA latency is 12.70%/11.84% lower. Two-repeat means are 76.245 ms CPU and
66.891 ms CUDA, a 12.27% reduction. CPU repetitions and both CPU/CUDA timing
pairs preserve all recorded workload and iteration counts; maximum second-pair
position disagreement is below 4.57e-12 m. Complete-scan means are
202.396/177.118 ms, subject to the unchanged-matching attribution limitation.
The final original-reference repeat is active; second-reference quality gates
remain pending. Evidence: `density-scaling-t32/maths_hard-window-repeat-comparison.json`.

## Maths_hard larger-window workload validated

All six larger-window maths_hard runs are complete (66/72). Two-repeat
optimization means are 113.917 ms original CPU, 76.245 ms CPU summaries, and
66.891 ms CUDA. Complete-scan means are 249.065/202.396/177.118 ms. CUDA
optimization latency is 12.27% lower than CPU summaries, winning both repeats.
Both accelerated backends pass all nine predeclared quality checks in both
repetitions. Every recorded workload and iteration count matches its paired
reference, with maximum accelerated/reference position disagreement below
6.04e-11 m. Original reference repetitions also preserve all counts.

Both enlarged workloads are now validated across all four sequences. The six
current-setting maths_hard runs remain; their first original-reference replay
is active. Within-sequence scaling against current maths_hard settings and the
final all-sequence conclusions await that baseline and its repetitions.
Evidence: `density-scaling-t32/maths_hard-window-final-comparison.json`,
`density-scaling-t32/maths_hard-window-final-agreement.json`, and the verified
checkpoint.

## Maths_hard first reference workload comparison

The first current-setting original-CPU run is complete (67/72), enabling this
first-reference comparison. Timings exclude 20 startup scans; counts average
all 2,440 scans. Every workload matches all 2,438 available ground-truth poses.

| Workload | Mean poses | Mean correspondences | Optimization ms | Complete scan ms | Mean 1 m RTE m | Mean 30 m RTE m |
|---|---:|---:|---:|---:|---:|---:|
| Current | 33.67 | 205,906 | 77.887 | 213.129 | 0.088381 | 0.804187 |
| More features | 33.61 | 518,633 | 124.659 | 396.214 | 0.085509 | 0.916191 |
| Larger window | 41.74 | 340,797 | 115.338 | 252.855 | 0.087143 | 0.919957 |

Denser selection realizes 2.52x correspondences with nearly unchanged retained
poses. The larger window realizes 1.24x poses and 1.66x correspondences. Both
enlarged settings slightly reduce mean short-distance error but increase mean
30 m error by about 14%. These are first-reference findings; accelerated
current-setting runs and repetitions remain pending. Do not treat these as
final CUDA scaling ratios. The first current CPU-summary replay is active.
Evidence: `density-scaling-t32/maths_hard-reference-workload-first-comparison.json`.

The first current-setting maths_hard CPU-summary comparison is complete (68/72).
Optimization averages 51.774 ms versus original 77.887 ms, a 1.504x speedup in
this first pair. Complete-scan processing is 178.059 ms versus 213.129 ms, with
matching 71.744 ms versus 78.477 ms despite unchanged matching code. All nine
quality gates pass and every recorded workload and iteration count matches the
reference. Maximum position disagreement is below 4.38e-11 m. The current CUDA
run is active; repeated backend comparisons remain pending. Evidence:
`density-scaling-t32/maths_hard-current-first-cpu-comparison.json`.

The first current-setting maths_hard CUDA comparison is complete (69/72).
CUDA optimization averages 53.690 ms versus CPU-summary 51.774 ms, 3.70%
higher latency in this first pair. All nine quality checks pass and every
recorded workload and iteration count matches the reference, with maximum
position disagreement below 7.55e-11 m. CUDA complete-scan time is 168.341 ms,
including matching 58.068 ms; the unchanged-matching attribution limitation
applies, so this lower scan time is not a CUDA optimization speedup. Current
settings retain CPU solving. CUDA repeat 2 is active; final repeated comparisons
remain pending. Evidence:
`density-scaling-t32/maths_hard-current-first-cuda-comparison.json`.

Both current-setting maths_hard CUDA repeats are complete (70/72).
Optimization is 53.690/52.594 ms (mean 53.142 ms), and complete-scan processing
is 168.341/162.131 ms. Every recorded workload and iteration count agrees
between CUDA repeats; maximum position disagreement is below 6.86e-11 m.
The final CPU-summary repeat is active. The second paired backend comparison
and final reference-paired quality checks remain pending. Evidence:
`density-scaling-t32/maths_hard-current-cuda-repeat-agreement.json`.

Both current-setting maths_hard CPU-summary/CUDA pairs are complete (71/72).
CPU-summary optimization is 51.774/51.790 ms versus CUDA 53.690/52.594 ms;
CUDA latency is 3.70%/1.55% higher. Two-repeat means are 51.782 ms CPU and
53.142 ms CUDA, a 2.63% increase. CPU repetitions and both CPU/CUDA pairs
preserve all recorded workload and iteration counts; maximum second-pair
position disagreement is below 3.62e-12 m. Complete-scan means are
177.831/165.236 ms, subject to the unchanged-matching attribution limitation.
Only the final original-reference repeat remains; second-reference quality
gates are pending. Evidence:
`density-scaling-t32/maths_hard-current-repeat-comparison.json`.

## Final suite verification

All 72 runs are complete. All 36 backend/workload/sequence groups are validated,
and all 432 paired quality checks were independently recomputed and pass.
The final maths_hard current-reference repeat preserves all workload counts;
its paired accelerated trajectories remain within 7.55e-11 m across both repeats.
Final current optimization means are 78.851 ms original CPU, 51.782 ms CPU
summaries, and 53.142 ms CUDA. The full suite exporter verified output hashes,
scan counts, timestamps, and experiment fingerprints. Frozen binary, source,
and build-metadata hashes match; only documentation changed since freezing.
All 38 final C++ tests pass. See the
[consolidated report](optimization-acceleration-results.md),
[full measurements](../benchmarks/results/density-scaling-t32/final.md), and
`benchmarks/results/final-verification/` for the final audit evidence. Historical
pending statements above describe their original checkpoints and are superseded
by this final result.
