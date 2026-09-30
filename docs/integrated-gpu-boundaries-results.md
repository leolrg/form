# Integrated GPU masks, selection, map and materialization

Investigation on 2026-09-30, A100 80 GB PCIe / 32-vCPU EPYC-Milan VM,
CUDA 13.0, GCC/libstdc++ 13; branch `codex/optimization-acceleration`,
base `1255543137f67ac54aea5434b9ef6093f1b9fe02` plus the archived source changes.
All new backends are opt-in; the default estimator is unchanged.

## Current-workload decision

**Move masks and selection to GPU with curvature and normals. Keep the current
CPU map/materialization path for the default workload.** The latter decision
reflects measured integration cost, not an inherent inability to construct maps
on GPU. GPU map construction is faster in isolation, but its current compatible
end-to-end path does not consistently improve on the resident frontend with a
CPU map. Do not describe a 1% difference between two noisy repeats as proof that
CPU map construction is intrinsically faster.

Corrected full-sequence runs, mean milliseconds per scan:

| Backend | Extraction | Map | Matching | Total | Total repeat means |
| --- | ---: | ---: | ---: | ---: | --- |
| GPU normals; CPU masks/selection/map/materialization | 6.264 | 3.930 | 12.512 | 41.399 | 40.892 / 41.906 |
| GPU frontend; CPU map/materialization | **2.638** | 4.133 | 12.622 | **37.764** | 37.563 / 37.964 |
| GPU frontend + map + gathering | 3.000 | **2.868** | 13.194 | 38.129 | 39.557 / 36.701 |

Resident masks/selection reduce complete extraction time by **57.9%** and total
latency by **8.8%** relative to the GPU-normal control. Both repeats improve.
The additional GPU map/gathering path reduces the map stage by **30.6%**, but
its total is **1.0% higher on average**, with opposite signs across repeats.
The extraction-only port reproduces the paired control's saved trajectories,
objectives and every tracked workload count exactly. The GPU map path retains
all counts and has maximum position difference **2.304e-13 m**. Both repetitions
of both corrected variants pass the full trajectory quality gate.

## What was implemented

`cuda-normals` is the corrected existing hybrid control: GPU curvature,
adjacent-row search and normal computation, CPU masks and ordered selection,
CPU world-map construction and final correspondence materialization. It already
uses the existing GPU matcher, QR summaries and dimension-dependent optimizer.

`cuda-frontend` keeps the scan, masks, curvature, sorting, ordered suppression,
selected indices, nearest-row search and normals on GPU. Independent sectors
are sorted in parallel; independent rows are selected in parallel, preserving
serial suppression within each row. Only completed feature indices and normals
return to the host adapter. Selection remains ordered even though it executes
on GPU. CPU TBB parallelism across scanlines did not make CPU placement necessary.

`cuda-materialization` adds GPU gathering of matched target coordinates,
normals, distances and accepted-query group order, retaining the CPU world map.
It preserves full host matches, raw arrays, deferred-factor ownership and API
behavior. It downloads 64 bytes/query rather than the original 16 bytes/query.

`cuda-map` adds GPU world transforms, voxel grouping and lookup construction
from retained local features, plus GPU gathering. Stable voxel grouping preserves
within-voxel insertion order, and nearest-neighbor traversal preserves its tie
rule. The local→world→local arithmetic path is retained. GPU map construction
moves work into snapshot preparation; comparing the vanished CPU world-map
counter alone would overstate the saving.

`cuda-pipeline` combines the resident frontend with GPU map construction and
gathering. Scan bookkeeping, host feature/raw-match objects and small optimizer
systems still use CPU. This is not a completely device-resident estimator.

## A rejected selection change and its correction

The first port sorted equal curvatures by original index. A matched CPU variant
(`cuda-stable-selection`) and the GPU variant produced identical trajectories,
but both failed the full original-order quality gate. Mean 30 m translation RTE
rose from **0.236248 to 0.438342 m**, and maximum trajectory separation reached
**0.478807 m**. Selected membership had changed on only five current-density
scans in the preceding all-scan probe; this still changed later optimization
and map updates substantially. The failed run's speedup is not a valid
quality-preserving acceleration result.

The corrected GPU sorter reproduces the original serial libstdc++ 13 introsort
permutation, including equal keys, partition traversal and heap fallback.
Its compatibility policy is explicitly restricted to that verified host library;
unsupported libraries/parallel-sort policies reject the original-order path.
The optional index-tie mode remains available as an explicit behavior change.
GPU sectors currently support at most 1024 samples.

The helper's upstream notices and license texts are included alongside the
implementation. Tests compare full sort and heap permutations for every size
1–1024 across all-equal, ascending, descending, organ-pipe, repeated and random
keys, and compare float/double resident extraction with CPU selection, including
1024-sample sectors, masks, zero quotas, ragged sectors and invalid-key recovery.

## Measurement and reproduction

Full runs replay all 1190 stairs scans; timing excludes the first 20 scans.
Every clean campaign has two serial repetitions, reversing backend order for
the second. No builds, sanitizers or competing benchmark jobs ran during clean
measurement. Instrumented profiles are kept separate. Dense-feature runs use
spacing 2 and caps 6/100 rather than spacing 5 and caps 3/50, with the same
recent-window cap 10. A 250-scan prefix does not establish full dense trajectory
quality or 30 m error.

The first frozen binary, SHA-256
`0a334b0b9f8d42f4df1d01db29955f571d177211f05390928a7677b3bb06845b`,
used changed index-tie semantics in `cuda-frontend` and `cuda-pipeline`.
Its full experiment is retained in `benchmarks/results/integrated-boundaries-full/`.
Map/materialization-only backends used the original selector and passed quality.

The corrected frozen binary, SHA-256
`1344a63446b1251a4e20faae8d6d326ddebbc693eccb023bce2cfde5b05d5c79`,
is in `benchmarks/results/integrated-boundaries-exact-build/`, alongside source
hashes, the source archive/diff and validation logs. The analyzer verifies
campaign completeness and fingerprints before comparing timing, workload counts
and trajectories. The predeclared translation gate permits each 1 m/30 m
mean/median/RMSE/max to increase by at most `max(5% of reference, 0.01 m)`;
matched-pose coverage must not decrease. Rotation is reported, without a
predeclared acceptance threshold.

## Isolating map construction and materialization

The first full campaign provides valid original-selector controls for these
three backends. Mean milliseconds per scan across two reversed-order repeats:

| Backend | Extraction | Map | Matching, including final materialization | Total |
| --- | ---: | ---: | ---: | ---: |
| GPU normals; CPU map/materialization | 6.061 | 3.883 | 12.221 | 40.804 |
| Add GPU final gathering only | 7.153 | 4.194 | 13.316 | 43.908 |
| Add GPU map construction and gathering | 7.268 | 3.024 | 13.332 | 42.573 |

All per-scan workload counts agree. Gathering-only trajectories match exactly;
GPU-map maximum position difference is 2.304e-13 m. Both pass the full quality
gate. Gathering alone regresses total time by 7.6%; adding GPU map construction
recovers some time, but remains 4.3% above this control. The map stage itself
improves by 22.1% relative to the control. Whole-pipeline latency, not the isolated
map-stage improvement, governs the placement decision.

Extraction is unchanged in these variants yet its measured time varies. The
repeat totals for gathering were 45.529/42.288 ms, and for map+gathering
42.838/42.308 ms. These are two repetitions on a shared VM, not confidence bounds
or a universal CPU/GPU break-even claim. The corrected-frontend campaign gives
a second integration context, with original selection now resident on GPU.

## Why final materialization regresses

Separate instrumented 250-scan current prefixes, two reversed repeats; mean
milliseconds per scan. These profiles explain stages and are not mixed with
unprofiled full-sequence latency:

| Scope | CPU map/materialization | GPU gathering only | GPU map + gathering |
| --- | ---: | ---: | ---: |
| CPU world map | 2.105 | 2.141 | 0.001 |
| Matcher snapshot, including GPU map where enabled | 2.317 | 2.307 | 2.981 |
| **Final materialization** | **2.641** | **3.412** | **3.735** |
| Download host scope | 0.214 | 1.091 | 1.317 |
| Host match reconstruction | 0.932 | 0.950 | 0.955 |
| Host group construction | 0.212 | 0.053 | 0.064 |

GPU gathering cuts host grouping by about 0.16 ms, but does not materially
reduce the roughly 0.95 ms host reconstruction stage. That stage still creates
full Match objects and writes their storage. Per-factor raw arrays and output
copies also remain. Removing the CPU inverse transforms is therefore a much
smaller opportunity than removing all host materialization.

Meanwhile the download scope grows by about 0.88 ms for gathering alone.
It includes host-vector allocation/initialization, the gather kernel launch,
a larger pageable-host download, stream synchronization and host validation;
it is **not an isolated PCIe transfer measurement**. Each query returns 64 bytes
rather than 16, including normal fields unused for point matches. That is roughly
0.86 MiB extra per scan at the full current workload's mean feature count. These
added boundary costs outweigh the modest grouping savings. Nested `match_rows`
and `match_ensure` counters must not be added to their enclosing scopes.

There are still optimization opportunities in this prototype. The GPU map
builder performs a redundant ordinary-reset coordinate extraction/synchronization
before its actual build, uses three radix-sort passes, and sizes its hash table
by point count rather than occupied voxel count. A more device-resident factor
API could avoid host raw objects entirely. These results justify the current
placement decision; they do not prove that every possible GPU map/materialization
implementation is slower.

## Denser workload pilot

Two reversed-order 250-scan runs, mean milliseconds per scan:

| Backend | Extraction | Map | Matching | Total |
| --- | ---: | ---: | ---: | ---: |
| GPU normals; CPU masks/selection/map/materialization | 7.223 | 10.247 | 20.474 | 44.858 |
| GPU frontend; CPU map/materialization | 3.712 | 10.247 | 20.247 | 41.048 |
| GPU final gathering only | 8.738 | 10.442 | 21.448 | 47.633 |
| GPU map and gathering | 7.995 | 5.870 | 21.192 | 41.665 |
| GPU frontend + map + gathering | 4.155 | 6.170 | 21.299 | **38.400** |

At this higher density, moving map construction does recover enough work to
pay for its integration: the combined pipeline is 6.5% faster than the resident
frontend with CPU map, and 14.4% faster than the GPU-normal control. Both repeats
improve for those comparisons. Gathering alone still regresses.

All workload counts agree with the paired control; GPU frontend and gathering
alone reproduce the prefix trajectory exactly, and GPU-map maximum separation
is 7.619e-14 m. However, this prefix contains no 30 m segments: the complete
quality gate is **missing**, not passed. This establishes a scaling candidate,
not a validated full-dense default or a measured switching threshold. More
GPU residence is justified by larger workloads here; it is not automatically
better for the current default.

## Verification

- CUDA CTest: **119/119 entries passed**, including the scalar-Eigen suite.
- CPU-only CTest: **50 passed, eight CUDA-dependent entries skipped**.
- Benchmark runner Python tests: **10 passed** in the evalio environment.
- Python extension rebuilt; extraction flags round-trip correctly.
- Corrected resident selector: four tests each under Compute Sanitizer memcheck
  and racecheck, zero errors/hazards. Earlier integrated map/materialization and
  selection tests also passed both sanitizers (ten tests per tool).
- Read-only code review covered map ownership/order/error handling and the
  original-order introsort correction. Review findings were addressed with
  invalid-key gating, grouped-result invalidation, host-policy restrictions,
  direct heap tests and maximum-sector GPU coverage.
- Both corrected full runs also pass against the saved original CPU reference,
  with every tracked workload count unchanged and maximum position separation
  below **8.9e-9 m**. See `original-cpu-quality.json` in the full-run directory.

## Reproduction

The frozen binary directories contain executable, source diff/archive, source
hashes and test logs. Each campaign's `suite.json` and `*.run.json` record binary
fingerprints, exact commands, workload settings, hardware/software provenance,
quality metrics and per-stage timing distributions. Use a fresh output directory;
the runner refuses to silently overwrite incompatible artifacts.

```sh
python benchmarks/run_suite.py \
  --binary benchmarks/results/integrated-boundaries-exact-build/form-replay \
  --output benchmarks/results/reproduce-boundaries-full \
  --sequences stairs --configs current --threads 32 --repeats 2 \
  --backends cuda-normals cuda-frontend cuda-pipeline \
  --cuda-solve-min-dimension 240 --gpu-sample-interval 0
```

For dense timing use `--configs features --limit 250` and include
`cuda-materialization cuda-map` among backends. For the separate current profiles
use `--configs current --limit 250 --profile` with
`cuda-normals cuda-materialization cuda-map`. Run with a Python environment
containing the benchmark dependencies, including evalio.

```sh
python benchmarks/analyze_integrated_boundaries.py \
  benchmarks/results/integrated-boundaries-exact-full \
  benchmarks/results/integrated-boundaries-exact-dense \
  benchmarks/results/integrated-boundaries-exact-profile \
  --output benchmarks/results/integrated-boundaries-comparison.json
```

Do not claim that every CPU operation has been proven faster there. Small
optimizer systems already have separate forced-placement evidence in
[detailed diagnostics](detailed-diagnostics-results.md). Host control and
supported host-facing factor objects remain interface responsibilities. The
new evidence specifically rejects CPU masks/selection as a necessary performance
boundary and supports retaining the current map/materialization boundary at the
default density, with a promising GPU-map option for larger workloads.
