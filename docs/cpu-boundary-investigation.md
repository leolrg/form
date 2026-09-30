# Masks, selection, map construction and materialization

Follow-up: [integrated GPU boundary results](integrated-gpu-boundaries-results.md)
now test masks/selection and map/materialization inside the full estimator.
That report supersedes the placement questions left open here; the measurements
below remain the earlier experiment.

Investigation on 2026-09-30, A100 80 GB PCIe / 32-vCPU EPYC-Milan VM, CUDA 13.0.
Baseline: the corrected `cuda-normals` implementation over `1255543` on
`codex/optimization-acceleration`. Production behavior is unchanged by this
investigation. The new CUDA program is an isolated diagnostic prototype, not an
integrated backend or a demonstrated estimator speedup.

The evidence supports pursuing a GPU-resident extraction pipeline. Masks and
ordered suppression can execute on GPU; equal-curvature sort ordering needs an
explicit compatibility decision. Map/materialization has a larger measured host
budget, with redundant representations and work already performed on GPU. Its
CPU placement has not been shown faster by a CPU/GPU implementation comparison.

## Existing production costs, after GPU normals

These are mean milliseconds per scan from two separate instrumented 250-scan
replays per workload, excluding the first 20 scans. They are current
`cuda-normals` profiles, not timings from the standalone prototype.

| Stage | Current | Dense features |
| --- | ---: | ---: |
| Validity mask | 0.539 | 0.539 |
| Planar sort/selection | 0.941 | 1.068 |
| Point mask and planar-exclusion pass | 0.706 | 0.712 |
| Point selection | 0.403 | 0.419 |
| **Masks + selection** | **2.589** | **2.737** |
| CPU world-map construction | 2.143 | 4.786 |
| Matcher snapshot preparation | 2.488 | 5.580 |
| Final raw materialization | 3.031 | 7.933 |
| **Map + snapshot + materialization** | **7.662** | **18.299** |

The two bold totals cover distinct work, but neither is a predicted removable
cost. Dense uses spacing 2 and caps 6/100, current spacing 5 and caps 3/50; both
use recent-window cap 10. These prefixes are not full-sequence latency/quality
measurements. Source: `benchmarks/results/cuda-normals-corrected-profile/`.

## What the extraction probe established

The standalone [probe](../benchmarks/extraction_boundary_probe.cu) mirrors the
current float range checks, curvature calculation, row/sector order, suppression,
point sampling, and existing `count > cap` stopping rule. It uses two parallel
mask kernels, a 256-thread bitonic sort per sector, and a block per scanline
whose first thread executes the dependent greedy selection. Thus a serial
dependency within a row does not require executing the row on CPU.

The all-scan correctness audit covers all 1,190 stairs scans at both densities
(2,380 configurations):

- GPU validity and point masks match the CPU reference everywhere.
- GPU greedy selection with the **original CPU-sorted indices** matches the CPU
  reference's ordered planar and point indices in every configuration.
- GPU sorting with an explicit `(curvature, original index)` order matches the
  CPU reference using that same total order in every configuration.
- That total order changes the original selector's ordered output on **6 current
  scans and 39 dense scans**. It changes actual selected membership on **5 current
  scans and 19 dense scans**; the remaining differences are permutations.

This is pre-normal candidate selection parity against a source-mirrored CPU
reference, not a full estimator trajectory check. No altered-sort variant has
passed an integrated quality gate. The current `std::sort` comparator only tests
curvature, so it does not specify an equal-key order. Suppression makes that order
observable. This is a semantic compatibility issue, not evidence of GPU slowness.

Timing sampled scans 0,20,...,1180 (60 scans), eight measured repetitions after
two warmups per scan/configuration, with operation order reversed on alternate
repeats. Mean microseconds:

| Isolated prototype scope | Current | Dense |
| --- | ---: | ---: |
| CPU prototype, both masks | 597.9 | 598.3 |
| CPU prototype, masks + sorting + selection | 2312.3 | 2383.3 |
| GPU masks, resident input, completed work | 19.1 | 19.3 |
| GPU masks, scan upload + mask downloads included | 575.7 | 587.5 |
| GPU masks + greedy selection, CPU-sorted indices already resident | 405.9 | 543.6 |
| GPU masks + index-stable sort + selection, resident input | 335.7 | 472.2 |
| GPU masks + index-stable sort + selection, input/output copies included | 1043.2 | 1251.1 |

Resident scopes include device synchronization but exclude input uploads/output
downloads. The ordered-selection scope excludes CPU sorting and sorted-index
upload. Its timing must not be treated as a complete semantics-preserving port.
The stable-sort GPU timings include sorting, but use the altered equal-key order.
The two paths have different preceding cache activity, so their difference does
not isolate sort cost.

All selection scopes start with precomputed curvature: curvature arithmetic,
normal construction and feature-object packing are excluded. The copy-inclusive
stable path uploads scan and precomputed curvature and downloads padded selected
index buffers. A future integrated path could retain existing scan/curvature
buffers and feed selected indices directly to GPU normals.

CPU timings are **prototype timings**, not production benchmarks: the mirror
copies curvature keys before sorting, uses smaller keys, combines the two range
checks, and has different temporary/output allocation and mask passes. Do not
divide these times into a production CPU/GPU speedup claim. The production table
above provides the separate current cost context. The preliminary sampled run
is retained separately; the table uses `extraction-sampled-final.csv`.

The masks-only result illustrates why the boundary matters: the transfer-inclusive
boundary, including host allocation and CUDA runtime overhead, dominates measured
latency. Keeping masks and selection with curvature and normals on-device is the
more useful experiment.
Planar selection currently uses TBB across rows; point selection still traverses
rows sequentially on CPU. No measurement establishes that either belongs on CPU.

## Map and materialization: where the host work comes from

The current code performs this sequence:

1. `KeypointMap::to_voxel_map` transforms retained local features and normals into
   world coordinates on CPU and inserts them into CPU voxel vectors.
2. `Snapshot::reset` traverses that map, copies world features into retained host
   metadata and CUDA upload records, packs queries, and builds pose-index arrays.
   `CudaMatcher::reset` validates records, builds a second lookup table on CPU,
   uploads it, and prepares search coordinates/local target data on GPU.
3. GPU matching already groups correspondences and builds compact QR summaries.
4. At the final rematch, results are downloaded as target indices/distances.
   `MatchBatch::ensure` reconstructs every full host match, transforms matched
   targets back to local coordinates on CPU, and groups accepted results again.
5. Per-group raw coordinate arrays are filled; full matches are copied into TBB
   concurrent vectors; map insertion reads only their query features/distances.

The snapshot packing host scope is **1.665 / 4.118 ms** (current/dense), within
the total **2.488 / 5.580 ms** snapshot scope. Final materialization includes only
**0.237 / 0.573 ms** in the result-download host scope, while target reconstruction
costs **1.045 / 2.927 ms** and host regrouping **0.159 / 0.369 ms**. Download and
reconstruction occur twice per scan, once for each feature type. These timings
include waits where applicable; download host time is not isolated PCIe time.
Nested `match_rows` and `match_ensure` counters must not be added to their parents.

These observations identify duplicate work, but do not prove a GPU replacement
will recover the whole budget. GPU reconstruction would transfer more data;
avoiding reconstruction can be preferable when its consumers need only summaries
or query/distance pairs.

### Which raw data is actually required?

For the measured summary-enabled backend with valid caches, full/semi
optimization, batching, freezing and marginalization consume `FeatureSummary`.
Connection/workload counts use deferred counts. Map insertion only reads
`match.query` (including scan ID) and `match.dist_sqrd`. None of those operations
requires the eagerly expanded raw correspondence arrays.

Raw arrays are nevertheless part of supported behavior: direct factor evaluation,
summary-disabled operation, invalidated-summary rebuilds, mutation, and callers
retaining correspondence objects. Therefore deleting their construction would
change that contract. Merely deleting the eager `ensureRaw()` loop is also
insufficient: full-match output still calls `MatchBatch::ensure`, and deferred
loaders keep a shared pointer to each batch's entire `HostMap`. Subsequent scans
then retain old map snapshots and allocate new ones. A lazy-only change could
increase memory while retaining the reconstruction cost.

### Concrete implementation options

**Extraction:** Keep scan, masks, curvature, selected indices and normals together
on GPU, returning only the final feature data required by the current adapter.
Either reproduce the current CPU equal-key permutation, or introduce a documented
total order in both CPU and GPU variants and benchmark against a matched CPU
variant plus original FORM quality. The latter gives reproducible semantics but
is a behavior change requiring full trajectory validation. Do not assume a stable
GPU sort is a drop-in replacement.

**Materialization:** Separate map-insertion output from raw-factor retention.
Produce compact query/distance or insertion-index output, and retain raw data in
a compact owned representation that can satisfy `ensureRaw()` without keeping
whole map snapshots. A compatible first experiment could gather final per-group
raw rows on GPU using existing local targets/group indices, download them once,
and avoid rebuilding target objects and regrouping on CPU. Measure the larger
transfer and memory cost against saved reconstruction. A more aggressive
summary-only mode would need an explicit API contract.

**World map:** Gather retained local features with an ordinal preserving current
scan/point iteration order, transform/bin on GPU, group voxels stably, and feed
the GPU matcher directly. Preserve point order within each voxel and the existing
27-neighbor traversal tie rule. Preserve floating-point transform/division/floor
behavior at voxel boundaries. Using original local coordinates instead of the
current local→world→local round trip also needs numerical validation. Pose updates,
scan insertion/removal and retained matcher snapshots need explicit invalidation
and ownership rules; caching yesterday's world coordinates is not sufficient.

These are concrete candidate designs, not implemented map-backend comparisons.
The reason the current map/materialization stays on CPU is its data model and
compatibility path. There is presently no measurement supporting a claim that
the corresponding work is inherently faster on CPU.

## Validation and reproduction

The prototype preserves production defaults and does not alter the prior normal
implementation. Synthetic checks exercise edges, invalid-range propagation,
fourth-coordinate range contribution, NaN comparison behavior and curvature ties.
Nonfinite real curvature keys are rejected by the probe; arbitrary sensor types,
Eigen reduction policies, shapes beyond its 256-element sector sort, and parameter
combinations are outside its scope. It is not a general replacement extractor.
The all-scan audit exits on any mask/ordered-selection/stable-reference mismatch.
Memory and race sanitizers passed synthetic cases plus the first real scan at
both densities, with zero errors/hazards. They were separate from timing runs.

```bash
mkdir -p benchmarks/results/cpu-boundary-investigation
nvcc -O3 -std=c++17 -arch=sm_80 --fmad=false \
  benchmarks/extraction_boundary_probe.cu -ltbb \
  -o benchmarks/results/cpu-boundary-investigation/extraction-probe
benchmarks/results/cpu-boundary-investigation/extraction-probe \
  /home/ubuntu/datasets/form-input/stairs.formpc 1190 1 0 \
  > benchmarks/results/cpu-boundary-investigation/extraction-all-parity.csv
benchmarks/results/cpu-boundary-investigation/extraction-probe \
  /home/ubuntu/datasets/form-input/stairs.formpc 1190 20 8 \
  > benchmarks/results/cpu-boundary-investigation/extraction-sampled-final.csv
python3 benchmarks/analyze_cpu_boundaries.py \
  --profiles benchmarks/results/cuda-normals-corrected-profile \
  --probe benchmarks/results/cpu-boundary-investigation \
  --output benchmarks/results/cpu-boundary-investigation/analysis.json
```

The ignored result directory contains frozen source/binary fingerprints in
`manifest.json`, CSVs, parity summaries, sanitizer logs, and `analysis.json` with
profile means and input hashes. Use a fresh output directory for new runs to
preserve the recorded experiment. CUDA/CPU prototype timing does not establish
trajectory quality, an exact CPU/GPU crossover, or gains on another GPU.
