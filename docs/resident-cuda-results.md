# Resident CPU/CUDA optimization results

The resident matrix architecture improves the larger-window CUDA optimization
from **37.31 to 30.52 ms** (18.2% less time) in the paired pilot. The current-size
all-CUDA path is slightly slower than the preceding CUDA batch backend, while
the new direct CPU optimizer improves that case. An explicit hybrid follow-up
combines CUDA summary preparation with CPU matrix processing for small systems.
That hybrid measures **9.27 / 12.95 / 30.21 ms** at current / denser-point /
larger-window settings, improving on paired direct CPU by **14% / 45% / 20%**.

## Measurement scope

- NVIDIA A100 80 GB PCIe; 32-vCPU EPYC Milan VM; 32 TBB threads.
- Newer College 2021 `stairs`, first 250 prepared scans, two reversed-order
  repeats. All comparisons use identical input and workload settings per pair.
- Optimization latency is `semi_ms + full_ms`; excludes marginalization,
  matching, extraction, replay loading and serialization. First 20 scans are
  excluded from latency statistics. Table entries are means across both repeats.
- FP64 throughout; original correspondence weights, objective, LM tolerances,
  damping policies and pose retraction are preserved. No workload reduction.
- These are screening runs, not full four-sequence acceptance: the prefix has
  no 30 m RTE segments. The complete predeclared quality gate remains unfulfilled.

## All-CUDA architecture comparison

Frozen implementation: `60dfaee`. All six backends were rerun together: 36 runs.
Previous CUDA backends use their existing selective solver threshold of 240 only
for the larger-window setting. `cuda-resident` keeps its matrix pipeline on GPU
at every size in this experiment.

| Workload | Original CPU | Previous CPU batch | Previous CUDA | Previous CUDA batch | Direct CPU LM | Resident CUDA LM |
|---|---:|---:|---:|---:|---:|---:|
| Current | 37.82 | 13.94 | 12.30 | 11.98 | 10.74 | 12.39 |
| More points | 107.44 | 25.73 | 15.45 | 15.10 | 23.79 | 15.32 |
| Larger window | 111.64 | 46.57 | 39.06 | 37.31 | 39.92 | 30.52 |

All latencies above are **ms per scan for optimization**. Relative to the previous
CUDA batch path, resident CUDA changes current latency by +3.4%, denser-point
latency by +1.5%, and larger-window latency by -18.2%. The direct CPU comparison
matters: resident CUDA is slower at current size, 35.6% faster with denser points,
and 23.5% faster with the larger window. These percentages describe elapsed-time
reductions, not speedup ratios.

| Workload | Original CPU growth | Direct CPU LM growth | Resident CUDA LM growth |
|---|---:|---:|---:|
| Current | 1.00× | 1.00× | 1.00× |
| More points | 2.84× | 2.21× | 1.24× |
| Larger window | 2.95× | 3.72× | 2.46× |

Each column's growth is relative to **its own current-size latency**.

## Hybrid follow-up: paired CPU scaling

Frozen implementation: `380bcf8`; 18 additional runs with fresh original CPU and
direct CPU controls. The same 250-scan prefix, settings, two reversed-order
repeats and warmup exclusion apply. Speedup claims here use paired runs within
this second pilot. The preceding six-backend pilot remains a separate experiment.

The hybrid uses CUDA QR preparation at every size. Below 240 scalar variables it
keeps the entire matrix pipeline on CPU; at/above 240 it uses resident CUDA.
Current and denser-point runs perform no CUDA dense solves. The larger-window
run averages 18.25 CUDA solves among 19.78 LM iterations per measured scan, with
zero fallbacks. The threshold is an explicit policy, not an autotuned optimum.

| Workload | Original CPU (ms) | Direct CPU LM (ms) | CUDA hybrid (ms) | Hybrid time reduction vs direct CPU |
|---|---:|---:|---:|---:|
| Current | 38.58 | 10.82 | 9.27 | 14.3% |
| More points | 104.59 | 23.57 | 12.95 | 45.1% |
| Larger window | 111.74 | 37.93 | 30.21 | 20.4% |

| Workload | Original CPU growth | Direct CPU growth | CUDA hybrid growth |
|---|---:|---:|---:|
| Current | 1.00× | 1.00× | 1.00× |
| More points | 2.71× | 2.18× | 1.40× |
| Larger window | 2.90× | 3.50× | 3.26× |

Growth is relative to each backend's own current latency. Both increased points
and increased windows are compared with CPU; the lower hybrid baseline is retained
when computing its growth. Hybrid speedup over original CPU is 4.16× / 8.08× /
3.70× across the three workloads. Those ratios include the earlier summary
acceleration as well as the new architecture.

The hybrid repeat means are 9.28/9.26 ms (current), 12.45/13.46 ms (more points),
and 30.24/30.18 ms (larger window). Two repeats establish screening evidence;
they do not establish a portable GPU crossover or broad statistical confidence.

For context, total estimator processing includes all unaccelerated stages:

| Workload | Original CPU total (ms) | Direct CPU total (ms) | Hybrid total (ms) |
|---|---:|---:|---:|
| Current | 112.55 | 74.75 | 63.68 |
| More points | 313.58 | 218.24 | 180.56 |
| Larger window | 186.11 | 104.69 | 89.09 |

Total-stage differences include variation outside optimization and should not be
attributed entirely to this change. Denser points still exceed a 100 ms scan
budget: optimization is about 13 ms of the roughly 181 ms processing time.
Further whole-estimator gains in that workload require work outside this stage.

The 12 accelerated/reference pairs in this second pilot have identical workload
and LM iteration counts. Maximum position difference is 4.15e-12 m. All available
1 m translation and coverage checks pass; 30 m checks are still unavailable.
Both new CPU/GPU transition tests also pass Compute Sanitizer memcheck and
racecheck with zero errors or hazards.

- [Hybrid machine-readable results](../benchmarks/results/resident-hybrid-pilot/final.json)
  and [stage/repeat tables](../benchmarks/results/resident-hybrid-pilot/final.md).
- Frozen binary/source: `benchmarks/results/resident-hybrid-backend/`; replay
  SHA256 `7435c3512be618ad06fbcdf120648cfb093820b7ca6caaa611688c6b1ba5794c`.
- [Reproduction and backend flags](../benchmarks/README.md#resident-matrix-pipeline).

## Realized workload

The settings change independently: current uses point/plane caps 3/50, spacing
5 and recent-window cap 10; denser points use 6/100 and spacing 2, keeping window
10; larger window uses cap 40 with the original point selection. Means below
include all 250 scans; actual retained poses also depend on keyframing.

| Workload | Mean selected point + plane features | Mean stored correspondences | Mean poses | Mean factors |
|---|---:|---:|---:|---:|
| Current | 18759 | 168122 | 14.13 | 98.56 |
| More points | 49624 | 493474 | 14.26 | 102.12 |
| Larger window | 18759 | 520164 | 38.56 | 762.90 |

The denser selection realizes 2.94× as many stored correspondences; the larger
window realizes 3.09× as many correspondences and 2.73× as many poses. These are
measured workload increases; the nominal window-cap increase is not substituted
for the actual number of optimization variables.

## What is resident, and what remains on CPU

At reset, the optimizer classifies exact summary factors, frozen Hessian factors,
and other nonlinear factors. CUDA retains roots, frozen G, assembly maps, the
undamped model, and damping/factorization storage. Each linearization uploads
poses, frozen-anchor displacements and auxiliary contributions. Damping, solve,
and linear-model errors consume the device matrix. Trial nonlinear cost uses
separate scratch so a rejected step can retry without rebuilding or transferring
the model. The CPU receives the step, scalar costs and status.

The direct CPU comparator uses the same decomposition and preassembles frozen G,
with Eigen Cholesky. Both paths retain GTSAM 4.2's LM state rules, acceptance,
convergence and iteration hooks. Exact subclasses keep virtual dispatch.
Constrained auxiliary Jacobians and other unsupported optimizer modes fail
explicitly instead of silently changing their semantics.

Pose retraction and LM decisions still execute on CPU, as do auxiliary factors,
reset/packing, previous-match freezing and marginalization. QR compression is
unchanged and its roots still pass through the host. Residency applies within an
optimization call; this is not a fully device-controlled estimator.

A separate current-size diagnostic profile found GPU/CPU summary preparation of
3.70/5.15 ms, and combined linearization/solve scopes of 3.47/2.74 ms. GPU solve
scopes include queued linearization work; they must not be interpreted as isolated
Cholesky kernel timings. Reset, trial-cost and graph-construction overhead remain
outside those two scopes. Profiling timings are not used for speedup claims.

## Validation and artifacts

The 36-run pilot has 30 accelerated/reference trace comparisons. All selected
feature, correspondence, factor, pose, rematch and LM iteration counts match
exactly. Maximum position difference is 1.35e-11 m. All available 1 m translation
and coverage checks pass; the 30 m checks are missing, not passed.

- [Machine-readable pilot](../benchmarks/results/resident-pilot/final.json) and
  [full stage/repeat tables](../benchmarks/results/resident-pilot/final.md).
- Frozen binary/source: `benchmarks/results/resident-backend/`; replay SHA256
  `9d4dea7e4fddb7ef877d7d3e97c9dde4ad541676f9df702e43271bf60d41cbfb`.
- Before the first pilot: 60 CUDA-build C++ tests pass; CPU-only build has 39
  passing tests and 8 CUDA skips; 27 benchmark Python tests pass. Memory check:
  8 resident tests, zero errors. Race check: 3 tests, zero hazards.
- The hybrid implementation adds threshold-transition and unary-dimension tests:
  62 CUDA-build C++ tests and 28 benchmark Python tests pass; CPU-only remains
  39 passing tests with 8 CUDA skips.

A read-only review checked frozen-factor math, trial-cost preservation, lifetime,
and local GTSAM LM behavior. Its constrained-Jacobian finding was fixed and tested.
The dense model-error kernel still uses one block; its cost at substantially
larger dimensions than this pilot remains an unmeasured optimization opportunity.
