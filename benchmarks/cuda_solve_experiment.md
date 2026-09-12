# Standalone FP64 dense Cholesky experiment

This compares fresh CPU Eigen Upper LLT + solve with cuSolver `cusolverDnDpotrf` and
`cusolverDnDpotrs` on actual damped LM systems. It is not integrated into FORM.
The implementation uses host cuSolver APIs only, so the complete executable is
compiled with ordinary C++ rather than NVCC. This preserves Eigen's normal CPU
vectorization; a compile-time guard rejects `EIGEN_GPUCC`. Runtime diagnostics
print the CPU compiler, enabled Eigen SIMD instruction sets, and Eigen thread
count. CPU flags match the production baseline: `-O3 -DNDEBUG -std=c++17`, with
no `-march=native`, OpenMP, fast-math, or reduced precision.

```bash
python benchmarks/compile_cuda_solve.py
/tmp/form-cuda-solve-benchmark --reader-only
# Coordinate a free GPU timing window before either command below.
/tmp/form-cuda-solve-benchmark --self-test-only --repeat 20
/tmp/form-cuda-solve-benchmark \
  --input-dir benchmarks/results/solve-stairs --repeat 30 \
  > benchmarks/results/tuning/cuda-solve.csv \
  2> benchmarks/results/tuning/cuda-solve.log
```

The input search is recursive and preserves relative filenames in the CSV. This
supports scan-200, scan-600, and scan-982 captures in one process. Input format:
8-byte ASCII `FORMSYS1`, uint32 little-endian N, N×N IEEE754 float64 coefficients
of the full symmetric A in column-major order, then N float64 RHS values. The
reader accepts 1≤N≤8192 and rejects bad magic, malformed dimensions, nonfinite
coefficients, asymmetric matrices, truncation, and trailing bytes. It does not
symmetrize or otherwise modify captured systems.

One cuSolver handle, stream, device workspace, and pinned input/output buffers
are retained for each N. Each sample copies the original full A and b into the
pinned buffers, uploads them, computes a fresh Cholesky factorization, solves,
copies the solution and both device-info codes back, synchronizes, checks every
status, and constructs an Eigen vector. Those operations are all inside the GPU
end-to-end timer. Host packing time and completed H2D/device factor-plus-solve
event times are reported separately. Device event timings are components and
must not be substituted for end-to-end latency. Warmup allocations, handle/event
creation, initial workspace sizing, reading, and validation are excluded.

CPU samples each allocate/factor a fresh Eigen Upper LLT from A and solve b.
The final GPU candidate uses CUBLAS_FILL_MODE_LOWER. Because the captured
matrices are fully symmetric, this is mathematically equivalent while allowing
each implementation its faster triangle. The CPU remains Upper to match
production. A separate Upper/Upper diagnostic is preserved for comparison. No cached
CPU factorization is used. CPU/GPU order alternates between repeats, with a
reproducible randomized initial order per system. Raw CSV samples preserve the
order, statuses, and timing components; summary rows report means and p50/p95.
Every captured solution must agree with CPU to relative 2-norm error ≤1e-6,
and both CPU/GPU scaled infinity-norm backward errors must be ≤1e-12:

`||A*x-b||∞ / (||A||∞*||x||∞ + ||b||∞)`.

Built-in SPD tests use dimensions 8/33/128 and prescribed spectral condition
numbers 1e2/1e10/1e12. Ill-conditioned synthetic tests explicitly allow relative
CPU/GPU solution disagreement up to 1e-2 while retaining the same backward-error
threshold; their CSV records this tolerance. This recognizes the unavoidable
sensitivity of forward solutions at those condition numbers. A separate
indefinite case checks that both paths reject a failed Cholesky factorization.

The two cuSolver operations are queued before the final synchronization; results
are accepted only when both host API statuses and both device info codes are
zero and all validation checks pass. A failed warmup or timed sample is reported
as failure and cannot contribute a successful timing summary. Runtime CUDA errors
abort with a nonzero exit status. The failure test itself is logged but never
included in timing summaries.

These are isolated dense solve timings. They exclude factor-graph assembly,
ordering, marginalization, map work, and CPU/GPU changes elsewhere in the
optimizer. No trajectory or convergence claim follows from a microbenchmark.

## Captured-system results (A100 80 GB PCIe)

Final comparison: production CPU Upper LLT against optimized GPU Lower Cholesky,
30 alternating repeats for each of 57 actual LM systems (1,710 accepted solve
pairs). All API statuses and device info codes were zero. Maximum relative
CPU/GPU solution difference was 1.31e-12; maximum GPU backward error was 5.41e-17.
The ill-conditioned SPD tests and indefinite rejection test also passed; Compute
Sanitizer memcheck reported zero errors.

| Scan | N | Systems | CPU mean µs | GPU total mean µs | CPU scan solve sum ms | GPU scan solve sum ms |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 108 | 20 | 82.79 | 182.92 | 1.656 | 3.658 |
| 600 | 174 | 18 | 237.93 | 300.51 | 4.283 | 5.409 |
| 982 | 258 | 19 | 627.34 | 508.58 | 11.920 | 9.663 |

GPU offload regressed the 108- and 174-variable systems. At N=258 it was 1.23×
faster, saving about 2.26 ms over that scan's nineteen solves. This supports only
further evaluation of a large-system threshold; it does not establish a safe
crossover between the sampled dimensions or justify unconditional GPU solving.
The measured H2D component varied substantially between diagnostic processes:
at N=258 the final Lower run measured roughly 226 µs versus 89 µs in the earlier
Upper run, offsetting a lower device factor/solve time. Means from these isolated
runs must not be promoted into an end-to-end trajectory speedup claim.

Artifacts under `benchmarks/results/tuning/`:

- `cuda-solve.csv`, `cuda-solve.log`: final CPU Upper / GPU Lower raw samples and summaries.
- `cuda-solve-aggregate.json`: final per-dimension means and validation maxima.
- `cuda-solve-build.json`, `cuda-solve-build.log`, `cuda-solve-run.json`: build/run provenance.
- `cuda-solve-memcheck.log`: SPD/ill-conditioned/failure-path memory checking.
- `cuda-solve-upper-upper.*`: matching-triangle diagnostic preserved separately.
- `cuda-solve-lower-diagnostic.*`: initial CPU Lower diagnostic; it does not match production.
