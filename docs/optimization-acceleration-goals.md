# FORM optimization acceleration: goals and requirements

Date: 2026-09-12

## Objective

Achieve the best practical optimization performance for FORM on a desktop NVIDIA GPU, while preserving reliable odometry. Build on the existing FORM codebase so results can be compared directly with the original estimator and evaluated through evalio.

The primary objective is lower optimization latency at the current problem size. A secondary objective is to use the available compute budget for more selected features or a larger pose window, and evaluate whether that improves accuracy or robustness while retaining real-time operation.

This document defines the intended outcomes and evaluation requirements. It does not select a kernel design, CPU/GPU partition, solver, or integration architecture.

## Context

FORM estimates poses over a fixed window of recent scans and retained keyscans. It uses point-to-plane and point-to-point correspondences, semi-linearized optimization during ICP, a final full nonlinear optimization, and marginalization. Its map is repaired using updated pose estimates.

The workload has many residuals relative to the number of pose variables. The paper identifies residual and Jacobian computation as a major optimization cost. The current code groups correspondences into scan-pair factors and constructs dense pose systems. Linearization, residual evaluation, and system construction are therefore leading acceleration candidates, but profiling must establish the actual bottlenecks on the target hardware.

GTSAM develop's CUDA code is a reference for useful conventions and techniques. Reusing its existing CUDA optimizers is not a requirement. The fastest validated approach may use custom CUDA, revised mathematical formulations, CPU optimizations, existing libraries, or a combination. Retaining a particular GTSAM interface is subordinate to the performance and correctness objectives.

## Scope

The primary scope is the optimization stage, including repeated residual evaluation, linearization, system construction, linear solving, and optimizer overhead. Measure related work such as marginalization separately so moving work across stage boundaries cannot create a misleading speedup.

Changes to feature selection and pose-window size are in scope as controlled workload experiments. Broader acceleration of extraction, matching, or map maintenance can be considered if measurements show that it materially limits the end-to-end benefit, but is not an initial deliverable.

The project does not initially require a new odometry system, new sensors, loop closure, a mapping application, or support for non-NVIDIA hardware. The original CPU implementation must remain available as a reproducible reference.

## Target environment and baseline

- Target class: desktop NVIDIA GPU. Record the exact GPU, CPU, memory, driver, CUDA toolkit, and software versions before performance comparisons.
- Starting FORM reference: commit `ec7094429f35c3244366815c7d4319be6b4367dc` (version 0.2.0).
- Use optimized builds and document CPU thread counts and other settings that affect timing.
- Establish a fresh baseline from this checkout. The author's approximately 12 Hz N21 result is historical context, not an acceptance threshold or a measured result for this machine. This checkout includes a post-paper change to point-feature extraction.
- If the reference version changes, record that change and rerun the corresponding baseline.

No minimum speedup multiplier is specified before measurement. The objective is a reproducible reduction in actual optimization time, with clear evidence of its effect on complete scan processing.

## Evaluation workloads

### 1. Current configuration

Compare the reference and accelerated implementations with the same input data, feature-selection settings, pose-window settings, and optimization tolerances. Preserve the estimation objective and convergence requirements. Reducing work by weakening those requirements must not be reported as an equivalent-work acceleration.

Use identical captured optimization inputs where appropriate to separate backend behavior from later changes in correspondence selection. Also evaluate complete sequential odometry runs, where numerical differences can affect subsequent scans.

### 2. Larger problems

Evaluate increased feature counts and larger active pose windows. Initially vary them separately to distinguish residual scaling from variable-count scaling. Record realized correspondence counts, active poses, and optimization iterations; configured limits alone do not describe the workload.

Compare CPU and accelerated implementations at matching enlarged settings. Separately compare the enlarged accelerated configuration with the original configuration to assess the accuracy-versus-latency tradeoff.

The real-time budget is 100 ms per incoming scan for the selected 10 Hz LiDAR. This budget applies to complete scan processing, not just optimization. Report latency tails and missed deadlines as well as average throughput. More features or poses are useful only if the measured quality or robustness benefit justifies their cost.

## Benchmark dataset

Use the approved four-sequence subset of **Newer College 2021 Multi-Cam**, captured with a 128-beam LiDAR. This is distinct from the 2020 Stereo-Cam dataset.

| Sequence | LiDAR scans | Ground-truth poses | Purpose |
|---|---:|---:|---|
| `stairs` | 1,190 | 1,190 | Narrow geometry and stair motion |
| `quad_easy` | 1,991 | 1,988 | Ordinary walking reference |
| `quad_hard` | 1,880 | 1,724 | Aggressive motion and difficult registration |
| `maths_hard` | 2,440 | 2,438 | Larger outdoor environment with challenging motion |
| **Total** | **7,501** | **7,340** | |

The original bags and ground-truth CSV files occupy approximately 39.6 GB under `/home/ubuntu/datasets/newer_college_2021/`. They were downloaded with evalio 0.6.1. Source-size and loading checks are recorded in `/home/ubuntu/datasets/n21-verification.json`; usage instructions are in `/home/ubuntu/datasets/README-FORM.md`.

Ground-truth coverage is not identical to scan coverage. Use consistent timestamp association and report the evaluated coverage. Use short segments for development, then all four complete sequences for final comparisons. The full 169 GB dataset is outside the currently approved download scope.

## Correctness and quality requirements

- For equivalent-work comparisons, preserve the factor objective, noise weighting, pose conventions, prior information, semi-linearized behavior, and marginalization semantics.
- Validate numerical agreement at the optimization level and trajectory quality at the system level. Bitwise identity is not required. Define numerical and trajectory tolerances before evaluating candidate results, with the reference behavior and conditioning taken into account.
- Report objective values, convergence behavior, tracking failures, and trajectory error. Include short-distance and longer-distance relative trajectory error to capture both smoothness and drift.
- Compare poses at the same output stage. A pose emitted immediately after scan processing must not be compared against a later smoothed pose as though both had the same latency.
- Treat changed objectives, approximations, reduced precision, or altered stopping rules as explicit experimental choices. Report their quality/performance tradeoffs separately from equivalent-work results.
- Exercise small batches, changing window contents, weak geometry, and difficult motion. Successful execution alone is insufficient evidence of correctness.

## Performance measurement requirements

Report optimization latency and end-to-end scan latency separately, including median, p95, p99, throughput, and the fraction of scans exceeding 100 ms. Report results per sequence; averages must not hide regressions or failures.

Include data preparation, transfers, synchronization, and required host work in the relevant timings. GPU launch submission time is not completed computation time. Separate startup and warm-up from steady-state performance, and exclude dataset download, loading, and result serialization from the estimator timing while documenting the boundary.

Record stage timings, workload counts, memory use, and repeated-run variability. Distinguish linearization, system construction, solving, and trial-error evaluation sufficiently to explain the observed speedup. An isolated kernel improvement is not sufficient if total optimization or scan processing becomes slower.

## Deliverables and success criteria

1. A reproducible CPU baseline and profile identifying where optimization time is spent.
2. An accelerated optimization path integrated into the existing FORM workflow, with the reference implementation still runnable.
3. Correctness and odometry-quality comparisons on the four downloaded sequences.
4. Performance results for current settings and controlled larger workloads, with scripts/configurations and environment details sufficient to reproduce them.
5. A concise assessment of the fastest validated configuration, remaining bottlenecks, and the useful operating range.

Primary success is a repeatable optimization speedup at equivalent settings, within the predeclared correctness and quality tolerances, with its end-to-end benefit demonstrated. Expanded-workload success is additional: a measured improvement in quality or robustness within the scan-processing budget, or a clear characterization of how much larger a problem can be handled at comparable latency. Neither outcome should be inferred solely from kernel timings or higher feature counts.

## References

- [FORM paper](https://arxiv.org/abs/2510.09966)
- [Original FORM repository](https://github.com/rpl-cmu/form)
- [Newer College 2021 Multi-Cam](https://ori-drs.github.io/newer-college-dataset/multi-cam/)
- [evalio](https://github.com/contagon/evalio)
- [GTSAM develop, as an implementation reference](https://github.com/borglab/gtsam/tree/develop)
