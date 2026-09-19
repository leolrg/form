# Certified selective rematching research plan

**Goal:** Establish whether certified partial correspondence search coupled to partial square-root summary updates delivers useful estimator acceleration with a defensible research contribution, or document a systematic negative result after testing plausible variants.

**Authorization:** The user approved autonomous research and implementation of this design, including further experiments, on a separate branch. No additional design approval is needed for routine research choices. Worktree: `/home/ubuntu/form-certified-rematching`; branch `codex/certified-rematching`; baseline `1255543`. Keep `/home/ubuntu/form` and its acceleration branch untouched.

**Architecture:** Preserve the fixed per-scan map snapshot, exact 27-voxel candidate semantics, four-dimensional distance arithmetic and deterministic tie order. Add an opt-in audit/selective matcher. First certify reuse only within an unchanged query voxel using a conservative nearest-competitor bound; fall back on uncertain cases. Then examine safe voxel-crossing certificates and spatially coherent GPU scheduling. Maintain per-query correspondence identity separately from pose-dependent distance. Incremental summaries operate on stable blocks of correspondence rows, rebuilding changed blocks and merging retained roots, rather than reusing whole factors only.

**Technology:** C++17, CUDA FP64, Eigen/GTSAM, existing replay/diagnostic framework. Use TDD for production changes and independent review of certificate validity. All performance claims require clean sequential repeated benchmarks with frozen binaries and identical workloads.

## Evidence and completion conditions

- Exact search indices and accepted groups agree with a full-search oracle at each tested rematch, including ties, voxel transitions, gate transitions, empty queries/maps, map replacement and invalid inputs. Reused distances must be recomputed because final map insertion uses them.
- Summary factors preserve cost and both-pose Gauss–Newton quantities up to justified floating-point differences; no rank truncation or Gram downdates. Partial grouping must preserve raw materialization and ownership lifetimes.
- Measure first-iteration overhead, certified fraction, changed identities, accepted/group changes, affected blocks, QR work, transfer volume, memory and total estimator latency. An unchanged correspondence percentage is not a speedup.
- Evaluate all 1190 stairs scans and additional available datasets. Include same-prefix current/dense/window CPU and GPU controls; do not claim a short prefix is full-sequence scaling.
- Separate baseline, audit-only, certified-search-only, block-summary-only and combined ablations. Include repeated clean runs with reversed order, correctness/trajectory/objective gates, separate traces and provenance.
- Compare prior-art guarantees/constructions; do not claim novelty for triangle inequality, cached ICP search, QR merging, or exact least-squares compression in isolation. State concrete differentiating mechanism and limitations; conference acceptance is not verifiable in this task.
- Positive completion requires credible useful end-to-end benefit and a supported limited contribution claim. Negative completion requires documenting tested failure mechanisms and exhausting reasonable certificate, block partition and scheduling variants, not merely a failed first prototype.

## Tasks

- [x] Inspect branch, processes, dependencies and create isolated research worktree.
- [x] Build and test unmodified baseline; freeze executable, build configuration and hash. All 108 C++ tests pass; frozen SHA256 `cb4fa9d5c9f66e049b7c9666c1be56b66f1cde1591527c8f151cd03d908b6ba2`.
- [x] Write focused prior-art comparison and mathematical certificate contract. See `docs/certified-rematching-prior-art.md`; Hamerly's assignment certificates plus delta statistics materially narrow the novelty hypothesis.
- [ ] Add opt-in search audit with failing tests: force a stable match through a small motion, verify audit identifies it, then compare to an independent full-search oracle. Include equal-distance points in different neighbor voxels, crossings that change nearest or first-hit order, nonzero padding, accepted/rejected transitions, ragged resets and stale-state rejection.
- [ ] Implement conservative certified search under the same contract, test oracle equivalence over adversarial and randomized poses, and run CUDA sanitizers. Record how floating-point bounds ensure a real-arithmetic gap also preserves the actual computed ordering.
- [ ] Run sequential 250-scan feasibility replays and analyze certificate yield, churn, block locality and first-search overhead. Continue to a full-stairs audit if promising; if weak, investigate entering/leaving candidate-domain bounds before deciding against the approach.
- [ ] Design stable block summaries from measured churn. Test one changed correspondence in a factor with otherwise unchanged blocks, insertions/deletions/group migration, rank deficiency and raw-result lifetimes before integration. Compare exact incremental versus fresh roots through cost/Jacobian/Hessian evaluations, not root entry equality.
- [ ] Implement GPU partial leaf/ancestor QR recomputation; compare fixed-query blocks, stable group slots and a compact full rebuild fallback where appropriate. Require actual partial work within a changed scan-pair factor.
- [ ] Review correctness and integrate the best combined variant. Freeze baseline/candidate and execute matched full/scaled CPU/GPU runs plus additional dataset validation.
- [ ] Investigate performance failures from measured overheads: certificate construction, full-search compaction, sparse QR launch/merge costs, group fragmentation and CPU/GPU transfer boundaries. Retain an experiment ledger with both wins and failures.
- [ ] Write final mathematical/complexity argument, reproducible result report and novelty assessment; run completion audit, commit and push research branch. Preserve the original branch exactly.

## Initial numerical contract

The source map need not have geometrically correct voxel placement: the public matcher accepts explicit voxel-to-point ranges, and existing tests use arbitrary placement. Therefore same-cell certification cannot rely on geometric bounds on unvisited points. Cross-cell certificates must either enforce/prove a stronger snapshot property or inspect newly entering candidate voxels and track retained winner membership/order.

Use immutable last-full-search anchors for certificates. Do not move the anchor after a skipped search without updating a valid competitor bound. A certificate must be strict enough to exclude computed-distance ties; otherwise fall back to the original lexicographic search. Failed snapshot/search operations must invalidate potentially stale certificate state.

Keep audit measurements separate from clean timing: any oracle full search, per-query downloads or host analysis is diagnostic overhead, not part of a reported acceleration.
