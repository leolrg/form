# CUDA correspondence matching and direct summary construction

The user authorized accelerating matching and connecting its output directly to
the existing exact summaries. Work continues on `codex/optimization-acceleration`.

The optional matcher must reproduce FORM's 27-voxel search, voxel floor convention,
strict distance threshold, and first-hit tie rule. The CPU-built world map remains
the source of truth and is uploaded once per incoming scan. Query features remain
on device across ICP rematches. Map snapshots preserve point order within each
voxel. Local target coordinates use the same world-to-local transformation as FORM.

The CUDA search produces nearest-point indices and distances. Accepted matches
are grouped by scan pair and packed into seven-column point/compact-plane features
on device, then passed directly to the existing FP64 QR pipeline. Compact roots
populate the existing immutable summary caches. Raw matches remain available on
the host for map insertion, correspondence counts, raw factor evaluation and
marginalization. This initial integration does not make the whole estimator resident.

The default matcher remains CPU. A replay backend selects GPU matching plus the
existing hybrid optimizer. A separate CPU-matching hybrid control isolates the
new contribution; original FORM and direct CPU controls retain equal workloads.
Numerical equivalence, threshold/boundary behavior, buffer reuse and CPU-only builds
are required. No fewer matches, lower precision, altered ICP stopping rules or
approximate nearest-neighbor search are used to obtain speedups.

Validation includes unit comparisons against `VoxelMap::find_closest`, raw-factor
cost/Hessian comparisons at multiple poses, Compute Sanitizer, and paired replay
at current, denser-point and larger-window settings. Report map-upload and combined
matching/optimization costs, since summary preparation moves between stage timers.
Run the 250-scan stairs screening matrix first, followed by longer/four-sequence
quality checks where available. Never label absent 30 m segments as passed.
