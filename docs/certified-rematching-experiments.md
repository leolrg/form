# Certified rematching experiment ledger

Research branch: `codex/certified-rematching`, isolated worktree
`/home/ubuntu/form-certified-rematching`. Established acceleration branch remains
unchanged at `1255543`. These are preliminary feasibility observations, not final
speed or novelty claims.

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
