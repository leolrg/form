"""Summarize saved GPU-normal profiles and the standalone extraction probe.

This does not turn prototype timings into estimator speedups or assert that
stable sorting preserves the production selector's equal-key order.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


STAGES = (
    "extract_validate_ms", "extract_planar_select_ms", "extract_point_mask_ms",
    "extract_point_select_ms", "map_world_wall_ms", "map_snapshot_wall_ms",
    "match_materialize_wall_ms", "diag_match_snapshot_pack_ms",
    "diag_matcher_reset_ms", "diag_match_download_ms", "diag_match_reconstruct_ms",
    "diag_match_host_group_ms", "diag_match_rows_ms", "diag_match_output_ms",
)


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"Empty CSV: {path}")
    return result


def summarize(profiles, probe):
    result = {"profile_ms": {}, "prototype_us": {}, "parity": {}, "inputs": {}}
    paths = []
    for config in ("current", "features"):
        repeats = []
        for repeat in (1, 2):
            path = profiles / f"stairs-{config}-cuda-normals-t32-r{repeat}.csv"
            paths.append(path)
            data = rows(path)
            if [int(r["scan"]) for r in data] != list(range(250)):
                raise ValueError(f"Expected complete 250-scan prefix: {path}")
            repeats.append({key: statistics.mean(float(r[key]) for r in data[20:])
                            for key in STAGES})
        result["profile_ms"][config] = {
            key: {"repeat_means": [r[key] for r in repeats],
                  "mean": statistics.mean(r[key] for r in repeats)} for key in STAGES
        }
    timed_path = probe / "extraction-sampled-final.csv"
    parity_path = probe / "extraction-all-parity.csv"
    paths.extend((timed_path, parity_path))
    timed, parity = rows(timed_path), rows(parity_path)
    for dense, name in ((0, "current"), (1, "features")):
        selected = [r for r in timed if int(r["dense"]) == dense]
        if {(int(r["scan"]), int(r["repeat"])) for r in selected} != {
                (scan, repeat) for scan in range(0, 1190, 20) for repeat in range(8)}:
            raise ValueError("Incomplete or unexpected timed sample")
        result["prototype_us"][name] = {
            key: statistics.mean(float(r[key]) for r in selected)
            for key in selected[0] if key.endswith("_us")
        }
        selected = [r for r in parity if int(r["dense"]) == dense]
        if [int(r["scan"]) for r in selected] != list(range(1190)):
            raise ValueError("Incomplete full-sequence parity audit")
        result["parity"][name] = {
            "scans": len(selected),
            "stable_sequence_changed_scans": [int(r["scan"]) for r in selected
                                               if int(r["stable_changes_sequence"])],
            "stable_membership_changed_scans": [int(r["scan"]) for r in selected
                                                 if int(r["stable_changes_membership"])],
        }
    for path in paths:
        result["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.profiles, args.probe)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("prototype_us", "parity")}, indent=2))
