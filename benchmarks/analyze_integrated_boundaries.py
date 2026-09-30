"""Verify integrated boundary campaigns and report paired quality and latency.

The stable CPU selector controls changed tie semantics. Map/materialization
comparisons retain the original selector. Every variant is also checked against
the original-order GPU-normal control; quality failure is never hidden by timing.
"""
import argparse
import json
import statistics
from pathlib import Path

from form_report import trace_agreement
from run_suite import aggregate, quality_gate

CONTROLS = {
    "cuda-stable-selection": "cuda-normals",
    "cuda-frontend": "cuda-normals",
    "cuda-materialization": "cuda-normals",
    "cuda-map": "cuda-materialization",
    "cuda-pipeline": "cuda-frontend",
}
STAGES = ("extract_ms", "map_ms", "match_ms", "total_ms", "optimization_ms")
PROFILE_STAGES = (
    "extract_curvature_ms", "extract_planar_select_ms", "extract_normals_ms", "extract_pack_ms",
    "map_world_wall_ms", "map_snapshot_wall_ms", "match_materialize_wall_ms",
    "diag_match_snapshot_pack_ms", "diag_matcher_reset_ms", "diag_match_download_ms",
    "diag_match_reconstruct_ms", "diag_match_host_group_ms", "diag_match_rows_ms", "diag_match_output_ms",
)


def analyze(directory):
    manifest = json.loads((directory / "suite.json").read_text())
    verified = aggregate(manifest, directory)
    if any(run["status"] != "complete" for run in verified["runs"]):
        raise ValueError(f"Incomplete or invalid campaign: {directory}")
    profile_modes = {"--profile" in spec["argv"] for spec in manifest["runs"]}
    if len(profile_modes) != 1:
        raise ValueError("Do not combine instrumented and clean runs in one campaign")
    stage_names = STAGES + PROFILE_STAGES if profile_modes == {True} else STAGES
    records = {}
    for spec in manifest["runs"]:
        record = json.loads((directory / (spec["id"] + ".run.json")).read_text())
        records[(spec["config"], spec["backend"], spec["repeat"])] = (spec, record)
    groups = []
    for config, backend in dict.fromkeys((c, b) for c, b, _ in records):
        runs = [r for (c, b, _), (_, r) in records.items() if (c, b) == (config, backend)]
        stages = {}
        for stage in stage_names:
            values = [r["analysis"]["timings"]["steady_state"][stage] for r in runs]
            stages[stage] = {"repeat_means": [v["mean_ms"] for v in values],
                             "mean": statistics.mean(v["mean_ms"] for v in values),
                             "mean_run_p95": statistics.mean(v["p95_ms"] for v in values)}
        groups.append({"config": config, "backend": backend, "stages": stages,
                       "peak_rss_kib": [r["resources"]["peak_rss_kib"] for r in runs]})
        print(config, backend, {s: round(v["mean"], 3) for s, v in stages.items()})
    comparisons = []
    for (config, backend, repeat), (spec, record) in records.items():
        if backend not in CONTROLS:
            continue
        for control in dict.fromkeys((CONTROLS[backend], "cuda-normals")):
            control_key = (config, control, repeat)
            if control_key not in records:
                continue
            cs, cr = records[control_key]
            gate = quality_gate(cr["analysis"]["trajectory"], record["analysis"]["trajectory"])
            agreement = trace_agreement(directory / cs["id"], directory / spec["id"])
            comparisons.append({"config": config, "candidate": backend, "control": control,
                                "repeat": repeat, "quality_gate": gate, "trace": agreement})
            print("pair", config, backend, control, repeat, gate["status"],
                  "position delta", agreement["max_translation_difference_m"],
                  "changed counts", agreement["changed_counts"])
    return {"directory": str(directory), "binary_sha256": manifest["provenance"]["binary_sha256"],
            "groups": groups, "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({"campaigns": [analyze(p) for p in args.directories]}, indent=2) + "\n")
