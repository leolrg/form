#!/usr/bin/env python3
"""Verify completed rematching runs and report repeat means and trajectory parity."""
import argparse
import json
import statistics
from pathlib import Path

from form_report import trace_agreement
from run_suite import save, verify_resume


def compare(directory):
    experiment = json.loads((directory/'experiment.json').read_text())
    records = {}
    for path in sorted(directory.glob('*.run.json')):
        record = json.loads(path.read_text())
        if record['status'] != 'complete':
            raise ValueError(f'Incomplete run: {path}')
        prefix = directory/path.name.removesuffix('.run.json')
        spec = record['spec']
        verify_resume(record, prefix, spec['expected_scans'], spec['fingerprint'])
        records[prefix.name] = (prefix, record)
    if len(records) != len(experiment['modes'])*experiment['repeats']:
        raise ValueError('Run count does not match completed experiment')
    references = [prefix for prefix, _ in records.values() if prefix.name.endswith('-off-r1')]
    if len(references) != 1:
        raise ValueError('One off-r1 reference required')
    reference = references[0]
    result = {}
    for mode in experiment['modes']:
        runs = []
        for repeat in range(1, experiment['repeats']+1):
            matched = [(p, r) for p, r in records.values() if p.name.endswith(f'-{mode}-r{repeat}')]
            # Exact prefix prevents "fast" from also matching "split-fast".
            prefix_stem = reference.name.removesuffix('-off-r1')
            matched = [(p, r) for p, r in matched if p.name == f'{prefix_stem}-{mode}-r{repeat}']
            if len(matched) != 1:
                raise ValueError(f'Missing/ambiguous mode {mode}, repeat {repeat}')
            prefix, record = matched[0]
            timing = record['analysis']['timings']['steady_state']
            runs.append({'id': prefix.name, 'total_ms': timing['total_ms']['mean_ms'],
                         'match_ms': timing['match_ms']['mean_ms'],
                         'agreement': trace_agreement(reference, prefix)})
        result[mode] = {
            'runs': runs,
            'mean_total_ms': statistics.mean(r['total_ms'] for r in runs),
            'mean_match_ms': statistics.mean(r['match_ms'] for r in runs),
            'max_translation_difference_m': max(r['agreement']['max_translation_difference_m'] for r in runs),
            'max_rotation_difference_rad': max(r['agreement']['max_rotation_difference_rad'] for r in runs),
            'count_mismatches': sum(sum(r['agreement']['changed_counts'].values()) for r in runs),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    save(args.directory/'comparison-verified.json', result)
    for mode, data in result.items():
        print(f"{mode:18s} total {data['mean_total_ms']:.3f} ms; match {data['mean_match_ms']:.3f} ms; "
              f"translation {data['max_translation_difference_m']:.3g} m; count mismatches {data['count_mismatches']}")


if __name__ == '__main__':
    main()
