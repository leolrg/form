#!/usr/bin/env python3
"""Sequential, provenance-recorded rematching experiments (clean or diagnostic)."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path

from run_suite import CONFIGS, ROOT, digest, execute, json_hash, provenance, save


# reuse, summary, kernel, runner-up, certificate, tree warp cap, tree bounds
MODES = {
    'off': ('off', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'baseline': ('off', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'cpu': ('off', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'reference': ('off', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'certified': ('certified', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'fast': ('certified', 'off', 'fused', 'occurrences', 'norm', 32, 0),
    'split': ('certified', 'off', 'split', 'distinct', 'norm', 32, 0),
    'split-fast': ('certified', 'off', 'split', 'occurrences', 'norm', 32, 0),
    'blocks': ('off', 'blocks64', 'fused', 'distinct', 'norm', 32, 0),
    'combined': ('certified', 'blocks64', 'fused', 'distinct', 'norm', 32, 0),
    'combined-fast': ('certified', 'blocks64', 'split', 'occurrences', 'norm', 32, 0),
    'audit': ('audit', 'off', 'fused', 'distinct', 'norm', 32, 0),
    'audit-fast': ('audit', 'off', 'split', 'occurrences', 'norm', 32, 0),
    'summary-audit': ('audit', 'audit64', 'fused', 'distinct', 'norm', 32, 0),
    'tree': ('off', 'tree64', 'fused', 'distinct', 'norm', 32, 0),
    'tree-combined': ('certified', 'tree64', 'split', 'occurrences', 'norm', 32, 0),
    'tree-audit': ('audit', 'audit-tree64', 'split', 'occurrences', 'norm', 32, 0),
    'tree128': ('off', 'tree64', 'fused', 'distinct', 'norm', 128, 0),
    'tree256': ('off', 'tree64', 'fused', 'distinct', 'norm', 256, 0),
    'tree-bounded': ('off', 'tree64', 'fused', 'distinct', 'norm', 32, 1),
    'tree-bounded128': ('off', 'tree64', 'fused', 'distinct', 'norm', 128, 1),
    'tree-bounded256': ('off', 'tree64', 'fused', 'distinct', 'norm', 256, 1),
    'squared': ('certified', 'off', 'fused', 'occurrences', 'squared', 32, 0),
    'squared-split': ('certified', 'off', 'split', 'occurrences', 'squared', 32, 0),
    'tree-squared128': ('certified', 'tree64', 'fused', 'occurrences', 'squared', 128, 1),
    'squared-audit': ('audit', 'audit-tree64', 'fused', 'occurrences', 'squared', 128, 1),
}
MODES['tree-squared-split128'] = ('certified', 'tree64', 'split', 'occurrences', 'squared', 128, 1)
MODES = {mode: (*settings, 'sorted') for mode, settings in MODES.items()}
MODES['queue128'] = (*MODES['tree-bounded128'][:-1], 'queue')
MODES['queue-combined'] = (*MODES['tree-squared-split128'][:-1], 'queue')
MODES['queue-audit'] = ('audit', 'audit-tree64', 'split', 'occurrences', 'squared', 128, 1, 'queue')
MODES = {mode: (*settings, 'capped') for mode, settings in MODES.items()}
MODES['compact'] = (*MODES['tree-bounded128'][:-1], 'compact')
MODES['compact-queue'] = (*MODES['queue128'][:-1], 'compact')
MODES['compact-sorted-combined'] = (*MODES['tree-squared-split128'][:-1], 'compact')
MODES['compact-combined'] = (*MODES['queue-combined'][:-1], 'compact')
MODES['compact-audit'] = (*MODES['queue-audit'][:-1], 'compact')
ENV_KEYS = ('FORM_CUDA_MATCH_REUSE', 'FORM_CUDA_SUMMARY_REUSE', 'FORM_CUDA_MATCH_KERNEL',
            'FORM_CUDA_MATCH_TOP2', 'FORM_CUDA_MATCH_CERTIFICATE', 'FORM_CUDA_TREE_WARPS',
            'FORM_CUDA_TREE_BOUNDS', 'FORM_CUDA_SUMMARY_ROWS', 'FORM_CUDA_TREE_LAYOUT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', default='stairs')
    parser.add_argument('--config', choices=CONFIGS, default='current')
    parser.add_argument('--limit', type=int, default=250)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=['off', 'certified'])
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--stats-only', action='store_true', help='Diagnostic counters/oracles without raw correspondence capture')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--gpu-sample-interval', type=float, default=0, help='Optional diagnostic-only GPU memory sampling in seconds')
    args = parser.parse_args()
    if args.stats_only and not args.diagnostic:
        parser.error('--stats-only requires --diagnostic')
    if args.gpu_sample_interval < 0 or (args.gpu_sample_interval and not args.diagnostic):
        parser.error('GPU memory sampling requires diagnostic mode and a nonnegative interval')
    if args.limit <= 20 or args.repeats < 1 or args.threads < 1:
        parser.error('limit > 20, repeats >= 1, threads >= 1 required')
    if 'baseline' in args.modes and args.baseline is None:
        parser.error('baseline mode requires --baseline')
    if any(MODES[m][0] == 'audit' for m in args.modes) and not args.diagnostic:
        parser.error('audit mode is diagnostic, never clean timing')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Hold lock throughout all children; prevent accidental overlapping suites.
    with (ROOT/'benchmarks/results/.rematching.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in (*ENV_KEYS, 'FORM_MATCH_STATS_PATH', 'FORM_MATCH_AUDIT_PATH'):
            os.environ.pop(key, None)
        binary = args.binary.resolve()
        prov = provenance(binary, output)
        prov['research_options'] = vars(args) | {'binary': str(binary), 'baseline': str(args.baseline), 'output': str(output)}
        input_dir = Path('/home/ubuntu/datasets/form-input')
        input_path = input_dir/(args.sequence+'.formpc')
        identity = {'binary': digest(binary), 'input': digest(input_path), 'config': CONFIGS[args.config],
                    'threads': args.threads, 'limit': args.limit, 'diagnostic': args.diagnostic, 'stats_only': args.stats_only,
                    'gpu_sample_interval': args.gpu_sample_interval,
                    'runtime_environment': prov['environment'], 'cpu_affinity': prov['cpu_affinity']}
        if args.baseline:
            identity['baseline'] = digest(args.baseline)
        for repeat in range(args.repeats):
            modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
            for mode in modes:
                name = f'{args.sequence}-{args.config}-{mode}-r{repeat+1}'
                os.environ.update({key: str(value) for key, value in zip(ENV_KEYS, MODES[mode])})
                if args.diagnostic and mode not in ('baseline', 'cpu', 'reference'):
                    os.environ['FORM_MATCH_STATS_PATH'] = str(output/(name+'.match-stats.csv'))
                    if args.stats_only:
                        os.environ.pop('FORM_MATCH_AUDIT_PATH', None)
                    else:
                        os.environ['FORM_MATCH_AUDIT_PATH'] = str(output/(name+'.matches.bin'))
                else:
                    os.environ.pop('FORM_MATCH_STATS_PATH', None)
                    os.environ.pop('FORM_MATCH_AUDIT_PATH', None)
                environment = {k: v for k, v in os.environ.items() if k.startswith('FORM_')}
                argv = [str(args.baseline.resolve() if mode == 'baseline' else binary),
                        '--input', str(input_path), '--output', str(output/name),
                        '--backend', 'cpu-extraction' if mode == 'cpu' else ('reference' if mode == 'reference' else 'cuda-extraction'),
                        '--threads', str(args.threads), '--limit', str(args.limit)]
                for key, value in CONFIGS[args.config].items():
                    argv += ['--'+key, str(value)]
                if args.diagnostic:
                    argv += ['--profile']
                spec = {'id': name, 'argv': argv, 'expected_scans': args.limit,
                        'ground_truth': str(input_dir/(args.sequence+'.gt.tum')),
                        'gpu_sample_interval_seconds': args.gpu_sample_interval,
                        'fingerprint': json_hash(identity | {'mode': mode, 'env': environment})}
                execute(spec, output, prov | {'experiment_environment': environment, 'identity': identity})
                record = json.loads((output/(name+'.run.json')).read_text())
                if record['status'] != 'complete':
                    raise RuntimeError(f'failed experiment: {name}')
                if args.diagnostic and mode not in ('baseline', 'cpu', 'reference'):
                    audit_path = None if args.stats_only else Path(environment['FORM_MATCH_AUDIT_PATH'])
                    stats_path = Path(environment['FORM_MATCH_STATS_PATH'])
                    if (audit_path is not None and (not audit_path.is_file() or audit_path.stat().st_size <= 8)) or not stats_path.is_file():
                        raise RuntimeError(f'missing diagnostic capture: {name}; verify executable build')
                    with stats_path.open() as stream:
                        stats = list(csv.DictReader(stream))
                    if not stats or any(int(row['mismatches']) for row in stats):
                        raise RuntimeError(f'empty or failing match audit: {name}')
                    if MODES[mode][0] == 'audit' and any(int(row['oracle_searched']) != int(row['total']) for row in stats):
                        raise RuntimeError(f'incomplete original-kernel oracle: {name}')
                    if MODES[mode][1] == 'audit-tree64' and not any(int(row.get('summary_tree', 0)) for row in stats):
                        raise RuntimeError(f'missing cached-tree path: {name}')
                    if MODES[mode][6] and MODES[mode][1] == 'audit-tree64' and not any(int(row.get('summary_bounded', 0)) for row in stats):
                        raise RuntimeError(f'missing bounded-tree path: {name}')
                    if MODES[mode][7] == 'queue' and not any(int(row.get('summary_queued', 0)) for row in stats):
                        raise RuntimeError(f'missing queued-summary path: {name}')
                    if MODES[mode][8] == 'compact' and not any(int(row.get('summary_compact', 0)) for row in stats):
                        raise RuntimeError(f'missing compact-summary path: {name}')
                    if MODES[mode][1] in ('audit64', 'audit-tree64') and not sum(int(row['summary_checks']) for row in stats):
                        raise RuntimeError(f'missing full summary reconstruction checks: {name}')
        save(output/'experiment.json', {'identity': identity, 'modes': args.modes, 'repeats': args.repeats})


if __name__ == '__main__':
    main()
