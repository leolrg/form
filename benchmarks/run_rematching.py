#!/usr/bin/env python3
"""Sequential, provenance-recorded rematching experiments (clean or diagnostic)."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path

from run_suite import CONFIGS, ROOT, digest, execute, json_hash, provenance, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', default='stairs')
    parser.add_argument('--config', choices=CONFIGS, default='current')
    parser.add_argument('--limit', type=int, default=250)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'off', 'audit', 'certified', 'cpu', 'reference', 'blocks', 'combined', 'summary-audit'), default=['off', 'certified'])
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()
    if args.limit <= 20 or args.repeats < 1 or args.threads < 1:
        parser.error('limit > 20, repeats >= 1, threads >= 1 required')
    if 'baseline' in args.modes and args.baseline is None:
        parser.error('baseline mode requires --baseline')
    if any(m in args.modes for m in ('audit', 'summary-audit')) and not args.diagnostic:
        parser.error('audit mode is diagnostic, never clean timing')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Hold lock throughout all children; prevent accidental overlapping suites.
    with (ROOT/'benchmarks/results/.rematching.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ('FORM_CUDA_MATCH_REUSE', 'FORM_CUDA_SUMMARY_REUSE', 'FORM_MATCH_STATS_PATH', 'FORM_MATCH_AUDIT_PATH'):
            os.environ.pop(key, None)
        binary = args.binary.resolve()
        prov = provenance(binary, output)
        prov['research_options'] = vars(args) | {'binary': str(binary), 'baseline': str(args.baseline), 'output': str(output)}
        input_dir = Path('/home/ubuntu/datasets/form-input')
        input_path = input_dir/(args.sequence+'.formpc')
        identity = {'binary': digest(binary), 'input': digest(input_path), 'config': CONFIGS[args.config],
                    'threads': args.threads, 'limit': args.limit, 'diagnostic': args.diagnostic,
                    'runtime_environment': prov['environment'], 'cpu_affinity': prov['cpu_affinity']}
        if args.baseline:
            identity['baseline'] = digest(args.baseline)
        for repeat in range(args.repeats):
            modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
            for mode in modes:
                name = f'{args.sequence}-{args.config}-{mode}-r{repeat+1}'
                os.environ['FORM_CUDA_MATCH_REUSE'] = 'audit' if mode in ('audit', 'summary-audit') else ('certified' if mode in ('certified', 'combined') else 'off')
                os.environ['FORM_CUDA_SUMMARY_REUSE'] = 'audit64' if mode == 'summary-audit' else ('blocks64' if mode in ('blocks', 'combined') else 'off')
                if args.diagnostic and mode not in ('baseline', 'cpu', 'reference'):
                    os.environ['FORM_MATCH_STATS_PATH'] = str(output/(name+'.match-stats.csv'))
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
                        'gpu_sample_interval_seconds': 0,
                        'fingerprint': json_hash(identity | {'mode': mode, 'env': environment})}
                execute(spec, output, prov | {'experiment_environment': environment, 'identity': identity})
                record = json.loads((output/(name+'.run.json')).read_text())
                if record['status'] != 'complete':
                    raise RuntimeError(f'failed experiment: {name}')
                if args.diagnostic and mode not in ('baseline', 'cpu', 'reference'):
                    audit_path = Path(environment['FORM_MATCH_AUDIT_PATH'])
                    stats_path = Path(environment['FORM_MATCH_STATS_PATH'])
                    if not audit_path.is_file() or audit_path.stat().st_size <= 8 or not stats_path.is_file():
                        raise RuntimeError(f'missing diagnostic capture: {name}; verify executable build')
                    with stats_path.open() as stream:
                        stats = list(csv.DictReader(stream))
                    if not stats or any(int(row['mismatches']) for row in stats):
                        raise RuntimeError(f'empty or failing match audit: {name}')
                    if mode in ('audit', 'summary-audit') and any(int(row['oracle_searched']) != int(row['total']) for row in stats):
                        raise RuntimeError(f'incomplete original-kernel oracle: {name}')
                    if mode == 'summary-audit' and not sum(int(row['summary_checks']) for row in stats):
                        raise RuntimeError(f'missing full summary reconstruction checks: {name}')
        save(output/'experiment.json', {'identity': identity, 'modes': args.modes, 'repeats': args.repeats})


if __name__ == '__main__':
    main()
