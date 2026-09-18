#!/usr/bin/env python3
"""Compare only matched old/new reuse experiments, checking artifacts and LM decisions."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from analyze_diagnostics import analyze_run,describe
from analyze_optimizer_policy import pair_calls
from form_report import trace_agreement
from run_suite import digest,quality_gate,save,verify_resume


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();base=args.directory
    manifest=json.loads((base/'suite.json').read_text());runs=[];lookup={};comparisons=[]
    for spec in manifest['runs']:
        prefix=base/spec['suite']/spec['id'];record=json.loads(Path(str(prefix)+'.run.json').read_text())
        verify_resume(record,prefix,spec['expected_scans'],spec['fingerprint'])
        profile='--profile' in spec['argv']
        if profile:
            if digest(Path(str(prefix)+'.optimizer.csv'))!=record['optimizer_sha256']:raise ValueError('optimizer artifact changed')
        result=analyze_run(prefix,profile=profile)
        run={**{k:spec[k] for k in ('id','suite','config','backend','repeat','version')},'prefix':str(prefix),
             'trajectory':record['analysis']['trajectory'],'resources':record['resources'],**result}
        if spec['suite']=='clean-full':
            with Path(str(prefix)+'.csv').open() as f:rows=list(csv.DictReader(f))[20:250]
            run['prefix_250']=dict(timed_scans=len(rows),means={k:statistics.mean(float(r[k]) for r in rows) for k in rows[0] if k not in ('scan','stamp_ns')},
                                   total=describe([float(r['total_ms']) for r in rows]))
        runs.append(run);lookup[(spec['suite'],spec['config'],spec['backend'],spec['repeat'],spec['version'])]=run
    for candidate in runs:
        if candidate['version']!='new':continue
        key=tuple(candidate[k] for k in ('suite','config','backend','repeat'))
        reference=lookup[(*key,'old')]
        agreement=trace_agreement(Path(reference['prefix']),Path(candidate['prefix']))
        if any(agreement['changed_counts'].values()):raise ValueError(f'changed workload: {key}: {agreement}')
        if agreement['max_translation_difference_m']>1e-8 or agreement['max_rotation_difference_rad']>1e-8:raise ValueError('pose parity failed')
        for field in ('full_initial_error','full_final_error'):
            if agreement[field]['max_scaled_difference']>1e-8:raise ValueError('objective parity failed')
        gate=quality_gate(reference['trajectory'],candidate['trajectory'])
        if gate['status']=='fail' or (candidate['scans']==1190 and gate['status']!='pass'):raise ValueError(f'quality gate: {key}: {gate}')
        row=dict(suite=key[0],config=key[1],backend=key[2],repeat=key[3],agreement=agreement,quality_gate=gate,
                 total_cpu_or_gpu_old_over_new=reference['total']['mean']/candidate['total']['mean'],
                 total_saved_ms=reference['total']['mean']-candidate['total']['mean'])
        if key[0]=='profile':
            def read(prefix):
                with Path(prefix+'.optimizer.csv').open() as f:return list(csv.DictReader(f))
            pairs=pair_calls(read(reference['prefix']),read(candidate['prefix']))
            fields=['diag_lm_accepted_calls','diag_lm_rejected_calls','diag_lm_solve_failed_calls']
            changed={k:sum(a[k]!=b[k] for _,a,b in pairs) for k in fields}
            if any(changed.values()) or any(a['backend']!=b['backend'] for _,a,b in pairs):raise ValueError('optimizer policy/decisions changed')
            row['paired_optimizer_calls']=len(pairs);row['changed_lm_decisions']=changed
        comparisons.append(row)
    grouped=defaultdict(list)
    for run in runs:
        grouped[(run['suite'],run['config'],run['backend'],run['version'])].append(run)
        if 'prefix_250' in run:grouped[('clean-prefix','current',run['backend'],run['version'])].append(run['prefix_250'])
    groups=[]
    for (suite,config,backend,version),items in grouped.items():
        keys=set.intersection(*(set(x['means']) for x in items))
        groups.append(dict(suite=suite,config=config,backend=backend,version=version,repeats=len(items),
                           means={k:statistics.mean(x['means'][k] for x in items) for k in sorted(keys)},
                           run_means=[x['total']['mean'] for x in items],repeat_statistics=describe([x['total']['mean'] for x in items]),
                           mean_per_run_p95=statistics.mean(x['total']['p95'] for x in items)))
    save(base/'analysis.json',dict(runs=runs,comparisons=comparisons,groups=groups))
    print('Verified',len(runs),'runs;',len(comparisons),'matched old/new pairs')
    for g in groups:print(g['suite'],g['config'],g['backend'],g['version'],g['run_means'])

if __name__=='__main__':main()
