#!/usr/bin/env python3
"""Analyze replay diagnostics without conflating inclusive timers/device timelines."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from form_report import trace_agreement
from run_suite import digest, now, quality_gate, save, verify_resume

STAGES=('extract_ms','map_ms','match_ms','semi_ms','full_ms','marginalize_ms','maintenance_ms')


def union_duration(intervals):
    total=0; end=None
    for a,b in sorted(intervals):
        if b<a: raise ValueError('negative interval')
        total+=max(0,b-max(a,end if end is not None else a))
        end=max(b,end if end is not None else b)
    return total


def percentile(values,q):
    values=sorted(values); p=(len(values)-1)*q; lo=int(p); hi=min(lo+1,len(values)-1)
    return values[lo]+(p-lo)*(values[hi]-values[lo])


def describe(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),
                p95=percentile(values,.95),p99=percentile(values,.99),min=min(values),max=max(values),
                sd=statistics.stdev(values) if len(values)>1 else 0.)


def summarize_optimizer(rows,timed_scans,warmup=20):
    groups=defaultdict(list)
    for row in rows:
        if int(row['scan'])>=warmup: groups[row['backend']+'/'+row['phase']].append(row)
    result={}
    for key,items in groups.items():
        fields=['wall_ms']+[k for k in items[0] if k.startswith('diag_')]
        totals={k:sum(float(r.get(k,0)) for r in items) for k in fields}
        dimensions=[int(r['dimension']) for r in items]
        result[key]=dict(calls=len(items),calls_per_scan=len(items)/timed_scans,
            dimension_min=min(dimensions),dimension_max=max(dimensions),
            per_scan={k:v/timed_scans for k,v in totals.items()},
            per_call={k:v/len(items) for k,v in totals.items()})
    return result


def ranks(values):
    ordered=sorted(range(len(values)),key=values.__getitem__); result=[0.]*len(values);i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and values[ordered[j]]==values[ordered[i]]: j+=1
        for k in ordered[i:j]:result[k]=(i+j-1)/2
        i=j
    return result


def correlations(rows):
    y=ranks([float(r['total_ms']) for r in rows]);result={}
    for field in ('poses','planar_features','point_features','rematches','lm_iterations','planar_correspondences','point_correspondences'):
        x=ranks([float(r[field]) for r in rows])
        result[field]=statistics.correlation(x,y) if len(set(x))>1 and len(set(y))>1 else None
    return result


def optimizer_manifest(directory, create=False):
    """Snapshot after campaign completion; subsequent analyses reject mutation."""
    artifacts={}
    for suite_path in sorted(directory.glob('*/suite.json')):
        for spec in json.loads(suite_path.read_text())['runs']:
            prefix=suite_path.parent/spec['id']
            record=json.loads(Path(str(prefix)+'.run.json').read_text())
            if record['status']!='complete':raise ValueError(f'incomplete: {prefix}')
            if '--profile' not in spec['argv']:continue
            path=Path(str(prefix)+'.optimizer.csv')
            with path.open() as f:
                if not list(csv.DictReader(f)):raise ValueError(f'empty optimizer table: {path}')
            artifacts[str(path.relative_to(directory))]=digest(path)
    path=directory/'optimizer-artifacts.json'
    if path.exists():
        if json.loads(path.read_text())['sha256']!=artifacts:raise ValueError('optimizer artifacts changed')
    elif create:
        save(path,dict(captured_at=now(),scope='Post-campaign snapshot, not completion-time hashes',sha256=artifacts))
    else:raise ValueError('run analyze_diagnostics first to snapshot optimizer artifacts')
    return artifacts


def analyze_run(prefix, profile=False):
    with Path(str(prefix)+'.csv').open() as f: rows=list(csv.DictReader(f))
    selected=rows[20:]
    if not selected:raise ValueError('no steady-state scans')
    fields=[k for k in selected[0] if k not in ('scan','stamp_ns')]
    means={k:statistics.mean(float(r[k]) for r in selected) for k in fields}
    residual=[float(r['total_ms'])-sum(float(r[k]) for k in STAGES) for r in selected]
    result=dict(scans=len(rows),timed_scans=len(selected),means=means,
        total=describe([float(r['total_ms']) for r in selected]),
        first_scan_ms=float(rows[0]['total_ms']),
        deadline_miss_fraction=sum(float(r['total_ms'])>100 for r in selected)/len(selected),
        stage_residual_ms=describe(residual),spearman_total=correlations(selected),
        slowest=[{k:r[k] for k in ('scan','total_ms',*STAGES,'poses','rematches','lm_iterations')} for r in sorted(selected,key=lambda r:float(r['total_ms']),reverse=True)[:10]])
    opt=Path(str(prefix)+'.optimizer.csv')
    if profile and not opt.exists():raise ValueError(f'missing optimizer table: {opt}')
    if opt.exists():
        with opt.open() as f: calls=list(csv.DictReader(f))
        if not calls:raise ValueError(f'empty optimizer table: {opt}')
        result['optimizer']=summarize_optimizer(calls,len(selected))
        # Exact event accounting: one decision per solve attempt, one record per
        # semi/full invocation. Counts are exact integers, not rounded means.
        for r in calls:
            if r.get('engine')!='resident': continue
            attempts=int(r['diag_cpu_solve_calls'])+int(r['diag_gpu_solve_calls'])
            assert attempts==int(r['diag_lm_accepted_calls'])+int(r['diag_lm_rejected_calls']),r
        if calls:
            for r in rows:
                actual=[c for c in calls if c['scan']==r['scan']]
                assert len(actual)==int(r['rematches'])+1,(r['scan'],len(actual))
                for field in ('cpu_solve','gpu_solve','cpu_reset','gpu_reset','lm_accepted','lm_rejected'):
                    key='diag_'+field+'_calls'
                    assert sum(int(c[key]) for c in actual)==int(r[key]),(r['scan'],key)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    optimizer_manifest(args.directory,create=True)
    runs=[]; groups=defaultdict(list);references={}
    for suite_path in sorted(args.directory.glob('*/suite.json')):
        suite=json.loads(suite_path.read_text())
        for spec in suite['runs']:
            prefix=suite_path.parent/spec['id'];record=json.loads(Path(str(prefix)+'.run.json').read_text())
            if record['status']!='complete':raise ValueError(f'incomplete: {prefix}')
            verify_resume(record,prefix,spec['expected_scans'],spec['fingerprint'])
            result=analyze_run(prefix,profile='--profile' in spec['argv'])
            run=dict(suite=suite_path.parent.name,id=spec['id'],sequence=spec['sequence'],backend=spec['backend'],config=spec['config'],
                     repeat=spec['repeat'],prefix=str(prefix),resources=record['resources'],trajectory=record['analysis']['trajectory'],**result)
            runs.append(run);groups[(run['suite'],run['sequence'],run['config'],run['backend'])].append(run)
            if run['backend']=='reference':references.setdefault((run['sequence'],run['config'],run['scans']),run['prefix'])
    agreements=[]
    lookup={r['prefix']:r for r in runs}
    for run in runs:
        key=(run['sequence'],run['config'],run['scans']);reference=references.get(key)
        if not reference: raise ValueError(f'No matched reference for {key}')
        if reference:
            agreement=trace_agreement(Path(reference),Path(run['prefix']))
            gate=quality_gate(lookup[reference]['trajectory'],run['trajectory'])
            agreements.append(dict(run=run['prefix'],reference=reference,quality_gate=gate,**agreement))
            if gate['status']=='fail':raise ValueError(f'quality failed: {agreements[-1]}')
            if run['scans']==1190 and gate['status']!='pass':raise ValueError('full-sequence quality metrics missing')
            if agreement['max_translation_difference_m']>1e-8 or agreement['max_rotation_difference_rad']>1e-8:
                raise ValueError(f'trajectory parity exceeded 1e-8: {agreements[-1]}')
            if any(agreement['changed_counts'].values()):raise ValueError(f'workload changed: {agreements[-1]}')
    grouped=[]
    for (suite,sequence,config,backend),items in groups.items():
        grouped.append(dict(suite=suite,sequence=sequence,config=config,backend=backend,repeats=len(items),
            total_run_means=describe([r['total']['mean'] for r in items]),
            mean_per_run_p95=statistics.mean(r['total']['p95'] for r in items),
            mean_per_run_p99=statistics.mean(r['total']['p99'] for r in items),
            means={k:statistics.mean(r['means'][k] for r in items) for k in items[0]['means']}))
    quality_counts={status:sum(a['quality_gate']['status']==status for a in agreements) for status in ('pass','missing','fail')}
    out=dict(warmup=20,run_count=len(runs),quality_status_counts=quality_counts,groups=grouped,runs=runs,trace_agreements=agreements)
    (args.directory/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Verified artifact/count/pose parity:',len(runs),'runs; trajectory quality gates:',quality_counts)
    for r in grouped:print(r['suite'],r['config'],r['backend'],round(r['total_run_means']['mean'],3))

if __name__=='__main__':main()
