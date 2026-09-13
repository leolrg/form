"""Export a compact, auditable acceleration report from a completed suite.

Per-group latency entries average the corresponding per-run statistic; they are
not pooled percentiles. Quality gates remain per-repeat, never averaged.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

STAGES=('optimization_ms','total_ms','extract_ms','map_ms','match_ms','semi_ms',
        'full_ms','marginalize_ms','maintenance_ms')
FIELDS=('mean_ms','median_ms','p95_ms','p99_ms','throughput_hz','deadline_miss_fraction')


def describe(values):
    if not values:
        return None
    return {'mean':statistics.mean(values),'min':min(values),'max':max(values),
            'sample_stddev':statistics.stdev(values) if len(values)>1 else None}


def grouped_results(records,comparisons):
    groups=defaultdict(list)
    gates={(c['sequence'],c['config'],c['candidate']):c for c in comparisons}
    for record in records:
        spec=record['spec']
        groups[(spec['sequence'],spec['config'],spec['backend'])].append(record)
    result=[]
    for (sequence,config,backend),items in groups.items():
        good=[r for r in items if r['status']=='complete']
        gate=gates.get((sequence,config,backend),{}).get('quality_gate_status','missing')
        status=('incomplete' if len(good)!=len(items) else 'validated' if backend=='reference' or gate=='pass'
                else 'quality-failed' if gate=='fail' else 'unvalidated')
        row={'sequence':sequence,'config':config,'backend':backend,'status':status,
             'completed_repeats':len(good),'planned_repeats':len(items),
             'source_run_ids':[r['spec']['id'] for r in items],'latency':{}}
        for stage in STAGES:
            values=[r['analysis']['timings']['steady_state'].get(stage) for r in good]
            values=[v for v in values if v]
            if values:
                row['latency'][stage]={f:describe([v[f] for v in values if v.get(f) is not None]) for f in FIELDS}
                row['latency'][stage]['scans_per_run']=statistics.mean(v['count'] for v in values)
        row['workload']={}
        names=set().union(*(r['analysis']['timings']['workload_all_scans'] for r in good))
        for name in names:
            row['workload'][name]={f:describe([r['analysis']['timings']['workload_all_scans'][name][f] for r in good]) for f in ('mean','max')}
        row['peak_rss_mib']=describe([r['resources']['peak_rss_kib']/1024 for r in good])
        row['sampled_gpu_peak_mib']=describe([r['gpu_memory']['sampled_peak_mib'] for r in good if r.get('gpu_memory',{}).get('sampled_peak_mib') is not None])
        row['trajectory_by_repeat']=[{'repeat':r['spec']['repeat'],**r['analysis']['trajectory']} for r in good]
        row['paired_comparison']=gates.get((sequence,config,backend))
        result.append(row)
    return result


def weighted_latency(rows,backend,stage,sequences,config='current'):
    selected=[r for r in rows if r['backend']==backend and r['config']==config and r['sequence'] in sequences]
    if {r['sequence'] for r in selected}!=set(sequences) or any(r['status']!='validated' for r in selected):
        return None
    if any(not r['latency'].get(stage,{}).get('mean_ms') for r in selected):
        return None
    count=sum(r['latency'][stage]['scans_per_run'] for r in selected)
    return sum(r['latency'][stage]['mean_ms']['mean']*r['latency'][stage]['scans_per_run'] for r in selected)/count


def scaling_results(groups):
    """Compare enlarged CUDA workloads with validated, matching CPU baselines.

    Ratios use means of per-run means. Quality deltas compare enlarged CUDA with
    current-workload CUDA on the same sequence, independently of speedup gates.
    Configured selection limits are never used as realized workload multipliers.
    """
    lookup={(r['sequence'],r['config'],r['backend']):r for r in groups}
    results=[]
    def ratio(a,b):
        return a/b if a is not None and b is not None and b>0 else None
    def latency(r,stage):
        return r['latency'][stage]['mean_ms']['mean']
    def workload(r,name):
        return r.get('workload',{}).get(name,{}).get('mean',{}).get('mean')
    def correspondences(r):
        plane,point=workload(r,'planar_correspondences'),workload(r,'point_correspondences')
        return plane+point if plane is not None and point is not None else None
    def rte(r,window):
        values=[t.get(window,{}).get('translation_m',{}).get('mean')
                for t in r['trajectory_by_repeat']]
        return statistics.mean(values) if values and all(v is not None for v in values) else None
    for row in groups:
        if row['backend']!='cuda' or row['config']=='current':
            continue
        seq,config=row['sequence'],row['config']
        current=lookup.get((seq,'current','cuda'))
        cpu=lookup.get((seq,config,'summary'))
        reference=lookup.get((seq,config,'reference'))
        required=[row,current,cpu,reference]
        result={'sequence':seq,'config':config}
        if any(r is None or r['status']!='validated' for r in required):
            result['status']='quality-failed' if any(r and r['status']=='quality-failed' for r in required) else 'incomplete'
            results.append(result)
            continue
        result.update(status='validated',
            mean_correspondences=correspondences(row),mean_poses=workload(row,'poses'),
            correspondence_multiplier=ratio(correspondences(row),correspondences(current)),
            pose_multiplier=ratio(workload(row,'poses'),workload(current,'poses')),
            cuda_optimization_latency_multiplier=ratio(latency(row,'optimization_ms'),latency(current,'optimization_ms')),
            cuda_scan_latency_multiplier=ratio(latency(row,'total_ms'),latency(current,'total_ms')),
            cpu_summary_over_cuda_optimization=ratio(latency(cpu,'optimization_ms'),latency(row,'optimization_ms')),
            cpu_summary_over_cuda_scan=ratio(latency(cpu,'total_ms'),latency(row,'total_ms')),
            reference_over_cuda_optimization=ratio(latency(reference,'optimization_ms'),latency(row,'optimization_ms')),
            reference_over_cuda_scan=ratio(latency(reference,'total_ms'),latency(row,'total_ms')))
        for window in ('rte_1m','rte_30m'):
            old,new=rte(current,window),rte(row,window)
            result[window+'_change_m']=new-old if new is not None and old is not None else None
            result[window+'_ratio']=ratio(new,old)
        result['coverage_change']=min(t['matched_poses'] for t in row['trajectory_by_repeat'])-min(t['matched_poses'] for t in current['trajectory_by_repeat'])
        results.append(result)
    return results


def markdown(summary):
    lines=['# FORM acceleration evaluation','',
           f"Verified completed runs: {summary['completed_runs']}/{summary['planned_runs']}.",
           '', 'Latencies below exclude the first 20 scans. Each entry averages the corresponding statistic across repeats; p95/p99 are averages of per-run percentiles. Raw records also report all scans, including startup. Quality gates apply to every repeat and all eight translational RTE statistics.',
           '', '| Workload | Sequence | Backend | Opt. mean ms | Scan mean ms | Scan p95 ms | Scan p99 ms | >100 ms % | Status |',
           '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary['groups']:
        def value(stage,field,scale=1):
            v=r['latency'].get(stage,{}).get(field)
            return f"{scale*v['mean']:.2f}" if v else '—'
        lines.append(f"| {r['config']} | {r['sequence']} | {r['backend']} | {value('optimization_ms','mean_ms')} | {value('total_ms','mean_ms')} | {value('total_ms','p95_ms')} | {value('total_ms','p99_ms')} | {value('total_ms','deadline_miss_fraction',100)} | {r['status']} |")
    lines.extend(['','## Accuracy and memory','',
                  'RTE entries average the per-repeat mean error. Coverage is the minimum matched-pose count across completed repeats. CPU/GPU memory entries are the maximum observed per-run peaks; GPU peaks are sampled and can miss short allocations.',
                  '', '| Workload | Sequence | Backend | RTE 1 m (m) | RTE 30 m (m) | Matched poses | CPU peak MiB | GPU sampled peak MiB |',
                  '|---|---|---|---:|---:|---:|---:|---:|'])
    for r in summary['groups']:
        def rte(window):
            values=[t[window]['translation_m']['mean'] for t in r['trajectory_by_repeat'] if t.get(window,{}).get('translation_m')]
            return f'{statistics.mean(values):.6f}' if values else '—'
        coverage=min((t['matched_poses'] for t in r['trajectory_by_repeat']),default='—')
        cpu=f"{r['peak_rss_mib']['max']:.1f}" if r['peak_rss_mib'] else '—'
        gpu=f"{r['sampled_gpu_peak_mib']['max']:.1f}" if r['sampled_gpu_peak_mib'] else '—'
        lines.append(f"| {r['config']} | {r['sequence']} | {r['backend']} | {rte('rte_1m')} | {rte('rte_30m')} | {coverage} | {cpu} | {gpu} |")
    if summary.get('scaling'):
        lines.extend(['','## Scaling tradeoffs','',
            'Workload and CUDA latency multipliers compare enlarged settings with current CUDA on the same sequence. Speedups compare matching enlarged CPU/CUDA settings. Ratios use means of per-run means. Positive RTE changes mean greater error; more correspondences do not imply better accuracy.',
            '', '| Sequence | Workload | Actual correspondences × | Actual poses × | CUDA opt. latency × | CPU summary / CUDA opt. | RTE 1 m change (m) | RTE 30 m change (m) | Status |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|'])
        for r in summary['scaling']:
            values=[f"{r[k]:.3f}" if r.get(k) is not None else '—' for k in
                    ('correspondence_multiplier','pose_multiplier','cuda_optimization_latency_multiplier',
                     'cpu_summary_over_cuda_optimization','rte_1m_change_m','rte_30m_change_m')]
            lines.append('| '+' | '.join([r['sequence'],r['config'],*values,r['status']])+' |')
    lines.extend(['','Machine-readable companion data contain all stages, repeat variability, objective values, LM iterations, realized feature/correspondence/pose counts, complete trajectory statistics, and per-repeat quality gates. Missing or failed comparisons do not qualify a backend as validated.',''])
    return '\n'.join(lines)


def trace_agreement(reference,candidate):
    import csv, math
    from decimal import Decimal
    def load(prefix):
        with Path(str(prefix)+'.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        with Path(str(prefix)+'.tum').open() as stream:
            poses=[line.split() for line in stream if line.strip() and not line.lstrip().startswith('#')]
        return rows,poses
    a,pa=load(reference);b,pb=load(candidate)
    if not a or len(a)!=len(b) or len(a)!=len(pa) or len(a)!=len(pb):
        raise ValueError('trace lengths differ or are empty')
    names=('poses','planar_features','point_features','rematches','lm_iterations','factors',
           'planar_correspondences','point_correspondences')
    changed={name:0 for name in names if name in a[0] and name in b[0]}
    objectives={name:[] for name in ('full_initial_error','full_final_error') if name in a[0] and name in b[0]}
    translation=[];rotation=[]
    for ar,br,ap,bp in zip(a,b,pa,pb):
        if ar['stamp_ns']!=br['stamp_ns'] or Decimal(ap[0])!=Decimal(bp[0]):
            raise ValueError('trace timestamps differ')
        for name in changed:
            changed[name]+=float(ar[name])!=float(br[name])
        for name in objectives:
            x,y=float(ar[name]),float(br[name]);objectives[name].append((abs(x-y),abs(x-y)/(1+abs(x))))
        av,bv=[float(v) for v in ap[1:]],[float(v) for v in bp[1:]]
        translation.append(math.sqrt(sum((x-y)**2 for x,y in zip(av[:3],bv[:3]))))
        qa,qb=av[3:],bv[3:]
        na,nb=math.sqrt(sum(x*x for x in qa)),math.sqrt(sum(x*x for x in qb))
        qa,qb=[x/na for x in qa],[x/nb for x in qb]
        if sum(x*y for x,y in zip(qa,qb))<0:
            qb=[-x for x in qb]
        difference=math.sqrt(sum((x-y)**2 for x,y in zip(qa,qb)))
        total=math.sqrt(sum((x+y)**2 for x,y in zip(qa,qb)))
        rotation.append(4*math.atan2(difference,total))
    return {'scans':len(a),'max_translation_difference_m':max(translation),
            'rms_translation_difference_m':math.sqrt(statistics.mean(x*x for x in translation)),
            'max_rotation_difference_rad':max(rotation),'changed_counts':changed,
            **{name:{'max_absolute_difference':max(x[0] for x in values),
                     'max_scaled_difference':max(x[1] for x in values)} for name,values in objectives.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True,help='Output prefix for JSON and Markdown')
    parser.add_argument('--tools-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    sys.path.insert(0,str(args.tools_dir))
    from run_suite import verify_resume
    manifests=[json.loads((suite/'suite.json').read_text()) for suite in args.suite]
    if len({m['provenance']['binary_sha256'] for m in manifests})!=1:
        raise ValueError('Cannot combine scaling suites built from different binaries')
    if len({json.dumps(m['provenance'].get('comparison_binary'),sort_keys=True) for m in manifests})!=1:
        raise ValueError('Cannot combine suites using different previous matching binaries')
    if len({json.dumps(m['quality_gate'],sort_keys=True) for m in manifests})!=1:
        raise ValueError('Cannot combine suites with different quality gates')
    records=[];comparisons=[];prefixes={};specs=[]
    for suite,manifest in zip(args.suite,manifests):
        report=json.loads((suite/'report.json').read_text())
        comparisons.extend(report['comparisons'])
        for spec in manifest['runs']:
            if spec['id'] in prefixes:
                raise ValueError('Duplicate run ID across suites: '+spec['id'])
            prefixes[spec['id']]=suite/spec['id']
            path=suite/(spec['id']+'.run.json')
            record=json.loads(path.read_text()) if path.exists() else {'status':'missing','spec':spec}
            if record['status']=='complete':
                verify_resume(record,suite/spec['id'],spec['expected_scans'],spec['fingerprint'])
            records.append(record);specs.append(spec)
    groups=grouped_results(records,comparisons)
    sequences=list(dict.fromkeys(s['sequence'] for s in specs))
    summary={'planned_runs':len(records),'completed_runs':sum(r['status']=='complete' for r in records),
             'all_quality_gates_pass':all(g['status']=='validated' for g in groups),
             'suites':[str(suite.resolve()) for suite in args.suite],
             'provenance':manifests[0]['provenance'],
             'provenance_by_suite':{str(suite.resolve()):manifest['provenance'] for suite,manifest in zip(args.suite,manifests)},
             'quality_gate_definition':manifests[0]['quality_gate'],'groups':groups,
             'scaling':scaling_results(groups),
             'current_weighted_latency_ms':{backend:{stage:weighted_latency(groups,backend,stage,sequences) for stage in ('optimization_ms','total_ms')}
                 for backend in dict.fromkeys(s['backend'] for s in specs)}}
    lookup={(r['spec']['sequence'],r['spec']['config'],r['spec']['repeat'],r['spec']['backend']):r for r in records}
    summary['trace_agreements']=[]
    for record in records:
        spec=record['spec']
        if spec['backend']=='reference' or record['status']!='complete':
            continue
        reference=lookup.get((spec['sequence'],spec['config'],spec['repeat'],'reference'),{})
        if reference.get('status')=='complete':
            summary['trace_agreements'].append({'candidate_run':spec['id'],'reference_run':reference['spec']['id'],
                **trace_agreement(prefixes[reference['spec']['id']],prefixes[spec['id']])})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(markdown(summary))
    print(f"Exported {summary['completed_runs']}/{summary['planned_runs']} verified completed runs")

if __name__=='__main__':
    main()
