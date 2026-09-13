#!/usr/bin/env python3
"""Run isolated, sequential FORM comparisons and aggregate verified results.

Defaults: four sequences, reference + summary, two repeats, three independent
workloads. Existing incomplete, running, failed, or incompatible runs are never
restarted or overwritten: use a new output directory after investigating them.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

from analyze_n21 import timing_metrics, trajectory_metrics

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ('stairs','quad_easy','quad_hard','maths_hard')
CONFIGS = {'current':{'points':3,'planes':50,'recent':10},
           'features':{'points':6,'planes':100,'feature-spacing':2,'recent':10},
           'window':{'points':3,'planes':50,'recent':40}}
GATE = {'metrics':[f'{window}.translation_m.{metric}'
                   for window in ('rte_1m','rte_30m')
                   for metric in ('mean','median','rmse','max')],
        'rule':'candidate <= reference + max(0.05 * reference, 0.01 metres)',
        'coverage':'candidate matched poses must not decrease',
        'rotation':'reported; no predeclared rotational acceptance threshold'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def capture(argv):
    try:
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=30)
        return {'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'argv':argv,'error':str(exc)}


def provenance(binary, output):
    diff=subprocess.run(['git','diff','HEAD','--binary'],cwd=ROOT,capture_output=True,check=True).stdout
    untracked=subprocess.run(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT,
                             capture_output=True,check=True).stdout.decode().split('\0')
    source_untracked={}
    for name in untracked:
        if not name:
            continue
        path=ROOT/name
        if path.is_relative_to(output) or path.is_relative_to(ROOT/'benchmarks/results'):
            continue
        if path.is_file():
            source_untracked[name]=digest(path)
    env_names=('PATH','LD_LIBRARY_PATH','CUDA_HOME','CUDA_PATH','CC','CXX',
               'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TBB_NUM_THREADS',
               'CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING','OMP_PROC_BIND','OMP_PLACES')
    return {'captured_at':now(),'binary':str(binary),'binary_sha256':digest(binary),
            'git_revision':capture(['git','rev-parse','HEAD'])['stdout'].strip(),
            'git_dirty_diff_sha256':hashlib.sha256(diff).hexdigest(),
            'git_tracked_dirty':bool(diff),'untracked_file_sha256':source_untracked,
            'git_status':capture(['git','status','--short']),
            'cpu':capture(['lscpu']),'gpu':capture(['nvidia-smi']),
            'cuda':capture(['nvcc','--version']), 'memory':Path('/proc/meminfo').read_text(),
            'platform':platform.platform(),'python':sys.version,'python_executable':sys.executable,
            'environment':{name:os.environ[name] for name in env_names if name in os.environ},
            'cpu_affinity':sorted(os.sched_getaffinity(0))}


def verify_outputs(prefix,expected_count):
    csv_path=Path(str(prefix)+'.csv');tum_path=Path(str(prefix)+'.tum')
    with csv_path.open() as stream:
        rows=list(csv.DictReader(stream))
    with tum_path.open() as stream:
        poses=[line.split() for line in stream if line.strip() and not line.lstrip().startswith('#')]
    if len(rows)!=expected_count or len(poses)!=expected_count:
        raise ValueError(f'expected {expected_count} records, found CSV={len(rows)}, TUM={len(poses)}')
    previous=None
    for index,(row,pose) in enumerate(zip(rows,poses)):
        stamp=int(row['stamp_ns'])
        if int(row['scan'])!=index or (previous is not None and stamp<=previous):
            raise ValueError('invalid scan order or timestamps')
        if len(pose)!=8 or int(Decimal(pose[0])*Decimal(1_000_000_000))!=stamp:
            raise ValueError('trajectory/timing timestamp mismatch')
        if not all(math.isfinite(float(v)) for v in pose[1:]):
            raise ValueError('non-finite trajectory')
        previous=stamp
    return {'count':len(rows),'csv_sha256':digest(csv_path),'tum_sha256':digest(tum_path)}


def verify_resume(record,prefix,expected_count,fingerprint):
    if record.get('status')!='complete':
        raise ValueError(f"run status is {record.get('status')}; refusing to restart existing output")
    if record.get('fingerprint')!=fingerprint:
        raise ValueError('run fingerprint changed; use a new output directory')
    if verify_outputs(prefix,expected_count)!=record.get('outputs'):
        raise ValueError('completed output content changed')


def quality_gate(reference,candidate):
    checks={};missing=False
    for window in ('rte_1m','rte_30m'):
        for metric in ('mean','median','rmse','max'):
            name=f'{window}.translation_m.{metric}'
            r=(reference.get(window,{}).get('translation_m') or {}).get(metric)
            c=(candidate.get(window,{}).get('translation_m') or {}).get(metric)
            if r is None or c is None or not math.isfinite(r) or not math.isfinite(c):
                checks[name]={'status':'missing'};missing=True
            else:
                threshold=r+max(.05*r,.01)
                checks[name]={'reference':r,'candidate':c,'maximum':threshold,'status':'pass' if c<=threshold else 'fail'}
    rc=reference.get('matched_poses');cc=candidate.get('matched_poses')
    checks['coverage']={'reference':rc,'candidate':cc,'status':'missing' if rc is None or cc is None
                        else ('pass' if cc>=rc else 'fail')}
    missing=missing or checks['coverage']['status']=='missing'
    status='fail' if any(c['status']=='fail' for c in checks.values()) else ('missing' if missing else 'pass')
    return {'status':status,'checks':checks}


def variability(values):
    if not values:
        return None
    mean=statistics.mean(values)
    return {'runs':len(values),'mean':mean,'min':min(values),'max':max(values),
            'sample_stddev':statistics.stdev(values) if len(values)>1 else None,
            'coefficient_of_variation':statistics.stdev(values)/mean if len(values)>1 and mean else None}


def aggregate(manifest,output):
    runs=[];records={}
    for spec in manifest['runs']:
        path=output/(spec['id']+'.run.json')
        if not path.exists():
            record={'status':'missing'}
        else:
            record=json.loads(path.read_text())
            if record.get('status')=='complete':
                try:
                    verify_resume(record,output/spec['id'],spec['expected_scans'],spec['fingerprint'])
                except (OSError,ValueError,KeyError) as exc:
                    record={**record,'status':'invalid','error':str(exc)}
        records[spec['id']]=record
        runs.append({'id':spec['id'],'status':record.get('status','missing'),
                     **({'error':record['error']} if 'error' in record else {})})
    groups={}
    for spec in manifest['runs']:
        key=(spec['sequence'],spec['config'],spec['backend'])
        groups.setdefault(key,[]).append((spec,records[spec['id']]))
    group_results=[]
    for (seq,config,backend),items in groups.items():
        good=[record for _,record in items if record.get('status')=='complete']
        latency={}
        for region in ('all_scans','steady_state'):
            latency[region]={}
            for stage in ('optimization_ms','total_ms','marginalize_ms'):
                latency[region][stage]={field:variability([r['analysis']['timings'][region][stage][field] for r in good
                                              if r['analysis']['timings'][region].get(stage)])
                                       for field in ('mean_ms','median_ms','p95_ms','p99_ms','throughput_hz','deadline_miss_fraction')}
        group_results.append({'sequence':seq,'config':config,'backend':backend,'planned_repeats':len(items),
                              'completed_repeats':len(good),'latency_variability':latency,
                              'peak_rss_kib':variability([r['resources']['peak_rss_kib'] for r in good]),
                              'sampled_gpu_peak_mib':variability([r['gpu_memory']['sampled_peak_mib'] for r in good
                                  if r.get('gpu_memory',{}).get('sampled_peak_mib') is not None])})
    comparisons=[]
    for (seq,config,backend),items in groups.items():
        if backend=='reference':
            continue
        refs={spec['repeat']:record for spec,record in groups.get((seq,config,'reference'),[])}
        pairs=[]
        for spec,candidate in items:
            reference=refs.get(spec['repeat'],{})
            if reference.get('status')!='complete' or candidate.get('status')!='complete':
                pairs.append({'repeat':spec['repeat'],'status':'missing','reference_status':reference.get('status','missing'),
                              'candidate_status':candidate.get('status','missing')})
                continue
            speedups={}
            for stage in ('optimization_ms','total_ms'):
                a=reference['analysis']['timings']['steady_state'].get(stage)
                b=candidate['analysis']['timings']['steady_state'].get(stage)
                speedups[stage]=a['mean_ms']/b['mean_ms'] if a and b and b['mean_ms'] else None
            gate=quality_gate(reference['analysis'].get('trajectory',{}),candidate['analysis'].get('trajectory',{}))
            pairs.append({'repeat':spec['repeat'],'status':'complete','steady_state_speedup':speedups,'quality_gate':gate})
        valid=[p for p in pairs if p['status']=='complete']
        status='fail' if any(p['quality_gate']['status']=='fail' for p in valid) else (
            'pass' if len(valid)==len(pairs) and all(p['quality_gate']['status']=='pass' for p in valid) else 'missing')
        comparisons.append({'sequence':seq,'config':config,'candidate':backend,'quality_gate_status':status,
                            'paired_runs':pairs,'steady_state_speedup_variability':{
                                stage:variability([p['steady_state_speedup'][stage] for p in valid if p['steady_state_speedup'][stage] is not None])
                                for stage in ('optimization_ms','total_ms')}})
    report={'generated_at':now(),'quality_gate_definition':GATE,
            'full_sequence_runs':all(s.get('full_sequence',False) for s in manifest['runs']),
            'runs':runs,'groups':group_results,'comparisons':comparisons}
    save(output/'report.json',report)
    lines=['# FORM suite report','',f"Completed {sum(r['status']=='complete' for r in runs)}/{len(runs)} planned runs.",'',
           'Speedups compare repeated runs at identical workload settings; quality gates apply to each repeat.',
           'Coverage: '+('complete sequences' if report['full_sequence_runs'] else 'includes limited development inputs')+'.','',
           '| Sequence | Workload | Backend | Optimization speedup | Scan speedup | Quality gate |',
           '|---|---|---|---:|---:|---|']
    for c in comparisons:
        def fmt(stage):
            value=c['steady_state_speedup_variability'][stage]
            return f"{value['mean']:.3f}x" if value else 'missing'
        lines.append(f"| {c['sequence']} | {c['config']} | {c['candidate']} | {fmt('optimization_ms')} | {fmt('total_ms')} | {c['quality_gate_status']} |")
    lines.extend(['','Incomplete or failed runs:',''])
    lines.extend(f"- {r['id']}: {r['status']} {r.get('error','')}" for r in runs if r['status']!='complete')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def gpu_memory_from_csv(text,pids):
    values=[]
    for line in text.splitlines():
        fields=line.split(',')
        if len(fields)!=2:
            continue
        try:
            pid,memory=int(fields[0].strip()),int(fields[1].strip())
        except ValueError:
            continue
        if pid in pids and memory>=0:
            values.append(memory)
    return sum(values) if values else None


def process_tree(pid):
    result={pid};pending=[pid]
    while pending:
        current=pending.pop()
        try:
            children=Path(f'/proc/{current}/task/{current}/children').read_text().split()
        except OSError:
            continue
        for child in map(int,children):
            if child not in result:
                result.add(child);pending.append(child)
    return result


def sample_gpu_memory(pid,interval,stop,result):
    result.update({'sample_interval_seconds':interval,'sampled_peak_mib':None,'queries':0,
                   'samples_with_target_process':0,'query_failures':0,
                   'limitation':'nvidia-smi process memory sampled across replay descendants; short allocations may be missed; includes CUDA context and allocator reservations. Sampling can perturb timing. Null means no target GPU process was observed.'})
    while not stop.is_set():
        start=time.monotonic()
        pids=process_tree(pid)
        try:
            sample=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'],capture_output=True,text=True,
                                  timeout=min(interval,2))
            result['queries']+=1
            if sample.returncode:
                result['query_failures']+=1
            else:
                used=gpu_memory_from_csv(sample.stdout,pids)
                if used is not None:
                    result['samples_with_target_process']+=1
                    result['sampled_peak_mib']=max(used,result['sampled_peak_mib'] or 0)
        except (OSError,subprocess.TimeoutExpired):
            result['query_failures']+=1
        stop.wait(max(0,interval-(time.monotonic()-start)))


def execute(spec,output,provenance_info):
    prefix=output/spec['id'];record_path=Path(str(prefix)+'.run.json')
    if record_path.exists():
        record=json.loads(record_path.read_text())
        verify_resume(record,prefix,spec['expected_scans'],spec['fingerprint'])
        print(f"resume verified: {spec['id']}",flush=True)
        return
    if any(output.glob(spec['id']+'.*')):
        raise ValueError(f'orphaned output exists for {spec["id"]}; use a new output directory')
    time_path=Path(str(prefix)+'.time.txt')
    argv=['/usr/bin/time','-f','%M %e %U %S %x','-o',str(time_path),'--',*spec['argv']]
    record={'status':'running','started_at':now(),'suite_pid':os.getpid(),'fingerprint':spec['fingerprint'],
            'spec':spec,'argv':argv,'provenance':provenance_info}
    save(record_path,record)
    try:
        with Path(str(prefix)+'.log').open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            record['child_pid']=child.pid;save(record_path,record)
            interval=spec.get('gpu_sample_interval_seconds',1)
            memory={};stop=threading.Event();sampler=None
            if interval>0:
                sampler=threading.Thread(target=sample_gpu_memory,args=(child.pid,interval,stop,memory),daemon=True)
                sampler.start()
            try:
                code=child.wait()
            finally:
                stop.set()
                if sampler is not None:
                    sampler.join(timeout=3)
                record['gpu_memory']=memory if interval>0 else {'enabled':False,'sampled_peak_mib':None}
        record['returncode']=code
        if code:
            raise RuntimeError(f'replay exited {code}; see {prefix}.log')
        record['outputs']=verify_outputs(prefix,spec['expected_scans'])
        resource=time_path.read_text().strip().split()
        record['resources']={'peak_rss_kib':int(resource[0]),'process_wall_seconds':float(resource[1]),
                             'user_cpu_seconds':float(resource[2]),'system_cpu_seconds':float(resource[3]),'exit_code':int(resource[4])}
        record['analysis']={'timings':timing_metrics(Path(str(prefix)+'.csv'),20),
                            'trajectory':trajectory_metrics(Path(str(prefix)+'.tum'),Path(spec['ground_truth']))}
        record['status']='complete'
    except Exception as exc:
        record['status']='failed';record['error']=str(exc)
    finally:
        record['finished_at']=now();save(record_path,record)
    print(f"{record['status']}: {spec['id']} {record.get('error','')}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build-accel/form-replay')
    parser.add_argument('--input-dir',type=Path,default=Path('/home/ubuntu/datasets/form-input'))
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/suite')
    parser.add_argument('--sequences',nargs='+',choices=SEQUENCES,default=SEQUENCES)
    parser.add_argument('--configs',nargs='+',choices=CONFIGS,default=list(CONFIGS))
    parser.add_argument('--sequence-first',action='store_true',
                        help='finish all workload settings for each sequence before the next')
    parser.add_argument('--backends',nargs='+',choices=('reference','summary','cuda','summary-batch','cuda-batch','summary-resident','cuda-resident','cuda-resident-hybrid'),default=['reference','summary'])
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--cuda-solve-min-dimension',type=int,
                        help='opt-in selective dense solving for cuda backend only')
    parser.add_argument('--cuda-solve-configs',nargs='+',choices=CONFIGS,
                        help='restrict legacy selective CUDA solving to these workloads; incompatible with resident hybrid')
    parser.add_argument('--gpu-sample-interval',type=float,default=1,help='GPU process memory polling seconds; zero disables sampling')
    parser.add_argument('--plan',action='store_true',help='print run descriptions; do not create outputs or run replay')
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    if args.cuda_solve_min_dimension is not None and args.cuda_solve_min_dimension < 1:
        parser.error('CUDA solve minimum dimension must be positive')
    if args.cuda_solve_configs is not None and 'cuda-resident-hybrid' in args.backends:
        parser.error('cuda-resident-hybrid selects every workload by dimension; omit --cuda-solve-configs')
    if args.cuda_solve_configs is not None and args.cuda_solve_min_dimension is None:
        parser.error('--cuda-solve-configs requires --cuda-solve-min-dimension')
    if args.threads<=0 or args.repeats<2 or args.gpu_sample_interval<0 or (args.limit is not None and args.limit<=0):
        parser.error('threads/limit must be positive, repeats at least two, GPU interval nonnegative')
    args.output=args.output.resolve();args.binary=args.binary.resolve();args.input_dir=args.input_dir.resolve()
    if args.aggregate_only:
        with (args.output/'.suite.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            aggregate(json.loads((args.output/'suite.json').read_text()),args.output)
        return
    prov=provenance(args.binary,args.output)
    identity={k:prov[k] for k in ('binary_sha256','git_revision','git_dirty_diff_sha256','untracked_file_sha256','environment','cpu_affinity')}
    runs=[]
    for outer in dict.fromkeys(args.sequences if args.sequence_first else args.configs):
        for inner in dict.fromkeys(args.configs if args.sequence_first else args.sequences):
            config,seq = (inner,outer) if args.sequence_first else (outer,inner)
            meta_path=args.input_dir/f'{seq}.json';meta=json.loads(meta_path.read_text())
            input_path=args.input_dir/f'{seq}.formpc';gt_path=args.input_dir/f'{seq}.gt.tum'
            if input_path.stat().st_size!=meta['bytes']:
                raise ValueError(f'input size does not match metadata: {seq}')
            expected=min(args.limit,meta['scans']) if args.limit else meta['scans']
            for repeat in range(1,args.repeats+1):
                backend_order=list(dict.fromkeys(args.backends))
                if repeat%2==0:
                    backend_order.reverse()
                for backend in backend_order:
                    run_id=f'{seq}-{config}-{backend}-t{args.threads}-r{repeat}'
                    prefix=args.output/run_id
                    argv=[str(args.binary),'--input',str(input_path),'--output',str(prefix),
                          '--backend',backend,'--threads',str(args.threads)]
                    for name,value in CONFIGS[config].items():
                        argv.extend(['--'+name,str(value)])
                    if args.limit:
                        argv.extend(['--limit',str(args.limit)])
                    if args.profile:
                        argv.append('--profile')
                    if (backend in ('cuda','cuda-batch','cuda-resident-hybrid') and args.cuda_solve_min_dimension is not None
                            and (args.cuda_solve_configs is None or config in args.cuda_solve_configs)):
                        argv.extend(['--cuda-solve-min-dimension',str(args.cuda_solve_min_dimension)])
                    spec={'id':run_id,'sequence':seq,'config':config,'backend':backend,'repeat':repeat,
                          'threads':args.threads,'settings':CONFIGS[config],'expected_scans':expected,
                          'gpu_sample_interval_seconds':args.gpu_sample_interval,
                          'source_scans':meta.get('source_scans',meta['scans']),
                          'full_sequence':meta.get('complete_sequence',False) and expected==meta['scans'],
                          'ground_truth':str(gt_path),'input_metadata_sha256':digest(meta_path),
                          'ground_truth_sha256':digest(gt_path),'argv':argv}
                    spec['fingerprint']=json_hash({'spec':spec,'build':identity})
                    runs.append(spec)
    manifest={'created_at':now(),'provenance':prov,'quality_gate':GATE,'runs':runs}
    if args.plan:
        print(json.dumps(manifest,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=args.output/'suite.json'
        if manifest_path.exists():
            existing=json.loads(manifest_path.read_text())
            if [r['fingerprint'] for r in existing['runs']] != [r['fingerprint'] for r in runs]:
                raise ValueError('suite configuration/source changed; use a new output directory')
            manifest=existing
        else:
            save(manifest_path,manifest)
        aggregate(manifest,args.output)
        for spec in manifest['runs']:
            if digest(args.binary)!=manifest['provenance']['binary_sha256']:
                raise ValueError('binary changed during suite; stopping')
            execute(spec,args.output,manifest['provenance'])
            aggregate(manifest,args.output)
        report=aggregate(manifest,args.output)
        if any(r['status']!='complete' for r in report['runs']):
            raise SystemExit(1)


if __name__=='__main__':
    main()
