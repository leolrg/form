#!/usr/bin/env python3
"""Compare CPU/GPU resident optimizers on matching real invocation workloads."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from analyze_diagnostics import optimizer_manifest


def pair_calls(cpu,gpu):
    def index(rows):
        out={}
        for r in rows:
            if int(r['scan'])<20:continue
            key=(int(r['scan']),int(r['call']),r['phase'],int(r['dimension']))
            if key in out:raise ValueError('duplicate invocation')
            out[key]=r
        return out
    a,b=index(cpu),index(gpu)
    if a.keys()!=b.keys():raise ValueError('CPU/GPU optimizer invocations differ')
    return [(key,a[key],b[key]) for key in sorted(a)]


def components(r):
    def ms(*names):return sum(float(r['diag_'+n+'_ms']) for n in names)
    out=dict(setup_ms=ms('optimizer_summary','optimizer_graph','optimizer_values','cpu_reset','gpu_reset'),
             reset_ms=ms('cpu_reset','gpu_reset'),
             linearize_ms=ms('cpu_linearize','gpu_linearize'),
             solve_ms=ms('cpu_solve','gpu_solve'),error_ms=ms('cpu_error','gpu_error'),
             retraction_ms=ms('retraction'),wall_ms=float(r['wall_ms']))
    out['other_ms']=out['wall_ms']-sum(out[k] for k in ('setup_ms','linearize_ms','solve_ms','error_ms','retraction_ms'))
    return out


def summarize(pairs):
    cpu=[components(a) for _,a,_ in pairs];gpu=[components(b) for _,_,b in pairs]
    means=lambda rows:{k:statistics.mean(r[k] for r in rows) for k in rows[0]}
    a,b=means(cpu),means(gpu)
    return dict(pairs=len(pairs),cpu=a,gpu=b,complete_call_cpu_over_gpu=a['wall_ms']/b['wall_ms'],
                solve_cpu_over_gpu=a['solve_ms']/b['solve_ms'] if b['solve_ms'] else None,
                changed_trial_counts={k:sum(x[k]!=y[k] for _,x,y in pairs) for k in ('diag_lm_accepted_calls','diag_lm_rejected_calls','diag_lm_solve_failed_calls')})


def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);args=p.parse_args();out={}
    optimizer_manifest(args.campaign)
    for config in ('current','window'):
        by_dimension=defaultdict(list);by_phase=defaultdict(list);per_repeat=[]
        for repeat in (1,2):
            def read(mode):
                path=args.campaign/mode/f'stairs-{config}-cuda-extraction-t32-r{repeat}.optimizer.csv'
                with path.open() as f:return list(csv.DictReader(f))
            paired=pair_calls(read('force-cpu'),read('force-gpu'))
            per_repeat.append(dict(repeat=repeat,phases={phase:summarize([p for p in paired if p[0][2]==phase]) for phase in sorted({p[0][2] for p in paired})}))
            for pair in paired:
                key,a,b=pair
                if a['backend']!='cpu' or b['backend']!='gpu':raise ValueError('forced policy not applied')
                by_dimension[(key[2],key[3])].append(pair);by_phase[key[2]].append(pair)
        out[config]=dict(per_repeat=per_repeat,phases={k:summarize(v) for k,v in by_phase.items()},
            dimensions=[dict(phase=phase,dimension=dimension,**summarize(v)) for (phase,dimension),v in sorted(by_dimension.items())])
    (args.campaign/'optimizer-policy.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for config,v in out.items():
        for phase,r in v['phases'].items():print(config,phase,'CPU/GPU call ratio',r['complete_call_cpu_over_gpu'],'solve ratio',r['solve_cpu_over_gpu'])

if __name__=='__main__':main()
