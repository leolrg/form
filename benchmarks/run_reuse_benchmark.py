#!/usr/bin/env python3
"""Sequential old/new optimizer-reuse controls; immutable binaries, matched workloads."""
import argparse
import fcntl
import json
from pathlib import Path
from run_suite import CONFIGS, digest, execute, json_hash, now, provenance, save


def schedule():
    runs=[]
    for suite,config,limit in [('clean-full','current',None),('clean-scaling','features',250),('clean-scaling','window',250)]:
        for repeat in (1,2):
            order=[('old','cuda-extraction'),('new','cuda-extraction'),('old','cpu-extraction'),('new','cpu-extraction')]
            if repeat==2:order.reverse()
            runs.extend((suite,config,limit,repeat,version,backend,False) for version,backend in order)
    # Third full CUDA pair improves resolution for the smaller normal-workload gain.
    runs.extend(('clean-full','current',None,3,version,'cuda-extraction',False) for version in ('old','new'))
    for config,limit in [('current',None),('window',250)]:
        runs.extend(('profile',config,limit,1,version,'cuda-extraction',True) for version in ('old','new'))
    return runs


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--previous-revision',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--input-dir',type=Path,default=Path('/home/ubuntu/datasets/form-input'))
    p.add_argument('--plan',action='store_true');args=p.parse_args()
    base=args.output.resolve();inputs=args.input_dir.resolve()
    binaries={'old':args.previous.resolve(),'new':args.binary.resolve()}
    hashes={k:digest(v) for k,v in binaries.items()}
    if hashes['old']==hashes['new']:raise ValueError('old/new binaries must differ')
    prov={k:{**provenance(v,base),'frozen_source_revision':args.previous_revision if k=='old' else args.revision} for k,v in binaries.items()}
    meta_path=inputs/'stairs.json';meta=json.loads(meta_path.read_text())
    if (inputs/'stairs.formpc').stat().st_size!=meta['bytes']:raise ValueError('input size changed')
    input_sha256=digest(inputs/'stairs.formpc')
    runs=[]
    for suite,config,limit,repeat,version,backend,profile in schedule():
        name=f'stairs-{config}-{version}-{backend}-r{repeat}'
        folder=base/suite;settings={**CONFIGS[config],**meta.get('replay_settings',{})}
        argv=[str(binaries[version]),'--input',str(inputs/'stairs.formpc'),'--output',str(folder/name),'--backend',backend,'--threads','32']
        for k,v in settings.items():argv.extend(['--'+k,str(v)])
        if limit:argv.extend(['--limit',str(limit)])
        if profile:argv.append('--profile')
        spec=dict(id=name,suite=suite,version=version,sequence='stairs',config=config,backend=backend,repeat=repeat,threads=32,
                  settings=settings,expected_scans=limit or meta['scans'],full_sequence=limit is None,
                  source_scans=meta['scans'],gpu_sample_interval_seconds=0,ground_truth=str(inputs/'stairs.gt.tum'),
                  input_sha256=input_sha256,
                  input_metadata_sha256=digest(meta_path),ground_truth_sha256=digest(inputs/'stairs.gt.tum'),
                  binary_sha256=hashes[version],frozen_source_revision=prov[version]['frozen_source_revision'],argv=argv)
        identity={k:prov[version][k] for k in ('environment','cpu_affinity')}
        spec['fingerprint']=json_hash(dict(spec=spec,runtime_identity=identity));runs.append(spec)
    manifest=dict(created_at=now(),provenance=prov,runs=runs,warmup=20,
                  current_prefix='For matched scaling, also analyze scans 20..249 of clean-full runs.')
    if args.plan:print(json.dumps(manifest,indent=2));return
    base.mkdir(parents=True,exist_ok=True)
    with (base/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=base/'suite.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if [r['fingerprint'] for r in saved['runs']]!=[r['fingerprint'] for r in runs]:raise ValueError('campaign changed')
            manifest=saved
        else:save(path,manifest)
        for spec in manifest['runs']:
            folder=base/spec['suite'];folder.mkdir(exist_ok=True)
            if digest(Path(spec['argv'][0]))!=spec['binary_sha256']:raise ValueError('binary changed')
            resumed=(folder/(spec['id']+'.run.json')).exists()
            execute(spec,folder,manifest['provenance'][spec['version']])
            record=json.loads((folder/(spec['id']+'.run.json')).read_text())
            if record['status']!='complete':raise RuntimeError(record.get('error','run failed'))
            if '--profile' in spec['argv']:
                path=folder/(spec['id']+'.optimizer.csv');hashed=digest(path)
                if resumed and 'optimizer_sha256' not in record:raise ValueError('resumed optimizer artifact lacks completion hash')
                if record.get('optimizer_sha256',hashed)!=hashed:raise ValueError('optimizer artifact changed')
                record['optimizer_sha256']=hashed
                save(folder/(spec['id']+'.run.json'),record)

if __name__=='__main__':main()
