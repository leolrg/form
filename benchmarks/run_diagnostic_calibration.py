#!/usr/bin/env python3
"""ABCCBA comparison of original, disabled, and enabled diagnostics."""
import argparse
import copy
import json
from pathlib import Path
from run_suite import digest,execute,json_hash,now,provenance,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--previous-revision',required=True)
    args=p.parse_args();base=args.campaign.resolve();output=base/'calibration'
    template=json.loads((base/'clean-full/suite.json').read_text())
    original=next(r for r in template['runs'] if r['backend']=='cuda-extraction')
    current=Path(original['argv'][0]);previous=args.previous.resolve()
    if digest(current)==digest(previous):raise ValueError('calibration requires distinct old/current binaries')
    output.mkdir(exist_ok=True)
    sources={key:{**provenance(binary,output),'frozen_source_revision':revision}
             for key,binary,revision in [('pre-instrumentation',previous,args.previous_revision),
                                       ('diagnostic-disabled',current,template['provenance']['git_revision']),
                                       ('diagnostic-enabled',current,template['provenance']['git_revision'])]}
    runs=[]
    for backend,repeat in [('pre-instrumentation',1),('diagnostic-disabled',1),('diagnostic-enabled',1),('diagnostic-enabled',2),('diagnostic-disabled',2),('pre-instrumentation',2)]:
        spec=copy.deepcopy(original);spec.pop('fingerprint',None)
        spec.update(backend=backend,repeat=repeat,id=f'stairs-current-{backend}-t32-r{repeat}')
        spec['argv'][0]=str(previous if backend=='pre-instrumentation' else current)
        spec['argv'][spec['argv'].index('--output')+1]=str(output/spec['id'])
        if backend=='diagnostic-enabled':spec['argv'].append('--profile')
        spec['binary_sha256']=digest(Path(spec['argv'][0]));spec['fingerprint']=json_hash(spec);runs.append(spec)
    manifest={'created_at':now(),'runs':runs,'provenance':sources,'order':'ABCCBA; old binary / diagnostics disabled / diagnostics enabled; no trace flags'}
    path=output/'suite.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if [r['fingerprint'] for r in saved['runs']]!=[r['fingerprint'] for r in runs]:raise ValueError('calibration changed')
    else:save(path,manifest)
    for spec in runs:
        if digest(Path(spec['argv'][0]))!=spec['binary_sha256']:raise ValueError('binary changed')
        execute(spec,output,sources[spec['backend']])
        record=json.loads((output/(spec['id']+'.run.json')).read_text())
        if record['status']!='complete':raise RuntimeError(record)

if __name__=='__main__':main()
