#!/usr/bin/env python3
"""Sequential diagnostic campaign. Freeze/review the executable before running.

All commands and completion records are kept by run_suite.py. A failed command
stops the campaign; existing incomplete runs are never overwritten.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,default=Path('/home/ubuntu/datasets/form-input'))
    p.add_argument('--plan',action='store_true');args=p.parse_args()
    commands=[]
    def suite(name,backends,configs,repeats=2,limit=None,profile=False,threshold=None):
        cmd=[sys.executable,str(ROOT/'benchmarks/run_suite.py'),'--binary',str(args.binary.resolve()),
             '--input-dir',str(args.input_dir.resolve()),'--output',str(args.output.resolve()/name),
             '--sequences','stairs','--backends',*backends,'--configs',*configs,
             '--threads','32','--repeats',str(repeats),'--gpu-sample-interval','0']
        if limit:cmd+=['--limit',str(limit)]
        if profile:cmd+=['--profile']
        if threshold:cmd+=['--cuda-solve-min-dimension',str(threshold)]
        commands.append(cmd)
    suite('clean-full',['reference','cpu-extraction','cuda-extraction'],['current'],3)
    suite('clean-scaling',['reference','cpu-extraction','cuda-extraction'],['features','window'],limit=250)
    suite('profile-full',['cpu-extraction','cuda-extraction'],['current'],profile=True)
    suite('profile-scaling',['cpu-extraction','cuda-extraction'],['features','window'],limit=250,profile=True)
    # Matching/extraction stay CUDA; only the post-QR optimizer policy changes.
    suite('force-cpu',['cuda-extraction'],['current','window'],limit=250,profile=True,threshold=6000)
    suite('force-gpu',['cuda-extraction'],['current','window'],limit=250,profile=True,threshold=1)
    for cmd in commands:
        print('RUN',cmd,flush=True)
        if not args.plan:subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
