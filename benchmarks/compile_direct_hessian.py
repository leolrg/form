#!/usr/bin/env python3
"""Build the isolated direct-Hessian experiment with this checkout's FORM flags.

Coordinate with other GPU benchmark work before invoking compilation or timing.
Does not modify CMake files or compile/link production targets.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=root/'build-accel')
    parser.add_argument('--output',type=Path,default=Path('/tmp/form-direct-hessian'))
    args=parser.parse_args();build=args.build_dir.resolve();output=args.output.resolve()
    nvcc=shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('nvcc is required')
    cuda=[nvcc,'-O3','-std=c++17','-arch=sm_80','-Xcompiler=-fPIC','-c',
          str(root/'benchmarks/direct_hessian.cu'),'-o',str(output)+'.cuda.o']
    commands=json.loads((build/'compile_commands.json').read_text())
    host=shlex.split(next(c['command'] for c in commands if c['file'].endswith('benchmarks/qr_benchmark.cpp')))
    host[host.index('-o')+1]=str(output)+'.host.o'
    host[-1]=str(root/'benchmarks/direct_hessian_benchmark.cpp')
    link=shlex.split((build/'CMakeFiles/form-qr-benchmark.dir/link.txt').read_text())
    link[link.index('-o')+1]=str(output)
    original=next(i for i,arg in enumerate(link) if arg.endswith('.o'))
    link[original:original+1]=[str(output)+'.host.o',str(output)+'.cuda.o']
    for command in (cuda,host,link):
        print(shlex.join(command),flush=True)
        subprocess.run(command,cwd=build,check=True)
    metadata={'built_at':datetime.now(timezone.utc).isoformat(),'argv':[cuda,host,link],
              'binary_sha256':digest(output),'libform_sha256':digest(build/'libFORM.a'),
              'source_sha256':{name:digest(root/name) for name in (
                  'benchmarks/direct_hessian.cu','benchmarks/direct_hessian.hpp',
                  'benchmarks/direct_hessian_benchmark.cpp','benchmarks/qr_capture.hpp',
                  'form/feature/cuda_qr.cu','form/feature/cuda_qr.hpp','form/feature/summary.cpp')}}
    Path(str(output)+'.build.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
