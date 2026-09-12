#!/usr/bin/env python3
"""Build the host-only cuSolver experiment with ordinary, vectorized C++.

Do not run compilation or GPU measurements concurrently with another benchmark.
No production or CMake files are modified.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/tmp/form-cuda-solve-benchmark'))
    parser.add_argument('--eigen-include',type=Path,default=root.parent/'form-deps/local/usr/include/eigen3')
    parser.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    args=parser.parse_args();output=args.output.resolve();source=root/'benchmarks/cuda_solve_benchmark.cpp'
    command=['/usr/bin/c++','-O3','-DNDEBUG','-std=c++17','-I'+str(args.eigen_include),
             '-I'+str(args.cuda/'include'),str(source),'-L'+str(args.cuda/'lib64'),
             '-Wl,-rpath,'+str(args.cuda/'lib64'),'-lcusolver','-lcudart','-o',str(output)]
    subprocess.run(command,check=True)
    result={'built_at':datetime.now(timezone.utc).isoformat(),'command':command,
            'source_sha256':sha(source),'binary_sha256':sha(output),
            'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),
            'cpu_flags_note':'Same O3/NDEBUG/C++17 baseline as production; no march=native, fast-math, OpenMP, or NVCC. Eigen SIMD and thread count printed by executable.'}
    Path(str(output)+'.build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
