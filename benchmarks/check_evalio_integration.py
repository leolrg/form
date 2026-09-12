#!/usr/bin/env python3
"""Check a built evalio extension and compare a bounded replay with C++.

Run with evalio's Python environment, e.g.:
  python benchmarks/check_evalio_integration.py --module-dir build-python/python \
    --input /home/ubuntu/datasets/form-input/stairs.formpc \
    --replay build-accel/form-replay --output /tmp/form-evalio-check

The prepared FORMPC input retains N21 point order and timestamps. Both paths use
0.1--50 m range, the input dimensions, identity IMU/LiDAR extrinsics, and the same
thread count. The comparison captures immediate world_T_lidar output per scan.
"""
import argparse
import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys


def load_extension(module_dir):
    sys.path.insert(0, str(module_dir.resolve()))
    import evalio
    core = importlib.import_module('_core')
    pipeline = core.FORMDev()
    defaults = pipeline.default_params()
    assert defaults['use_summary'] is False
    assert defaults['use_cuda_summaries'] is False
    assert defaults['use_cuda_dense_solver'] is False
    assert defaults['cuda_solve_min_dimension'] == 240
    assert defaults['feature_spacing'] == 0
    assert isinstance(pipeline, evalio.pipelines.Pipeline)
    return pipeline



def check_abi_guard(module_dir):
    # Run in a child: a regression here used to abort the entire interpreter.
    code = """
import sys, evalio
sys.path.insert(0, sys.argv[1])
evalio._cpp.abi_tag = lambda: 'incompatible-test-ABI'
try:
    import _core
except ImportError as error:
    assert 'incompatible nanobind ABIs' in str(error), str(error)
else:
    raise AssertionError('Expected ImportError for incompatible evalio ABI')
"""
    subprocess.run([sys.executable, '-c', code, str(module_dir.resolve())], check=True)


def run_pipeline(args, backend):
    import numpy as np
    from evalio.types import LidarMeasurement, LidarParams, SE3, Stamp
    pipeline = load_extension(args.module_dir)
    options = {'use_summary': backend != 'reference',
               'use_cuda_summaries': backend == 'cuda', 'num_threads': args.threads,
               'feature_spacing': args.feature_spacing}
    if backend == 'cuda' and args.cuda_solve_min_dimension is not None:
        options.update(use_cuda_dense_solver=True,
                       cuda_solve_min_dimension=args.cuda_solve_min_dimension)
    # evalio returns unrecognized parameters; an empty map confirms acceptance.
    unrecognized = pipeline.set_params(options)
    assert not unrecognized, unrecognized
    matrices = []
    stamps = []
    with args.input.open('rb') as stream:
        magic, rows, columns = struct.unpack('<8sII', stream.read(16))
        assert magic == b'FORMPC01' and (rows, columns) == (128, 1024)
        pipeline.set_lidar_params(LidarParams(num_rows=rows, num_columns=columns,
                                              min_range=.1, max_range=50.))
        pipeline.set_imu_T_lidar(SE3.identity())
        pipeline.initialize()
        for index in range(args.limit):
            stamp, count = struct.unpack('<qI', stream.read(12))
            assert count == rows*columns
            raw = stream.read(count*16)
            assert len(raw) == count*16
            xyz = np.asfortranarray(np.frombuffer(raw, dtype='<f4').reshape(count, 4)[:, :3],
                                    dtype=np.float64)
            measurement = LidarMeasurement.from_vec_positions(Stamp.from_nsec(stamp), xyz)
            pipeline.add_lidar(measurement)
            estimates = pipeline.saved_estimates()
            assert len(estimates) == 1, (index, len(estimates))
            out_stamp, pose = estimates[0]
            assert out_stamp.to_nsec() == stamp
            matrices.append(pose.to_mat())
            stamps.append(stamp)
    np.savez(args.output / f'{backend}-python.npz', stamps=stamps, matrices=matrices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cuda-solve-min-dimension', type=int)
    parser.add_argument('--feature-spacing', type=int, default=0)
    parser.add_argument('--import-only', action='store_true')
    parser.add_argument('--worker-backend', choices=('reference', 'summary', 'cuda'))
    args = parser.parse_args()
    if args.feature_spacing < 0:
        parser.error('feature spacing must be nonnegative')
    if args.cuda_solve_min_dimension is not None and args.cuda_solve_min_dimension < 1:
        parser.error('CUDA solve minimum dimension must be positive')
    if args.import_only:
        check_abi_guard(args.module_dir)
        print(json.dumps(load_extension(args.module_dir).default_params(), sort_keys=True))
        return
    if not all((args.input, args.replay, args.output)):
        parser.error('--input, --replay, and --output are required for replay validation')
    if args.limit < 1 or args.threads < 1:
        parser.error('--limit and --threads must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_backend:
        run_pipeline(args, args.worker_backend)
        return
    import numpy as np
    from decimal import Decimal
    from evalio.types import SO3
    results = []
    for backend in ('reference', 'summary', 'cuda'):
        prefix = args.output / f'{backend}-cpp'
        command = [str(args.replay.resolve()), '--input', str(args.input),
                        '--output', str(prefix), '--backend', backend,
                        '--limit', str(args.limit), '--threads', str(args.threads),
                        '--feature-spacing', str(args.feature_spacing)]
        if backend == 'cuda' and args.cuda_solve_min_dimension is not None:
            command.extend(['--cuda-solve-min-dimension', str(args.cuda_solve_min_dimension)])
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                        '--worker-backend', backend], check=True)
        py = np.load(args.output / f'{backend}-python.npz')
        rows = [line.split() for line in prefix.with_suffix('.tum').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
        assert len(rows) == args.limit == len(py['matrices'])
        cpp = []
        for index, row in enumerate(rows):
            assert int(Decimal(row[0])*10**9) == py['stamps'][index]
            matrix = np.eye(4)
            matrix[:3, 3] = np.array(row[1:4], dtype=float)
            qx, qy, qz, qw = map(float, row[4:8])
            matrix[:3, :3] = SO3(qw=qw, qx=qx, qy=qy, qz=qz).to_mat()
            cpp.append(matrix)
        maximum = float(np.max(np.abs(np.array(cpp) - py['matrices'])))
        np.testing.assert_allclose(py['matrices'], cpp, rtol=0, atol=1e-8,
                                   err_msg=f'{backend}: Python/C++ pose mismatch')
        results.append({'backend': backend, 'scans': args.limit,
                        'max_absolute_pose_matrix_difference': maximum})
        print(json.dumps(results[-1]), flush=True)
    (args.output / 'integration.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
