#!/usr/bin/env python3
"""Summarize FORM replay latency and evalio trajectory errors in the LiDAR frame.

All timing rows are reported, plus steady state after the first 20 rows.
Association uses unique nearest timestamps within 50 ms, without interpolation.
ATE aligns the first matched pose (no fitted rotation, translation, or scale).
RTE uses evalio distance windows of 1 m and 30 m, in metres and degrees.
"""
import argparse
import csv
from decimal import Decimal
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def summarize_ms(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('latencies must be finite and nonnegative')
    return {'count':len(values), 'mean_ms':float(values.mean()),
            'median_ms':float(np.median(values)), 'p95_ms':float(np.percentile(values,95)),
            'p99_ms':float(np.percentile(values,99)), 'max_ms':float(values.max()),
            'throughput_hz':float(1000/values.mean()) if values.mean() else None,
            'deadline_miss_fraction':float(np.mean(values > 100))}


def timing_metrics(path, warmup=20):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'no timing rows: {path}')
    columns = {name:np.array([float(row[name]) for row in rows])
               for name in rows[0] if name.endswith('_ms')}
    if 'semi_ms' in columns and 'full_ms' in columns:
        columns['optimization_ms'] = columns['semi_ms']+columns['full_ms']
    result = {'scans':len(rows), 'warmup_rows':warmup,
              'timing_boundary':'Estimator processing only; replay loading and output serialization excluded.',
              'optimization_definition':'semi_ms + full_ms; marginalize_ms reported separately.',
              'factor_timer_note':'factor_*_cpu_ms are summed per-call elapsed durations and may overlap across TBB threads; they are not wall latency or additive stages.',
              'all_scans':{name:summarize_ms(values) for name,values in columns.items()},
              'steady_state':{name:summarize_ms(values[warmup:]) for name,values in columns.items()}}
    counts = ('poses','planar_features','point_features','rematches','lm_iterations','factors','planar_correspondences','point_correspondences','full_initial_error','full_final_error')
    result['workload_all_scans'] = {name:{'mean':float(np.mean([float(row[name]) for row in rows])),
                                        'max':float(max(float(row[name]) for row in rows))}
                                   for name in counts if name in rows[0]}
    return result


def associate_stamps(estimate, ground_truth, max_delta_ns):
    """Greedy globally closest one-to-one association; ties resolved by index."""
    estimate = np.asarray(estimate, dtype=np.int64)
    ground_truth = np.asarray(ground_truth,dtype=np.int64)
    if max_delta_ns < 0 or np.any(np.diff(estimate) <= 0) or np.any(np.diff(ground_truth) <= 0):
        raise ValueError('timestamps must increase strictly and tolerance must be nonnegative')
    candidates = []
    for i,stamp in enumerate(estimate):
        low = np.searchsorted(ground_truth,stamp-max_delta_ns,side='left')
        high = np.searchsorted(ground_truth,stamp+max_delta_ns,side='right')
        candidates.extend((abs(int(stamp)-int(ground_truth[j])),i,j) for j in range(low,high))
    used_e,used_g,pairs = set(),set(),[]
    for _,i,j in sorted(candidates):
        if i not in used_e and j not in used_g:
            used_e.add(i)
            used_g.add(j)
            pairs.append((i,j))
    pairs.sort()
    if not pairs:
        return np.array([],dtype=int),np.array([],dtype=int)
    return np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])


def load_tum(path):
    from evalio.types import SE3, SO3, Stamp, Trajectory
    result = Trajectory()
    with path.open() as stream:
        for index,line in enumerate(stream,1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f'{path}:{index}: expected eight TUM columns')
            stamp = int(Decimal(fields[0])*Decimal(1_000_000_000))
            values = np.array([float(v) for v in fields[1:]])
            if not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]),1,atol=1e-4):
                raise ValueError(f'{path}:{index}: invalid pose or quaternion')
            if result.stamps and stamp <= result.stamps[-1].to_nsec():
                raise ValueError(f'{path}:{index}: non-increasing timestamp')
            q = values[3:]/np.linalg.norm(values[3:])
            result.append(Stamp.from_nsec(stamp),SE3(SO3(qx=q[0],qy=q[1],qz=q[2],qw=q[3]),values[:3]))
    if not len(result):
        raise ValueError(f'empty trajectory: {path}')
    return result


def error_summary(error):
    def stats(values):
        values = values[np.isfinite(values)]
        if not len(values):
            return None
        return {'mean':float(values.mean()), 'median':float(np.median(values)),
                'rmse':float(np.sqrt(np.mean(values**2))), 'max':float(values.max())}
    return {'samples':int(np.isfinite(error.trans).sum()),
            'translation_m':stats(error.trans), 'rotation_deg':stats(error.rot)}


def trajectory_metrics(estimate_path, gt_path, max_delta_ms=50):
    from evalio import stats
    from evalio.types import Trajectory
    estimate,gt = load_tum(estimate_path),load_tum(gt_path)
    est_ns = np.array([s.to_nsec() for s in estimate.stamps],dtype=np.int64)
    gt_ns = np.array([s.to_nsec() for s in gt.stamps],dtype=np.int64)
    ei,gi = associate_stamps(est_ns,gt_ns,int(max_delta_ms*1e6))
    result = {'estimate_poses':len(estimate),'ground_truth_poses':len(gt),'matched_poses':len(ei),
              'estimate_coverage_fraction':len(ei)/len(estimate),'ground_truth_coverage_fraction':len(gi)/len(gt),
              'association':'greedy globally closest unique timestamp pairs; no interpolation',
              'max_allowed_stamp_difference_ms':max_delta_ms, 'frame':'world_T_lidar',
              'alignment':'first matched pose; rigid SE3, no scale fitting',
              'evalio_version':importlib.metadata.version('evalio')}
    if len(ei) < 2:
        result['error'] = 'fewer than two associated poses'
        return result
    result['max_actual_stamp_difference_ms'] = float(np.max(np.abs(est_ns[ei]-gt_ns[gi]))/1e6)
    est_matched = Trajectory(stamps=[estimate.stamps[i] for i in ei],poses=[estimate.poses[i] for i in ei])
    gt_matched = Trajectory(stamps=[gt.stamps[i] for i in gi],poses=[gt.poses[i] for i in gi])
    stats.align_poses(est_matched,gt_matched)
    result['ate_first_pose_aligned'] = error_summary(stats.ate(est_matched,gt_matched))
    for meters in (1,30):
        result[f'rte_{meters}m'] = error_summary(stats.rte(est_matched,gt_matched,stats.WindowMeters(meters)))
    result['matched_start_ns'] = int(est_ns[ei[0]])
    result['matched_end_ns'] = int(est_ns[ei[-1]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings',type=Path,required=True)
    parser.add_argument('--trajectory',type=Path)
    parser.add_argument('--ground-truth',type=Path)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--max-delta-ms',type=float,default=50)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if bool(args.trajectory) != bool(args.ground_truth):
        parser.error('--trajectory and --ground-truth must be used together')
    if args.warmup < 0 or args.max_delta_ms < 0:
        parser.error('warmup and association tolerance must be nonnegative')
    result = {'timings':timing_metrics(args.timings,args.warmup)}
    if args.trajectory:
        result['trajectory'] = trajectory_metrics(args.trajectory,args.ground_truth,args.max_delta_ms)
    content = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content,end='')


if __name__ == '__main__':
    main()
