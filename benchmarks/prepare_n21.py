#!/usr/bin/env python3
"""Prepare bounded-memory FORM replay inputs; requires evalio 0.6.1 and numpy.

Wire format: FORMPC01, uint32 rows/cols, then repeated int64 stamp_ns,
uint32 count, count float32 (x,y,z,zero). All integers/floats little endian.
EOF ends the records. No filtering, deskew, point reordering, or IMU input.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import time

import numpy as np

MAGIC = b'FORMPC01'
SEQUENCES = ('stairs', 'quad_easy', 'quad_hard', 'maths_hard')
ROWS, COLS = 128, 1024


def validate_dimensions(rows, cols):
    if not (0 < rows <= 2**32-1 and 0 < cols <= 2**32-1 and rows*cols <= 2**32-1):
        raise ValueError(f'invalid dimensions: {rows} x {cols}')


def write_header(stream, rows, cols):
    validate_dimensions(rows, cols)
    stream.write(struct.pack('<8sII', MAGIC, rows, cols))


def read_header(stream):
    header = stream.read(16)
    if len(header) != 16:
        raise ValueError('truncated file header')
    magic, rows, cols = struct.unpack('<8sII', header)
    if magic != MAGIC:
        raise ValueError('invalid FORMPC magic')
    validate_dimensions(rows, cols)
    return rows, cols


def write_frame(stream, stamp_ns, xyz, rows, cols):
    validate_dimensions(rows, cols)
    if xyz.shape != (rows*cols, 3):
        raise ValueError(f'point shape {xyz.shape} does not match dimensions')
    points = np.zeros((rows*cols, 4), dtype='<f4')
    points[:, :3] = xyz
    stream.write(struct.pack('<qI', stamp_ns, len(points)))
    stream.write(points.tobytes())


def read_frames(stream, rows, cols):
    validate_dimensions(rows, cols)
    while header := stream.read(12):
        if len(header) != 12:
            raise ValueError('truncated frame header')
        stamp, count = struct.unpack('<qI', header)
        if count != rows*cols:
            raise ValueError('frame count does not match dimensions')
        data = stream.read(count*16)
        if len(data) != count*16:
            raise ValueError('truncated points')
        yield stamp, np.frombuffer(data, dtype='<f4').reshape(count, 4)


def pointcloud_xyz(msg, rows, cols):
    """Copy XYZ only, honoring endian, field offsets, row padding and point stride."""
    validate_dimensions(rows, cols)
    if (msg.height, msg.width) != (rows, cols):
        raise ValueError(f'PointCloud2 dimensions {msg.height}x{msg.width}, expected {rows}x{cols}')
    if msg.point_step <= 0 or msg.row_step < cols*msg.point_step or len(msg.data) != rows*msg.row_step:
        raise ValueError('invalid PointCloud2 strides/data size')
    fields = {field.name: field for field in msg.fields}
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.datatype != 7 or field.count != 1 or not 0 <= field.offset <= msg.point_step-4:
            raise ValueError(f'unsupported float32 field: {name}')
    dtype = np.dtype({'names':['x','y','z'], 'formats':[('>' if msg.is_bigendian else '<')+'f4']*3,
                      'offsets':[fields[n].offset for n in ('x','y','z')], 'itemsize':msg.point_step})
    source = np.ndarray((rows,cols), dtype=dtype, buffer=msg.data, strides=(msg.row_step,msg.point_step))
    xyz = np.empty((rows*cols,3), dtype='<f4')
    for axis,name in enumerate(('x','y','z')):
        xyz[:,axis] = source[name].reshape(-1)
    return msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec, xyz


def export_ground_truth(dataset, path):
    from evalio.types import SE3, SO3

    def normalized(pose):
        q = pose.rot
        values = np.array([q.qx,q.qy,q.qz,q.qw])
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('invalid ground-truth/calibration quaternion')
        values /= norm
        return SE3(SO3(qx=values[0],qy=values[1],qz=values[2],qw=values[3]),pose.trans),abs(norm-1)

    gt = dataset.ground_truth_raw()
    imu_T_gt,_ = normalized(dataset.imu_T_gt())
    imu_T_lidar,_ = normalized(dataset.imu_T_lidar())
    gt_T_lidar = imu_T_gt.inverse()*imu_T_lidar
    max_norm_deviation = 0.0
    with path.open('w') as stream:
        stream.write('# timestamp tx ty tz qx qy qz qw; world_T_lidar\n')
        for stamp, pose in gt:
            pose,deviation = normalized(pose)
            max_norm_deviation = max(max_norm_deviation,deviation)
            pose = pose*gt_T_lidar
            sec, nsec = divmod(stamp.to_nsec(), 1_000_000_000)
            q = pose.rot
            values = [*pose.trans, q.qx, q.qy, q.qz, q.qw]
            stream.write(f'{sec}.{nsec:09d} '+' '.join(f'{v:.17g}' for v in values)+'\n')
    return {'poses':len(gt), 'frame':'world_T_lidar',
            'conversion':'world_T_gt * inverse(imu_T_gt) * imu_T_lidar',
            'quaternion_normalization':'Source GT and extrinsic quaternions normalized before SE3 composition',
            'max_source_quaternion_norm_deviation':max_norm_deviation,
            'imu_T_lidar':imu_T_lidar.to_mat().tolist(),
            'imu_T_gt':imu_T_gt.to_mat().tolist(),
            'gt_T_lidar':gt_T_lidar.to_mat().tolist()}


def convert(dataset, output, limit=None):
    sequence = dataset.seq_name
    dest = output/f'{sequence}.formpc'
    meta_path = output/f'{sequence}.json'
    if dest.exists() or meta_path.exists():
        raise FileExistsError(f'refusing to overwrite {dest} or {meta_path}')
    iterator = dataset.data_iter()
    try:
        count = min(len(iterator), limit) if limit else len(iterator)
        required = 16+count*(12+ROWS*COLS*16)
        if shutil.disk_usage(output).free < required+2*1024**3:
            raise OSError(f'not enough disk space for {required} bytes plus 2 GiB reserve')
        selected = {0,count//2,count-1}
        checks = []
        started = time.monotonic()
        tmp = dest.with_suffix('.formpc.partial')
        with tmp.open('xb', buffering=8*1024*1024) as stream:
            write_header(stream, ROWS, COLS)
            actual = 0
            for index,(connection,_,raw) in enumerate(iterator.reader.messages(connections=iterator.connections_lidar)):
                if index >= count:
                    break
                msg = iterator.reader.deserialize(raw, connection.msgtype)
                stamp, xyz = pointcloud_xyz(msg, ROWS, COLS)
                if index in selected:
                    reference = iterator._lidar_conversion(msg)
                    expected = np.asarray(reference.to_vec_positions(), dtype='<f4')
                    np.testing.assert_array_equal(xyz, expected, err_msg=f'{sequence} frame {index}: evalio xyz/order mismatch')
                    if stamp != reference.stamp.to_nsec():
                        raise ValueError('evalio timestamp mismatch')
                    check = {'frame':index, 'stamp_ns':stamp, 'xyz_order_exact':True,
                             'xyz_sha256':hashlib.sha256(xyz.tobytes()).hexdigest()}
                    checks.append(check)
                    print(f'{sequence}: frame {index} exactly matches evalio xyz/order/stamp', flush=True)
                write_frame(stream, stamp, xyz, ROWS, COLS)
                actual += 1
                if actual % 100 == 0:
                    print(f'{sequence}: {actual}/{count}, {time.monotonic()-started:.1f}s', flush=True)
            if actual != count:
                raise ValueError(f'expected {count} scans, got {actual}')
        if tmp.stat().st_size != required:
            raise ValueError('incorrect output size')
        gt_info = export_ground_truth(dataset, output/f'{sequence}.gt.tum')
        tmp.rename(dest)
        meta = {'sequence':sequence, 'format':'FORMPC01', 'rows':ROWS, 'cols':COLS,
                'scans':actual, 'source_scans':len(iterator), 'bytes':required,
                'complete_sequence':actual==len(iterator), 'deskew':False, 'point_order':'row-major',
                'stamp':'PointCloud2.header.stamp, scan start, Unix nanoseconds',
                'source_files':[{'path':str(dataset.folder/name),'bytes':(dataset.folder/name).stat().st_size}
                                for name in dataset.files()],
                'evalio_version':importlib.metadata.version('evalio'), 'sample_checks':checks,
                'ground_truth':gt_info, 'elapsed_seconds':time.monotonic()-started}
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
        print(f'{sequence}: complete, {actual} scans, {required} bytes', flush=True)
    finally:
        iterator.reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/datasets'))
    parser.add_argument('--output', type=Path, default=Path('/home/ubuntu/datasets/form-input'))
    parser.add_argument('--sequences', nargs='+', choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument('--limit', type=int, help='development conversion only; default is complete sequences')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    from evalio import datasets
    datasets.set_data_dir(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    total = sum(min(datasets.NewerCollege2021[s].quick_len(),args.limit) if args.limit
                else datasets.NewerCollege2021[s].quick_len() for s in args.sequences)*(12+ROWS*COLS*16)
    if shutil.disk_usage(args.output).free < total+2*1024**3:
        raise OSError('insufficient disk for requested sequences plus 2 GiB reserve')
    for seq in args.sequences:
        convert(datasets.NewerCollege2021[seq],args.output,args.limit)


if __name__ == '__main__':
    main()
