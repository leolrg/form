#!/usr/bin/env python3
"""Prepare Hilti/MCD replay inputs through evalio's sensor-specific LiDAR loader."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
from prepare_n21 import write_header, write_frame, export_ground_truth

SEQUENCES = ('hilti_2022/basement_2', 'multi_campus/tuhh_day_04')

def convert(dataset, output, limit=None):
    name=dataset.seq_name
    dest=output/f'{name}.formpc'
    meta_path=output/f'{name}.json'
    gt_path=output/f'{name}.gt.tum'
    tmp=dest.with_suffix('.formpc.partial')
    if any(p.exists() for p in (dest,meta_path,gt_path,tmp)):
        raise FileExistsError(f'refusing to overwrite outputs for {name}')
    params=dataset.lidar_params()
    rows,cols=params.num_rows,params.num_columns
    source_count=dataset.quick_len()
    expected=min(source_count,limit) if limit else source_count
    required=16+expected*(12+rows*cols*16)
    if shutil.disk_usage(output).free < required+2*1024**3:
        raise OSError('insufficient disk space plus 2 GiB reserve')
    selected={0,expected//2,expected-1}
    checks=[]; actual=0; previous=None
    started=time.monotonic()
    iterator=iter(dataset.lidar())
    try:
        with tmp.open('xb',buffering=8*1024*1024) as stream:
            write_header(stream,rows,cols)
            for scan in iterator:
                if limit and actual>=limit:break
                xyz=np.asarray(scan.to_vec_positions(),dtype='<f4')
                stamp=scan.stamp.to_nsec()
                if previous is not None and stamp<=previous:
                    raise ValueError('non-increasing scan timestamp')
                write_frame(stream,stamp,xyz,rows,cols)
                if actual in selected:
                    checks.append({'frame':actual,'stamp_ns':stamp,
                        'xyz_sha256':hashlib.sha256(xyz.tobytes()).hexdigest()})
                previous=stamp;actual+=1
                if actual%100==0:print(f'{name}: {actual}/{expected}, {time.monotonic()-started:.1f}s',flush=True)
    finally:
        if hasattr(iterator,'close'):iterator.close()
    if actual!=expected or tmp.stat().st_size!=required:
        raise ValueError(f'scan count/output size mismatch: {actual} versus {expected}')
    gt=export_ground_truth(dataset,gt_path)
    meta={'sequence':name,'dataset':str(dataset.full_name),'format':'FORMPC01','rows':rows,'cols':cols,
          'scans':actual,'source_scans':source_count,'complete_sequence':actual==source_count,'bytes':required,
          'deskew':False,'point_order':'evalio row-major, including invalid-point padding',
          'stamp':'evalio scan start (includes sensor-specific header adjustment)',
          'replay_settings':{'min-range':params.min_range,'max-range':params.max_range},
          'evalio_version':importlib.metadata.version('evalio'),'ground_truth':gt,'sample_checks':checks,
          'source_files':[{'path':str(dataset.folder/f),'bytes':(dataset.folder/f).stat().st_size} for f in dataset.files()],
          'elapsed_seconds':time.monotonic()-started}
    tmp.rename(dest)
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{name}: complete, {actual} scans, {required} bytes',flush=True)
    return meta

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sequences',nargs='+',choices=SEQUENCES,default=SEQUENCES)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<=0:parser.error('--limit must be positive')
    from evalio import datasets as ds
    ds.set_data_dir(args.data_root)
    datasets={'hilti_2022/basement_2':ds.Hilti2022.basement_2,'multi_campus/tuhh_day_04':ds.MultiCampus.tuhh_day_04}
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.sequences:convert(datasets[name],args.output,args.limit)

if __name__=='__main__':main()
