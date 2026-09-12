#!/usr/bin/env python3
"""Validate FORMQR01 captures independently and summarize exact ragged shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read_batch(path):
    with path.open('rb') as stream:
        remaining=path.stat().st_size
        def read(count):
            nonlocal remaining
            if count>remaining:
                raise ValueError('truncated QR batch')
            data=stream.read(count)
            if len(data)!=count:
                raise ValueError('truncated QR batch')
            remaining-=count
            return data
        if read(8)!=b'FORMQR01':
            raise ValueError('invalid magic')
        count,=struct.unpack('<I',read(4))
        if not 0<count<=100000 or count>remaining//12:
            raise ValueError('invalid matrix count')
        result=[]
        for _ in range(count):
            rows,cols=struct.unpack('<QI',read(12))
            if cols not in (7,13) or rows>2**31-1 or rows>remaining//(8*cols):
                raise ValueError('invalid matrix shape')
            result.append(np.frombuffer(read(rows*cols*8),dtype='<f8').reshape((rows,cols),order='F'))
        if remaining:
            raise ValueError('trailing batch data')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    batches=[]
    for path in sorted(args.directory.glob('*.formqr')):
        matrices=read_batch(path)
        with path.open('rb') as stream:
            sha=hashlib.file_digest(stream,'sha256').hexdigest()
        batches.append({'file':path.name,'shapes':[list(a.shape) for a in matrices],
                        'bytes':path.stat().st_size,'sha256':sha,'all_finite':all(np.isfinite(a).all() for a in matrices)})
    if not batches:
        raise ValueError('no captured batches')
    manifest_path=args.directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['batches']!=[{'file':b['file'],'shapes':b['shapes']} for b in batches]:
            raise ValueError('manifest shapes/order differ from batch contents')
    print(json.dumps({'batches':batches},indent=2))


if __name__=='__main__':
    main()
