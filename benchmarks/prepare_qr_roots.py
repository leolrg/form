#!/usr/bin/env python3
"""Create clean QR-root FORMQR01 fixtures; CPU work only, never GPU execution.

Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 while other experiments run.
The nine rebuilt-pair captures are not the whole active estimator window.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
from inspect_qr_capture import read_batch


def clean_root(raw):
    if raw.ndim!=2 or raw.shape[1] not in (7,13) or not np.isfinite(raw).all():
        raise ValueError('expected finite 7/13-column features')
    width=raw.shape[1]
    root=np.zeros((width,width),dtype='<f8',order='F')
    if len(raw):
        reduced=np.linalg.qr(raw,mode='r')
        root[:min(len(raw),width),:]=np.triu(reduced)
    gram=raw.T@raw
    error=float(np.linalg.norm(root.T@root-gram)/max(1.,np.linalg.norm(gram)))
    if not np.isfinite(root).all() or error>1e-11:
        raise ValueError(f'QR root Gram agreement failed: {error}')
    return root,error


def write_batch(path,matrices):
    if not matrices or len(matrices)>100000:
        raise ValueError('invalid batch size')
    with path.open('xb') as stream:
        stream.write(b'FORMQR01'+struct.pack('<I',len(matrices)))
        for matrix in matrices:
            rows,cols=matrix.shape
            if cols not in (7,13) or rows>2**31-1:
                raise ValueError('unsupported matrix shape')
            stream.write(struct.pack('<QI',rows,cols))
            stream.write(np.asarray(matrix,dtype='<f8').tobytes(order='F'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=root/'benchmarks/results/qr-stairs200')
    parser.add_argument('--output',type=Path,default=root/'benchmarks/results/tuning/qr-stairs200-roots')
    args=parser.parse_args()
    files=sorted(args.input_dir.glob('*.formqr'))
    if not files:
        raise ValueError('no input captures')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'format':'FORMQR01','fixture':'clean QR upper roots','source':str(args.input_dir.resolve()),
              'scope':'Only the captured rebuilt pairs; not the complete active window.',
              'numpy_version':np.__version__,'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
              'batches':[],'validation':[]}
    for path in files:
        raw=read_batch(path);roots=[];errors=[]
        for matrix in raw:
            factor,error=clean_root(matrix);roots.append(factor);errors.append(error)
        target=args.output/path.name;write_batch(target,roots)
        reread=read_batch(target)
        for expected,actual in zip(roots,reread):
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(np.tril(actual,-1),np.zeros_like(actual))
        weights=np.concatenate([r[:,0] for r in roots if r.shape[1]==7])
        manifest['batches'].append({'file':path.name,'shapes':[list(r.shape) for r in roots]})
        manifest['validation'].append({'file':path.name,'input_sha256':sha(path),'output_sha256':sha(target),
             'maximum_gram_relative_error':max(errors),'original_shapes':[list(a.shape) for a in raw],
             'point_weights':{'rows':len(weights),'nonunit_rows':int(np.sum(weights!=1)),
                              'zero_rows':int(np.sum(weights==0)),'negative_rows':int(np.sum(weights<0)),
                              'minimum':float(weights.min()),'maximum':float(weights.max())}})
    # Separate directory: edge cases must not enter the nine-batch timing aggregate.
    edge_dir=args.output/'point-weight-edges';edge_dir.mkdir()
    edges=[]
    base=np.triu(np.arange(49,dtype=np.float64).reshape(7,7)/11)
    for weight in (-3.,0.,.5,2.5):
        matrix=base.copy();matrix[0,0]=weight;edges.append(matrix)
    edges.append(np.zeros((7,7)))
    edge_path=edge_dir/'batch-000000.formqr';write_batch(edge_path,edges)
    edge_meta={'format':'FORMQR01','fixture':'Synthetic point upper roots with signed/nonunit/zero homogeneous weights and an all-zero root',
               'point_row0_weights':[-3.,0.,.5,2.5,0.],
               'batches':[{'file':edge_path.name,'shapes':[[7,7]]*len(edges)}]}
    (edge_dir/'manifest.json').write_text(json.dumps(edge_meta,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'batches':len(files),'root_matrices':sum(len(b['shapes']) for b in manifest['batches']),
                      'maximum_gram_relative_error':max(v['maximum_gram_relative_error'] for v in manifest['validation']),
                      'point_rows_nonunit_weight':sum(v['point_weights']['nonunit_rows'] for v in manifest['validation']),
                      'point_rows_zero_weight':sum(v['point_weights']['zero_rows'] for v in manifest['validation']),
                      'output':str(args.output),'separate_edge_fixture':str(edge_dir)},indent=2))


if __name__=='__main__':
    main()
