#!/usr/bin/env python3
"""Attribute Nsight Systems CUDA activity to its issuing NVTX host range.

Device durations and host API durations overlap; never add them. NVTX scope
attribution follows launch correlation, not the later synchronization call.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from analyze_diagnostics import union_duration


def attribute_apis(ranges,apis):
    by_thread=defaultdict(list)
    for r in ranges:by_thread[r['globalTid']].append(r)
    positions={t:0 for t in by_thread};stacks=defaultdict(list);result={}
    for entries in by_thread.values():entries.sort(key=lambda r:(r['start'],-r['end']))
    for api in sorted(apis,key=lambda r:r['start']):
        tid=api['globalTid'];entries=by_thread[tid];pos=positions.get(tid,0);stack=stacks[tid]
        while pos<len(entries) and entries[pos]['start']<=api['start']:
            r=entries[pos]
            while stack and stack[-1]['end']<=r['start']:stack.pop()
            stack.append(r);pos+=1
        positions[tid]=pos
        while stack and stack[-1]['end']<=api['start']:stack.pop()
        scans=[r for r in stack if r['name'].startswith('scan/')]
        if not scans:continue
        scan=int(scans[-1]['name'].split('/')[1])
        stages=[r['name'] for r in stack if r['name'].startswith(('stage/','optimization/'))]
        result[(api['globalTid']//0x1000000,api['correlationId'])]=dict(scan=scan,scope=stack[-1]['name'],stage=stages[-1] if stages else 'unassigned',api=api)
    return result


def analyze(path,warmup=20):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges=[]
    for row in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r=dict(row);r['name']=r['text'] or strings.get(r['textId'],'')
        ranges.append(r)
    scans={int(r['name'].split('/')[1]):r for r in ranges if r['name'].startswith('scan/') and int(r['name'].split('/')[1])>=warmup}
    if not scans:raise ValueError('no timed scan NVTX ranges')
    apis=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    attributed=attribute_apis(ranges,apis)
    kernels=defaultdict(lambda:dict(count=0,ns=0));copies=defaultdict(lambda:dict(count=0,ns=0,bytes=0))
    scopes=defaultdict(lambda:dict(kernel_ns=0,copy_ns=0,memset_ns=0,kernels=0,copies=0,bytes=0,h2d_bytes=0,d2h_bytes=0,api_ns=0,sync_ns=0,sync_calls=0))
    api_summary=defaultdict(lambda:dict(count=0,ns=0))
    busy=[];kernel_intervals=[];copy_intervals=[];unmatched=0;unmatched_timed=0;outside_scan_ns=0
    kinds={r['id']:r['label'] for r in db.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}
    for a in attributed.values():
        if a['scan'] not in scans:continue
        api=a['api'];name=strings[api['nameId']];duration=api['end']-api['start']
        api_summary[name]['count']+=1;api_summary[name]['ns']+=duration
        scope=scopes[a['scope']];scope['api_ns']+=duration
        if 'Synchronize' in name:scope['sync_ns']+=duration;scope['sync_calls']+=1
    for table,category in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        if table not in tables:continue
        for row in db.execute('SELECT * FROM '+table):
            r=dict(row);a=attributed.get((r['globalPid']//0x1000000,r['correlationId']))
            if not a:
                unmatched+=1
                if any(v['start']<=r['start']<v['end'] for v in scans.values()):unmatched_timed+=1
                continue
            if a['scan'] not in scans:continue
            duration=r['end']-r['start'];scope=scopes[a['scope']];scope[category+'_ns']+=duration
            scan=scans[a['scan']];lo=max(r['start'],scan['start']);hi=min(r['end'],scan['end'])
            clipped=max(0,hi-lo);outside_scan_ns+=duration-clipped
            if clipped:busy.append((lo,hi))
            if category=='kernel':
                scope['kernels']+=1;name=strings[r['shortName']]
                kernels[name]['count']+=1;kernels[name]['ns']+=duration
                if clipped:kernel_intervals.append((lo,hi))
            elif category=='copy':
                scope['copies']+=1;scope['bytes']+=r['bytes'];name=kinds[r['copyKind']]
                copies[name]['count']+=1;copies[name]['ns']+=duration;copies[name]['bytes']+=r['bytes']
                if r['copyKind']==1:scope['h2d_bytes']+=r['bytes']
                if r['copyKind']==2:scope['d2h_bytes']+=r['bytes']
                if clipped:copy_intervals.append((lo,hi))
    scan_ns=sum(r['end']-r['start'] for r in scans.values())
    return dict(trace=str(path),warmup=warmup,timed_scans=len(scans),scan_wall_ms=scan_ns*1e-6,
        device_busy_union_ms=union_duration(busy)*1e-6,kernel_union_ms=union_duration(kernel_intervals)*1e-6,
        copy_union_ms=union_duration(copy_intervals)*1e-6,device_busy_fraction=union_duration(busy)/scan_ns,
        unmatched_device_events=unmatched,unmatched_during_timed_scans=unmatched_timed,
        attributed_device_ns_outside_issuing_scan=outside_scan_ns,
        kernels=dict(kernels),copies=dict(copies),scopes=dict(scopes),apis=dict(api_summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=analyze(args.sqlite);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('scans',result['timed_scans'],'device busy',result['device_busy_fraction'],'unmatched timed',result['unmatched_during_timed_scans'])

if __name__=='__main__':main()
