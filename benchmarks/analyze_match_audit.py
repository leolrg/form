"""Analyze diagnostic correspondence streams; never used for clean timings."""
import argparse
from collections import Counter
import json
import math
import struct
from itertools import zip_longest


def read_records(stream):
    def exact(n):
        data = stream.read(n)
        if len(data) != n:
            raise ValueError('truncated match audit')
        return data

    if exact(8) != b'FORMMAT1':
        raise ValueError('invalid match audit magic')
    while True:
        header = stream.read(32)
        if not header:
            return
        if len(header) != 32:
            raise ValueError('truncated match audit header')
        scan, kind, count, group_count = struct.unpack('<4Q', header)
        if kind not in (0, 1) or count > 100_000_000 or group_count > 65535:
            raise ValueError('invalid audit dimensions')
        groups = struct.unpack('<' + 'Q'*group_count, exact(8*group_count))
        if len(set(groups)) != len(groups):
            raise ValueError('duplicate target scan')
        matches = list(struct.iter_unpack('<iid', exact(16*count)))
        for index, group, distance in matches:
            if index < -1 or group < -1 or group >= group_count or not math.isfinite(distance) or distance < 0:
                raise ValueError('invalid match record')
            if group >= 0 and index < 0:
                raise ValueError('accepted missing match')
        yield dict(scan=scan, kind=kind, groups=groups, matches=matches)


def compare_records(reference, candidate):
    counts = dict(searches=0, queries=0)
    for a, b in zip_longest(reference, candidate):
        if a is None or b is None or (a['scan'], a['kind']) != (b['scan'], b['kind']):
            raise ValueError('different rematch streams')
        def canonical(record):
            return [(i, record['groups'][g] if g >= 0 else None, d) for i, g, d in record['matches']]
        if canonical(a) != canonical(b):
            raise ValueError(f"matching mismatch at scan {a['scan']}, kind {a['kind']}")
        counts['searches'] += 1
        counts['queries'] += len(a['matches'])
    return counts


def summarize_records(records, block_sizes=(32, 64, 128, 256), warmup=20):
    def empty():
        return dict(searches=0, queries=0, accepted=0, raw_identity_changes=0,
                    summary_row_changes=0, full_qr_tiles=0,
                    blocks={str(b): dict(active_after=0, dirty_union=0,
                        dirty_surviving=0, rebuild_rows=0, padded_rows=0) for b in block_sizes})

    report = dict(first=empty(), steady=empty(), by_kind={}, by_iteration={})
    previous = {}
    epoch = None
    for record in records:
        scan, kind = record['scan'], record['kind']
        if scan < warmup:
            continue
        if epoch is not None and scan < epoch:
            raise ValueError('scan order is not monotone')
        if scan != epoch:
            previous.clear()
            epoch = scan
        groups, matches = record['groups'], record['matches']
        rows = [(groups[g], i) if g >= 0 else None for i, g, _ in matches]
        old, iteration = previous.get(kind, (None, 0))
        if old is not None and len(old['matches']) != len(matches):
            raise ValueError('query count changed inside snapshot')
        old_rows = old['rows'] if old else [None]*len(rows)
        changed = [i for i, (a, b) in enumerate(zip(old_rows, rows)) if a != b]
        phase = 'steady' if old else 'first'
        targets = [report[phase],
                   report['by_kind'].setdefault(str(kind), {}).setdefault(phase, empty()),
                   report['by_iteration'].setdefault(str(iteration), empty())]
        counts = Counter(r[0] for r in rows if r is not None)
        for target in targets:
            target['searches'] += 1
            target['queries'] += len(matches)
            target['accepted'] += sum(counts.values())
            target['full_qr_tiles'] += sum((n+63)//64 for n in counts.values())
            if old:
                target['raw_identity_changes'] += sum(a[0] != b[0] for a, b in zip(old['matches'], matches))
                target['summary_row_changes'] += len(changed)
        for block in block_sizes:
            active = Counter((r[0], i//block) for i, r in enumerate(rows) if r is not None)
            dirty = {(r[0], i//block) for i in changed for r in (old_rows[i], rows[i]) if r is not None}
            for target in targets:
                stats = target['blocks'][str(block)]
                stats['active_after'] += len(active)
                stats['dirty_union'] += len(dirty)
                stats['dirty_surviving'] += sum(k in active for k in dirty)
                stats['rebuild_rows'] += sum(active[k] for k in dirty)
                stats['padded_rows'] += block*sum(k in active for k in dirty)
        previous[kind] = (dict(matches=matches, rows=rows), iteration+1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    with open(args.input, 'rb') as source:
        result = summarize_records(read_records(source), warmup=args.warmup)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        with open(args.output, 'w') as dest:
            dest.write(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
