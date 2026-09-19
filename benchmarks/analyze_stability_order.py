"""Offline stable64 ordering experiment; counts QR work, never predicts timing.

Policies: ordinary query order; future change-count ordering (unimplementable
oracle heuristic, not an optimum); and reorder once at rematch 1 using the first
observed change flag, paying for a complete rebuild. Later incoming rows always
use ascending original query IDs and first holes. One scan's histories are held
at a time. Actual QR counters include zero-filled dirty leaves, persistent
highwater holes, and singleton parent copies, matching the cached-tree work
rules rather than the optimistic live-row-only StableBlocks counters.
"""
import argparse
import json

from analyze_match_audit import read_records
from analyze_stable_blocks import StableBlocks


POLICIES = ('query_order', 'oracle_future_changes', 'first_change_flag')


class PrioritizedBlocks(StableBlocks):
    """Sort only the initial arrivals; maintain caller indices thereafter."""
    def __init__(self, kind, priority=None):
        super().__init__(64, kind, hole_policy='first')
        self.priority = priority

    def update(self, rows):
        old_rows = self.previous
        old_locations = list(self.locations)
        if old_rows is None and self.priority is not None:
            if len(self.priority) != len(rows):
                raise ValueError('priority must have one entry per query')
            order = sorted(range(len(rows)), key=lambda q: (self.priority[q], q))
            result = super().update([rows[q] for q in order])
            locations = [None] * len(rows)
            for packed, query in enumerate(order):
                locations[query] = self.locations[packed]
            self.locations = locations
            self.slots = {g: [None if q is None else order[q] for q in slots]
                          for g, slots in self.slots.items()}
            self.previous = list(rows)
        else:
            result = super().update(rows)
        dirty = {}
        for q, row in enumerate(rows):
            if old_rows is not None and old_rows[q] == row:
                continue
            locations = (self.locations[q],)
            if old_rows is not None:
                locations += (old_locations[q],)
            for location in locations:
                if location is not None:
                    group, slot = location
                    dirty.setdefault(group, set()).add(slot // 64)
        merges = inputs = copies = 0
        for group, changed in dirty.items():
            extent = len(self.slots[group]) // 64
            while extent > 1:
                parents = {i // self.fanout for i in changed}
                for parent in parents:
                    children = min(self.fanout, extent - parent * self.fanout)
                    if children > 1:
                        merges += 1
                        inputs += children
                    else:
                        copies += 1
                changed = parents
                extent = (extent + self.fanout - 1) // self.fanout
        leaves = sum(map(len, dirty.values()))
        result.update(actual_leaf_qr64=leaves, actual_cached_merge_qr64=merges,
                      actual_cached_merge_input_roots=inputs,
                      actual_cached_copy_nodes=copies, actual_qr64_tasks=leaves + merges)
        return result


def add_counts(target, counts):
    for key, value in counts.items():
        target[key] = target.get(key, 0) + value


def analyze_snapshot(history, kind):
    """Analyze one feature-kind history with immutable original query indices."""
    report = {p: {'first': {}, 'steady': {}, 'reorder': {}, 'total': {}}
              for p in POLICIES}
    if not history:
        return report
    count = len(history[0])
    if any(len(rows) != count for rows in history):
        raise ValueError('query count changed inside snapshot')
    future = [0] * count
    for old, new in zip(history, history[1:]):
        for q, (a, b) in enumerate(zip(old, new)):
            future[q] += a != b
    first_flag = ([int(a != b) for a, b in zip(history[0], history[1])]
                  if len(history) > 1 else [0] * count)
    for policy in POLICIES:
        state = PrioritizedBlocks(kind, future if policy == 'oracle_future_changes' else None)
        for iteration, rows in enumerate(history):
            reorder = policy == 'first_change_flag' and iteration == 1
            if reorder:
                state = PrioritizedBlocks(kind, first_flag)
            counters = state.update(rows)
            if reorder:
                counters['changed_rows'] = sum(a != b for a, b in zip(history[0], rows))
                add_counts(report[policy]['reorder'], counters)
            phase = 'first' if iteration == 0 else 'steady'
            add_counts(report[policy][phase], counters)
            add_counts(report[policy]['total'], counters)
    return report


def summarize_records(records, warmup=20):
    report = dict(snapshots=0, feature_snapshots=0, max_snapshot_rows=0,
                  policies={p: {'first': {}, 'steady': {}, 'reorder': {}, 'total': {}}
                            for p in POLICIES}, by_kind={})
    histories = {}
    epoch = None

    def flush():
        if not histories:
            return
        report['snapshots'] += 1
        report['max_snapshot_rows'] = max(report['max_snapshot_rows'],
            sum(len(rows) for history in histories.values() for rows in history))
        for kind, history in histories.items():
            report['feature_snapshots'] += 1
            result = analyze_snapshot(history, kind)
            kind_target = report['by_kind'].setdefault(str(kind), {})
            for policy, phases in result.items():
                for phase, counters in phases.items():
                    add_counts(report['policies'][policy][phase], counters)
                    target = kind_target.setdefault(policy, {}).setdefault(phase, {})
                    add_counts(target, counters)

    for record in records:
        scan = record['scan']
        if epoch is not None and scan < epoch:
            raise ValueError('scan order is not monotone')
        if scan != epoch:
            flush()
            histories.clear()
            epoch = scan
        if scan < warmup:
            continue
        rows = [(record['groups'][group], index) if group >= 0 else None
                for index, group, _ in record['matches']]
        histories.setdefault(record['kind'], []).append(rows)
    flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    with open(args.input, 'rb') as stream:
        result = summarize_records(read_records(stream), args.warmup)
    result.update(model=__doc__, input=args.input, warmup=args.warmup,
                  leaf_size=64, priority_direction='ascending, query ID breaks ties')
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
