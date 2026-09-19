"""Offline work model for stable per-target-scan QR leaves, not a timing model.

Within each immutable matching snapshot, query rows retain slots while their
target scan is unchanged. Removed slots become holes. Incoming queries, ordered
by query index, fill holes in dirty leaves first, then other existing holes, then
new leaves. The optional first-hole policy instead uses the lowest vacant slot.
Leaves never compact; empty leaves retain their indices. This models
packing live rows within each rebuilt leaf (not processing its padding).

Two merge estimates are reported: rebuilding each affected group's entire root
from active leaves, and updating only ancestors in a fixed-index merge tree.
The latter assumes cached internal nodes and zero roots for empty leaves. Both
are work counts; neither includes allocation, assignment, compaction or GPU
launch overhead and neither predicts measured acceleration.
"""
import argparse
from collections import Counter
import json

from analyze_match_audit import read_records


def merge_work(active, dirty, fanout, extent=None):
    """Count nonempty input roots consumed by dirty ancestors, including deletes."""
    active, dirty = set(active), set(dirty)
    if extent is None:
        extent = max(active | dirty, default=-1) + 1
    nodes = inputs = 0
    while extent > 1:
        parents = {i // fanout for i in dirty}
        inputs += sum(i // fanout in parents for i in active)
        nodes += len(parents)
        active = {i // fanout for i in active}
        dirty = parents
        extent = (extent + fanout - 1) // fanout
    return nodes, inputs


class StableBlocks:
    def __init__(self, block_size, kind, hole_policy='dirty'):
        if block_size <= 0 or kind not in (0, 1) or hole_policy not in ('dirty', 'first'):
            raise ValueError('positive block size and kind 0/1 required')
        self.block = block_size
        self.hole_policy = hole_policy
        self.columns = 13 if kind == 0 else 7
        self.fanout = 64 // self.columns
        self.previous = None
        self.locations = []
        self.slots = {}

    def update(self, rows):
        first = self.previous is None
        if first:
            self.previous = [None] * len(rows)
            self.locations = [None] * len(rows)
        if len(rows) != len(self.previous):
            raise ValueError('query count changed inside snapshot')
        dirty = set()
        incoming = {}
        changed = 0
        for query, (old, new) in enumerate(zip(self.previous, rows)):
            if old == new:
                continue
            changed += 1
            if old is not None:
                group, slot = self.locations[query]
                dirty.add((group, slot // self.block))
                if new is None or new[0] != old[0]:
                    self.slots[group][slot] = None
                    self.locations[query] = None
            if new is not None and (old is None or new[0] != old[0]):
                incoming.setdefault(new[0], []).append(query)
        for group, queries in incoming.items():
            slots = self.slots.setdefault(group, [])
            holes = [i for i, query in enumerate(slots) if query is None]
            if self.hole_policy == 'dirty':
                holes.sort(key=lambda i: ((group, i // self.block) not in dirty, i))
            missing = len(queries) - len(holes)
            if missing > 0:
                old_size = len(slots)
                slots.extend([None] * (((missing + self.block - 1) // self.block) * self.block))
                holes.extend(range(old_size, len(slots)))
            for query, slot in zip(queries, holes):
                slots[slot] = query
                self.locations[query] = (group, slot)
                dirty.add((group, slot // self.block))

        active = Counter((group, slot // self.block)
                         for group, slots in self.slots.items()
                         for slot, query in enumerate(slots) if query is not None)
        group_counts = Counter(row[0] for row in rows if row is not None)
        affected = {group for group, _ in dirty}
        flat_inputs = sum(group in affected for group, _ in active)
        leaf_tiles = leaf_merge_nodes = leaf_merge_inputs = 0
        for leaf in dirty:
            count = (active[leaf] + 63) // 64
            leaf_tiles += count
            nodes, inputs = merge_work(range(count), range(count), self.fanout)
            leaf_merge_nodes += nodes
            leaf_merge_inputs += inputs
        tree_nodes = tree_inputs = full_nodes = full_inputs = 0
        for group in self.slots:
            if group in affected:
                nodes, inputs = merge_work([i for g, i in active if g == group],
                                          [i for g, i in dirty if g == group], self.fanout,
                                          extent=len(self.slots[group]) // self.block)
                tree_nodes += nodes
                tree_inputs += inputs
            n = (group_counts[group] + 63) // 64
            nodes, inputs = merge_work(range(n), range(n), self.fanout)
            full_nodes += nodes
            full_inputs += inputs
        self.previous = list(rows)
        return dict(searches=1, queries=len(rows), accepted=sum(group_counts.values()),
                    changed_rows=changed, full_qr64_leaves=sum((n + 63) // 64 for n in group_counts.values()),
                    active_leaves=len(active), allocated_leaves=sum(len(s) // self.block for s in self.slots.values()),
                    dirty_leaves=len(dirty), dirty_surviving_leaves=sum(k in active for k in dirty),
                    rebuild_rows=sum(active[k] for k in dirty),
                    rebuild_padded_rows=self.block * sum(k in active for k in dirty),
                    produced_leaf_root_rows=sum(min(self.columns, active[k]) for k in dirty),
                    rebuild_qr64_tiles=leaf_tiles, leaf_qr64_merge_nodes=leaf_merge_nodes,
                    leaf_qr64_merge_input_roots=leaf_merge_inputs,
                    affected_groups=len(affected), flat_merge_input_roots=flat_inputs,
                    tree_merge_nodes=tree_nodes, tree_merge_input_roots=tree_inputs,
                    full_qr64_merge_nodes=full_nodes, full_qr64_merge_input_roots=full_inputs)


def summarize_records(records, block_sizes=(32, 64, 128, 256), warmup=20, hole_policy='dirty'):
    report = {'hole_policy': hole_policy,
              'blocks': {str(b): {'first': {}, 'steady': {}, 'by_kind': {}} for b in block_sizes}}
    states = {}
    epoch = None
    for record in records:
        scan, kind = record['scan'], record['kind']
        if scan < warmup:
            continue
        if epoch is not None and scan < epoch:
            raise ValueError('scan order is not monotone')
        if scan != epoch:
            states.clear()
            epoch = scan
        rows = [(record['groups'][group], index) if group >= 0 else None
                for index, group, _ in record['matches']]
        for block in block_sizes:
            first = (kind, block) not in states
            state = states.setdefault((kind, block), StableBlocks(block, kind, hole_policy))
            counters = state.update(rows)
            phase = 'first' if first else 'steady'
            target = report['blocks'][str(block)]
            targets = [target[phase], target['by_kind'].setdefault(str(kind), {}).setdefault(phase, {})]
            for target in targets:
                for key, value in counters.items():
                    target[key] = target.get(key, 0) + value
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--hole-policy', choices=('dirty', 'first'), default='dirty')
    parser.add_argument('--block-sizes', type=int, nargs='+', default=(32, 64, 128, 256))
    args = parser.parse_args()
    with open(args.input, 'rb') as stream:
        result = summarize_records(read_records(stream), block_sizes=args.block_sizes,
                                   warmup=args.warmup, hole_policy=args.hole_policy)
    result['model'] = __doc__
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        with open(args.output, 'w') as dest:
            dest.write(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
