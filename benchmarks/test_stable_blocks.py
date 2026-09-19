import random
import unittest

from analyze_stable_blocks import StableBlocks, merge_work, summarize_records


class StableBlockTests(unittest.TestCase):
    def test_unchanged_leaf_survives_target_change(self):
        state = StableBlocks(2, 0)
        rows = [(9, i) for i in range(4)]
        state.update(rows)
        rows[0] = (9, 100)
        result = state.update(rows)
        self.assertEqual(result['changed_rows'], 1)
        self.assertEqual(result['active_leaves'], 2)
        self.assertEqual(result['dirty_leaves'], 1)
        self.assertEqual(result['rebuild_rows'], 2)
        self.assertEqual(state.locations[2], (9, 2))

    def test_migration_and_rejection_touch_both_groups(self):
        state = StableBlocks(2, 1)
        state.update([(9, 1), (9, 2), (10, 3), (10, 4)])
        result = state.update([(9, 1), (10, 8), None, (10, 4)])
        self.assertEqual(result['changed_rows'], 2)
        self.assertEqual(result['dirty_leaves'], 2)
        self.assertEqual(result['rebuild_rows'], 3)
        self.assertEqual(state.locations[1], (10, 0))
        self.assertEqual(state.locations[3], (10, 1))
        self.assertIsNone(state.locations[2])

    def test_fill_dirty_block_before_clean_hole(self):
        state = StableBlocks(2, 0)
        state.update([(9, i) for i in range(4)] + [None])
        state.update([None, (9, 1), (9, 2), (9, 3), None])
        result = state.update([None, (9, 1), None, (9, 3), (9, 4)])
        self.assertEqual(state.locations[4], (9, 2))
        self.assertEqual(result['dirty_leaves'], 1)
        self.assertEqual(result['rebuild_rows'], 2)

    def test_first_hole_policy_uses_lowest_slot(self):
        state = StableBlocks(2, 0, hole_policy='first')
        state.update([(9, i) for i in range(4)] + [None])
        state.update([None, (9, 1), (9, 2), (9, 3), None])
        result = state.update([None, (9, 1), None, (9, 3), (9, 4)])
        self.assertEqual(state.locations[4], (9, 0))
        self.assertEqual(result['dirty_leaves'], 2)
        self.assertEqual(result['rebuild_rows'], 3)

    def test_rejection_removes_last_leaf_and_reacceptance_reuses_it(self):
        state = StableBlocks(2, 0)
        state.update([(9, 1)])
        result = state.update([None])
        self.assertEqual(result['dirty_leaves'], 1)
        self.assertEqual(result['dirty_surviving_leaves'], 0)
        self.assertEqual(result['rebuild_rows'], 0)
        state.update([(9, 2)])
        self.assertEqual(state.locations[0], (9, 0))
        self.assertEqual(len(state.slots[9]), 2)

    def test_stream_resets_snapshots_and_canonicalizes_groups(self):
        records = [dict(scan=20, kind=0, groups=[9, 10], matches=[(1, 0, .1)]),
                   dict(scan=20, kind=0, groups=[10, 9], matches=[(1, 1, .2)]),
                   dict(scan=21, kind=0, groups=[9], matches=[(2, 0, .1)])]
        result = summarize_records(records, (2,))['blocks']['2']
        self.assertEqual(result['first']['searches'], 2)
        self.assertEqual(result['steady']['searches'], 1)
        self.assertEqual(result['steady']['changed_rows'], 0)
        self.assertEqual(result['steady']['dirty_leaves'], 0)

    def test_cached_tree_counts_only_dirty_ancestors(self):
        self.assertEqual(merge_work(range(10), {0}, 4), (2, 7))
        self.assertEqual(merge_work(range(10), set(), 4), (0, 0))
        self.assertEqual(merge_work(range(10), range(10), 4), (4, 13))
        self.assertEqual(merge_work({0}, {0}, 4, extent=16), (2, 2))

    def test_large_virtual_leaf_accounts_for_internal_qr64_work(self):
        state = StableBlocks(128, 0)
        result = state.update([(9, i) for i in range(100)])
        self.assertEqual(result['dirty_surviving_leaves'], 1)
        self.assertEqual(result['rebuild_qr64_tiles'], 2)
        self.assertEqual(result['leaf_qr64_merge_input_roots'], 2)
        self.assertEqual(result['tree_merge_input_roots'], 0)

    def test_random_updates_preserve_ownership_and_unchanged_slots(self):
        rng = random.Random(71903)
        state = StableBlocks(4, 0)
        old = [None] * 41
        for _ in range(100):
            rows = [None if rng.random() < .2 else (rng.randrange(4), rng.randrange(100)) for _ in old]
            previous_locations = list(state.locations)
            state.update(rows)
            found = {}
            for group, slots in state.slots.items():
                for slot, query in enumerate(slots):
                    if query is not None:
                        self.assertNotIn(query, found)
                        found[query] = (group, slot)
            self.assertEqual(set(found), {q for q, row in enumerate(rows) if row is not None})
            for query, row in enumerate(rows):
                self.assertEqual(state.locations[query], found.get(query))
                if row is not None:
                    self.assertEqual(found[query][0], row[0])
                if row is not None and old[query] is not None and row[0] == old[query][0]:
                    self.assertEqual(state.locations[query], previous_locations[query])
            old = rows


if __name__ == '__main__':
    unittest.main()
