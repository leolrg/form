import unittest

from analyze_stability_order import PrioritizedBlocks, analyze_snapshot, summarize_records
from analyze_stable_blocks import StableBlocks


class StabilityOrderTests(unittest.TestCase):
    def test_one_changed_block_and_no_change(self):
        state = PrioritizedBlocks(0)
        rows = [(9, i) for i in range(128)]
        self.assertEqual(state.update(rows)['actual_leaf_qr64'], 2)
        self.assertEqual(state.update(rows)['actual_leaf_qr64'], 0)
        rows[0] = (9, 999)
        result = state.update(rows)
        self.assertEqual(result['actual_leaf_qr64'], 1)
        self.assertEqual(result['actual_cached_merge_qr64'], 1)
        self.assertEqual(result['actual_cached_merge_input_roots'], 2)

    def test_priority_retains_original_query_ids_and_later_first_holes(self):
        state = PrioritizedBlocks(1, [1, 0, 0, 0])
        rows = [(9, 0), (9, 1), (9, 2), None]
        state.update(rows)
        self.assertEqual(state.slots[9][:3], [1, 2, 0])
        self.assertEqual(state.previous, rows)
        state.update([(9, 0), None, (9, 2), (9, 3)])
        self.assertEqual(state.locations[3], (9, 0))
        self.assertEqual(state.locations[0], (9, 2))

    def test_migration_dirties_old_and_new_groups_including_empty_leaf(self):
        state = PrioritizedBlocks(0)
        state.update([(9, 0), (10, 1)])
        result = state.update([(10, 2), (10, 1)])
        self.assertEqual(result['actual_leaf_qr64'], 2)
        self.assertEqual(result['actual_cached_merge_qr64'], 0)
        self.assertEqual(state.locations, [(10, 1), (10, 0)])

    def test_oracle_groups_changes_and_learned_pays_rebuild(self):
        first = [(9, i) for i in range(128)]
        second = list(first)
        second[0], second[64] = (9, 1000), (9, 1001)
        third = list(second)
        third[0], third[64] = (9, 2000), (9, 2001)
        report = analyze_snapshot([first, second, third], 0)
        self.assertEqual(report['query_order']['steady']['actual_leaf_qr64'], 4)
        self.assertEqual(report['oracle_future_changes']['steady']['actual_leaf_qr64'], 2)
        learned = report['first_change_flag']
        self.assertEqual(learned['steady']['actual_leaf_qr64'], 3)
        self.assertEqual(learned['reorder']['actual_leaf_qr64'], 2)
        self.assertEqual(learned['steady']['changed_rows'], 4)

    def test_learned_rebuild_cost_even_when_first_transition_unchanged(self):
        rows = [(9, i) for i in range(65)]
        result = analyze_snapshot([rows, rows, rows], 1)['first_change_flag']
        self.assertEqual(result['steady']['actual_leaf_qr64'], 2)
        self.assertEqual(result['reorder']['searches'], 1)
        self.assertEqual(result['steady']['changed_rows'], 0)

    def test_singleton_parent_is_copy_and_empty_holes_still_participate(self):
        state = PrioritizedBlocks(0)
        rows = [(9, i) for i in range(320)]
        result = state.update(rows)
        self.assertEqual(result['actual_cached_merge_qr64'], 2)
        self.assertEqual(result['actual_cached_copy_nodes'], 1)
        rows[-1] = (9, 500)
        result = state.update(rows)
        self.assertEqual(result['actual_cached_merge_qr64'], 1)
        self.assertEqual(result['actual_cached_copy_nodes'], 1)

    def test_stream_canonicalizes_group_ids_and_resets_snapshots(self):
        records = [dict(scan=20, kind=0, groups=[9, 10], matches=[(2, 0, .1)]),
                   dict(scan=20, kind=0, groups=[10, 9], matches=[(2, 1, .2)]),
                   dict(scan=21, kind=0, groups=[9], matches=[(2, 0, .1)])]
        result = summarize_records(iter(records))
        self.assertEqual(result['snapshots'], 2)
        self.assertEqual(result['policies']['query_order']['steady']['changed_rows'], 0)
        self.assertEqual(result['policies']['query_order']['first']['searches'], 2)

    def test_changed_query_count_rejected(self):
        with self.assertRaises(ValueError):
            analyze_snapshot([[(9, 0)], []], 0)

    def test_zero_priorities_match_existing_first_hole_model(self):
        rows = [(9 + (q % 3), q) for q in range(201)]
        baseline = StableBlocks(64, 0, hole_policy='first')
        prioritized = PrioritizedBlocks(0, [0] * len(rows))
        for step in range(4):
            if step:
                rows[step] = None
                rows[step + 64] = (12, step)
            expected = baseline.update(rows)
            actual = prioritized.update(rows)
            self.assertEqual({k: actual[k] for k in expected}, expected)
            self.assertEqual(prioritized.locations, baseline.locations)
            self.assertEqual(prioritized.slots, baseline.slots)


if __name__ == '__main__':
    unittest.main()
