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

    def test_two_transition_rebuild_and_remaining_work(self):
        first = [(9, i) for i in range(128)]
        history = [first]
        for step in range(1, 4):
            rows = list(history[-1])
            rows[0], rows[64] = (9, 1000 * step), (9, 1000 * step + 1)
            history.append(rows)
        result = analyze_snapshot(history, 0)
        learned = result['two_transition_persistence']
        self.assertEqual(learned['total']['actual_leaf_qr64'], 7)
        self.assertEqual(learned['steady']['actual_leaf_qr64'], 5)
        self.assertEqual(learned['reorder']['actual_leaf_qr64'], 2)
        self.assertEqual(learned['after_rematch2']['actual_leaf_qr64'], 1)
        self.assertEqual(result['query_order']['after_rematch2']['actual_leaf_qr64'], 2)
        self.assertEqual(learned['steady']['changed_rows'], 6)

    def test_two_transition_policy_does_not_rebuild_short_history(self):
        rows = [(9, i) for i in range(65)]
        result = analyze_snapshot([rows, rows], 0)['two_transition_persistence']
        self.assertEqual(result['total']['actual_leaf_qr64'], 2)
        self.assertEqual(result['reorder'], {})
        self.assertEqual(result['after_rematch2'], {})

    def test_two_transition_priority_uses_latest_change_weight(self):
        first = [(9, i) for i in range(257)]
        second = list(first)
        # Initial-only unstable query gets score1; late change gets score2.
        second[0] = (9, 1000)
        third = list(second)
        third[1] = (9, 1001)
        fourth = list(third)
        fourth[0] = (9, 2000)
        result = analyze_snapshot([first, second, third, fourth], 0)
        # Scores put original query0 at slot255, query1 at256. The changed
        # query0 therefore dirties the four-input merge, not the singleton tail.
        learned = result['two_transition_persistence']['after_rematch2']
        self.assertEqual(learned['actual_leaf_qr64'], 1)
        self.assertEqual(learned['actual_cached_merge_input_roots'], 6)

    def test_dirty_partition_preserves_clean_leaves_and_row_ownership(self):
        rows = [(9, q) for q in range(192)]
        state = PrioritizedBlocks(0, dirty_partition=True)
        normal = PrioritizedBlocks(0)
        state.update(rows)
        normal.update(rows)
        clean = list(state.slots[9][64:128])
        rows[0], rows[128], rows[129] = (9, 1000), (9, 1001), None
        result = state.update(rows)
        expected = normal.update(rows)
        self.assertEqual(state.slots[9][64:128], clean)
        self.assertEqual(state.slots[9][:2], [0, 128])
        self.assertIsNone(state.slots[9][-1])
        self.assertEqual(sorted(q for q in state.slots[9] if q is not None),
                         [q for q, row in enumerate(rows) if row is not None])
        for slot, query in enumerate(state.slots[9]):
            if query is not None:
                self.assertEqual(state.locations[query], (9, slot))
        self.assertEqual(result['actual_qr64_tasks'], expected['actual_qr64_tasks'])
        self.assertEqual(state.previous, rows)
        self.assertEqual(state.update(rows)['actual_qr64_tasks'], 0)

    def test_dirty_partition_migrations_keep_groups_and_feature_multisets(self):
        rows = [(9, q) for q in range(70)] + [(10, q) for q in range(70, 140)]
        state = PrioritizedBlocks(1, dirty_partition=True)
        state.update(rows)
        rows[0], rows[75], rows[76] = (10, 1000), (9, 1001), None
        state.update(rows)
        for group, slots in state.slots.items():
            actual = sorted(rows[q] for q in slots if q is not None)
            expected = sorted(row for row in rows if row is not None and row[0] == group)
            self.assertEqual(actual, expected)
        self.assertIsNone(state.locations[76])

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
