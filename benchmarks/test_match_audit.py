import io
import struct
import unittest

from analyze_match_audit import read_records, summarize_records


def record(scan, kind, groups, matches):
    data = struct.pack('<4Q', scan, kind, len(matches), len(groups))
    data += struct.pack('<' + 'Q'*len(groups), *groups)
    for index, group, distance in matches:
        data += struct.pack('<iid', index, group, distance)
    return data


class MatchAuditTests(unittest.TestCase):
    def test_reads_complete_records_and_rejects_truncation(self):
        raw = b'FORMMAT1' + record(2, 0, [1, 3], [(4, 0, .2), (-1, -1, 1e300)])
        records = list(read_records(io.BytesIO(raw)))
        self.assertEqual(records[0]['groups'], (1, 3))
        self.assertEqual(records[0]['matches'][0], (4, 0, .2))
        for truncated in (b'', raw[:9], raw[:-1]):
            with self.assertRaises(ValueError):
                list(read_records(io.BytesIO(truncated)))

    def test_dirty_blocks_count_old_and_new_target_groups(self):
        # Query 1 migrates from group 0 to group 1; query 2 becomes rejected.
        raw = b'FORMMAT1'
        raw += record(20, 0, [9, 10], [(1, 0, .1), (2, 0, .1), (3, 1, .1), (4, 1, .1)])
        raw += record(20, 0, [9, 10], [(1, 0, .2), (8, 1, .2), (3, -1, .9), (4, 1, .2)])
        report = summarize_records(list(read_records(io.BytesIO(raw))), block_sizes=(2,))
        self.assertEqual(report['steady']['queries'], 4)
        self.assertEqual(report['steady']['raw_identity_changes'], 1)
        self.assertEqual(report['steady']['summary_row_changes'], 2)
        self.assertEqual(report['steady']['blocks']['2']['dirty_union'], 3)
        self.assertEqual(report['steady']['blocks']['2']['active_after'], 3)

    def test_distances_change_without_changing_summary_and_epochs_reset(self):
        raw = b'FORMMAT1'
        for scan in (20, 21):
            raw += record(scan, 1, [1], [(0, 0, .1), (1, -1, .9)])
            raw += record(scan, 1, [1], [(0, 0, .2), (2, -1, 1.2)])
        report = summarize_records(list(read_records(io.BytesIO(raw))), block_sizes=(2,))
        self.assertEqual(report['first']['queries'], 4)
        self.assertEqual(report['steady']['queries'], 4)
        self.assertEqual(report['steady']['summary_row_changes'], 0)
        self.assertEqual(report['steady']['raw_identity_changes'], 2)

    def test_reordered_groups_are_compared_by_target_scan(self):
        raw = b'FORMMAT1'
        raw += record(20, 0, [4, 5], [(2, 0, .1)])
        raw += record(20, 0, [5, 4], [(2, 1, .1)])
        report = summarize_records(list(read_records(io.BytesIO(raw))), block_sizes=(2,))
        self.assertEqual(report['steady']['summary_row_changes'], 0)

    def test_rejects_invalid_group_and_nonfinite_distance(self):
        for matches in ([(0, 1, .1)], [(0, 0, float('nan'))], [(-1, 0, .1)]):
            with self.assertRaises(ValueError):
                list(read_records(io.BytesIO(b'FORMMAT1' + record(20, 0, [1], matches))))


if __name__ == '__main__':
    unittest.main()
