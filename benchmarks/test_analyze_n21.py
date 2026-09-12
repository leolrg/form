import tempfile
from pathlib import Path
import unittest

import numpy as np
import analyze_n21 as a


class AnalysisTests(unittest.TestCase):
    def test_summary_reports_tails_and_deadlines(self):
        result = a.summarize_ms([10, 20, 30, 140])
        self.assertEqual(result['median_ms'], 25)
        self.assertEqual(result['mean_ms'], 50)
        self.assertEqual(result['deadline_miss_fraction'], .25)
        self.assertEqual(result['throughput_hz'], 20)
        self.assertIsNone(a.summarize_ms([]))
        with self.assertRaises(ValueError):
            a.summarize_ms([float('nan')])

    def test_association_enforces_tolerance_and_unique_matches(self):
        est, gt = a.associate_stamps([0,10,11,30,100], [0,12,31,60], 3)
        self.assertEqual(est.tolist(), [0,2,3])
        self.assertEqual(gt.tolist(), [0,1,2])
        with self.assertRaises(ValueError):
            a.associate_stamps([1,0], [0,1], 3)

    def test_tum_preserves_nanosecond_stamp_and_rejects_bad_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'poses.tum'
            path.write_text('1625132454.927972608 1 2 3 0 0 0 1\n')
            trajectory = a.load_tum(path)
            self.assertEqual(trajectory.stamps[0].to_nsec(), 1625132454927972608)
            np.testing.assert_array_equal(trajectory.poses[0].trans,[1,2,3])
            path.write_text('1 1 2 3 0 0 0 0\n')
            with self.assertRaises(ValueError):
                a.load_tum(path)

    def test_evalio_identity_has_zero_errors_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'poses.tum'
            path.write_text(''.join(f'{i}.000000000 {i} 0 0 0 0 0 1\n' for i in range(35)))
            result = a.trajectory_metrics(path,path)
            self.assertEqual(result['matched_poses'],35)
            self.assertEqual(result['estimate_coverage_fraction'],1)
            self.assertEqual(result['ate_first_pose_aligned']['translation_m']['rmse'],0)
            self.assertEqual(result['rte_30m']['translation_m']['rmse'],0)

class TimingFileTests(unittest.TestCase):
    def test_warmup_excluded_and_optimization_includes_both_solvers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'timing.csv'
            path.write_text('scan,stamp_ns,total_ms,semi_ms,full_ms,marginalize_ms,poses\n'
                            '0,1,200,80,20,10,1\n'
                            '1,2,40,5,10,4,2\n'
                            '2,3,60,10,15,4,3\n')
            result = a.timing_metrics(path,1)
            self.assertEqual(result['all_scans']['total_ms']['count'],3)
            self.assertEqual(result['steady_state']['total_ms']['count'],2)
            self.assertEqual(result['steady_state']['optimization_ms']['mean_ms'],20)
            self.assertEqual(result['steady_state']['marginalize_ms']['mean_ms'],4)
            self.assertEqual(result['workload_all_scans']['poses']['max'],3)


if __name__ == '__main__':
    unittest.main()
