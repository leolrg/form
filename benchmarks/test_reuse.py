import unittest
from collections import defaultdict
from run_reuse_benchmark import schedule

class ReuseScheduleTests(unittest.TestCase):
    def test_every_run_has_exact_old_new_workload_pair(self):
        pairs=defaultdict(set)
        for suite,config,limit,repeat,version,backend,profile in schedule():
            pairs[(suite,config,limit,repeat,backend,profile)].add(version)
        self.assertEqual(len(schedule()),30)
        self.assertTrue(all(v=={'old','new'} for v in pairs.values()))
        for config in ['features','window']:
            for backend in ['cpu-extraction','cuda-extraction']:
                self.assertIn(('clean-scaling',config,250,2,backend,False),pairs)
    def test_second_repeat_reverses_control_order(self):
        first=[r[4:6] for r in schedule() if r[0]=='clean-full' and r[3]==1]
        second=[r[4:6] for r in schedule() if r[0]=='clean-full' and r[3]==2]
        self.assertEqual(first,list(reversed(second)))
