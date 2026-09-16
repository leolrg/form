import unittest
import json
from pathlib import Path
import tempfile
from analyze_diagnostics import union_duration, summarize_optimizer

class DiagnosticsTests(unittest.TestCase):
    def test_union_does_not_double_count_overlap(self):
        self.assertEqual(union_duration([(0,10),(3,6),(8,20),(25,30)]),25)
        self.assertEqual(union_duration([]),0)
    def test_optimizer_groups_by_actual_backend_and_phase(self):
        rows=[dict(scan='20',backend='cpu',phase='semi',dimension='234',wall_ms='2',diag_cpu_solve_ms='1',diag_cpu_solve_calls='3'),
              dict(scan='21',backend='gpu',phase='semi',dimension='240',wall_ms='4',diag_gpu_solve_ms='2',diag_gpu_solve_calls='5'),
              dict(scan='0',backend='gpu',phase='full',dimension='6',wall_ms='1000')]
        out=summarize_optimizer(rows,timed_scans=2)
        self.assertEqual(len(out),2)
        self.assertEqual(out['cpu/semi']['dimension_min'],234)
        self.assertEqual(out['gpu/semi']['dimension_max'],240)
        self.assertEqual(out['gpu/semi']['per_scan']['wall_ms'],2)
        self.assertEqual(out['gpu/semi']['per_call']['diag_gpu_solve_calls'],5)

class TraceAttributionTests(unittest.TestCase):
    def test_launch_scope_and_scan_not_sync_scope(self):
        from analyze_cuda_trace import attribute_apis
        ranges=[dict(start=0,end=100,name='scan/20',globalTid=7),
                dict(start=10,end=30,name='stage/matching',globalTid=7),
                dict(start=15,end=20,name='match_search_launch',globalTid=7),
                dict(start=30,end=50,name='wait',globalTid=7)]
        apis=[dict(start=16,end=17,globalTid=7,correlationId=1),
              dict(start=31,end=45,globalTid=7,correlationId=2),
              dict(start=101,end=103,globalTid=7,correlationId=3)]
        out=attribute_apis(ranges,apis)
        self.assertEqual(out[(0,1)]['scope'],'match_search_launch')
        self.assertEqual(out[(0,1)]['stage'],'stage/matching')
        self.assertEqual(out[(0,2)]['scope'],'wait')
        self.assertEqual(out[(0,1)]['scan'],20)
        self.assertNotIn((0,3),out)

class PolicyTests(unittest.TestCase):
    def test_pairs_require_matching_invocations(self):
        from analyze_optimizer_policy import pair_calls
        a=dict(scan='20',call='1',phase='semi',dimension='120',wall_ms='2')
        b={**a,'wall_ms':'3'}
        self.assertEqual(len(pair_calls([a],[b])),1)
        with self.assertRaises(ValueError):pair_calls([a],[{**b,'dimension':'126'}])
        with self.assertRaises(ValueError):pair_calls([a,a],[b])

class ArtifactTests(unittest.TestCase):
    def test_profile_artifacts_required_and_mutation_rejected(self):
        from analyze_diagnostics import optimizer_manifest
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);suite=base/'profile';suite.mkdir()
            (suite/'suite.json').write_text(json.dumps({'runs':[{'id':'run','argv':['--profile']}]}))
            (suite/'run.run.json').write_text(json.dumps({'status':'complete'}))
            with self.assertRaises(FileNotFoundError):optimizer_manifest(base,create=True)
            artifact=suite/'run.optimizer.csv';artifact.write_text('scan,wall_ms\n')
            with self.assertRaises(ValueError):optimizer_manifest(base,create=True)
            artifact.write_text('scan,wall_ms\n20,1\n')
            optimizer_manifest(base,create=True);optimizer_manifest(base)
            artifact.write_text('scan,wall_ms\n20,2\n')
            with self.assertRaises(ValueError):optimizer_manifest(base)
