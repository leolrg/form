import csv
import json
from pathlib import Path
import tempfile
import unittest

import run_suite as r


class SuiteTests(unittest.TestCase):
    def test_additional_dataset_ranges_apply_to_every_backend(self):
        import contextlib, io, sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'basement_2.formpc').write_bytes(b'x')
            (root/'basement_2.gt.tum').write_text('')
            (root/'basement_2.json').write_text(json.dumps({'scans':1,'bytes':1,
                'replay_settings':{'min-range':.5,'max-range':120.}}))
            prov={k:None for k in ('binary_sha256','git_revision','git_dirty_diff_sha256',
                                  'untracked_file_sha256','environment','cpu_affinity')}
            argv=['run_suite.py','--plan','--input-dir',tmp,'--sequences','basement_2',
                  '--configs','current','--backends','reference','summary-resident','cuda-matching']
            output=io.StringIO()
            with patch.object(sys,'argv',argv), patch.object(r,'provenance',return_value=prov), contextlib.redirect_stdout(output):
                r.main()
            runs=json.loads(output.getvalue())['runs']
            self.assertEqual(len(runs),6)
            for run in runs:
                self.assertEqual(float(run['argv'][run['argv'].index('--min-range')+1]),.5)
                self.assertEqual(float(run['argv'][run['argv'].index('--max-range')+1]),120.)
                self.assertEqual(run['settings']['min-range'],.5)

    def test_previous_matcher_uses_frozen_binary_and_reversed_order(self):
        import contextlib, io, sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'stairs.formpc').write_bytes(b'x')
            (root/'stairs.gt.tum').write_text('')
            (root/'stairs.json').write_text(json.dumps({'scans':1,'bytes':1}))
            frozen=root/'previous'; frozen.write_bytes(b'old')
            prov={k:None for k in ('binary_sha256','git_revision','git_dirty_diff_sha256',
                                  'untracked_file_sha256','environment','cpu_affinity')}
            argv=['run_suite.py','--plan','--input-dir',tmp,'--sequences','stairs',
                  '--configs','current','--backends','reference','cuda-matching-previous','cuda-matching',
                  '--previous-matching-binary',str(frozen),'--cuda-solve-min-dimension','600']
            output=io.StringIO()
            with patch.object(sys,'argv',argv), patch.object(r,'provenance',return_value=prov), contextlib.redirect_stdout(output):
                r.main()
            manifest=json.loads(output.getvalue()); runs=manifest['runs']
            self.assertEqual([x['backend'] for x in runs[:3]],list(reversed([x['backend'] for x in runs[3:]])))
            for run in runs:
                if run['backend']=='cuda-matching-previous':
                    self.assertEqual(run['argv'][0],str(frozen))
                    self.assertEqual(run['argv'][run['argv'].index('--backend')+1],'cuda-matching')
                    self.assertEqual(run['binary_sha256'],r.digest(frozen))
                    self.assertEqual(run['argv'][run['argv'].index('--cuda-solve-min-dimension')+1],'600')
            self.assertEqual(manifest['provenance']['comparison_binary']['sha256'],r.digest(frozen))

    def test_hybrid_rejects_legacy_config_restriction(self):
        import contextlib, io, sys
        from unittest.mock import patch
        argv = ['run_suite.py', '--output', '/unused', '--backends', 'cuda-resident-hybrid',
                '--cuda-solve-min-dimension', '240', '--cuda-solve-configs', 'window']
        with patch.object(sys, 'argv', argv), patch.object(r, 'provenance', side_effect=AssertionError('must reject before provenance')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                r.main()
            self.assertEqual(error.exception.code, 2)

    def test_matching_and_hybrid_receive_same_threshold(self):
        import contextlib, io, sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'stairs.formpc').write_bytes(b'x')
            (root/'stairs.gt.tum').write_text('')
            (root/'stairs.json').write_text(json.dumps({'scans':1,'bytes':1}))
            prov={k:None for k in ('binary_sha256','git_revision','git_dirty_diff_sha256',
                                  'untracked_file_sha256','environment','cpu_affinity')}
            argv=['run_suite.py','--plan','--input-dir',tmp,'--sequences','stairs',
                  '--configs','current','--backends','cuda-matching','cuda-resident-hybrid',
                  '--cuda-solve-min-dimension','600']
            output=io.StringIO()
            with patch.object(sys,'argv',argv), patch.object(r,'provenance',return_value=prov), contextlib.redirect_stdout(output):
                r.main()
            for run in json.loads(output.getvalue())['runs']:
                self.assertIn('--cuda-solve-min-dimension',run['argv'])
                i=run['argv'].index('--cuda-solve-min-dimension')
                self.assertEqual(run['argv'][i+1],'600')

    def test_workload_variants_change_independently(self):
        self.assertEqual(r.CONFIGS['current'], {'points':3,'planes':50,'recent':10})
        self.assertEqual(r.CONFIGS['features'], {'points':6,'planes':100,'feature-spacing':2,'recent':10})
        self.assertEqual(r.CONFIGS['window'], {'points':3,'planes':50,'recent':40})

    def test_quality_gate_has_absolute_floor_and_requires_all_metrics(self):
        def report(value):
            return {'matched_poses':100,**{f'rte_{n}m':{'translation_m':{metric:value for metric in ('mean','median','rmse','max')}}
                                          for n in (1,30)}}
        self.assertEqual(r.quality_gate(report(.1),report(.109))['status'],'pass')
        self.assertEqual(r.quality_gate(report(.1),report(.111))['status'],'fail')
        self.assertEqual(r.quality_gate(report(1),report(1.04))['status'],'pass')
        self.assertEqual(r.quality_gate(report(1),report(1.06))['status'],'fail')
        self.assertEqual(r.quality_gate(report(1),{})['status'],'missing')
        for window in ('rte_1m','rte_30m'):
            for metric in ('median','max'):
                candidate=report(.1)
                candidate[window]['translation_m'][metric]=.111
                self.assertEqual(r.quality_gate(report(.1),candidate)['status'],'fail')
                del candidate[window]['translation_m'][metric]
                self.assertEqual(r.quality_gate(report(.1),candidate)['status'],'missing')
        self.assertEqual(len(r.GATE['metrics']),8)
        candidate=report(.1);candidate['matched_poses']=99
        self.assertEqual(r.quality_gate(report(.1),candidate)['status'],'fail')

    def test_verify_outputs_rejects_incomplete_and_wrong_stamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'run'
            Path(str(prefix)+'.csv').write_text('scan,stamp_ns,total_ms\n0,1000000000,2\n1,2000000000,3\n')
            Path(str(prefix)+'.tum').write_text('1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n')
            self.assertEqual(r.verify_outputs(prefix,2)['count'],2)
            with self.assertRaises(ValueError):
                r.verify_outputs(prefix,3)
            Path(str(prefix)+'.tum').write_text('1 0 0 0 0 0 0 1\n3 0 0 0 0 0 0 1\n')
            with self.assertRaises(ValueError):
                r.verify_outputs(prefix,2)

    def test_completed_resume_requires_matching_fingerprint_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'run'
            Path(str(prefix)+'.csv').write_text('scan,stamp_ns,total_ms\n0,1000000000,2\n')
            Path(str(prefix)+'.tum').write_text('1 0 0 0 0 0 0 1\n')
            outputs=r.verify_outputs(prefix,1)
            record={'status':'complete','fingerprint':'abc','outputs':outputs}
            r.verify_resume(record,prefix,1,'abc')
            with self.assertRaises(ValueError):
                r.verify_resume(record,prefix,1,'different')
            Path(str(prefix)+'.csv').write_text('scan,stamp_ns,total_ms\n0,1000000000,4\n')
            with self.assertRaises(ValueError):
                r.verify_resume(record,prefix,1,'abc')
            record['status']='running'
            with self.assertRaises(ValueError):
                r.verify_resume(record,prefix,1,'abc')


class RunnerIntegrationTests(unittest.TestCase):
    def test_sequential_execution_resume_and_missing_aggregation(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            worker=root/'fake_replay.py'
            worker.write_text("""import sys
from pathlib import Path
prefix=Path(sys.argv[1])
Path(str(prefix)+'.csv').write_text('scan,stamp_ns,total_ms,semi_ms,full_ms,marginalize_ms\\n'+''.join(f'{i},{(i+1)*1000000000},10,2,1,1\\n' for i in range(35)))
Path(str(prefix)+'.tum').write_text(''.join(f'{i+1} {i} 0 0 0 0 0 1\\n' for i in range(35)))
""")
            gt=root/'gt.tum'
            gt.write_text(''.join(f'{i+1} {i} 0 0 0 0 0 1\n' for i in range(35)))
            specs=[]
            for backend in ('reference','summary'):
                spec={'id':backend,'sequence':'stairs','config':'current','backend':backend,
                      'repeat':1,'expected_scans':35,'fingerprint':'fixed',
                      'ground_truth':str(gt),'argv':[sys.executable,str(worker),str(root/backend)]}
                specs.append(spec)
            manifest={'runs':specs}
            before=r.aggregate(manifest,root)
            self.assertEqual(before['comparisons'][0]['quality_gate_status'],'missing')
            for spec in specs:
                r.execute(spec,root,{})
                record=root/(spec['id']+'.run.json')
                saved=record.read_bytes()
                r.execute(spec,root,{})
                self.assertEqual(record.read_bytes(),saved)
            after=r.aggregate(manifest,root)
            self.assertEqual(after['comparisons'][0]['quality_gate_status'],'pass')
            self.assertTrue(all(item['status']=='complete' for item in after['runs']))
            self.assertGreater(json.loads((root/'reference.run.json').read_text())['resources']['peak_rss_kib'],0)


class GpuMemoryTests(unittest.TestCase):
    def test_sums_only_target_process_memory_and_ignores_na(self):
        sample='123, 12\n456, 1024\n123, 20\n789, [N/A]\n'
        self.assertEqual(r.gpu_memory_from_csv(sample,{123,789}),32)
        self.assertIsNone(r.gpu_memory_from_csv(sample,{999}))


if __name__ == '__main__':
    unittest.main()
