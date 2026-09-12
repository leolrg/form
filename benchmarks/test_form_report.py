import unittest
from form_report import grouped_results, weighted_latency, scaling_results


def record(seq='a', backend='reference', ms=10, count=100, status='complete'):
    return {'spec':{'id':f'{seq}-{backend}','sequence':seq,'config':'current','backend':backend,'repeat':1},
            'status':status,'analysis':{'timings':{'steady_state':{'optimization_ms':{'mean_ms':ms,'count':count},'total_ms':{'mean_ms':2*ms,'count':count}},'workload_all_scans':{}},
                                      'trajectory':{'matched_poses':count}},'resources':{'peak_rss_kib':1}}


class ReportTests(unittest.TestCase):
    def test_scaling_uses_real_counts_and_reports_quality_tradeoff(self):
        def group(config,backend,ms,correspondences,poses,rte):
            return {'sequence':'a','config':config,'backend':backend,'status':'validated',
                    'latency':{stage:{'mean_ms':{'mean':value}} for stage,value in
                               [('optimization_ms',ms),('total_ms',2*ms)]},
                    'workload':{'planar_correspondences':{'mean':{'mean':correspondences}},
                                'point_correspondences':{'mean':{'mean':0}},
                                'poses':{'mean':{'mean':poses}}},
                    'trajectory_by_repeat':[]}
        # Construct trajectories separately to keep the synthetic data explicit.
        rows=[]
        for config,backend,ms,count,poses,rte in [('current','cuda',10,100,10,.1),
              ('features','cuda',15,170,10,.12),('features','summary',30,170,10,.12),
              ('features','reference',60,170,10,.12)]:
            row=group(config,backend,ms,count,poses,rte)
            row['trajectory_by_repeat']=[{'matched_poses':100,'rte_1m':{'translation_m':{'mean':rte}},
                                         'rte_30m':{'translation_m':{'mean':2*rte}}}]
            rows.append(row)
        result=scaling_results(rows)[0]
        self.assertEqual(result['status'],'validated')
        self.assertAlmostEqual(result['correspondence_multiplier'],1.7)
        self.assertEqual(result['pose_multiplier'],1)
        self.assertEqual(result['cuda_optimization_latency_multiplier'],1.5)
        self.assertEqual(result['cpu_summary_over_cuda_optimization'],2)
        self.assertEqual(result['reference_over_cuda_optimization'],4)
        self.assertAlmostEqual(result['rte_1m_change_m'],.02)
        self.assertAlmostEqual(result['rte_30m_change_m'],.04)

    def test_scaling_cannot_validate_missing_or_failed_baselines(self):
        row={'sequence':'a','config':'features','backend':'cuda','status':'validated'}
        result=scaling_results([row])[0]
        self.assertEqual(result['status'],'incomplete')
        self.assertNotIn('cpu_summary_over_cuda_optimization',result)

    def test_weighting_uses_scans_and_requires_all_sequences(self):
        rows=grouped_results([record('a',ms=10,count=100),record('b',ms=30,count=300)],[])
        self.assertEqual(weighted_latency(rows,'reference','optimization_ms',['a','b']),25)
        self.assertIsNone(weighted_latency(rows,'reference','optimization_ms',['a','b','c']))

    def test_incomplete_or_failed_repeat_cannot_qualify(self):
        good=record(backend='cuda')
        bad=record(backend='cuda',status='failed');bad['spec']['repeat']=2
        rows=grouped_results([good,bad],[])
        self.assertEqual(rows[0]['status'],'incomplete')
        self.assertIsNone(weighted_latency(rows,'cuda','optimization_ms',['a']))

    def test_quality_failure_is_retained(self):
        rows=grouped_results([record(backend='cuda')],[{'sequence':'a','config':'current','candidate':'cuda','quality_gate_status':'fail'}])
        self.assertEqual(rows[0]['status'],'quality-failed')
        self.assertIsNone(weighted_latency(rows,'cuda','optimization_ms',['a']))

class TraceTests(unittest.TestCase):
    def test_pose_difference_and_changed_workload_are_reported(self):
        from form_report import trace_agreement
        from pathlib import Path
        import tempfile, math
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            Path(str(a)+'.csv').write_text('stamp_ns,lm_iterations,full_final_error\n1000000000,2,9\n2000000000,4,10\n')
            Path(str(b)+'.csv').write_text('stamp_ns,lm_iterations,full_final_error\n1000000000,2,9.0001\n2000000000,5,10\n')
            Path(str(a)+'.tum').write_text('1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n')
            Path(str(b)+'.tum').write_text(f'1 3 4 0 0 0 {2**-.5} {2**-.5}\n2 0 0 0 0 0 0 -1\n')
            result=trace_agreement(a,b)
            self.assertEqual(result['max_translation_difference_m'],5)
            self.assertAlmostEqual(result['max_rotation_difference_rad'],math.pi/2)
            self.assertEqual(result['changed_counts']['lm_iterations'],1)
            self.assertAlmostEqual(result['full_final_error']['max_absolute_difference'],.0001)
