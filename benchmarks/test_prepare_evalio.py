import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import prepare_evalio as p
from prepare_n21 import read_header, read_frames

class ConversionTests(unittest.TestCase):
    def test_preserves_evalio_order_start_timestamp_and_sensor_ranges(self):
        xyz = np.array([[4,5,6], [0,0,0], [1,2,3], [7,8,9]], dtype=np.float32)
        scan = NS(stamp=NS(to_nsec=lambda:123456789), to_vec_positions=lambda:xyz)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=NS(full_name='test/sensor', seq_name='sensor', folder=root,
                       files=lambda:[], quick_len=lambda:1, lidar=lambda:iter([scan]),
                       lidar_params=lambda:NS(num_rows=2,num_columns=2,min_range=.5,max_range=120.))
            with patch.object(p,'export_ground_truth',return_value={'poses':1}):
                meta=p.convert(dataset,root)
            with (root/'sensor.formpc').open('rb') as stream:
                self.assertEqual(read_header(stream),(2,2))
                records=list(read_frames(stream,2,2))
            self.assertEqual(records[0][0],123456789)
            np.testing.assert_array_equal(records[0][1][:,:3],xyz)
            self.assertTrue(meta['complete_sequence'])
            self.assertEqual(meta['replay_settings'],{'min-range':.5,'max-range':120.})
            with self.assertRaises(FileExistsError):p.convert(dataset,root)

if __name__=='__main__':unittest.main()
