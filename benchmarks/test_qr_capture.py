import struct
import tempfile
from pathlib import Path
import unittest
import numpy as np
import inspect_qr_capture as q


class CaptureFormatTests(unittest.TestCase):
    def test_ragged_column_major_including_empty_matrix(self):
        a=np.arange(21,dtype='<f8').reshape(3,7)
        payload=b'FORMQR01'+struct.pack('<I',2)+struct.pack('<QI',3,7)+a.tobytes(order='F')+struct.pack('<QI',0,13)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'batch.formqr';path.write_bytes(payload)
            matrices=q.read_batch(path)
            np.testing.assert_array_equal(matrices[0],a)
            self.assertEqual(matrices[1].shape,(0,13))

    def test_reject_bad_header_dimensions_truncation_and_trailing_data(self):
        good=b'FORMQR01'+struct.pack('<I',1)+struct.pack('<QI',0,7)
        invalid=[b'BADQR001'+good[8:],b'FORMQR01',good[:-1],good+b'X',
                 b'FORMQR01'+struct.pack('<I',0),
                 b'FORMQR01'+struct.pack('<I',1)+struct.pack('<QI',3,8),
                 b'FORMQR01'+struct.pack('<I',1)+struct.pack('<QI',2**63,7),
                 b'FORMQR01'+struct.pack('<I',1)+struct.pack('<QI',1,7)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'batch.formqr'
            for payload in invalid:
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    q.read_batch(path)


if __name__=='__main__':
    unittest.main()
