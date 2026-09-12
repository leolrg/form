import tempfile
from pathlib import Path
import unittest
import numpy as np
import prepare_qr_roots as p
from inspect_qr_capture import read_batch


class RootFixtureTests(unittest.TestCase):
    def test_short_rank_deficient_and_empty_inputs_have_clean_padded_roots(self):
        for raw in (np.array([[1.,2,3,4,5,6,7],[1.,2,3,4,5,6,7]]),np.zeros((0,13))):
            root,error=p.clean_root(raw)
            self.assertEqual(root.shape,(raw.shape[1],raw.shape[1]))
            np.testing.assert_array_equal(np.tril(root,-1),np.zeros_like(root))
            np.testing.assert_allclose(root.T@root,raw.T@raw,atol=1e-12)
            self.assertLess(error,1e-12)

    def test_nonunit_point_weights_and_wire_roundtrip(self):
        root,_=p.clean_root(np.column_stack([np.ones(9),np.arange(54).reshape(9,6)]))
        self.assertAlmostEqual(abs(root[0,0]),3)
        self.assertTrue((root[1:,0]==0).all())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'points.formqr'
            p.write_batch(path,[root])
            np.testing.assert_array_equal(read_batch(path)[0],root)
            with self.assertRaises(FileExistsError):
                p.write_batch(path,[root])

    def test_rejects_nonfinite_or_wrong_width(self):
        for raw in (np.zeros((3,8)),np.full((3,7),np.nan)):
            with self.assertRaises(ValueError):
                p.clean_root(raw)


if __name__=='__main__':
    unittest.main()
