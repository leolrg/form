import io
import struct
import unittest
from types import SimpleNamespace as NS

import numpy as np
import prepare_n21 as p


class RecordTests(unittest.TestCase):
    def test_round_trip_and_exact_wire_format(self):
        stream = io.BytesIO()
        xyz = np.array([[1, 2, 3], [-4, 5, 6]], dtype=np.float32)
        p.write_header(stream, 1, 2)
        p.write_frame(stream, 1625132454927972608, xyz, 1, 2)
        expected = (b'FORMPC01' + struct.pack('<IIqI', 1, 2, 1625132454927972608, 2)
                    + struct.pack('<8f', 1, 2, 3, 0, -4, 5, 6, 0))
        self.assertEqual(stream.getvalue(), expected)
        stream.seek(0)
        self.assertEqual(p.read_header(stream), (1, 2))
        stamp, out = next(p.read_frames(stream, 1, 2))
        self.assertEqual(stamp, 1625132454927972608)
        np.testing.assert_array_equal(out[:, :3], xyz)
        self.assertEqual(list(p.read_frames(stream, 1, 2)), [])

    def test_reject_invalid_dimensions_and_point_count(self):
        for dims in [(0, 2), (2, 0), (-1, 2), (2**32, 2)]:
            with self.assertRaises(ValueError):
                p.write_header(io.BytesIO(), *dims)
        with self.assertRaises(ValueError):
            p.write_frame(io.BytesIO(), 0, np.zeros((3, 3)), 1, 2)
        for wire in [b'FORMPC00'+struct.pack('<II', 1, 2),
                     b'FORMPC01'+struct.pack('<II', 0, 2), b'FORMPC01']:
            with self.assertRaises(ValueError):
                p.read_header(io.BytesIO(wire))

    def test_reject_truncation_and_bad_frame_count(self):
        for wire in [b'x', struct.pack('<qI', 3, 1), struct.pack('<qI', 3, 2)+b'x']:
            with self.assertRaises(ValueError):
                list(p.read_frames(io.BytesIO(wire), 1, 2))

    def test_pointcloud_handles_row_padding_and_big_endian(self):
        dtype = np.dtype({'names':['z','x','y'], 'formats':['>f4']*3,
                          'offsets':[0, 4, 8], 'itemsize':16})
        data = bytearray(80)
        points = np.ndarray((2,2), dtype=dtype, buffer=data, strides=(40,16))
        points['x'] = [[1,2],[3,4]]
        points['y'] = 5
        points['z'] = 6
        fields = [NS(name=n, offset=o, datatype=7, count=1)
                  for n,o in [('z',0),('x',4),('y',8)]]
        msg = NS(height=2, width=2, point_step=16, row_step=40,
                 is_bigendian=True, fields=fields, data=data,
                 header=NS(stamp=NS(sec=17,nanosec=123)))
        stamp, xyz = p.pointcloud_xyz(msg, 2, 2)
        self.assertEqual(stamp, 17000000123)
        np.testing.assert_array_equal(xyz, [[1,5,6],[2,5,6],[3,5,6],[4,5,6]])
        with self.assertRaises(ValueError):
            p.pointcloud_xyz(msg, 1, 4)
        msg.row_step = 8
        with self.assertRaises(ValueError):
            p.pointcloud_xyz(msg, 2, 2)

class GroundTruthTests(unittest.TestCase):
    def test_normalizes_source_rotation_before_sensor_frame_transform(self):
        import tempfile
        from pathlib import Path
        from evalio.types import SE3, SO3, Stamp, Trajectory
        source = Trajectory(stamps=[Stamp.from_nsec(123)],
                            poses=[SE3(SO3(qx=0,qy=0,qz=1,qw=1),np.zeros(3))])
        dataset = NS(ground_truth_raw=lambda:source,
                     imu_T_gt=SE3.identity,
                     imu_T_lidar=lambda:SE3(SO3.identity(),np.array([1.,0,0])))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gt.tum'
            meta=p.export_ground_truth(dataset,path)
            values=np.loadtxt(path)
            np.testing.assert_allclose(values[1:4],[0,1,0],atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(values[4:]),1)
            self.assertIn('quaternion_normalization',meta)


if __name__ == '__main__':
    unittest.main()
