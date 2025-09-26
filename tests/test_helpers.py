import unittest
import numpy as np
from relign.helpers import sensor_size_min, scene_objects


class TestHelpers(unittest.TestCase):
    def test_sensor_size_min(self):
        sensor_size = sensor_size_min(fov=90, focal_length=3, sensor_resolution=[200, 200])
        np.testing.assert_array_almost_equal(sensor_size, np.array([6.0, 6.0]))

        sensor_size = sensor_size_min(fov=90, focal_length=3, sensor_resolution=[100, 200])
        np.testing.assert_array_almost_equal(sensor_size, np.array([6.0, 12]))

        sensor_size = sensor_size_min(fov=90, focal_length=3, sensor_resolution=[200, 100])
        np.testing.assert_array_almost_equal(sensor_size, np.array([12, 6.0]))

    def test_scene_objects(self):
        dct_a = scene_objects(
            fov=84.1,
            focal_length=2.895,
            sensor_resolution=[865, 896],
            exit_pupil_angle=13.45,
            ref_pattern_res=[865, 896],
        )

        dct_b = {
            'sensor_scale': np.array([5.22249573, 5.40966031]),
            'emitter_distance': np.float64(-2000.58559728545),
            'emitter_scale': np.array([2506.63518571, 2596.46835421]),
        }

        for k, v in dct_a.items():
            np.testing.assert_array_almost_equal(v, dct_b[k])
