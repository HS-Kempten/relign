import unittest
import numpy as np
from relign import helpers


class TestHelpers(unittest.TestCase):
    def test_compute_z_positions_scene_objects(self):
        n_lenses = np.arange(2, 5)
        dist_lenses = 2
        dist_sensor = 5
        results_expected = {
            2: {"sensor": 6, "lenses": [-1, 1]},
            3: {"sensor": 7, "lenses": [-2, 0, 2]},
            4: {"sensor": 8, "lenses": [-3, -1, 1, 3]},
        }
        for i in n_lenses:
            z_positions = helpers.compute_z_positions_scene_objects(
                dist_sensor=dist_sensor,
                dist_lenses=dist_lenses,
                n_lenses=i,
            )

            np.testing.assert_equal(
                results_expected[i], z_positions, err_msg=f"computation fails for n_lenses=={i}"
            )

        # additional test for floats
        np.testing.assert_equal(
            helpers.compute_z_positions_scene_objects(
                dist_sensor=0.5,
                dist_lenses=0.4,
                n_lenses=3,
            ),
            {"sensor": 0.9, "lenses": [-0.4, 0, 0.4]},
            err_msg="computation fails for floats",
        )
