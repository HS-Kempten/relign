import unittest

from relign.generator import (
    GaussianIntensityEnv,
    LensEnv,
    EnvWrapper,
)
from relign.algorithms.scan import Scan
from relign.algorithms.gp import GP
from relign.algorithms.random_search import RandomSearch
from relign.algorithms.forest import ForestOptimization
from relign.algorithms.coordinate_walk import CoordinateWalk


class TestScan(unittest.TestCase):

    def test_scan_2d_action(self):
        env = GaussianIntensityEnv(n_actions=2, seed=42)
        env_wrapper = EnvWrapper(env)
        env_wrapper.reset()
        algo = Scan(n_coordinate_steps=40, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(env_wrapper)

        self.assertTupleEqual(x_opt.shape, (2,))
        env_wrapper(x_opt)
        self.assertLess((env.get_optimal_move() ** 2).sum(), 1.0e-6)

    def test_scan_3d_actions(self):
        env = GaussianIntensityEnv(n_actions=3)
        env_wrapper = EnvWrapper(env)
        env_wrapper.reset()
        algo = Scan(n_coordinate_steps=30, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(env_wrapper)

        self.assertTupleEqual(x_opt.shape, (3,))
        env_wrapper(x_opt)
        self.assertLess((env.get_optimal_move() ** 2).sum(), 1e-4)

    def test_scan_2d_action_la(self):
        env = LensEnv(n_actions=2)
        env_wrapper = EnvWrapper(env)
        env_wrapper.reset()
        algo = Scan(n_coordinate_steps=10, v_min=0.45, v_max=0.55)
        x_opt, _, _ = algo.align(env_wrapper)

        self.assertTupleEqual(x_opt.shape, (2,))
        self.assertLess(env_wrapper(x_opt), 0.15)


class TestGP(unittest.TestCase):

    def test_align(self):
        self.env = GaussianIntensityEnv(n_actions=2)
        self.env_wrapper = EnvWrapper(self.env)
        self.env_wrapper.reset()
        algo = GP(n_steps=20, v_min=0.1, v_max=0.9)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.05)

    def test_gp_2d_action_la(self):
        self.env = LensEnv(n_actions=2)
        self.env_wrapper = EnvWrapper(self.env)
        self.env_wrapper.reset()
        algo = GP(n_steps=20, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.15)


class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.env = GaussianIntensityEnv(n_actions=2)
        self.env_wrapper = EnvWrapper(self.env)

    def test_gp(self):
        self.env_wrapper.reset()
        algo = GP(n_steps=20, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.1)

    def test_scan(self):
        self.env_wrapper.reset()
        algo = Scan(n_steps=200, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.1)

    def test_random(self):
        self.env_wrapper.reset()
        algo = RandomSearch(n_steps=200, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.1)

    def test_forest(self):
        self.env_wrapper.reset()
        algo = ForestOptimization(n_steps=50, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.1)

    def test_coordinate_walk(self):
        self.env_wrapper.reset()
        algo = CoordinateWalk(n_steps=50, v_min=0.4, v_max=0.6)
        x_opt, func_vals, x_iters = algo.align(self.env_wrapper)
        self.assertLess(self.env_wrapper(x_opt), 0.1)
