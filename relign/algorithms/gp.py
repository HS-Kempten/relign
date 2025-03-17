import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_objective
from .baseline_algorithms import (
    AlignmentAlgorithm,
    Stop,
    )
class GP(AlignmentAlgorithm):
    """
    Gaussian Process Optimization.
    """
    def align(self, env, x0=None):
        if x0 is None:
            x0 = np.random.uniform(self.v_min, self.v_max, env.n_actions).tolist()
        res = gp_minimize(
            env,
            dimensions=np.array(env.n_actions * [(self.v_min, self.v_max)]),
            x0=x0,
            n_calls=self.n_steps,
            verbose=False,
            callback=Stop(self.stopping_threshold),
        )
        x_opt = np.array(res.x_iters[res.func_vals.argmin()])
        return x_opt, res.func_vals, res.x_iters

    def plot(self):
        if self.res is not None:
            _ = plot_objective(self.res, n_samples=40)
