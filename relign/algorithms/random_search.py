import numpy as np
from skopt import dummy_minimize
from skopt.plots import plot_objective
from .baseline_algorithms import (
    AlignmentAlgorithm,
    Stop,
    )
class RandomSearch(AlignmentAlgorithm):
    """
    Random Search Optimization.
    """
    def align(self, env):
        self.res = dummy_minimize(
            env,
            dimensions=env.n_actions * [(self.v_min, self.v_max)],
            n_calls=self.n_steps,
            verbose=False,
            callback=Stop(self.stopping_threshold),
        )
        x_opt = np.array(self.res.x_iters[self.res.func_vals.argmin()])
        return x_opt, self.res.func_vals, self.res.x_iters

    def plot(self):
        if self.res is not None:
            _ = plot_objective(self.res, n_samples=40)
