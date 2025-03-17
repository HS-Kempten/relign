import numpy as np
from skopt import forest_minimize
from skopt.plots import plot_objective
from .baseline_algorithms import (
    AlignmentAlgorithm,
    Stop,
)


class ForestOptimization(AlignmentAlgorithm):
    """
    Optimization using Random Forest.
    """

    def align(self, env, x0=None, y0=None):
        self.res = forest_minimize(
            env,
            dimensions=env.n_actions * [(self.v_min, self.v_max)],
            n_calls=self.n_steps,
            verbose=False,
            callback=Stop(self.stopping_threshold),
            x0=x0,
            y0=y0,
        )
        x_opt = np.array(self.res.x_iters[self.res.func_vals.argmin()])
        return x_opt, self.res.func_vals, self.res.x_iters

    def plot(self):
        if self.res is not None:
            _ = plot_objective(self.res, n_samples=40)
