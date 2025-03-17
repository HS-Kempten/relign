import numpy as np
from .baseline_algorithms import (
    AlignmentAlgorithm,
    )
from itertools import product

class Scan(AlignmentAlgorithm):
    """
    Scans coordinate by coordinate
    """
    def __init__(self, n_coordinate_steps=5, n_steps=-1, v_min=-0.5, v_max=0.5):
        super().__init__(n_steps=n_steps, v_min=v_min, v_max=v_max)
        self.n_coordinate_steps=n_coordinate_steps

    def align(self, env):
        """
        Needs `EnvWrapper`
        """
        if self.n_steps >0:
            self.n_coordinate_steps = int(self.n_steps ** (1 / env.n_actions))

        self.coordinate_range = np.linspace(self.v_min, self.v_max, self.n_coordinate_steps)
        r = np.zeros(env.n_actions)

        func_vals = []
        x_iters = []
        for indices in product(range(self.n_coordinate_steps), repeat=env.n_actions):
            for dim, i in enumerate(indices):
                r[dim] = self.coordinate_range[i]
            dist = env(r)
            func_vals.append(dist.copy())
            x_iters.append(r.copy())
        x_iters=np.stack(x_iters)
        func_vals=np.stack(func_vals)
        x_opt = x_iters[func_vals.argmin()]
        return x_opt, func_vals, x_iters
