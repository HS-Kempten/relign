import numpy as np
from .baseline_algorithms import (
    AlignmentAlgorithm,
    )
class CoordinateWalk (AlignmentAlgorithm):
    def __init__(self, n_coordinate_steps=5, n_steps=-1, v_min=-0.5, v_max=0.5,
                 rounds=1, start=None):
        super().__init__(n_steps=n_steps, v_min=v_min, v_max=v_max)
        self.n_coordinate_steps=n_coordinate_steps
        self.rounds = rounds
        self.start = start

    def align(self, env):
        if self.n_steps >0:
            self.n_coordinate_steps = int(self.n_steps / (env.n_actions*self.rounds))
        self.coordinate_range = np.linspace(self.v_min, self.v_max, self.n_coordinate_steps)
        r = np.zeros(env.n_actions)
        if self.start is None:
            indices = np.random.randint(0, self.n_coordinate_steps, size=env.n_actions)
        else:
            indices = self.start
        for dim, i in enumerate(indices):
                r[dim] = self.coordinate_range[i]
        func_vals = []
        x_iters = []
        for i in range(self.rounds):
            for dim in range(env.n_actions):
                for idx in range(self.n_coordinate_steps):
                    r[dim] = self.coordinate_range[idx]
                    dist = env(r)
                    func_vals.append(dist)
                    x_iters.append(r.copy())

                argmin = np.argmin(func_vals[-self.n_coordinate_steps:])
                r[dim] = self.coordinate_range[argmin]
        return r, func_vals, x_iters
