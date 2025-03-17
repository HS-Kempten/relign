import numpy as np
from scipy.optimize import minimize
from .baseline_algorithms import AlignmentAlgorithm, Stop

class TrustNCG(AlignmentAlgorithm):
    """
    Optimization using scipy's minimize with the 'trust-ncg' method.
    """
    def align(self, env):
        def func_wrapper(x):
            return env(x)

        x0 = np.full(env.n_actions, self.v_min)
        #bounds = [(self.v_min, self.v_max)] * env.n_actions

        res = minimize(
            fun=func_wrapper,
            x0=x0,
            method='trust-ncg',
            jac='2-point',
            hess='2-point',
            options={'maxiter': self.n_steps},
            callback=lambda xk: Stop(self.stopping_threshold)
        )

        x_opt = res.x
        func_vals = np.array([res.fun])
        x_iters = np.array([x_opt])

        return x_opt, func_vals, x_iters
