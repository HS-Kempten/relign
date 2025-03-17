from skopt.callbacks import EarlyStopper
from abc import (
    ABC,
    abstractmethod,
)
class Stop(EarlyStopper):
    def __new__(cls, threshold):
        if threshold is None:
            return None
        return super().__new__(cls)

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def _criterion(self, result):
        return result.func_vals[-1] < self.threshold

class AlignmentAlgorithm(ABC):

    def __init__(self, n_steps=100, stopping_threshold=None, v_min=0., v_max=1.):
        self.n_steps = n_steps
        self.stopping_threshold = stopping_threshold
        self.v_min = v_min
        self.v_max = v_max
        self.res = None

    @abstractmethod
    def align(self, env, x0=None):
        """
        Optimize the given environment's parameters.

        Args:
            env: An object representing the environment, expected to have attributes:
                - n_actions (int): Number of actions/dimensions to optimize.
                - __call__ : evaluation funtions get a numpy.ndarray with
                                shape=env.n_actions and return a float
        Returns:
            tuple:
                - x_opt (numpy.ndarray, shape=( env.n_actions,)):
                        Optimal input parameters found during optimization.
                - func_vals (numpy.ndarray, shape=(n_steps,)):
                        Function values at all evaluated points.
                - x_iters (numpy.ndarray shape=(n_steps, env.actions)):
                        Input parameter combinations evaluated.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
