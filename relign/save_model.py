import os
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)


class SaveCallback(BaseCallback):
    """
    A callabck to save a model.

    Args:
        model_save_path (str): saving path for the model
        threshold (float| Threshold): Threshold for path name
    """

    parent: EvalCallback

    def __init__(self, model_save_path, threshold):
        super().__init__(verbose=0)

        self.model_save_path = model_save_path
        self.threshold = threshold
        os.makedirs(self.model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.parent.model.save(
            os.path.join(self.model_save_path, f'best_model_{self.threshold:.2}'.replace('0.', ''))
        )
        return True
