import os
import unittest
from stable_baselines3 import PPO  # noqa
from relign.generator import (  # noqa
    GymnasiumEnv,  # noqa
    LensEnv,  # noqa
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # noqa


class TestRL(unittest.TestCase):
    def test_train_with_stable_baselines_gaussian_intensity(self):
        env = GymnasiumEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10)

    def test_train_with_stable_baselines_lens_alignment(self):
        env = GymnasiumEnv(
            env_cls=LensEnv, n_actions=5, n_lenses=2, noise_angle=0, noise_translation=0
        )
        model = PPO("MlpPolicy", env, n_steps=2, n_epochs=4, batch_size=2, verbose=1)
        model.learn(total_timesteps=5)

    def test_with_potential_reward(self):
        env = GymnasiumEnv(reward='potential')
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10)
