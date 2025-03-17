import unittest
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO  # noqa

from relign.generator import (  # noqa
    GymnasiumEnv,  # noqa
    GaussianIntensityEnv,  # noqa
)

from relign.curriculum import (
    CLCallbackSteps,
    Threshold,
    EvaluationCallback,
)


class TestThreshold(unittest.TestCase):

    def test_comparisons(self):

        threshold = Threshold(1)

        self.assertLess(threshold, 2)
        self.assertGreater(threshold, 0)
        self.assertEqual(threshold, 1)
        self.assertNotEqual(threshold, 0)


class TestCL(unittest.TestCase):

    def test_curriculum_learning_with_gaussian_intensity(self):

        threshold = Threshold(2)

        env_train = make_vec_env(
            env_id=GymnasiumEnv,
            n_envs=5,
            env_kwargs={
                'env_cls': GaussianIntensityEnv,
                'score_goal_threshold': threshold,
                'max_episode_steps': 10,
            },
        )

        env_eval = make_vec_env(
            env_id=GymnasiumEnv,
            n_envs=5,
            env_kwargs={
                'env_cls': GaussianIntensityEnv,
                'score_goal_threshold': threshold,
                'max_episode_steps': 50,
            },
        )

        curriculum_callback = CLCallbackSteps(threshold=threshold, factor=0.5)

        model = PPO("MlpPolicy", env_train, verbose=1)
        model.learn(
            total_timesteps=20,
            callback=EvaluationCallback(
                eval_env=env_eval,
                eval_freq=200,  # eval after 4 roleouts
                n_eval_episodes=2,
                callback_after_eval=curriculum_callback,
            ),
        )

        self.assertEqual(curriculum_callback.last_adjustment, 10)
        self.assertEqual(curriculum_callback.threshold.threshold, 0.5)
