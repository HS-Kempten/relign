import unittest
import numpy as np
from numpy.random import default_rng
from relign.network import CustomCNN
from relign.generator import (
    GaussianIntensityEnv,
    LensEnv,
    GymnasiumEnv,
    make_stacked_vec_env,
)
from relign.helpers import rmse
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO


def get_random_action(seed, n_actions):
    rng = default_rng(seed)
    return rng.uniform(low=-0.5, high=0.5, size=n_actions)


class TestGenerator(unittest.TestCase):
    def test_optimum_5d_actions(self):
        n_actions = 5
        env = GaussianIntensityEnv(n_actions=n_actions)
        env.reset()
        r_optimal = env.get_optimal_move()
        img = env.step(r_optimal)["img"]

        dist_to_gt = (img - env._gt).sum()

        self.assertEqual(dist_to_gt, 0)

    def test_optimum_random_starting_location(self):
        n_actions = 2
        env = GaussianIntensityEnv(n_actions=n_actions)
        env.reset()
        env.step(get_random_action(42, n_actions))
        r_optimal = env.get_optimal_move()
        img = env.step(r_optimal)["img"]
        dist_to_gt = (img - env._gt).sum()

        self.assertEqual(dist_to_gt, 0)

    def test_optimum_2d_action(self):
        n_actions = 2
        env = GaussianIntensityEnv(n_actions=n_actions)
        env.reset()
        r_optimal = env.get_optimal_move()
        img = env.step(r_optimal)["img"]
        dist_to_gt = (img - env._gt).sum()

        self.assertEqual(dist_to_gt, 0)

    def test_move_with_two_actions(self):
        env = GaussianIntensityEnv(n_actions=2)
        env.reset()
        img = env.step(np.array([0.1, 0.2]))["img"]
        self.assertTupleEqual(img.shape, (env.WIDTH, env.HEIGHT, 1))

    def test_compute_move_to_position(self):
        env = GaussianIntensityEnv(n_actions=2)
        env.reset()

        # Compute an action that places close to boundary
        a = env._compute_move_to_position(np.array([0.9, 0.9]))
        env.step(a)
        np.testing.assert_array_almost_equal(env.r, np.array([0.9, 0.9]))

    def test_hard_clipping_two_actions(self):
        env = GaussianIntensityEnv(n_actions=2, clipping='hard')
        env.reset()
        a = env._compute_move_to_position(env.r + np.array([0, 10]))
        r_old = env.r.copy()
        env.step(a)

        # Second dimension is exceeded and should be clipped
        np.testing.assert_array_almost_equal(
            env.r,
            np.array([r_old[0], 1]),
        )

    def test_hard_clipping_multiple_actions_positive_direction(self):
        env = GaussianIntensityEnv(n_actions=5, clipping='hard')
        env.reset()
        a = env._compute_move_to_position(env.r + np.ones(5))
        env.step(a)

        self.assertTrue(np.all(env.r <= 1))
        self.assertTrue(np.all(env.r >= 0))

        # One positive boundary should be touched
        self.assertTrue(np.any(env.r == 1))

    def test_clipping_multiple_actions_negative_direction(self):
        env = GaussianIntensityEnv(n_actions=5, clipping='hard')
        env.reset()
        a = env._compute_move_to_position(env.r - np.ones(5))
        env.step(a)

        self.assertTrue(np.all(env.r <= 1))
        self.assertTrue(np.all(env.r >= 0))

        # One zero boundary should be touched
        self.assertTrue(np.any(env.r == 0))

    def test_correct_clipping(self):

        for clipping in ['soft', 'hard']:
            env = GaussianIntensityEnv(n_actions=5, clipping=clipping)
            env.reset()
            # Move to center
            a = env._compute_move_to_position(0.5 * np.ones(5))
            env.step(a)

            # Move along unit ray to outreach
            a = env._compute_move_to_position(5 * np.ones(1))
            env.step(a)

            # Position should be exactly the corner
            np.testing.assert_array_almost_equal(env.r, np.ones(5), decimal=3)


class TestLensEnv(unittest.TestCase):
    def test_movement_all_possible_n_actions(self):
        env = LensEnv()
        for n_actions in range(env.N_ACTIONS_MAX):
            env = LensEnv(n_actions=n_actions)
            env.reset()
            img = env.step(get_random_action(42, n_actions))["img"]
            self.assertTupleEqual(img.shape, (env.WIDTH, env.HEIGHT, 1))
            self.assertFalse(np.array_equal(img, np.zeros((env.WIDTH, env.HEIGHT, 1))))

    def test_init_image_not_ground_gruth(self):
        env = LensEnv(n_actions=4, seed=10)
        img_init = env.reset()
        dist_to_gt = ((env._gt - img_init) ** 2).mean()
        self.assertGreater(dist_to_gt, 0.03)

    def test_vertex_positioning(self):
        n_actions = 5
        env = LensEnv(n_actions=n_actions, seed=42, noise_angle=0, noise_translation=0)
        img_start = env.reset()

        start_vertices_start = env.lens_vertex_coords_start
        vp_start = np.array(env.params["lens_0.vertex_positions"])

        img_new = env.step(get_random_action(12, n_actions))["img"]
        img_new = env.step(get_random_action(12, n_actions))["img"]

        start_vertices_after_move = env.lens_vertex_coords_start
        vp_after_move = np.array(env.params["lens_0.vertex_positions"])

        np.testing.assert_array_equal(start_vertices_start, start_vertices_after_move)
        self.assertFalse(np.array_equal(vp_start, vp_after_move))
        self.assertFalse(np.array_equal(img_start, img_new))

    def test_reset_with_noise(self):
        env = LensEnv(n_actions=5, seed=1337, noise_angle=0.25, noise_translation=0.25)

        img_start_0 = env.reset()
        start_vertices_0 = env.lens_vertex_coords_start
        w_0 = env.W
        vp_0 = env.params["lens_0.vertex_positions"]

        img_start_1 = env.reset()
        start_vertices_1 = env.lens_vertex_coords_start
        w_1 = env.W
        vp_1 = env.params["lens_0.vertex_positions"]

        self.assertFalse(np.array_equal(img_start_0, img_start_1))
        self.assertFalse(np.array_equal(start_vertices_0, start_vertices_1))
        self.assertFalse(np.array_equal(w_0, w_1))
        self.assertFalse(np.array_equal(vp_0, vp_1))

        env = LensEnv(n_actions=5, seed=42, noise_angle=0.25, noise_translation=0.25)
        img_start_2 = env.reset()
        self.assertFalse(np.array_equal(img_start_0, img_start_2))

    def test_reset_without_noise(self):
        env = LensEnv(n_actions=5, seed=10, noise_angle=0, noise_translation=0)

        img_start_0 = env.reset()
        start_vertices_0 = env.lens_vertex_coords_start
        w_0 = env.W
        vp_0 = env.params["lens_0.vertex_positions"]

        img_start_1 = env.reset()
        start_vertices_1 = env.lens_vertex_coords_start
        w_1 = env.W
        vp_1 = env.params["lens_0.vertex_positions"]

        self.assertFalse(np.array_equal(img_start_0, img_start_1))
        np.testing.assert_array_equal(start_vertices_0, start_vertices_1)
        self.assertFalse(np.array_equal(w_0, w_1))
        self.assertFalse(np.array_equal(vp_0, vp_1))

        env = LensEnv(n_actions=5, seed=10, noise_angle=0, noise_translation=0)
        img_start_2 = env.reset()
        np.testing.assert_array_equal(img_start_0, img_start_2)


class TestGymEnv(unittest.TestCase):
    def test_env(self):
        check_env(GymnasiumEnv(), warn=True, skip_render_check=True)

    def test_optimum_5d_actions(self):
        n_actions = 5
        env = GymnasiumEnv(n_actions=n_actions)
        env.reset()
        r_optimal = env.get_optimal_move()
        img, _, _, _, _ = env.step(r_optimal)
        dist_to_gt = (img.ravel() - env.gt.ravel()).sum()
        self.assertEqual(dist_to_gt, 0)

    def test_move_with_two_actions(self):
        env = GymnasiumEnv(n_actions=2)
        env.reset()
        img, _, _, _, _ = env.step(np.array([0.1, 0.2]))
        self.assertTupleEqual(img.shape, (env.obs_shape))

    def test_ppo_training_with_mlp_policy(self):
        env = make_vec_env(
            env_id=GymnasiumEnv,
            n_envs=2,
        )

        model = PPO(
            policy='MlpPolicy',
            env=env,
            n_steps=4,
        )

        model.learn(total_timesteps=2 * 2 * 4)  # two times backprob

    def test_ppo_training_with_cnn_policy(self):
        env = make_stacked_vec_env(env_name='gi')
        model = PPO(
            policy='CnnPolicy',
            env=env,
            n_steps=4,
            policy_kwargs=dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=256),
            ),
        )

        model.learn(total_timesteps=2 * 2 * 4)  # two times backprob

    def test_score_threshold(self):
        env = GymnasiumEnv(score_goal_threshold=0.1)
        env.reset()
        action = env.get_optimal_move()
        img, reward, terminated, _, _ = env.step(action)
        self.assertTrue(terminated)

    def test_potential_reward(self):
        env = GymnasiumEnv(score_goal_threshold=0.1, reward='potential')
        img_start, _ = env.reset()
        action = env.get_optimal_move()
        img_final, reward, _, _, _ = env.step(action)
        print(env.env._scores)
        print(env.gt.shape)

        score_start = rmse(img_start[0, :, :], env.gt[:, :, 0])
        score_final = rmse(img_final[0, :, :], env.gt[:, :, 0])
        self.assertAlmostEqual(reward, score_start - score_final, places=4)
        self.assertGreater(reward, 0)

        _, reward, _, _, _ = env.step(-action)
        self.assertAlmostEqual(reward, score_final - score_start, places=4)
        self.assertLess(reward, 0)

    def test_stacking_and_vectorizing(self):

        n_envs = 10
        n_stack = 5

        # Vectorize environment for parallel trainings
        vec_env = make_vec_env(
            env_id=GymnasiumEnv,
            n_envs=n_envs,
        )

        # Stack latest observations into one
        env = VecFrameStack(vec_env, n_stack=n_stack, channels_order='first')
        obs = env.reset()

        self.assertTupleEqual(
            obs.shape, (n_envs, n_stack, env.envs[0].env.env.WIDTH, env.envs[0].env.env.WIDTH)
        )
