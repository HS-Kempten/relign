"""
Core module holding the implementation of the Lens environment.
"""

import os
import logging
import numpy as np
from scipy.stats import multivariate_normal
import mitsuba as mi
from abc import (
    ABC,
    abstractmethod,
)
from importlib.resources import files
from relign.helpers import (
    get_device,
    compute_z_positions_scene_objects,
    compute_focal_length,
    get_gt_filename_from_params,
    rmse,
)
import gymnasium as gym

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    N_ACTIONS_MAX: int
    WIDTH: int
    HEIGHT: int

    def __init__(
        self,
        seed=None,
        n_actions=5,
        reward="minus_one",
        max_episode_steps=200,
        score_goal_threshold=0.05,
        noise_movement=0.02,
        clipping="soft",
    ):
        if n_actions > self.N_ACTIONS_MAX:
            raise ValueError(f"n_actions must be <= {self.N_ACTIONS_MAX}")

        if clipping not in ['soft', 'hard']:
            raise ValueError('Invalid clipping')

        self.max_episode_steps = max_episode_steps
        self.clipping = clipping
        self.score_goal_threshold = score_goal_threshold
        self.random = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.R = 0.5 * np.ones(self.N_ACTIONS_MAX)
        self._gt = self._load_gt()
        self.r = np.zeros(n_actions)
        self.noise_movement = noise_movement
        # This can be considered as a view on self.R as the full R is needed when generating an
        # image
        self.r_optimal = self.R[0 : self.n_actions]
        self._setup_reward(reward)

    def reset(self):
        """
        Resets the environment for a new episode.
        Should be called after every episode to reset the internal state of the environment.
        - Generates a new dependence matrix `W` with added noise.
        - Initializes a random starting position offset.

        Returns:
            img: The image representation of the current state.

        Raises:
            ValueError: If the generated matrix `W` is singular.
        """
        self._elapsed_steps = 0
        self._scores = []

        # Generate dependence matrix
        self.W = np.eye(self.n_actions) + np.abs(
            self.random.normal(size=(self.n_actions, self.n_actions), scale=self.noise_movement)
        )
        self.W = self.W / self.W.sum(axis=1)[:, np.newaxis]

        if np.linalg.det(self.W) == 0:
            raise ValueError("Matrix not singular")

        self.r_offset = self._init_random_starting_position()

        self.r = np.clip(
            self.r_optimal + self.r_offset,
            np.zeros(self.n_actions),
            np.ones(self.n_actions),
        )
        R = self.fill_action_with_optimal_values()
        img = self._make_image(R)
        self._scores.append(self.compute_distance_to_gt(img))
        return img

    def _setup_reward(self, reward):
        self.reward_mappings = {
            "minus_one": self.compute_minus_one_reward,
            "potential": self.compute_potential_reward,
            "minus_one_edge": self.compute_minus_one_edge_reward,
            "minus_one_plus": self.compute_minus_one_plus,
        }
        if reward not in self.reward_mappings.keys():
            raise ValueError(f"given 'reward style' must be in {list(self.reward_mappings.keys())}")
        else:
            self.comp_reward = self.reward_mappings[reward]

    def _init_random_starting_position(self):
        return self.random.uniform(-0.5, 0.5, size=self.n_actions)

    def compute_minus_one_reward(self, *_):
        return -1

    def compute_potential_reward(self, *_):
        return self._scores[-2] - self._scores[-1]

    def compute_minus_one_plus(self, *_):
        if self._scores[-1] < self.score_goal_threshold * 2:
            return -self._scores[-1]
        else:
            return -1

    def compute_minus_one_edge_reward(self, *_):
        if self.r.any() == 1 or self.r.any() == 0:
            return -2
        else:
            return -1

    def compute_distance_to_optimum(self):
        return rmse(self.r_optimal, self.r)

    def compute_distance_to_gt(self, img):
        if self._gt is None:
            raise ValueError(
                "No precomputed ground truth image was found for the current setup. Please run "
                "`BaseEnv.create_gt()` to generate it. "
                "Note that this process may take a few minutes."
            )
        return rmse(img[:, :, 0], self._gt[:, :, 0])

    def _compute_move_to_position(self, r):
        return np.linalg.inv(self.W).dot((r - self.r).T)

    def _update(self, a):

        if not a.shape[0] == self.n_actions:
            raise ValueError("moving vector must have same dimension as `Environment.n_actions`")

        r_proposed = self.W.dot(a) + self.r

        if np.all((r_proposed >= 0) & (r_proposed <= 1)):
            self.r = r_proposed
        else:
            if self.clipping == 'hard':
                self.r = self._clip_hard(r_proposed)
            else:
                self.r = self._clip_soft(r_proposed)

    def _clip_hard(self, r):
        """
        Clips the proposed direction such that position lies within unit box
        """

        a = r - self.r

        intersections = []

        normals = np.concatenate([np.eye(self.n_actions), -np.eye(self.n_actions)])
        intercepts = np.concatenate([np.ones(self.n_actions), np.zeros(self.n_actions)])

        for n, b in zip(normals, intercepts):
            t = (b - n.dot(self.r)) / (n.dot(a))
            intersections.append(t)

        intersections = np.array(intersections)

        # Get smallest non-negative element in intersections
        tmin = intersections[intersections >= 0].min()
        return self.r + tmin * a

    def _clip_soft(self, r):
        return np.clip(
            r,
            np.zeros(self.n_actions),
            np.ones(self.n_actions),
        )

    def get_optimal_move(self):
        return np.linalg.inv(self.W).dot(self.r_optimal - self.r)

    def fill_action_with_optimal_values(self):
        return np.concatenate([self.r, self.R[self.n_actions :]])

    def step(self, action):
        """Executes a single step in the environment using the given action.

        Args:
            action (list or np.array): The action to be taken in the environment.

        Returns:
            dict: A dictionary containing the following keys:
                - `img` (np.ndarray): The generated image after taking the action.
                - `reward` (float): The computed reward based on the current state
                - `truncated` (bool): Whether the maximum number of steps has been reached.
                - `terminated` (bool): Whether the goal threshold was reached.

        """
        if isinstance(action, list):
            action = np.array(action)

        self._elapsed_steps += 1
        self._update(action)

        R = self.fill_action_with_optimal_values()

        img = self._make_image(R)
        score = self.compute_distance_to_gt(img)
        self._scores.append(score)

        truncated = self._elapsed_steps >= self.max_episode_steps
        terminated = score < self.score_goal_threshold

        return {
            "img": img,
            "reward": self.comp_reward(truncated, terminated),
            "truncated": truncated,
            "terminated": terminated,
        }

    def get_steps(self):
        return self._elapsed_steps

    def _load_gt_from_filename(self, filename):
        path = os.path.join(
            str(files("relign")),
            "data",
            f"{self.__class__.__name__}",
            filename,
        )

        try:
            return np.load(path)
        except FileNotFoundError:
            return None

    @abstractmethod
    def _load_gt(self):
        """defines filename and loads ground truth image."""
        raise NotImplementedError

    @abstractmethod
    def _make_image(self, r):
        """takes clipped r and creates image."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_gt():
        """computes ground truth image."""
        raise NotImplementedError


class GaussianIntensityEnv(BaseEnvironment):
    N_ACTIONS_MAX = 5
    WIDTH = 50
    HEIGHT = 50

    def _make_image(self, r):
        center_x = r[0]
        center_y = r[1]
        cov_x = r[2]
        cov_y = r[3]
        angle = r[4]

        img = np.zeros(shape=(self.WIDTH, self.HEIGHT, 1), dtype=np.float32)

        cov_x = 0.05 + (cov_x) ** 2 / 5
        cov_y = 0.05 + (cov_y) ** 2 / 5

        center_x = center_x * self.WIDTH
        center_y = center_y * self.HEIGHT

        angle = np.pi * angle / 2

        X, Y = np.meshgrid(
            np.linspace(-1, 1, self.WIDTH, dtype=np.float32),
            np.linspace(-1, 1, self.HEIGHT, dtype=np.float32),
        )
        xx = np.dstack((X, Y))

        # Must be positive seminite
        cov = np.array([[cov_x, 0], [0, cov_y]])

        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        # Rotate the covariance matrix and make it positive seminate again
        cov_rotated = np.matmul(np.matmul(rotation, cov), rotation.T)

        rv = multivariate_normal(
            mean=[-1 + 2 * center_x / self.WIDTH, -1 + 2 * center_y / self.HEIGHT],
            cov=cov_rotated,
        )
        img[:, :, 0] = img[:, :, 0] + rv.pdf(xx)
        return img

    def _load_gt(self):
        filename = f"gt__{self.WIDTH}x{self.HEIGHT}.npy"
        return super()._load_gt_from_filename(filename)

    @staticmethod
    def create_gt():
        env = GaussianIntensityEnv(seed=0, n_actions=5)
        env.reset()

        filename = f'gt__{env.WIDTH}x{env.HEIGHT}.npy'
        path = os.path.join(
            str(files("relign")),
            "data",
            env.__class__.__name__,
            filename,
        )

        np.save(path, env._make_image(env.R))

    @staticmethod
    def starting_goal_threshold():
        env = GaussianIntensityEnv()
        img_gt = env._load_gt()
        return rmse(np.zeros((env.WIDTH, env.HEIGHT, 1)), img_gt)


class LensEnv(BaseEnvironment):
    """
    Environment for mitsuba3.
    """

    N_LENSES_MAX = 3
    N_ACTIONS_MAX = 5
    WIDTH = 50
    HEIGHT = 50

    def __init__(
        self,
        n_lenses=2,
        score_goal_threshold=None,
        n_actions=5,
        noise_angle=0.0,
        noise_translation=0.0,
        sample_count=512,
        *args,
        **kwargs,
    ):
        from relign.config import Config  # to avoid circular imports

        if score_goal_threshold is None:
            score_goal_threshold = Config.targets_optimum["la"]({"n_lenses": n_lenses})

        if get_device() == "cpu" or get_device() == "mps":
            mi.set_variant("scalar_rgb")
            # mi.set_variant('llvm_ad_rgb') is bugged (memory leak)
        else:
            mi.set_variant("cuda_ad_rgb")

        if not ((0 <= noise_angle <= 1) and (0 <= noise_translation <= 1)):
            raise ValueError(
                "`noise_angle` and `noise_translation` must be between 0 (no noise at all)"
                f" and 1 (maximum noise). ({noise_angle} and {noise_translation} given)"
            )
        if 0 < n_lenses <= self.N_LENSES_MAX:
            self.n_lenses = n_lenses
        else:
            raise ValueError(
                f"can only create between 0 and (including) {self.N_LENSES_MAX} lenses"
            )

        z_positions = compute_z_positions_scene_objects(
            n_lenses=self.n_lenses,
            dist_lenses=0.35,
            dist_sensor=1 / self.n_lenses,
        )

        if n_lenses == 1:
            self.z_sensor = compute_focal_length() - 0.27 / 2
        elif n_lenses == 2:
            self.z_sensor = 0.8
        elif n_lenses == 3:
            self.z_sensor = 0.7
        else:
            self.z_sensor = z_positions["sensor"]

        self.noise_angle = noise_angle
        self.noise_translation = noise_translation
        self.z_lenses = z_positions["lenses"]
        self.sample_count = sample_count
        # loads ground truth image
        super().__init__(
            score_goal_threshold=score_goal_threshold, n_actions=n_actions, *args, **kwargs
        )

    def reset(self):
        """
        Resets the environment and cleans up resources.

        Returns:
            The result of the superclass's `reset` method.
        """

        from relign.helpers import clean_up

        logger.debug("resetting env and cleaning up dr.jit")
        clean_up()

        self.scene = self._create_optimal_scene()
        self.params = mi.traverse(self.scene)

        self._set_starting_lens_vertices()
        if self.noise_angle > 0 or self.noise_translation > 0:
            self._add_noise_to_lens_positions()
        return super().reset()

    def _set_starting_lens_vertices(self):
        self.lens_vertex_coords_start = np.array(
            [
                np.array([self.params[f"lens_{lens}.vertex_positions"]]).reshape(-1, 3)
                for lens in range(self.n_lenses)
            ]
        )

    def _create_optimal_scene(self):
        scene_dict = self._create_base_scene_dict()
        self._add_lenses_to_scene_dict(scene_dict)
        return mi.load_dict(scene_dict, parallel=False)

    def render_scene_overview(
        self, sensor_distance=None, axis=[1, 0, 0], sample_count=512, r=np.array([])
    ):
        if r.size == 0:
            r = self.r
        if isinstance(axis, list):
            axis = np.array(axis)
        elif not isinstance(axis, np.ndarray):
            raise ValueError("axis must be list-like object. E. g. [1, 0, 0]")

        if sensor_distance is None:
            sensor_distance = 4.0 * np.log(self.n_lenses)

        scene_dict = self._create_base_scene_dict()
        self._add_lenses_to_scene_dict(scene_dict)

        scene_dict["Light"] = {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 0.1,
            },
        }
        bitmap = mi.Bitmap(self._make_image(r))
        texture = {
            'type': 'bitmap',
            'bitmap': bitmap,
            'wrap_mode': 'clamp',
        }
        material = {
            'type': 'diffuse',
            'reflectance': texture,
        }

        scene_dict["sensor"]["bsdf"] = material

        if np.array_equal(axis, np.array([1, 0, 0])):
            for n in range(self.n_lenses):
                scene_dict[f"lens_{n}"]["bsdf"] = {
                    'type': 'roughdielectric',
                    'distribution': 'beckmann',
                    'alpha': 0.15,
                    'int_ior': 'bk7',
                    'ext_ior': 'air',
                }

        scene = mi.load_dict(scene_dict, parallel=False)
        params = mi.traverse(scene)

        for n in range(self.n_lenses):
            params[f"lens_{n}.vertex_positions"] = self.params[f"lens_{n}.vertex_positions"]

        new_sensor = mi.load_dict(
            {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=axis * (-1 * sensor_distance),
                    target=[0, 0, 0],
                    up=[0, 1, 0],  # positive y-axis is "up"
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": 200,
                    "height": 200,
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {
                    "type": "independent",
                    "sample_count": sample_count,
                },
            }
        )

        params.update()
        image = mi.render(scene, sensor=new_sensor)
        img = image.numpy()[..., 0].reshape(200, 200, 1)
        return img

    def _add_lenses_to_scene_dict(self, scene_dict):
        for i, z in zip(range(self.n_lenses), self.z_lenses):
            scene_dict[f"lens_{i}"] = {
                "type": "ply",
                "face_normals": True,
                "filename": os.path.join(
                    str(files("relign")),
                    "data",
                    f"{self.__class__.__name__}",
                    "lens.PLY",
                ),
                "to_world": mi.ScalarTransform4f().translate([0.0, 0.0, z]),
                "bsdf": {
                    "type": "dielectric",
                    "int_ior": 1.5168,  # Interior refractive index nbk 7
                    "ext_ior": 1.0,  # Exterior refractive index (air)
                },
            }

    def _add_noise_to_lens_positions(self):
        noise_angles_xy = self.random.normal(
            loc=0, scale=self.noise_angle * 25, size=(self.n_lenses, 2)
        )
        noise_translation_xyz = np.hstack(
            [
                self.random.normal(
                    loc=0, scale=self.noise_translation * 0.025, size=(self.n_lenses, 2)
                ),
                np.zeros((self.n_lenses, 1)),
            ]
        )

        for lens, angles, translation, z_position in zip(
            range(self.n_lenses), noise_angles_xy, noise_translation_xyz, self.z_lenses
        ):
            lens_vertices = np.array(self.params[f"lens_{lens}.vertex_positions"]).reshape(-1, 3)

            # this could be done in one matrix

            # translation in origin.
            lens_vertices -= [0, 0, z_position]

            # rotation
            rotation_matrix = self._construct_rotation_matrix(angles)
            lens_vertices = lens_vertices @ rotation_matrix.T

            # apply translation noise
            lens_vertices += translation

            # translate back to starting position
            lens_vertices += [0, 0, z_position]

            self.params[f"lens_{lens}.vertex_positions"] = lens_vertices.flatten()

        self._set_starting_lens_vertices()

    def _create_base_scene_dict(self):
        return {
            "type": "scene",
            "Integrator": {
                "type": "path",
            },
            "sensor": {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f()
                .translate([0.0, 0.0, self.z_sensor])
                .rotate(axis=[1, 0, 0], angle=180)
                .scale([0.5, 0.5, 0.5]),
                'sensor': {
                    'type': 'irradiancemeter',
                    'sampler': {
                        'type': 'independent',
                        'sample_count': self.sample_count,
                        'seed': int(self.random.integers(2**32 - 1)),
                    },
                    "film": {
                        "type": "hdrfilm",
                        "width": self.WIDTH,
                        "height": self.HEIGHT,
                        "rfilter": {
                            "type": "box",
                        },
                    },
                },
            },
            "Light": {
                "type": "envmap",
                "filename": os.path.join(
                    str(files("relign")),
                    "data",
                    f"{self.__class__.__name__}",
                    "env_map_200.png",
                ),
                "to_world": mi.ScalarTransform4f().rotate(axis=np.array([0, 1, 0]), angle=180),
            },
        }

    def _render_scene(self, r):
        """
        At first, scales all parameters from [0, 1] to their individual range, depending on amount
        of lenses. For one lens:
            xy: [-0.4, 0.4].
            z: [-0.5, 0.5].
            angles xy: [-30, 30].
        Then applies rotation on saved original 3d mesh around coordinate center.
        After that, performs translation to desired position.
        """
        # scale into [-1, 1]
        r = 2 * r - 1
        # Scale angles
        angles = 30 * r[3:]
        z = 1.0 / (self.n_lenses * self.n_lenses) * r[2]
        xy = 0.4 * r[:2]

        # First, we have to rotate
        rotated_vertices = self.rotate(angles)
        # After rotation, position needs to be translated
        self.translate(
            rotated_vertices=rotated_vertices,
            translation_vector=np.array([xy[0], -xy[1], z]),
        )

        self.params.update()
        image = mi.render(self.scene, seed=int(self.random.integers(2**32 - 1)))

        return image.numpy()[..., 0].reshape(self.WIDTH, self.HEIGHT, 1)

    def _construct_rotation_matrix(self, angles):
        # Construct rotation matrix by multiplying the rotation matrices of all axes
        rotation_matrix = np.identity(3)
        for i, angle in enumerate(angles):
            axis = np.eye(3)[i]
            R = np.array(mi.scalar_rgb.Transform4f().rotate(axis=axis, angle=angle).matrix)[
                :3, :3
            ].T
            rotation_matrix = rotation_matrix @ R
        return rotation_matrix

    def rotate(self, angles):
        rotated_vertices = self.lens_vertex_coords_start

        if angles is not None:
            rotation_matrix = self._construct_rotation_matrix(angles)
            rotated_vertices = rotated_vertices @ rotation_matrix.T

        return rotated_vertices

    def translate(self, rotated_vertices, translation_vector=None):
        if translation_vector is None:
            translation_vector = np.array([0, 0, 0])

        # applies general translation of objective (to all lenses)
        translated_vertices = rotated_vertices + translation_vector

        # apply transformation to lenses
        for i in range(self.n_lenses):
            self.params[f"lens_{i}.vertex_positions"] = translated_vertices[i].flatten()

    def _make_image(self, r):
        """r is always between [0, 1] here."""
        return self._render_scene(r)

    def get_optimal_move(self):
        raise ValueError("optimal move is not available for `LensEnv`!")

    def _load_gt(self, n_samples=1000):
        filename = get_gt_filename_from_params(
            n_lenses=self.n_lenses,
            n_samples=n_samples,
            width=self.WIDTH,
            height=self.HEIGHT,
        )
        return super()._load_gt_from_filename(filename)

    @staticmethod
    def create_gt(n_samples=1000, n_lenses=2, sample_count=512):
        env = LensEnv(
            n_actions=5,
            n_lenses=n_lenses,
            noise_angle=0.0,
            noise_translation=0.0,
            sample_count=sample_count,
        )
        optimal_r = env.R
        imgs = []

        for _ in np.arange(n_samples):
            env.reset()
            imgs.append(env._make_image(optimal_r))

        mean_image = np.mean(imgs, axis=0)

        filename = f'gt__n_lenses_{n_lenses}__n_samples_{n_samples}__{env.WIDTH}x{env.HEIGHT}.npy'
        path = os.path.join(
            str(files("relign")),
            "data",
            env.__class__.__name__,
            filename,
        )

        np.save(path, mean_image)
        return mean_image, imgs

    @staticmethod
    def compute_mean_env_variation(n_samples=1000, n_lenses=2):
        gt, imgs = LensEnv.create_gt(n_samples, n_lenses)
        return np.mean([rmse(img[:, :, 0], gt[:, :, 0]) for img in imgs])

    @staticmethod
    def starting_goal_threshold(n_samples=1000):
        env = LensEnv()
        img_gt = env._load_gt(n_samples=n_samples)
        return rmse(np.zeros((env.WIDTH, env.HEIGHT, 1)), img_gt)


class EnvWrapper:
    """
    Allows to take absolute steps
    """

    def __init__(self, env):
        self.env = env
        self.r = np.zeros(self.env.n_actions)

    def reset(self):
        obs = self.env.reset()
        self.r = self.env.r
        self.x = []
        self.x_env = []
        return obs

    def move(self, r):
        r = np.array(r)
        img = self.env.step(r - self.r)["img"]
        self.r = r
        return img

    def __call__(self, r):
        img = self.move(r)
        self.x.append(r)
        self.x_env.append(self.env.r)
        return self.env.compute_distance_to_gt(img)

    @property
    def n_actions(self):
        return self.env.n_actions


class GymnasiumEnv(gym.Env):

    def __init__(
        self,
        env_cls=GaussianIntensityEnv,
        *args,
        **kwargs,
    ):
        self.env = env_cls(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=0, high=3, shape=(1, self.env.WIDTH, self.env.HEIGHT), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.2, high=0.2, shape=(self.env.n_actions,), dtype=np.float32
        )

    def _preprocess_obs(self, img):
        # TODO: in future, this should be: img.reshape(1, 50, 50)
        return np.swapaxes(img, 0, 2)

    def get_optimal_move(self):
        return self.env.get_optimal_move()

    @property
    def obs_shape(self):
        # TODO: in future, this should be: (width, height, 1)
        return (1, self.env.WIDTH, self.env.HEIGHT)

    @property
    def gt(self):
        return self.env._gt

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        img = self.env.reset()
        return self._preprocess_obs(img), {}

    def step(self, action):

        dct = self.env.step(action)
        return (
            self._preprocess_obs(dct["img"]),
            dct["reward"],
            bool(dct["terminated"]),
            bool(dct["truncated"]),
            {},
        )


def make_stacked_vec_env(
    env_name,
    score_threshold=0.3,
    max_episode_steps=200,
    n_envs=10,
    n_stack=5,
    seed=None,
    reward="minus_one",
    env_args={},
):
    from relign.config import Config  # to avoid circular imports

    env_kwargs = {
        'score_goal_threshold': score_threshold,
        'max_episode_steps': max_episode_steps,
    }

    env_kwargs["env_cls"] = Config.envs[env_name]["cls"]
    env_kwargs["n_actions"] = Config.envs[env_name]["cls"].N_ACTIONS_MAX
    env_kwargs["seed"] = seed
    env_kwargs["reward"] = reward
    env_kwargs.update(env_args)

    env = make_vec_env(
        env_id=GymnasiumEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    if n_stack > 1:
        # Stack latest observations into one
        env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
    return env
