"""
Core module holding the implementation of the Lens environment.
"""

import os
import logging
import warnings
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
    get_gt_filename_from_params,
    rmse,
    construct_mitsuba_rotation_matrix,
    clean_up_drjit,
)
from relign.config import Config
import gymnasium as gym

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    N_ACTIONS_MAX: int

    def __init__(
        self,
        seed=None,
        n_actions=5,
        reward="minus_one",
        max_episode_steps=200,
        score_goal_threshold=0.05,
        noise_movement=0.01,
        clipping="soft",
        width=50,
        height=50,
        score="combined",
    ):
        if n_actions > self.N_ACTIONS_MAX:
            raise ValueError(f"n_actions must be <= {self.N_ACTIONS_MAX}")

        if clipping not in ['soft', 'hard']:
            raise ValueError('Invalid clipping')

        self.width = width
        self.height = height
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
        self._setup_score(score)

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

    def _distort_movement(self, r, a):
        return self.W.dot(a) + r

    def _setup_reward(self, reward):
        self.reward_mappings = {
            "minus_one": self.compute_minus_one_reward,
            "potential": self.compute_potential_reward,
            "minus_one_edge": self.compute_minus_one_edge_reward,
            "minus_one_plus": self.compute_minus_one_plus,
            "score": self.compute_score_reward,
        }
        if reward not in self.reward_mappings.keys():
            raise ValueError(f"given 'reward style' must be in {list(self.reward_mappings.keys())}")
        else:
            self.reward_func = self.reward_mappings[reward]

    def _setup_score(self, score):
        self.score_mappings = {
            "rmse": self._score_rmse,
            "rmse_binary": self._score_rmse_binary,
            "combined": self._score_combined,
            "distance": self._score_distance,
        }
        if score not in self.score_mappings.keys():
            raise ValueError(f"given score must be in {list(self.score_mappings.keys())}")
        else:
            self.score_func = self.score_mappings[score]

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

    def compute_score_reward(self, *_):
        return -self._scores[-1]

    def compute_distance_to_optimum(self):
        return rmse(self.r_optimal, self.r)

    def _score_rmse(self, img):
        return rmse(img, self._gt)

    def _score_rmse_binary(self, img):
        threshold = 0.02
        bin_img0 = (img > threshold).astype(int)
        bin_img1 = (self._gt > threshold).astype(int)
        return rmse(bin_img0, bin_img1)

    def _score_combined(self, img):
        optimum_region_expected = 0.1
        distance_to_center = np.linalg.norm(self.r - self.R[: self.n_actions])
        if distance_to_center > optimum_region_expected:
            score_r = distance_to_center
        else:
            score_r = 0

        return rmse(img, self._gt) + score_r

    def _score_distance(self, _):
        return np.linalg.norm(self.r - self.R[: self.n_actions])

    def compute_distance_to_gt(self, img):
        if self._gt is None:
            warnings.warn(
                "No precomputed ground truth image was found for the current setup. Please run "
                "`BaseEnv().create_gt()` to generate it. "
                "Note that this process may take a few minutes."
            )
            return np.inf

        return self.score_func(img)

    def _compute_move_to_position(self, r):
        return np.linalg.inv(self.W).dot((r - self.r).T)

    def _update(self, a):
        if not a.shape[0] == self.n_actions:
            raise ValueError("moving vector must have same dimension as `Environment.n_actions`")

        r_proposed = self._distort_movement(r=self.r, a=a)

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
            "reward": self.reward_func(truncated, terminated),
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

    @abstractmethod
    def create_gt(self):
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

        img = np.zeros(shape=(self.width, self.height, 1), dtype=np.float32)

        cov_x = 0.05 + (cov_x) ** 2 / 5
        cov_y = 0.05 + (cov_y) ** 2 / 5

        center_x = center_x * self.width
        center_y = center_y * self.height

        angle = np.pi * angle / 2

        X, Y = np.meshgrid(
            np.linspace(-1, 1, self.width, dtype=np.float32),
            np.linspace(-1, 1, self.height, dtype=np.float32),
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
            mean=[-1 + 2 * center_x / self.width, -1 + 2 * center_y / self.height],
            cov=cov_rotated,
        )
        img[:, :, 0] = img[:, :, 0] + rv.pdf(xx)
        return img

    def _load_gt(self):
        filename = f"gt__{self.width}x{self.height}.npy"
        return super()._load_gt_from_filename(filename)

    def create_gt(self):
        self.reset()

        filename = f'gt__{self.width}x{self.height}.npy'
        path = os.path.join(
            str(files("relign")),
            "data",
            self.__class__.__name__,
            filename,
        )

        np.save(path, self._make_image(self.R))

    @staticmethod
    def starting_goal_threshold():
        env = GaussianIntensityEnv()
        img_gt = env._load_gt()
        return rmse(np.zeros((env.width, env.height, 1)), img_gt)


class LensEnv(BaseEnvironment):
    """
    Environment for mitsuba3.
    """

    N_ACTIONS_MAX = 5

    def __init__(
        self,
        spec="od",
        score_goal_threshold=None,
        n_actions=5,
        noise_objects=0.0,
        sample_count=512,
        ref_pattern="siemens",
        hr=False,
        config=None,
        width=50,
        height=50,
        mitsuba_variant=None,
        *args,
        **kwargs,
    ):
        self._check_for_deprecated_arguments(kwargs)
        self._setup_mitsuba(mitsuba_variant)

        if score_goal_threshold is None:
            score_goal_threshold = Config.targets_optimum["la"][spec].get(
                (width, height, sample_count), 0.0
            )

        if not (0 <= noise_objects <= 1):
            raise ValueError("`noise_objects` must be between 0 (no noise) and 1 (maximum noise).")
        if config is None:
            self.config = {
                "lens_env_specs": Config.lens_env_specs[spec],
                "lens_object_specs": Config.lens_object_specs[spec],
                "lens_noise_funcs": Config.lens_noise_funcs[spec],
                "lens_scene_specs": Config.lens_env_specs[spec]["func"](
                    **Config.lens_env_specs[spec]["lens_specs"],
                    ref_pattern_res=[300, 300],
                    sensor_resolution=[
                        kwargs.get("width", 50),
                        kwargs.get("height", 50),
                    ],  # TODO: this is bit hacky and will fail if default changes.
                ),
            }
        else:
            self.config = config

        self.spec = spec
        self.hr = hr
        self.noise_objects = noise_objects
        self.sample_count = sample_count
        self.ref_pattern = ref_pattern
        # loads ground truth image
        super().__init__(
            score_goal_threshold=score_goal_threshold,
            n_actions=n_actions,
            width=width,
            height=height,
            *args,
            **kwargs,
        )

    def _check_for_deprecated_arguments(self, kwargs):
        if "n_lenses" in kwargs.keys():
            raise RuntimeError(
                "`n_lenses` parameter is deprecated and was removed. "
                "Use `spec` argument instead with one of the following options: "
                f"{[s for s in Config.lens_setups]}"
            )

        if "noise_angle" in kwargs.keys() or "noise_translation" in kwargs.keys():
            raise RuntimeError(
                "`noise_angle` and `noise_translation` parameters are deprecated and were removed. "
                "Use single argument `noise_objects` instead."
            )

    def _setup_mitsuba(self, variant):
        if variant is not None:
            mi.set_variant(variant)
        elif get_device() == "cpu" or get_device() == "mps":
            mi.set_variant("scalar_rgb")
            # mi.set_variant('llvm_ad_rgb') is bugged (memory leak)
        else:
            mi.set_variant("cuda_ad_mono")

        # TODO: This is currently needed, as issues with vertices produce too many warnings.
        mi.set_log_level(mi.LogLevel.Error)

    def reset(self):
        """
        Resets the environment and cleans up resources.

        Returns:
            The result of the superclass's `reset` method.
        """
        clean_up_drjit()
        self.scene = self._create_optimal_scene()
        self.params = mi.traverse(self.scene)
        self._set_starting_lens_vertices()
        if self.noise_objects > 0.0:
            self._add_noise_to_lens_positions()
        return super().reset()

    def _set_starting_lens_vertices(self):
        self.lens_vertex_coords_start = [
            np.array([self.params[f"{obj}.vertex_positions"]]).reshape(-1, 3)
            for obj in self.config["lens_object_specs"].keys()
        ]

    def _create_optimal_scene(self):
        scene_dict = self._create_base_scene_dict()
        self._add_lenses_to_scene_dict(scene_dict)
        return mi.load_dict(scene_dict, parallel=False, optimize=False)

    def render_scene_overview(
        self,
        sensor_distance=None,
        axis=[1, 0, 0],
        sample_count=512,
        r=np.array([]),
        width=200,
        height=200,
    ):
        if r.size == 0:
            r = self.fill_action_with_optimal_values()
        if isinstance(axis, list):
            axis = np.array(axis)
        elif not isinstance(axis, np.ndarray):
            raise ValueError("axis must be list-like object. E. g. [1, 0, 0]")

        if sensor_distance is None:
            sensor_distance = self.config["lens_env_specs"].get("sensor_dist_overview", 7)

        scene_dict = self._create_base_scene_dict()
        self._add_lenses_to_scene_dict(scene_dict, overview=True)

        scene_dict["Light"] = {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 0.1,
            },
        }

        img_sensor = self._make_image(r) * self.config["lens_env_specs"].get(
            "bitmap_contrast_factor", 1
        )
        bitmap = mi.Bitmap(img_sensor)
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

        scene = mi.load_dict(scene_dict, parallel=False, optimize=False)
        params = mi.traverse(scene)

        for obj in self.config["lens_object_specs"].keys():
            params[f"{obj}.vertex_positions"] = self.params[f"{obj}.vertex_positions"]

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
                    "width": width,
                    "height": height,
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
        img = image.numpy()[..., 0].reshape(width, height, 1)
        return img

    def _add_lenses_to_scene_dict(self, scene_dict, overview=False):
        specs = self.config["lens_object_specs"]

        for obj_name, obj_specs in specs.items():
            scene_dict[obj_name] = {
                "type": "ply",
                "face_normals": False,
                "filename": os.path.join(
                    str(files("relign")),
                    "data",
                    f"{self.__class__.__name__}",
                    "high_res" if self.hr else "",
                    obj_specs["filename"],
                ),
                "to_world": mi.ScalarTransform4f()
                .translate(obj_specs["translate"])
                .scale(obj_specs.get("scale", np.ones(3))),
                "bsdf": (
                    obj_specs.get("bsdf_overview", obj_specs["bsdf"])
                    if overview
                    else obj_specs["bsdf"]
                ),
            }

    def _add_noise_to_lens_positions(self):
        for obj, noise_func in self.config["lens_noise_funcs"].items():
            vertices = np.array(self.params[f"{obj}.vertex_positions"]).reshape(-1, 3)
            vertices_after_noise = noise_func(
                vertices=vertices,
                rng=self.random,
                scale_factor=self.noise_objects,
                pos_z=self.config["lens_object_specs"][obj]["translate"][-1],
            )
            self.params[f"{obj}.vertex_positions"] = vertices_after_noise.flatten()

        self._set_starting_lens_vertices()

        return None

    def _create_base_scene_dict(self):
        return {
            "type": "scene",
            "Integrator": {
                "type": "path",
            },
            "sensor": {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f()
                .translate([0.0, 0.0, self.config["lens_env_specs"]["sensor_z"]])
                .rotate(axis=[1, 0, 0], angle=180)
                .scale(np.append(self.config["lens_scene_specs"]["sensor_scale"] / 2.0, 1)),
                'sensor': {
                    'type': 'irradiancemeter',
                    'sampler': {
                        'type': 'independent',
                        'sample_count': self.sample_count,
                        'seed': int(self.random.integers(2**32 - 1)),
                    },
                    "film": {
                        "type": "hdrfilm",
                        "width": self.width,
                        "height": self.height,
                        "rfilter": {
                            "type": "box",
                        },
                    },
                },
            },
            "Light": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0, 0, self.config["lens_scene_specs"]["emitter_distance"]],
                    target=[0, 0, 0],
                    up=[0, 1, 0],
                )
                @ mi.ScalarTransform4f().scale(
                    np.append(self.config["lens_scene_specs"]["emitter_scale"] / 2, 1)
                ),
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "bitmap",
                        "filename": os.path.join(
                            str(files("relign")),
                            "data",
                            "ref_patterns",
                            self.ref_pattern + ".png",
                        ),
                    },
                },
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

        angles = self.config["lens_env_specs"]["rotation_scaling"] * r[3:]
        translation_vector = self.config["lens_env_specs"]["translation_scaling"] * r[:3]

        # First, we have to rotate
        rotated_vertices = self.rotate(angles)
        # After rotation, position needs to be translated
        self.translate(
            rotated_vertices=rotated_vertices,
            translation_vector=np.array(translation_vector),
        )

        self.params.update()
        image = mi.render(self.scene, seed=int(self.random.integers(2**32 - 1)))

        return image.numpy()[..., 0].reshape(self.height, self.width, 1)

    def rotate(self, angles):
        rotated_vertices = self.lens_vertex_coords_start

        if angles is not None:
            rotation_matrix = construct_mitsuba_rotation_matrix(angles)
            rotated_vertices = [r @ rotation_matrix.T for r in rotated_vertices]

        return rotated_vertices

    def translate(self, rotated_vertices, translation_vector=None):
        if translation_vector is None:
            translation_vector = np.array([0, 0, 0])

        translated_vertices = [t + translation_vector for t in rotated_vertices]

        for i, obj in enumerate(self.config["lens_object_specs"].keys()):
            self.params[f"{obj}.vertex_positions"] = translated_vertices[i].flatten()

    def _make_image(self, r):
        """r is always between [0, 1] here."""
        return self._render_scene(r)

    def get_optimal_move(self):
        raise ValueError("optimal move is not available!")

    def _load_gt(self, sample_count=2**16):
        filename = get_gt_filename_from_params(
            width=self.width,
            height=self.height,
            ref_pattern=self.ref_pattern,
            sample_count=sample_count,
            spec=self.spec,
        )
        return super()._load_gt_from_filename(filename)

    def create_gt(self, sample_count=2**16):
        self.sample_count = sample_count
        self.reset()

        img = self._make_image(self.R)

        filename = (
            "gt"
            + f"__spec_{self.spec}"
            + f"__ref_pattern_{self.ref_pattern}"
            + f"__sample_count_{sample_count}"
            + f"__{self.width}x{self.height}.npy"
        )

        path = os.path.join(
            str(files("relign")),
            "data",
            self.__class__.__name__,
            filename,
        )

        np.save(path, img)
        return img

    def compute_mean_env_variation(self, n_samples=1000):
        if self._gt is None:
            self._gt = self.create_gt()
            warnings.warn("Creating ground truth for current setup. This may take some time.")

        imgs = []
        for _ in range(n_samples):
            self.reset()
            img = self._make_image(self.R)
            imgs.append(img)
        return np.mean([self.compute_distance_to_gt(img) for img in imgs])

    def starting_goal_threshold(self):
        return rmse(np.zeros((self.width, self.height, 1)), self._gt)


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
            low=0, high=3.5, shape=(1, self.env.width, self.env.height), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.2, high=0.2, shape=(self.env.n_actions,), dtype=np.float32
        )

    def _preprocess_obs(self, img):
        return img.reshape(-1, self.env.width, self.env.height)

    def get_optimal_move(self):
        return self.env.get_optimal_move()

    @property
    def obs_shape(self):
        return (1, self.env.width, self.env.height)

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
    env_kwargs = {
        'score_goal_threshold': score_threshold,
        'max_episode_steps': max_episode_steps,
    }

    env_kwargs["env_cls"] = Config.get_envs()[env_name]["cls"]
    env_kwargs["n_actions"] = Config.get_envs()[env_name]["cls"].N_ACTIONS_MAX
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
