from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
import numpy as np
import torch
from sb3_contrib import (
    RecurrentPPO,
    TRPO,
)
from stable_baselines3 import (
    PPO,
    A2C,
)
from relign.helpers import update_dict, get_lens_specs_from_file, scene_objects
from relign.network import CustomCNN
from relign import noise


@dataclass(frozen=True)
class Config:
    lens_setups = ["l2", "l3", "od"]

    algorithms = MappingProxyType(
        {
            "PPO": {
                "cls": PPO,
                "args": {
                    "n_epochs": 5,
                    "ent_coef": 0.1,
                    "vf_coef": 0.5,
                    "clip_range": 0.2,
                    "gae_lambda": 0.95,
                    "batch_size": 500,
                    "max_grad_norm": 0.5,
                    "use_sde": False,
                },
            },
            "RPPO": {
                "cls": RecurrentPPO,
                "args": {
                    "n_epochs": 5,
                    "ent_coef": 0.1,
                    "vf_coef": 0.5,
                    "clip_range": 0.2,
                    "gae_lambda": 0.95,
                    "batch_size": 100,
                    "max_grad_norm": 0.5,
                    "policy": "CnnLstmPolicy",
                    "use_sde": False,
                },
            },
            "A2C": {
                "cls": A2C,
                "args": {
                    "max_grad_norm": 0.5,
                    "use_sde": False,
                },
            },
            "TRPO": {
                "cls": TRPO,
                "args": {"batch_size": 50, "use_sde": False},
            },
        }
    )

    training_args = MappingProxyType(
        {
            "model": "PPO",
            "env": "la",
            "n_envs": 5,
            "curriculum": False,
            "n_stack": 5,
            "total_steps": 5e6,
            "seed": None,
        }
    )

    default_model_args = MappingProxyType(
        {
            "policy": "CnnPolicy",
            "policy_kwargs": dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=256),
                log_std_init=-3,
            ),
            "device": torch.device('cuda:0'),
            "gamma": 0.9,  # discount factor
            "learning_rate": 3e-4,
            "use_sde": False,
            'sde_sample_freq': 8,
            "normalize_advantage": True,
            # Window size for rollout statistics, are reported whenever an environment is `done`,
            # which depends on the simulation-time and the steps
            "stats_window_size": 10,
            "verbose": 0,
            "n_steps": 100,  # Number of environment-steps before an update-step is done
        }
    )

    targets_optimum = MappingProxyType(
        {
            "la": {
                lens_setups[0]: {
                    (50, 50, 512): 0.022074252,
                },
                lens_setups[1]: {
                    (50, 50, 512): 0.039180785,
                },
                lens_setups[2]: {
                    (50, 50, 512): 0.0015,
                    (200, 200, 512): 0.30,
                    (200, 200, 256): 0.03,  # 0.27,
                    (200, 200, 128): 0.035,  # 0.27,
                    (200, 200, 64): 0.04,
                },
            },
            "gi": 0.0,
        }
    )

    lens_benchmarks = MappingProxyType(
        {
            "b_L2_N000": {
                "spec": lens_setups[0],
                "noise_objects": 0.0,
            },
            "b_L2_N025": {
                "spec": lens_setups[0],
                "noise_objects": 0.25,
            },
            "b_L2_N050": {
                "spec": lens_setups[0],
                "noise_objects": 0.5,
            },
            "b_L3_N000": {
                "spec": lens_setups[1],
                "noise_objects": 0.0,
            },
            "b_L3_N025": {
                "spec": lens_setups[1],
                "noise_objects": 0.25,
            },
            "b_L3_N050": {
                "spec": lens_setups[1],
                "noise_objects": 0.5,
            },
            "b_OD_N000_64": {
                "spec": lens_setups[2],
                "noise_objects": 0.0,
                "width": 200,
                "height": 200,
                "sample_count": 64,
                "sgt": 0.051,
            },
            "b_OD_N000_256": {
                "spec": lens_setups[2],
                "noise_objects": 0.0,
                "width": 200,
                "height": 200,
                "sample_count": 256,
                "sgt": 0.038,
            },
            "b_OD_N025_64": {
                "spec": lens_setups[2],
                "noise_objects": 0.25,
                "width": 200,
                "height": 200,
                "sample_count": 64,
                "sgt": 0.0555,
            },
            "b_OD_N025_256": {
                "spec": lens_setups[2],
                "noise_objects": 0.25,
                "width": 200,
                "height": 200,
                "sample_count": 256,
                "sgt": 0.047,
            },
            "b_OD_N050_64": {
                "spec": lens_setups[2],
                "noise_objects": 0.5,
                "width": 200,
                "height": 200,
                "sample_count": 64,
                "sgt": 0.060,
            },
        }
    )

    lens_env_specs = MappingProxyType(
        {
            lens_setups[0]: {
                "sensor_z": 0.8,
                "translation_scaling": [0.4, 0.4, 0.25],
                "rotation_scaling": [30, 30],
                "sensor_dist_overview": 4,
                "bitmap_contrast_factor": 1,
                "n_objects": 2,
                "func": lambda **_: {
                    "sensor_scale": np.ones(2),
                    "emitter_distance": np.array([-65]),
                    "emitter_scale": np.array([50, 50]),
                },
                "lens_specs": {},
            },
            lens_setups[1]: {
                "sensor_z": 0.7,
                "translation_scaling": [0.4, 0.4, 0.11],
                "rotation_scaling": [30, 30],
                "sensor_dist_overview": 4,
                "bitmap_contrast_factor": 1,
                "n_objects": 3,
                "func": lambda **_: {
                    "sensor_scale": np.ones(2),
                    "emitter_distance": np.array([-65]),
                    "emitter_scale": np.array([50, 50]),
                },
                "lens_specs": {},
            },
            lens_setups[2]: {
                "sensor_z": sum(get_lens_specs_from_file(filename="config.txt")[1])
                - sum(get_lens_specs_from_file(filename="config.txt")[1][:-2]) / 2,
                "translation_scaling": [1.0, 1.0, 1.0],
                "rotation_scaling": [7, 7],
                "sensor_dist_overview": 50,
                "bitmap_contrast_factor": 10,
                "n_objects": 7,
                "func": scene_objects,
                "lens_specs": {
                    "fov": 84.1,
                    "focal_length": 2.895,
                    "exit_pupil_angle": 13.45,
                },
            },
        }
    )

    lens_object_specs = MappingProxyType(
        {
            lens_setups[0]: {  # 2 same lenses
                f"lens_{i}": {
                    "translate": [0.0, 0.0, z],
                    "scale": [1.0, 1.0, 1.0],
                    "filename": "lens.PLY",
                    "bsdf": {
                        "type": "dielectric",
                        "int_ior": "bk7",
                        "ext_ior": "air",
                    },
                    "bsdf_overview": {
                        'type': 'roughdielectric',
                        'distribution': 'beckmann',
                        'alpha': 0.15,
                        'int_ior': 'bk7',
                        'ext_ior': 'air',
                    },
                }
                for i, z in enumerate([-0.175, 0.175])
            },
            lens_setups[1]: {  # 3 same lenses
                f"lens_{i}": {
                    "translate": [0.0, 0.0, z],
                    "scale": [1.0, 1.0, 1.0],
                    "filename": "lens.PLY",
                    "bsdf": {
                        "type": "dielectric",
                        "int_ior": "bk7",
                        "ext_ior": "air",
                    },
                    "bsdf_overview": {
                        'type': 'roughdielectric',
                        'distribution': 'beckmann',
                        'alpha': 0.15,
                        'int_ior': 'bk7',
                        'ext_ior': 'air',
                    },
                }
                for i, z in enumerate([-0.35, 0, 0.35])
            },
            lens_setups[2]: {  # optic design lens
                **{
                    f"lens_{i}": {
                        "translate": [
                            0.0,
                            0.0,
                            -sum(get_lens_specs_from_file(filename="config.txt")[1][:-2]) / 2,
                        ],
                        "scale": np.ones(3),
                        "filename": f"lens_{i}.PLY",
                        "bsdf": {
                            "type": "dielectric",
                            "int_ior": get_lens_specs_from_file(filename="config.txt")[0][i],
                            "ext_ior": "air",
                        },
                        "bsdf_overview": {
                            'type': 'roughdielectric',
                            'distribution': 'beckmann',
                            'alpha': 0.35,
                            'int_ior': 'bk7',
                            'ext_ior': 'air',
                        },
                    }
                    for i in range(4)
                },
                **{
                    f"stop_{i}": {
                        "translate": [
                            0.0,
                            0.0,
                            -sum(get_lens_specs_from_file(filename="config.txt")[1][:-2]) / 2,
                        ],
                        "scale": np.ones(3),
                        "filename": f"stop_{s}.PLY",
                        "bsdf": {
                            "type": "diffuse",
                            "reflectance": {'type': 'rgb', 'value': [0.0, 0.0, 0.0]},
                        },
                        "bsdf_overview": {
                            'type': 'roughdielectric',
                            'distribution': 'beckmann',
                            'alpha': 0.0,
                            'int_ior': 1.0,
                            'ext_ior': 'air',
                        },
                    }
                    for s, i in enumerate(range(4, 6))
                },
                **{
                    "mount": {
                        "translate": [
                            0.0,
                            0.0,
                            -sum(get_lens_specs_from_file(filename="config.txt")[1][:-2]) / 2,
                        ],
                        "scale": np.ones(3),
                        "filename": "lens_mount.PLY",
                        "bsdf": {
                            "type": "diffuse",
                            "reflectance": {'type': 'rgb', 'value': [0.0, 0.0, 0.0]},
                        },
                        "bsdf_overview": {
                            'type': 'roughdielectric',
                            'distribution': 'beckmann',
                            'alpha': 0.25,
                            'int_ior': 1.0,
                            'ext_ior': 'air',
                        },
                    }
                },
            },
        }
    )

    lens_system_specs = MappingProxyType(
        {
            lens_setups[0]: {
                "fov": 84.1,
                "focal_length": 0.2895,
                "exit_pupil_angle": 13.45,
                "func": lambda _: {},
            },
            lens_setups[1]: {
                "fov": 84.1,
                "focal_length": 2.895,
                "exit_pupil_angle": 13.45,
                "func": lambda _: {},
            },
            lens_setups[2]: {
                "fov": 84.1,
                "focal_length": 2.895,
                "exit_pupil_angle": 13.45,
                "func": scene_objects,
            },
        },
    )

    lens_noise_funcs = MappingProxyType(
        {
            lens_setups[0]: {
                f"lens_{i}": partial(noise.gaussian_trans_rot, scale_rot=3, scale_trans=0.03)
                for i in range(2)
            },
            lens_setups[1]: {
                f"lens_{i}": partial(noise.gaussian_trans_rot, scale_rot=3, scale_trans=0.03)
                for i in range(3)
            },
            lens_setups[2]: {
                f"lens_{i}": partial(noise.gaussian_trans_rot, scale_rot=1.5, scale_trans=0.00005)
                for i in range(1, 3)
            },
        }
    )

    @staticmethod
    def set_benchmark_params(params: dict, benchmark: str | None) -> dict:
        if benchmark is not None:
            params.update(Config.lens_benchmarks[benchmark])
        return params

    @staticmethod
    def setup_params(params: dict) -> tuple:
        default_model_args = update_dict(Config.default_model_args, params)
        training_args = update_dict(Config.training_args, params)
        env_args = update_dict(Config.get_envs()[params["env"]]["args"], params)
        model_args = update_dict(Config.algorithms[params["model"]]["args"], params)

        return env_args, {**default_model_args, **model_args}, training_args

    @staticmethod
    def get_envs():
        # lazy import to avoid circular imports
        from relign.generator import GaussianIntensityEnv, LensEnv, GymnasiumEnv

        return MappingProxyType(
            {
                "gi": {
                    "cls": GaussianIntensityEnv,
                    "args": {},
                },
                "la": {
                    "cls": LensEnv,
                    "args": {
                        "spec": "od",
                        "noise_objects": 0.0,
                        "reward": "minus_one",
                        "sample_count": 64,
                        "width": 200,
                        "height": 200,
                    },
                },
                "gymnasium": {
                    "cls": GymnasiumEnv,
                    "args": {},
                },
            }
        )
