from dataclasses import dataclass
from types import MappingProxyType
import torch
from sb3_contrib import (
    RecurrentPPO,
    TRPO,
)
from stable_baselines3 import (
    PPO,
    A2C,
)
from relign.helpers import update_dict
from relign.network import CustomCNN
import relign.generator


@dataclass(frozen=True)
class Config:
    envs = MappingProxyType(
        {
            "gi": {
                "cls": relign.generator.GaussianIntensityEnv,
                "args": {},
            },
            "la": {
                "cls": relign.generator.LensEnv,
                "args": {
                    "n_lenses": 2,
                    "noise_angle": 0.0,
                    "noise_translation": 0.0,
                    "reward": "minus_one",
                    "sample_count": 512,
                },
            },
            "gymnasium": {
                "cls": relign.generator.GymnasiumEnv,
                "args": {},
            },
        }
    )

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
                    "batch_size": 200,
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
                    "batch_size": 200,
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
            "la": lambda env_args: {
                1: 0.025,
                2: 0.022074252,
                3: 0.039180785,
            }[env_args["n_lenses"]],
            "gi": lambda _: 0.0,
        }
    )

    lens_benchmarks = MappingProxyType(
        {
            "b_L2_N000": {
                "n_lenses": 2,
                "noise_angle": 0.0,
                "noise_translation": 0.0,
            },
            "b_L2_N025": {
                "n_lenses": 2,
                "noise_angle": 0.25,
                "noise_translation": 0.25,
            },
            "b_L2_N050": {
                "n_lenses": 2,
                "noise_angle": 0.5,
                "noise_translation": 0.5,
            },
            "b_L3_N000": {
                "n_lenses": 3,
                "noise_angle": 0.0,
                "noise_translation": 0.0,
            },
            "b_L3_N025": {
                "n_lenses": 3,
                "noise_angle": 0.25,
                "noise_translation": 0.25,
            },
            "b_L3_N050": {
                "n_lenses": 3,
                "noise_angle": 0.0,
                "noise_translation": 0.0,
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
        env_args = update_dict(Config.envs[params["env"]]["args"], params)
        model_args = update_dict(Config.algorithms[params["model"]]["args"], params)

        return env_args, {**default_model_args, **model_args}, training_args
