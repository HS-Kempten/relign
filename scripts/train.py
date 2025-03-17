import json
from typing import (
    Annotated,
    Optional,
)
from typer import (
    run,
    Option,
    Context,
)
import logging
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from relign.config import Config
from relign.helpers import setup_logger
from relign.generator import make_stacked_vec_env
from relign.save_model import SaveCallback
from relign.curriculum import (
    CLCallbackSteps,
    Threshold,
    EvaluationCallback,
)
from stable_baselines3.common.callbacks import (
    CallbackList,
)


working_dir = os.environ["WORKING_DIR"]


def _make_env(env_name, n_envs, n_stack, max_episode_steps, score_threshold, env_args, seed):
    return make_stacked_vec_env(
        env_name=env_name,
        n_envs=n_envs,
        n_stack=n_stack,
        score_threshold=score_threshold,
        max_episode_steps=max_episode_steps,
        env_args=env_args,
        seed=seed,
    )


def _get_slurm_id():
    slurm_array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    slurm_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    full_job_id = ""
    if slurm_array_job_id and slurm_task_id:
        full_job_id = f"{slurm_array_job_id}_{slurm_task_id}"
    elif slurm_job_id:
        full_job_id = slurm_job_id
    return {"SLURM Job ID": full_job_id}


def train(params):
    env_args, model_args, training_args = Config.setup_params(params)
    target_optimum = Config.targets_optimum[training_args["env"]](env_args)

    if training_args['curriculum']:
        # Start with a larger threshold

        if env_args["reward"] == 'potential':
            score_goal_threshold = Threshold(
                Config.envs[training_args["env"]]["cls"].starting_goal_threshold() * 0.9
            )

        else:
            score_goal_threshold = Threshold(
                Config.envs[training_args["env"]]["cls"].starting_goal_threshold() * 0.9 * 0.5
            )

        curriculum_callback = CLCallbackSteps(
            threshold=score_goal_threshold,
            factor=0.9,
            max_steps=20,
            look_back=5,
        )

    else:
        curriculum_callback = None
        score_goal_threshold = 5 * target_optimum

    env_eval, env_train = (
        _make_env(
            env_name=training_args["env"],
            n_envs=training_args['n_envs'],
            n_stack=training_args['n_stack'],
            max_episode_steps=model_args["n_steps"],
            score_threshold=score_goal_threshold,
            seed=training_args["seed"],
            env_args=env_args,
        )
        for _ in range(2)
    )

    run = wandb.init(
        project='relign',
        sync_tensorboard=True,
        config={**env_args, **model_args, **training_args, **_get_slurm_id()},
    )
    save_model_callback = SaveCallback(
        model_save_path=f"{working_dir}/relign/models/{run.id}", threshold=score_goal_threshold
    )

    eval_callback = EvaluationCallback(
        eval_env=env_eval,
        deterministic=False,
        eval_freq=2 * model_args["n_steps"],
        callback_after_eval=curriculum_callback,
        callback_on_new_best=save_model_callback,
    )
    os.makedirs(f"{working_dir}/relign/models/{run.id}", exist_ok=True)
    with open(f"{working_dir}/relign/models/{run.id}/env_args.json", "w") as file:
        json.dump(env_args, file)

    model_args["env"] = env_train
    model_args["tensorboard_log"] = f"{working_dir}/relign/logs/{run.id}"

    model = Config.algorithms[training_args["model"]]["cls"](**model_args)
    model.learn(
        total_timesteps=training_args["total_steps"],
        callback=CallbackList([WandbCallback(verbose=2), eval_callback]),
    )
    run.finish()


def main(
    env: Annotated[str, Option(help="Environment used")],
    model: Annotated[str, Option(help="Reinforcement Learning Model")],
    n_lenses: Annotated[
        Optional[int], Option(help="Number of lenses in optical lens setup")
    ] = None,
    noise_movement: Annotated[Optional[float], Option(help="Noise for every movement")] = None,
    noise_translation: Annotated[
        Optional[float], Option(help="Translated single lens noise")
    ] = None,
    noise_angle: Annotated[Optional[float], Option(help="Rotated single lens noise")] = None,
    learning_rate: Annotated[Optional[float], Option(help="Learning rate for the model")] = None,
    ent_coef: Annotated[Optional[float], Option(help="Entropy coefficient for the model")] = None,
    vf_coef: Annotated[
        Optional[float], Option(help="Value function coefficient for the model")
    ] = None,
    n_envs: Annotated[
        Optional[int], Option(help="Number of environments to run in parallel")
    ] = None,
    gamma: Annotated[Optional[float], Option(help="Discount factor for the environment")] = None,
    seed: Annotated[Optional[int], Option(help="Random seed for reproducibility")] = None,
    n_stack: Annotated[
        Optional[int], Option(help="Number of frames to stack in the environment")
    ] = None,
    n_steps: Annotated[Optional[int], Option(help="Number of steps per update")] = None,
    max_grad_norm: Annotated[
        Optional[float], Option(help="Maximum gradient norm for clipping")
    ] = None,
    total_steps: Annotated[Optional[int], Option(help="Total training steps")] = None,
    normalize_advantage: Annotated[
        Optional[bool], Option(help="Normalize advantage estimates", is_flag=True)
    ] = None,
    clip_range: Annotated[Optional[float], Option(help="Clip range for PPO")] = None,
    verbose: Annotated[Optional[bool], Option(help="Enables debug loggings", is_flag=True)] = None,
    curriculum: Annotated[
        Optional[bool], Option(help="Use curriculum learning", is_flag=True)
    ] = None,
    use_sde: Annotated[
        Optional[bool], Option(help="Use generalized State Dependent Exploration", is_flag=True)
    ] = None,
    sample_count: Annotated[
        Optional[int], Option(help="Number of samples per pixel (Mitsuba render)", is_flag=True)
    ] = None,
    reward: Annotated[Optional[str], Option(help="The reward to use")] = None,
    benchmark: Annotated[
        Optional[str], Option(help="Benchmark string. Warning: Overrides parameters!")
    ] = None,
    ctx: Context = Option(None, hidden=True),
):
    params = ctx.params
    params = Config.set_benchmark_params(params, benchmark)

    if verbose:
        setup_logger(logging.DEBUG)

    train(params)


if __name__ == '__main__':
    run(main)
