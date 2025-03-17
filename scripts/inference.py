import numpy as np
import os
import glob
import argparse
import pickle
from stable_baselines3 import PPO  
import wandb
from relign.generator import make_stacked_vec_env
import json
from relign.algorithms.gp import GP
from relign.algorithms.random_search import RandomSearch
from relign.algorithms.forest import ForestOptimization
from scipy.optimize import minimize
from relign.generator import (
    LensEnv,
    EnvWrapper,
)

baseline_algo = {'forest': ForestOptimization, 'gp': GP, 'random_search': RandomSearch}
working_dir = os.environ["WORKING_DIR"]


def main(
    tags, steps, num_runs, save_name, ids, render_overview, seed, save_no_observation, env_args={}
):
    all_results = {}
    for model_id in ids:

        if model_id == "trus-reg":
            vmin = 0.3
            vmax = 0.7
            bnds = ((vmin, vmax), (vmin, vmax), (vmin, vmax), (vmin, vmax), (vmin, vmax))
            x0 = np.random.uniform(vmin, vmax, 5)
            # x0 = 0.5*np.ones(5)
            env = LensEnv(n_actions=5)
            env_wrapper = EnvWrapper(env)
            actions, obs_list, score, r, overview = [], [], [], [], []
            for run_number in range(num_runs):
                obs = env.reset()
                if render_overview:
                    overview.append([])
                env_wrapper.reset()
                minimize(
                    env_wrapper,
                    x0,
                    method='trust-constr',
                    tol=1e-12,
                    options={'maxiter': steps // 6},
                    bounds=bnds,
                )
                actions.append([])
                obs_list.append([])
                r.append([])
                s = env._scores
                while len(s) < steps + 1:
                    score.append(1)
                score.append(s[:steps])

        elif model_id in ["forest", "gp", "random_search"]:
            env = LensEnv(n_actions=5, **env_args)

            env_wrapper = EnvWrapper(env)

            algo = baseline_algo[model_id](n_steps=steps, v_min=0.2, v_max=0.8)
            actions, obs_list, score, r, overview = [], [], [], [], []
            for run_number in range(num_runs):
                obs = env.reset()
                if render_overview:
                    overview.append([])
                env_wrapper.reset()
                x_opt, func_vals, x_iters = algo.align(env_wrapper)
                actions.append([])
                obs_list.append([])
                r.append([])
                score.append(env._scores[1:])
        else:
            print(f"Processing model: {model_id}")
            model_path = np.sort(
                glob.glob(os.path.join(working_dir, f"relign/models/{model_id}/best*"))
            )[0]
            model = PPO.load(model_path)

            env_name = 'la'
            n_stack = 5
            with open(
                os.path.join(working_dir, f"relign/models/{model_id}/env_args.json"), "r"
            ) as file:
                env_args = json.load(file)

            env_args['seed'] = seed
            env = make_stacked_vec_env(
                env_name=env_name,
                n_envs=1,
                n_stack=n_stack,
                score_threshold=0.0,
                max_episode_steps=200,
                env_args=env_args,
            )

            actions, obs_list, score, r, overview = [], [], [], [], []

            for run_number in range(num_runs):
                obs = env.reset()
                score.append(
                    [env.venv.envs[0].env.env.compute_distance_to_gt(obs[0, -1, :, np.newaxis])]
                )
                actions.append([np.zeros((1, 5))])
                obs_list.append([obs[-1, -1]])
                r.append([env.venv.envs[0].env.env.r])
                if render_overview:
                    overview.append([])

                for step in range(steps):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = env.step(action)

                    actions[run_number].append(action)
                    obs_list[run_number].append(obs[-1, -1])
                    score[run_number].append(
                        env.venv.envs[0].env.env.compute_distance_to_gt(obs[0, -1, ..., np.newaxis])
                    )
                    r[run_number].append(env.venv.envs[0].env.env.r)
                    if render_overview:
                        overview[run_number].append(
                            env.venv.envs[0].env.env.render_scene_overview()
                        )

        score = np.array(score)
        print(score.shape)
        r = np.array(r)
        if save_no_observation:
            obs_list = None
        else:
            obs_list = np.array(obs_list)
        actions = np.array(actions)

        best_results = np.minimum.accumulate(score, axis=1)

        all_results[f"{model_id}"] = {
            "score": score,
            "best_results": best_results,
            "mean_results": np.mean(best_results, axis=0),
            "median_results": np.median(best_results, axis=0),
            "quantile_20_results": np.percentile(best_results, 20, axis=0),
            "quantile_80_results": np.percentile(best_results, 80, axis=0),
            "r": r,
            "obs": obs_list,
            "actions": actions,
            "env_args": env_args,
            "overview": overview,
        }

    with open(save_name, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO models and save results.")
    parser.add_argument(
        "--tags", type=str, default="default_tag", help="Tags for identifying the run"
    )
    parser.add_argument("--num_runs", type=int, default=50, help="Numberruns per model")
    parser.add_argument("--steps", type=int, default=15, help="Number of steps per run")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the run")
    parser.add_argument("--noise", type=float, default=0, help="Seed for the run")
    parser.add_argument(
        "--save_name", type=str, default="all_results.pkl", help="File to save results"
    )
    parser.add_argument(
        "--render_overview", action="store_true", help="Render an overview of the results"
    )
    parser.add_argument(
        "--save_no_observation", action="store_true", help="save_no_observations to save space"
    )

    args = parser.parse_args()
    api = wandb.Api()
    runs = api.runs("relign", filters={"tags": {"$eq": str(args.tags)}})

    ids = [run.id for run in runs]
    if args.tags == "baseline":
        ids = ids + ['gp', 'forest', 'random_search']

        for n_lenses in [2, 3]:
            env_args = {
                'n_lenses': n_lenses,
                'noise_angle': args.noise,
                'noise_translation': args.noise,
            }

            main(
                args.tags,
                args.steps,
                args.num_runs,
                f'l_{n_lenses}_n_{args.noise}_{args.save_name}',
                ids,
                args.render_overview,
                args.seed,
                args.save_no_observation,
                env_args=env_args,
            )
    else:
        main(
            args.tags,
            args.steps,
            args.num_runs,
            args.save_name,
            ids,
            args.render_overview,
            args.seed,
            args.save_no_observation,
        )
