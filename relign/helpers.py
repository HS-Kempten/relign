import sys
import os
import argparse
import logging
from pathlib import Path
from types import MappingProxyType
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time


def get_device():
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        return "mps"

    if sys.platform == "linux" and torch.cuda.is_available():
        return torch.device("cuda:0")

    return "cpu"


def plot_trajectory(trajectory, optimum):
    matplotlib.use('agg')

    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = plt.get_cmap("coolwarm")

    ax.scatter(optimum[0], optimum[1], color="black", s=500, marker="x")

    for i in range(len(trajectory) - 1):
        color = cmap(i / len(trajectory))
        ax.scatter(
            trajectory[i][0],
            trajectory[i][1],
            color=color,
            s=5,
            marker="o",
        )
        ax.arrow(
            trajectory[i][0],
            trajectory[i][1],
            trajectory[i + 1][0] - trajectory[i][0],
            trajectory[i + 1][1] - trajectory[i][1],
            shape="full",
            color=color,
            lw=1,
            alpha=i / len(trajectory),
            length_includes_head=True,
            head_width=0.05,
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    plt.close()  # see `figure.max_open_warning`
    return fig


def save_torch_model(
    model,
    reward,
    n_episodes,
    hparams,
):
    path_data = Path(str(os.getenv("WORKING_DIR"))) / "relign" / "models"
    str_hparams = "__".join([f"{k}_{v}" for k, v in hparams.items()])
    model_name = (
        f"model__rew_{reward}__" f"hparams__{str_hparams}__" f"n_episodes__{n_episodes}" ".pt"
    )

    torch.save(model.state_dict(), path_data / model_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-log',
        '--loglevel',
        default='info',
        help='Provide logging level. Example --loglevel debug, default=info',
    )
    parser.add_argument(
        '--n_episodes',
        default=20000,
        help='Sets n_episodes. Example --n_episodes 200, default=20000',
    )
    parser.add_argument(
        '--n_actions', default=5, help='Sets n_actions. Example --n_actions 2, default=5'
    )
    parser.add_argument(
        '--logging',
        default='True',
        help='Flag to enable logging by wandb. Example --logging False, default=True',
    )
    parser.add_argument(
        '--env',
        default='gaussian_intensity',
        help='Sets environment. Example --env lens, default=gaussian_intensity',
    )
    return parser.parse_args()


def setup_logger(loglevel):
    if not os.getenv("WORKING_DIR"):
        raise ValueError("os.env `WORKING_DIR` is needed for logging, but not set")

    log_file_name = f"{time.strftime('%Y_%m_%d__%H_%M_%S')}"
    log_file_path = Path(str(os.getenv("WORKING_DIR"))) / "relign" / "logs" / log_file_name
    log_file_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=f"{log_file_path}.log",
        level=loglevel,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def plot_log_densities(log_densities):
    matplotlib.use('agg')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.arange(len(log_densities)), log_densities)
    ax.set_xlabel('steps in one episode')
    ax.set_ylabel('log_densities')
    plt.close()
    return fig


def compute_z_positions_scene_objects(n_lenses=3, dist_lenses=2, dist_sensor=5):
    z_lenses = np.arange(0, np.round(dist_lenses * n_lenses, 3), dist_lenses) - (
        dist_lenses * (n_lenses - 1) / 2
    )
    r = {"lenses": z_lenses}
    r["sensor"] = z_lenses[-1] + dist_sensor
    return r


def update_dict(dict_a: dict | MappingProxyType, dict_b: dict | MappingProxyType) -> dict:
    """overrides values in dict_a from dict_b if they're not None and returns updated dict"""
    return {k: v if (dict_b.get(k) is None) else dict_b[k] for k, v in dict_a.items()}


def clean_up():
    import drjit as dr
    import gc

    gc.collect()
    gc.collect()

    dr.kernel_history_clear()
    dr.flush_malloc_cache()
    dr.malloc_clear_statistics()
    dr.flush_kernel_cache()

    """ this seems to break drjit with cuda at some point
    if hasattr(dr, 'sync_thread'):
        dr.sync_thread()
        dr.registry_clear()
        dr.set_flags(dr.JitFlag.Default)
    """


def compute_focal_length(n=1.5168, r1=1.0, r2=-1.0, d=0.27):
    return 1 / ((n - 1) * ((1 / r1) - (1 / r2) + ((n - 1) * d) / ((n * r1 * r2))))


def get_gt_filename_from_params(n_lenses, n_samples, width, height):
    return f'gt__n_lenses_{n_lenses}__n_samples_{n_samples}__{width}x{height}.npy'


def rmse(arr0, arr1):
    return np.sqrt(np.mean((arr0 - arr1) ** 2))


def plot_scene(env, action, img, step, r=np.array([]), sample_count=512, description=True):
    score = env.compute_distance_to_gt(img.reshape(50, 50, 1))
    nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 6))

    overview_x = env.render_scene_overview(
        sensor_distance=4, axis=[1, 0, 0], sample_count=sample_count, r=r
    )
    axs[0].imshow(overview_x, cmap="gray", vmin=0.0, vmax=0.1)
    axs[0].set_axis_off()
    axs[0].set_title("Scene Viewed Along Z-axis")
    axs[1].imshow(img, cmap="inferno")
    axs[1].set_axis_off()
    axs[1].set_title("Irradiance Measured on Sensor")
    axs[2].imshow(env._gt, cmap="inferno")
    axs[2].set_axis_off()
    axs[2].set_title("Reference Pattern")

    action_str = np.array2string(action, precision=3, floatmode='fixed')

    action_str = (
        f"Tx: {action[0]: .2f}, "
        f"Ty: {action[1]: .2f}, "
        f"Tz: {action[2]: .2f}, "
        f"Rx: {action[3]: .2f}, "
        f"Ry: {action[4]: .2f}"
    )

    if description:
        fig.suptitle(
            "Active Alignment by Reinforcement Learning Agent", fontsize=16, fontweight="bold"
        )

        fig.text(
            0.5,
            0.88,
            f"Alignment Step: {step}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            color="black",
        )

        fig.text(
            0.5,
            0.15,
            "Relative Movement:",
            fontsize=10,
            fontweight="bold",
            ha="center",
            color="darkblue",
        )
        fig.text(
            0.5,
            0.12,
            f"{action_str}",
            fontsize=10,
            ha="center",
            color="darkblue",
        )

        fig.text(
            0.4,
            0.07,
            "Score (RMSE):",
            fontsize=10,
            ha="center",
            fontweight="bold",
            color="darkblue",
        )

        fig.text(
            0.6,
            0.07,
            "Difference to Render Variance:",
            fontsize=10,
            ha="center",
            fontweight="bold",
            color="darkblue",
        )

        cmap = mcolors.LinearSegmentedColormap.from_list("RdGn", ["green", "yellow", "red"])
        norm = mcolors.Normalize(vmin=0.025, vmax=0.20)
        color = cmap(norm(score))

        fig.text(
            0.4,
            0.04,
            f"{score:.3f}",
            fontsize=10,
            ha="center",
            color=color,
        )

        fig.text(
            0.6,
            0.04,
            f"{score - 0.0243:.3f}",
            fontsize=10,
            ha="center",
            color=color,
        )
    return fig


def create_inference_image_series(path_video, path_model, seed, steps):
    from stable_baselines3 import PPO
    from relign.generator import make_stacked_vec_env

    model = PPO.load(sorted(path_model.glob("best*"))[0])
    venv = make_stacked_vec_env(
        env_name="la",
        n_envs=1,
        n_stack=5,
        score_threshold=0.00,
        max_episode_steps=500,
        seed=seed,
    )
    obs = venv.reset()

    fig = plot_scene(
        env=venv.venv.envs[0].env.env,
        action=np.zeros(5),
        img=obs[-1][-1],
        step=0,
    )
    fig.savefig(path_video / "img_0.png")
    for step in range(1, steps + 1):

        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = venv.step(action)

        fig = plot_scene(
            env=venv.venv.envs[0].env.env,
            action=action[0],
            img=obs[-1][-1],
            step=step,
        )
        fig.savefig(path_video / f"img_{step}.png")


def create_interpolated_inference_image_series(
    model_id,
    seed,
    n_interp=10,
    steps=10,
    fps=2,
):
    from stable_baselines3 import PPO
    from relign.generator import make_stacked_vec_env
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    path_model = Path(os.environ["WORKING_DIR"]) / "relign" / "models" / model_id
    path = (
        Path(os.environ["WORKING_DIR"])
        / "relign"
        / "videos"
        / model_id
        / str(seed)
        / f"interpolated_{n_interp}"
    )
    path.mkdir(parents=True, exist_ok=True)

    model = PPO.load(sorted(path_model.glob("best*"))[0])
    venv = make_stacked_vec_env(
        env_name="la",
        n_envs=1,
        n_stack=5,
        score_threshold=0.00,
        max_episode_steps=500,
        seed=seed,
    )

    obs = venv.reset()

    actions = []
    states = [venv.venv.envs[0].env.env.r]
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = venv.step(action)
        actions.append(action[0])
        states.append(venv.venv.envs[0].env.env.r)

    points = states
    all_interpolated_points = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i + 1]
        interpolated_points = np.linspace(point1, point2, n_interp + 2)
        interpolated_points = interpolated_points[1:-1]
        all_interpolated_points.append(interpolated_points)
    interpolated_points = np.vstack(all_interpolated_points)

    env = venv.venv.envs[0].env.env

    for i, a in enumerate(interpolated_points):
        plot_scene(
            env=env,
            action=actions[int(i / n_interp)],
            img=env._make_image(a),
            step=int(i / n_interp),
            sample_count=512,
            r=a,
        )
        plt.savefig(path / f"img_{i}.png")

    img_series = [f"{path}/img_{i}.png" for i in range(len(interpolated_points))]

    clip = ImageSequenceClip(sequence=img_series, fps=fps)
    clip.write_videofile(f"{path}/inference.mp4", codec="libx264")


def concatenate_images_to_video(path, n_images):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    img_series = [f"{path}/img_{i}.png" for i in range(n_images)]
    clip = ImageSequenceClip(sequence=img_series, fps=2)
    clip.write_videofile(f"{path}/inference.mp4", codec="libx264")


def make_inference_video(model_id="l8aeci4i", seed=42, steps=10):

    path_video = Path(os.environ["WORKING_DIR"]) / "relign" / "videos" / model_id / str(seed)
    path_video.mkdir(parents=True, exist_ok=True)
    path_model = Path(os.environ["WORKING_DIR"]) / "relign" / "models" / model_id

    create_inference_image_series(path_video, path_model, seed, steps)
    concatenate_images_to_video(path_video, steps)
