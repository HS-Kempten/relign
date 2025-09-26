import sys
import os
import argparse
import logging
import re
from importlib.resources import files
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


def update_dict(dict_a: dict | MappingProxyType, dict_b: dict | MappingProxyType) -> dict:
    """overrides values in dict_a from dict_b if they're not None and returns updated dict"""
    return {k: v if (dict_b.get(k) is None) else dict_b[k] for k, v in dict_a.items()}


def compute_focal_length(n=1.5168, r1=1.0, r2=-1.0, d=0.27):
    return 1 / ((n - 1) * ((1 / r1) - (1 / r2) + ((n - 1) * d) / (n * r1 * r2)))


def get_gt_filename_from_params(width, height, ref_pattern, sample_count, spec, **kwargs):
    return (
        "gt"
        + "".join([f"__{k}_{v}" for k, v in kwargs.items()])
        + f"__spec_{spec}"
        + f"__ref_pattern_{ref_pattern}"
        + f"__sample_count_{sample_count}"
        + f"__{width}x{height}.npy"
    )


def rmse(arr0, arr1):
    return np.sqrt(np.mean((arr0 - arr1) ** 2))


def plot_scene(
    env,
    action,
    img,
    step,
    r=np.array([]),
    description=True,
    sample_count_overview=512,
    env_overview=None,
):
    score = env.compute_distance_to_gt(img.reshape(env.width, env.height, 1))
    nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 6))

    if env_overview is not None:
        overview_x = env_overview.render_scene_overview(
            axis=[1, 0, 0], sample_count=sample_count_overview, r=r
        )
    else:
        overview_x = env.render_scene_overview(
            axis=[1, 0, 0], sample_count=sample_count_overview, r=r
        )

    axs[0].imshow(overview_x, cmap="gray", vmin=0.0, vmax=0.1)
    axs[0].set_axis_off()
    axs[0].set_title("Lens System with Sensor")
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
            "Relative Movement by RL Agent:",
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
            "Score:",
            fontsize=10,
            ha="center",
            fontweight="bold",
            color="darkblue",
        )

        fig.text(
            0.6,
            0.07,
            "Score Difference to Goal:",
            fontsize=10,
            ha="center",
            fontweight="bold",
            color="darkblue",
        )

        from relign.config import Config

        score_goal_treshold = Config.targets_optimum["la"][env.spec][
            env.width, env.height, env.sample_count
        ]

        cmap = mcolors.LinearSegmentedColormap.from_list("RdGn", ["green", "yellow", "red"])
        norm = mcolors.Normalize(vmin=score_goal_treshold, vmax=0.054)
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
            f"{score - score_goal_treshold:.3f}",
            fontsize=10,
            ha="center",
            color=color,
        )
    return fig


def create_inference_image_series(path_video, path_model, seed, steps, sample_count_overview=512):
    from stable_baselines3 import PPO
    from relign.generator import make_stacked_vec_env
    import json

    model = PPO.load(sorted(path_model.glob("best*"))[0])
    with open(path_model / "env_args.json", "r") as f:
        env_args = json.load(f)

    env_args["hr"] = True

    venv = make_stacked_vec_env(
        env_name="la",
        n_envs=1,
        n_stack=5,
        score_threshold=0.00,
        max_episode_steps=500,
        seed=seed,
        env_args=env_args,
    )

    obs = venv.reset()

    fig = plot_scene(
        env=venv.venv.envs[0].env.env,
        action=np.zeros(5),
        img=obs[-1][-1],
        step=0,
        sample_count_overview=sample_count_overview,
    )

    path_video.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_video / "img_0.png")

    for step in range(1, steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = venv.step(action)

        fig = plot_scene(
            env=venv.venv.envs[0].env.env,
            action=action[0],
            img=obs[-1][-1],
            step=step,
            sample_count_overview=sample_count_overview,
        )
        fig.savefig(path_video / f"img_{step}.png")


def create_interpolated_inference_image_series(
    model_id,
    seed,
    n_interp=10,
    steps=10,
    fps=2,
    sample_count_overview=None,
):
    from stable_baselines3 import PPO
    from relign.generator import make_stacked_vec_env
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    import json

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
    with open(path_model / "env_args.json", "r") as f:
        env_args = json.load(f)
    # env_args["hr"] = True

    venv = make_stacked_vec_env(
        env_name="la",
        n_envs=1,
        n_stack=5,
        score_threshold=0.00,
        max_episode_steps=500,
        seed=seed,
        env_args=env_args,
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

    from relign.generator import LensEnv

    env_overview = LensEnv(**env_args, hr=True)
    env_overview.reset()

    for i, a in enumerate(interpolated_points):
        plot_scene(
            env=env,
            action=actions[int(i / n_interp)],
            img=env._make_image(a),
            step=int(i / n_interp),
            sample_count_overview=sample_count_overview,
            r=a,
            env_overview=env_overview,
        )
        plt.savefig(path / f"img_{i}.png")

    img_series = [f"{path}/img_{i}.png" for i in range(len(interpolated_points))]

    clip = ImageSequenceClip(sequence=img_series, fps=fps)
    clip.write_videofile(f"{path}/inference.mp4", codec="libx264")


def concatenate_images_to_video(path, n_images, fps=2):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    img_series = [f"{path}/img_{i}.png" for i in range(n_images)]
    clip = ImageSequenceClip(sequence=img_series, fps=fps)
    clip.write_videofile(f"{path}/inference.mp4", codec="libx264")


def make_inference_video(model_id="l8aeci4i", seed=42, steps=10):
    path_video = Path(os.environ["WORKING_DIR"]) / "relign" / "videos" / model_id / str(seed)
    path_video.mkdir(parents=True, exist_ok=True)
    path_model = Path(os.environ["WORKING_DIR"]) / "relign" / "models" / model_id

    create_inference_image_series(path_video, path_model, seed, steps, 2**12)
    concatenate_images_to_video(path_video, steps)


def extract_number(path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def make_gif(path_imgs, path_output, duration=25):
    from PIL import Image

    imgs_sorted = sorted(path_imgs.glob('*.png'), key=extract_number)

    images = [Image.open(image_path) for image_path in imgs_sorted]
    images[0].save(
        path_output / "alignment.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def get_lens_specs_from_file(filename="config.txt", dist_indices=(), rgb_idx=-1):
    path = os.path.join(
        str(files("relign")),
        "data",
        "LensEnv",
        filename,
    )
    with open(path) as f:
        lines = f.readlines()

    refr_idc = _extract_refractive_indices(lines, rgb_idx=rgb_idx)
    obj_dists = _extract_object_distances(lines, dist_indices)

    return refr_idc, obj_dists


def _extract_refractive_indices(lines, rgb_idx=-1):
    """
    Extracts the refractive indices from the given lines of a file.
    rgb_idx = 1 for red, 2 for green, 3 for blue.
    rgb_idx = -1 for monochromatic file.
    """
    refr_idx_start = 0
    refr_idx_end = 0
    for i, f in enumerate(lines):
        if f == "REFRACTIVE INDICES\n":
            refr_idx_start = i + 2
        if "SOLVES" in f:
            refr_idx_end = i - 1
            break
    return [
        float([f.rstrip() for f in lens.split(" ") if f][rgb_idx])
        for lens in lines[refr_idx_start:refr_idx_end]
    ]


def _extract_object_distances(lines, dist_indices):
    """returns distances from given indices. If no indices given, return all"""
    dists_idx_start = 0
    dists_idx_end = 0
    for i, f in enumerate(lines):
        if "RDY" in f:
            dists_idx_start = i + 1
        if "SPECIFICATION" in f and not dist_indices:
            dists_idx_end = i - 1
            dist_indices = np.arange(1, dists_idx_end - dists_idx_start)
            break
    return [
        float([s for s in lines[dists_idx_start + dist_idx].split(" ") if s][2])
        for dist_idx in dist_indices
    ]


def plot_dual_view(env, sample_count=1024, return_fig=False):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))

    axs[0].imshow(env.render_scene_overview(sample_count=sample_count), cmap="gray")
    axs[0].set_axis_off()
    axs[0].set_title("Lens System")

    axs[1].imshow(env._make_image(env.fill_action_with_optimal_values()))
    axs[1].set_axis_off()
    axs[1].set_title("Sensor")

    return fig if return_fig else None


def construct_mitsuba_rotation_matrix(angles):
    # Construct rotation matrix by multiplying the rotation matrices of all axes
    import mitsuba as mi

    rotation_matrix = np.identity(3)
    for i, angle in enumerate(angles):
        axis = np.eye(3)[i]
        R = np.array(mi.scalar_rgb.Transform4f().rotate(axis=axis, angle=angle).matrix)[:3, :3].T
        rotation_matrix = rotation_matrix @ R
    return rotation_matrix


def sensor_size_min(fov, focal_length, sensor_resolution):
    """
    Calculates the minimum sensor size to catch all the light rays from given lens with field of
    view `fov` and focal length `focal_length`. Assuming to work only with rectangle sensors.
    """

    min_width_height = np.array(
        [2 * focal_length * np.tan(np.radians(fov / 2.0)) for _ in range(2)]
    )

    resolution_rate = max(sensor_resolution) / min(sensor_resolution)

    min_width_height[np.argmax(sensor_resolution)] *= resolution_rate

    return min_width_height


def pixel_pitch(size, resolution):
    return size[0] / resolution[0]


def emitter_distance_min(pixel_pitch, alpha, focal_length):
    """Calculates minimum emitter distance."""
    safety_factor = 3.0
    return (
        (pixel_pitch + 2 * np.tan(np.radians(alpha)) * focal_length)
        * focal_length
        * -safety_factor
        / pixel_pitch
    )


def emitter_size(ref_pattern_res, fov, emitter_distance):
    """
    Calculates the physical size (width and height) of the emitter based on the reference pattern
    resolution, field of view (fov), and emitter distance.

    Parameters:
        ref_pattern_res (tuple or list): Resolution of the reference pattern as (width, height).
        fov (float): Field of view in degrees.
        emitter_distance (float): Distance from the emitter to the sensor or lens.

    Returns:
        np.ndarray: Array containing the emitter's width and height.
    """
    beta = np.rad2deg(np.arctan(ref_pattern_res[1] / ref_pattern_res[0]))
    d_l = abs(emitter_distance) * np.tan(np.radians(fov / 2.0))

    return np.array([2 * d_l * np.cos(np.radians(beta)), 2 * d_l * np.cos(np.radians(90 - beta))])


def scene_objects(fov, focal_length, sensor_resolution, exit_pupil_angle, ref_pattern_res):
    sensor_dims = sensor_size_min(fov, focal_length, sensor_resolution)
    pp = pixel_pitch(size=sensor_dims, resolution=sensor_resolution)
    emitter_dist = emitter_distance_min(pp, exit_pupil_angle, focal_length)
    return {
        "sensor_scale": sensor_dims,
        "emitter_distance": emitter_dist,
        "emitter_scale": emitter_size(ref_pattern_res, fov, emitter_dist),
    }


def clean_up_drjit():
    import drjit as dr
    import gc

    gc.collect()
    gc.collect()

    dr.kernel_history_clear()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()
