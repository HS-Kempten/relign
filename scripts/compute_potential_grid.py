import numpy as np
import pandas as pd
from relign.helpers import rmse
from relign.generator import LensEnv, GymnasiumEnv
from itertools import product, combinations

def compute_distance_to_gt(img, env):
    return rmse(img[0, :, :], env.env._gt[:, :, 0])

if __name__ == '__main__':

    env = GymnasiumEnv(
        env_cls=LensEnv,
        spec="od",
        noise_objects=0.0,
        noise_movement=0.0,
        n_actions=5,
        width=200,
        height=200,
        sample_count=64,
    )
    env.reset()

    distances = np.linspace(0.1, 0.9, 11)

    for idx_x, idx_y in combinations(range(5), 2):
        potentials = []
        positions = []
        for x, y in product(distances, repeat=2):

            pos = 0.5 * np.ones(5)
            pos[idx_x] = x
            pos[idx_y] = y
            a = env.env._compute_move_to_position(pos)
            img, _, _, _, _ = env.step(a)
            print(env.env.r)

            potential = compute_distance_to_gt(img, env)
            potentials.append(potential)
            positions.append(pos)

        df_potentials = pd.DataFrame(
            data=positions,
            columns=[f"a{i}" for i in range(5)],
        )
        df_potentials['potential'] = potentials
        df_potentials.to_csv(f'potential_grid_{idx_x}_{idx_y}.csv')
