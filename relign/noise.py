from relign.helpers import construct_mitsuba_rotation_matrix
import numpy as np


def gaussian_trans_rot(vertices, rng, pos_z, scale_rot=2, scale_trans=0.05, scale_factor=0.5):
    angles_xy = rng.normal(loc=0, scale=scale_rot * scale_factor, size=2)
    translation_xyz = np.hstack(
        [
            rng.normal(loc=0, scale=scale_trans * scale_factor, size=2),
            np.zeros(1),
        ]
    )
    # translation in origin
    vertices = vertices - [0, 0, pos_z]

    # apply rotation noise
    rotation_matrix = construct_mitsuba_rotation_matrix(angles_xy)
    vertices_rotated = vertices @ rotation_matrix.T

    # apply translation noise
    vertices_translated = vertices_rotated + translation_xyz

    # translate back to starting position
    vertices_origin = vertices_translated + [0, 0, pos_z]

    return vertices_origin
