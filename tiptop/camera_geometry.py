"""Camera transforms usable before loading simulator or device dependencies."""

import numpy as np


def _rigid_transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix.copy()


def camera_world_pose(parent_from_camera, *, mount="wrist", world_from_ee=None):
    """Resolve optical camera pose using column-vector transforms T_A_B.

    wrist: calibration is T_ee_camera, combined with FK at observation time.
    fixed: calibration is T_world_camera; robot motion must not alter it.
    Camera axes must match Frame intrinsics (+Z depth, +Y down).
    """
    calibration = _rigid_transform(parent_from_camera, "camera calibration")
    if mount == "fixed":
        return calibration
    if mount != "wrist":
        raise ValueError(f"Unknown camera mount {mount!r}; choose 'wrist' or 'fixed'")
    if world_from_ee is None:
        raise ValueError("A wrist camera requires the current world_from_ee from robot FK")
    return _rigid_transform(world_from_ee, "world_from_ee") @ calibration


def fixed_camera_envelopes(profile):
    """Camera body/bracket boxes in base_link using the declared physical rig.

    The rig is at the body's front-face centre (+Z forward, +X along its width).
    It is separate from the colour optical calibration and image rotation.
    Dimensions/offsets/margin must already be validated by the profile loader.
    """
    base_from_rig = np.asarray(profile["parent_from_rig"], float)
    margin = float(profile["collision_margin_m"])
    result = []
    for name, dims, offset in [
        ("fixed_camera_body", profile["body_dims_m"], [0, 0, -profile["body_dims_m"][2] / 2]),
        ("fixed_camera_bracket", profile["bracket_dims_m"], profile["rig_from_bracket_xyz_m"]),
    ]:
        transform = base_from_rig.copy()
        transform[:3, 3] += transform[:3, :3] @ np.asarray(offset)
        result.append(
            dict(name=name, dims_m=(np.asarray(dims) + 2 * margin).tolist(), base_from_obstacle=transform.tolist())
        )
    return result
