"""Camera extrinsics shared by wrist and fixed observation sources.

Transforms use column vectors: T_A_B maps B coordinates into A. Camera axes must
already use the optical convention of Frame intrinsics (+Z depth, +Y down).
"""

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
    """Resolve calibrated camera pose without assuming where it is mounted.

    wrist: calibration is T_ee_camera, combined with FK at observation time.
    fixed: calibration is T_world_camera; robot motion must not alter it.
    Neither mode supplies or estimates an object's pose.
    """
    calibration = _rigid_transform(parent_from_camera, "camera calibration")
    if mount == "fixed":
        return calibration
    if mount != "wrist":
        raise ValueError(f"Unknown camera mount {mount!r}; choose 'wrist' or 'fixed'")
    if world_from_ee is None:
        raise ValueError("A wrist camera requires the current world_from_ee from robot FK")
    return _rigid_transform(world_from_ee, "world_from_ee") @ calibration
