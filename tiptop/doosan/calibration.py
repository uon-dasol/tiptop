"""Offline M1013 calibration math. Transforms map child coordinates to parent.

Hand-eye inputs are world_from_parent and camera_from_board, with a stationary
board and camera rigidly attached to that parent (normally link_6). No robot or
camera is opened here. A small residual is a consistency check, not an absolute
accuracy certificate or permission to move hardware.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tiptop.perception.cameras.geometry import _rigid_transform


def _poses(values, name, minimum=6):
    poses = np.asarray([_rigid_transform(value, name) for value in values])
    if len(poses) < minimum:
        raise ValueError(f"{name} needs at least {minimum} poses")
    return poses


def _rotation_excitation(poses):
    rotations = poses[0, :3, :3].T @ poses[:, :3, :3]
    vectors = Rotation.from_matrix(rotations).as_rotvec()
    singular = np.linalg.svd(vectors, compute_uv=False)
    if singular[1] < np.deg2rad(10):
        raise ValueError("Calibration needs rotations about at least two independent axes")
    return singular.tolist()


def solve_hand_eye(world_from_parent, camera_from_board):
    """Return parent_from_camera and stationary-board consistency residuals.

    OpenCV PARK solves the eye-in-hand AX=XB problem. The camera images must
    already use the declared pinhole K; do not undistort an undistorted image a
    second time. Units of both input translations must be metres.
    """
    parents = _poses(world_from_parent, "world_from_parent", minimum=10)
    boards = _poses(camera_from_board, "camera_from_board", minimum=10)
    if len(parents) != len(boards):
        raise ValueError("Each robot pose needs exactly one board pose from the same observation")
    excitation = _rotation_excitation(parents)
    rotation, translation = cv2.calibrateHandEye(
        list(parents[:, :3, :3]),
        list(parents[:, :3, 3]),
        list(boards[:, :3, :3]),
        list(boards[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation.reshape(3)
    transform = _rigid_transform(transform, "estimated parent_from_camera")
    fixed = parents @ transform @ boards
    mean_position = fixed[:, :3, 3].mean(0)
    mean_rotation = Rotation.from_matrix(fixed[:, :3, :3]).mean()
    position_errors = np.linalg.norm(fixed[:, :3, 3] - mean_position, axis=1)
    rotation_errors = (mean_rotation.inv() * Rotation.from_matrix(fixed[:, :3, :3])).magnitude()
    return dict(
        parent_from_camera=transform.tolist(),
        samples=len(parents),
        method="OpenCV PARK",
        translation_units="metres",
        rotation_excitation_singular_values_rad=excitation,
        board_position_rms_m=float(np.sqrt(np.mean(position_errors**2))),
        board_position_max_m=float(position_errors.max()),
        board_rotation_max_deg=float(np.rad2deg(rotation_errors.max())),
        consistent=bool(position_errors.max() <= 0.005 and rotation_errors.max() <= np.deg2rad(1)),
        absolute_accuracy_verified=False,
    )


def solve_pivot(world_from_parent):
    """Estimate a fixed TCP point by pivoting about a stationary contact point.

    Solves R_i * parent_p_tcp + t_i = world_p_pivot. This estimates translation
    only; the TCP orientation and DB-grasp alignment still need measured datums.
    """
    poses = _poses(world_from_parent, "world_from_parent", minimum=10)
    _rotation_excitation(poses)
    matrix = np.concatenate([poses[:, :3, :3], np.broadcast_to(-np.eye(3), (len(poses), 3, 3))], axis=2)
    matrix = matrix.reshape(-1, 6)
    target = -poses[:, :3, 3].reshape(-1)
    solution, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != 6 or singular[-1] / singular[0] < 1e-4:
        raise ValueError("Pivot poses are degenerate; use a wider range of independent orientations")
    points = np.einsum("nij,j->ni", poses[:, :3, :3], solution[:3]) + poses[:, :3, 3]
    errors = np.linalg.norm(points - solution[3:], axis=1)
    return dict(
        parent_p_tcp_m=solution[:3].tolist(),
        world_p_pivot_m=solution[3:].tolist(),
        rms_m=float(np.sqrt(np.mean(errors**2))),
        maximum_error_m=float(errors.max()),
        consistent=bool(errors.max() <= 0.002),
        tcp_orientation_calibrated=False,
    )


def fit_rigid_datums(source_points, target_points):
    """Fit target_from_source from at least three measured, non-collinear datums."""
    source, target = np.asarray(source_points, float), np.asarray(target_points, float)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape or len(source) < 3:
        raise ValueError("Matching source/target datum points must have shape [N,3], N>=3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Datum points must be finite metres")
    left, right = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(left, tol=1e-8) < 2 or np.linalg.matrix_rank(right, tol=1e-8) < 2:
        raise ValueError("Datum points must be non-collinear")
    u, _, vt = np.linalg.svd(left.T @ right)
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ sign @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return dict(
        target_from_source=transform.tolist(),
        rms_m=float(np.sqrt(np.mean(errors**2))),
        maximum_error_m=float(errors.max()),
        translation_units="metres",
    )
