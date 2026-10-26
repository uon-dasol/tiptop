"""Calibration frame direction, observability and residual regressions; no devices."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiptop.doosan.calibration import solve_hand_eye, solve_pivot, fit_rigid_datums


def pose(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    value[:3, 3] = translation
    return value


def test_hand_eye_recovers_camera_to_flange_and_rejects_unobservable_samples():
    rng = np.random.default_rng(12)
    parent_camera = pose([0.3, -0.2, 0.1], [0.08, 0.015, -0.03])
    world_board = pose([0.1, 0.2, -0.1], [0.6, 0.1, 0.35])
    parents = np.stack([pose(rng.normal(0, 0.5, 3), rng.uniform([0.2, -0.3, 0.5], [0.7, 0.3, 0.9])) for _ in range(16)])
    boards = np.linalg.inv(parents @ parent_camera) @ world_board
    result = solve_hand_eye(parents, boards)
    np.testing.assert_allclose(result["parent_from_camera"], parent_camera, atol=1e-8)
    assert result["consistent"]
    assert not result["absolute_accuracy_verified"]
    with pytest.raises(ValueError, match="independent axes"):
        solve_hand_eye(np.repeat(parents[:1], 16, axis=0), boards)
    with pytest.raises(ValueError, match="exactly one"):
        solve_hand_eye(parents, boards[:-1])
    # Reversing the board transform must not be accepted as a valid calibration.
    with pytest.raises(ValueError, match="rigid"):
        solve_hand_eye(parents, np.linalg.inv(boards))


def test_pivot_recovers_translation_without_claiming_orientation():
    rng = np.random.default_rng(5)
    point, pivot = np.array([0.02, -0.01, 0.24]), np.array([0.6, 0.1, 0.4])
    matrices = np.stack([pose(rng.normal(0, 0.5, 3), [0, 0, 0]) for _ in range(16)])
    matrices[:, :3, 3] = pivot - matrices[:, :3, :3] @ point
    result = solve_pivot(matrices)
    np.testing.assert_allclose(result["parent_p_tcp_m"], point, atol=1e-10)
    assert result["consistent"] and not result["tcp_orientation_calibrated"]


def test_measured_datums_preserve_handedness_and_reject_collinear_data():
    source = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.02, 0.01, 0.1]])
    expected = pose([0.2, -0.4, 0.1], [0.03, -0.02, 0.05])
    target = source @ expected[:3, :3].T + expected[:3, 3]
    np.testing.assert_allclose(fit_rigid_datums(source, target)["target_from_source"], expected, atol=1e-10)
    with pytest.raises(ValueError, match="non-collinear"):
        fit_rigid_datums(np.zeros((3, 3)), np.zeros((3, 3)))


def test_printable_board_is_detectable_on_the_undistorted_colour_grid():
    from tiptop.scripts.calibrate_m1013 import make_board, detect_board
    from tiptop.perception.cameras.frame import Frame
    import cv2

    spec = dict(columns=7, rows=5, square_m=0.025, marker_m=0.018)
    board = make_board(spec)
    grey = cv2.copyMakeBorder(board.generateImage((1400, 1000)), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    k = np.array([[1400, 0, rgb.shape[1] / 2], [0, 1400, rgb.shape[0] / 2], [0, 0, 1.0]])
    frame = Frame(serial="synthetic-board", timestamp=0.0, rgb=rgb, intrinsics=k)
    transform, rms, count = detect_board(frame, spec)
    assert count >= 6 and rms < 0.1 and transform[2, 3] > 0
