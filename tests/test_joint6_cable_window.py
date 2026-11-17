import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from tiptop.doosan.known_task import (
    FINGER_SYMMETRY_RAD,
    first_successful_solutions,
    fold_j6_by_symmetry,
    j6_symmetry_shifts,
    turn_grasps,
    z_rotations,
)
from tiptop.doosan.model import ASSETS

LAB_HOME = Path(__file__).resolve().parents[1] / "simulation/isaac_sim/config/lab_home.json"


def joint_limits_deg(joint):
    limit = ET.parse(ASSETS / "m1013_uon.urdf").getroot().find(f"joint[@name='{joint}']/limit")
    return math.degrees(float(limit.get("lower"))), math.degrees(float(limit.get("upper")))


def joint_6_limits_deg():
    return joint_limits_deg("joint_6")


def test_joint_6_window_is_the_lab_home_angle_plus_minus_180_degrees():
    home = json.loads(LAB_HOME.read_text())["joint_positions_deg"][5]
    lower, upper = joint_6_limits_deg()
    assert lower == pytest.approx(home - 180, abs=1e-3)
    assert upper == pytest.approx(home + 180, abs=1e-3)


@pytest.mark.parametrize("index", [3, 4])
def test_wrist_joints_are_limited_to_the_lab_home_angle_plus_minus_90_degrees(index):
    home = json.loads(LAB_HOME.read_text())["joint_positions_deg"][index]
    lower, upper = joint_limits_deg(f"joint_{index + 1}")
    assert lower == pytest.approx(home - 90, abs=1e-3)
    assert upper == pytest.approx(home + 90, abs=1e-3)


def test_symmetry_shift_moves_j6_within_sixty_degrees_of_the_window_middle():
    limits = np.deg2rad([-88.62, 271.38])
    neutral = limits.mean()
    j6 = np.deg2rad(np.linspace(-88, 271, 50))
    shifts = j6_symmetry_shifts(j6, limits)
    np.testing.assert_allclose(np.round(shifts / FINGER_SYMMETRY_RAD), shifts / FINGER_SYMMETRY_RAD, atol=1e-9)
    assert np.abs(j6 + shifts - neutral).max() <= FINGER_SYMMETRY_RAD / 2 + 1e-9


def test_z_rotations_turn_the_tcp_about_its_own_axis():
    rotated = np.eye(4)[None] @ z_rotations([np.pi / 2])
    np.testing.assert_allclose(rotated[0, :3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_folded_ik_solutions_reach_the_turned_grasp_and_stay_near_the_window_middle():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Planning GPU required for IK")
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose

    from tiptop.doosan.model import get_m1013_uon_ik_solver

    far_box = Cuboid(name="far", dims=[0.1, 0.1, 0.1], pose=[5.0, 5.0, 5.0, 1, 0, 0, 0])
    ik = get_m1013_uon_ik_solver(WorldConfig(cuboid=[far_box]), num_seeds=16)
    limits = ik.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
    rng = np.random.default_rng(0)
    q = rng.uniform(np.deg2rad([60, -40, 60, -30, 60, 0]), np.deg2rad([120, 0, 120, 30, 100, 250]), (24, 6))
    goals = ik.kinematics.get_state(torch.as_tensor(q, dtype=torch.float32, device="cuda")).ee_pose.get_matrix()
    solved = ik.solve_batch(Pose.from_matrix(goals), return_seeds=4)
    before = first_successful_solutions(solved)[:, 5].clone()
    shifts = fold_j6_by_symmetry(ik, solved, limits[:, 5])
    ok = solved.success.any(dim=1).cpu().numpy()
    assert ok.sum() >= 12
    after = first_successful_solutions(solved)
    np.testing.assert_allclose((after[:, 5] - before).cpu().numpy()[ok], shifts[ok], atol=1e-5)
    neutral = limits[:, 5].mean()
    assert np.abs(after[:, 5].cpu().numpy()[ok] - neutral).max() <= FINGER_SYMMETRY_RAD / 2 + 1e-3
    turned = goals.cpu().numpy() @ z_rotations(shifts)
    reached = ik.kinematics.get_state(after).ee_pose.get_matrix().cpu().numpy()
    np.testing.assert_allclose(reached[ok], turned[ok], atol=2e-3)


def test_turn_grasps_keeps_a_gpu_tensor_on_its_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    grasps = torch.eye(4, device="cuda", dtype=torch.float32).repeat(3, 1, 1)
    turned = turn_grasps(grasps, np.array([0.0, np.pi / 2, -np.pi / 2]))
    assert turned.device == grasps.device and turned.dtype == grasps.dtype
    assert torch.equal(turned[0], grasps[0])
    np.testing.assert_allclose(turned[1, :3, 0].cpu().numpy(), [0, 1, 0], atol=1e-6)
