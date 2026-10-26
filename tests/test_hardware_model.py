"""Measured frame application and original-asset preservation; no hardware access."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiptop.doosan.hardware import derive_mount_urdf, load_hardware_profile, prepare_hardware_model

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tiptop/doosan/assets/m1013_uon"


def matrix(r, t):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(r).as_matrix()
    value[:3, 3] = t
    return value


def synthetic_profile(tmp_path):
    calibration = dict(
        calibrated=True,
        consistent=True,
        world_frame="base_link",
        parent_link="link_6",
        camera_frame="undistorted_color_optical",
        camera_serial="SYNTHETIC-NO-DEVICE",
        parent_from_camera=np.eye(4).tolist(),
    )
    (tmp_path / "hand-eye.json").write_text(json.dumps(calibration))
    profile = json.loads((ROOT / "tiptop/config/m1013_hardware.example.json").read_text())
    profile.update(calibrated=True, camera_calibration="hand-eye.json", provenance="SYNTHETIC TEST ONLY")
    profile["tool"] = dict(
        calibrated=True,
        gripper_geometry_matches_reference=True,
        link_6_from_gripper_body=matrix([0.1, 0, 0.2], [0.01, 0.02, 0.03]).tolist(),
        gripper_body_from_tcp=matrix([0, -0.2, 0.1], [0.01, -0.02, 0.25]).tolist(),
        db_grasp_from_tcp=matrix([0.1, 0.2, 0], [0.005, 0, 0.002]).tolist(),
    )
    profile["camera_profile"] = json.loads((ROOT / "simulation/isaac_sim/config/wrist_camera.json").read_text())[
        "profile"
    ]
    profile["camera_profile"]["mount_calibrated"] = True
    profile["workspace"] = dict(
        calibrated=True,
        fixed_obstacles=[
            dict(
                name="full_table",
                dims_m=[0.8, 0.8, 0.04],
                base_from_obstacle=matrix([0, 0, 0], [0.65, 0, 0.38]).tolist(),
            )
        ],
    )
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(profile))
    return path


def test_unmeasured_profile_is_rejected_without_loading_any_device():
    with pytest.raises(ValueError, match="completed measured"):
        load_hardware_profile(ROOT / "tiptop/config/m1013_hardware.example.json")


def test_derived_mount_and_tcp_are_named_transforms_and_original_is_unchanged(tmp_path):
    from yourdfpy import URDF

    source = ASSETS / "m1013_uon.urdf"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mount = matrix([0.1, -0.2, 0.3], [0.01, 0.02, 0.03])
    tcp = matrix([-0.1, 0.2, 0.1], [0, 0.01, 0.25])
    destination = derive_mount_urdf(
        source, tmp_path / "model.urdf", link_6_from_gripper_body=mount, gripper_body_from_tcp=tcp
    )
    robot = URDF.load(str(destination), load_meshes=False)
    np.testing.assert_allclose(robot.get_transform("gripper_body", "link_6"), mount, atol=1e-9)
    np.testing.assert_allclose(robot.get_transform("gripper_tcp", "gripper_body"), tcp, atol=1e-9)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="separate URDF"):
        derive_mount_urdf(source, source, link_6_from_gripper_body=mount, gripper_body_from_tcp=tcp)


def test_measured_tcp_tool_spheres_and_camera_model_share_the_derived_frames(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Planning GPU required for model preparation")
    from scipy.spatial import cKDTree
    from yourdfpy import URDF

    profile = load_hardware_profile(synthetic_profile(tmp_path))
    prepared = prepare_hardware_model(profile, tmp_path / "derived")
    original = URDF.load(str(ASSETS / "m1013_uon.urdf"), load_meshes=False)
    body_from_old_tcp = original.get_transform("gripper_tcp", "gripper_body")
    body_from_new_tcp = np.asarray(profile["tool"]["gripper_body_from_tcp"])
    new_from_old = np.linalg.inv(body_from_new_tcp) @ body_from_old_tcp
    spheres = np.load(ASSETS / "gripper_spheres_open.npy")
    expected = spheres.copy()
    expected[:, :3] = spheres[:, :3] @ new_from_old[:3, :3].T + new_from_old[:3, 3]
    assert prepared["gripper_spheres"].shape == expected.shape
    assert cKDTree(expected).query(prepared["gripper_spheres"])[0].max() < 1e-6
    np.testing.assert_allclose(prepared["grasp_from_tcp"], profile["tool"]["db_grasp_from_tcp"])
    assert prepared["fixed_obstacles"][0].name == "configured_full_table"
    names = prepared["robot_config"]["robot_cfg"]["kinematics"]["collision_link_names"]
    assert "wrist_camera_body" in names and "wrist_camera_bracket" in names


def test_physical_plan_is_bound_to_measured_mount_and_optical_calibration(tmp_path):
    from tiptop.doosan.client import validate_plan_backend
    from tiptop.doosan.hardware import hardware_profile_fingerprint

    path = synthetic_profile(tmp_path)
    plan = dict(
        robot="m1013_uon",
        robot_system=0,
        hardware_profile=str(path),
        hardware_profile_sha256=hardware_profile_fingerprint(load_hardware_profile(path)),
    )
    state = dict(success=True, robot_system=0, gripper=dict(backend="uon_native"))
    validate_plan_backend(plan, state)
    calibration = json.loads((tmp_path / "hand-eye.json").read_text())
    calibration["parent_from_camera"][0][3] += 0.01
    (tmp_path / "hand-eye.json").write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="changed after planning"):
        validate_plan_backend(plan, state)
