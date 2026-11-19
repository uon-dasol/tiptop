"""Measured frame application and original-asset preservation; no hardware access."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiptop.doosan import hardware as hardware_module
from tiptop.doosan.hardware import (
    cached_hardware_model,
    derive_mount_urdf,
    load_hardware_profile,
    prepare_hardware_model,
    validate_hardware_observation,
)
from tiptop.robot_limits import DEFAULTS, save_robot_limits

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tiptop/doosan/assets/m1013_uon"


def matrix(r, t):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(r).as_matrix()
    value[:3, 3] = t
    return value


def synthetic_profile(tmp_path, *, mount="wrist"):
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
    if mount == "fixed":
        calibration.update(
            parent_link="base_link", image_rotation_deg=180,
            parent_from_camera=matrix([0.2, 0.1, 0], [0.01, 0.7, 1.02]).tolist(),
        )
        (tmp_path / "hand-eye.json").write_text(json.dumps(calibration))
        profile["camera_profile"].update(
            mount="fixed", parent_link="base_link",
            parent_from_rig=matrix([0, np.pi, 0.3], [0, 0.7, 1.02]).tolist(),
        )
        profile["workspace"]["pick_place_bounds_xy_m"] = [[-0.3, 0.5], [0.3, 1.1]]
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.mark.parametrize("mount", ["wrist", "fixed"])
def test_unmeasured_profile_is_rejected_without_loading_any_device(tmp_path, mount):
    path = synthetic_profile(tmp_path, mount=mount)
    document = json.loads(path.read_text())
    document["calibrated"] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="completed measured"):
        load_hardware_profile(path)


def test_assumed_geometry_is_explicitly_planning_only(tmp_path):
    from tiptop.camera_geometry import fixed_camera_envelopes

    path = synthetic_profile(tmp_path, mount="fixed")
    document = json.loads(path.read_text())
    expected_boxes = fixed_camera_envelopes(document["camera_profile"])
    document.update(calibrated=False, geometry_source="assumed_lab_model")
    document["tool"].update(calibrated=False, gripper_geometry_matches_reference=False)
    document["workspace"]["calibrated"] = False
    document["camera_profile"]["mount_calibrated"] = False
    path.write_text(json.dumps(document))
    calibration_path = tmp_path / document["camera_calibration"]
    calibration = json.loads(calibration_path.read_text())
    calibration.update(calibrated=False, consistent=False, pose_source="assumed")
    calibration_path.write_text(json.dumps(calibration))
    for options in [{}, {"allow_assumed_camera": True}]:
        with pytest.raises(ValueError, match="completed measured"):
            load_hardware_profile(path, **options)
    profile = load_hardware_profile(path, allow_assumed_camera=True, allow_assumed_model=True)
    assert profile["calibrated"] is False
    assert profile["tool"]["calibrated"] is False
    assert profile["workspace"]["calibrated"] is False
    assert profile["_camera_calibration"]["pose_source"] == "assumed"
    # Marking optical calibration as assumed does not move the collision body.
    assert fixed_camera_envelopes(profile["camera_profile"]) == expected_boxes


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


def test_fixed_envelopes_follow_measured_body_pose_and_ignore_image_rotation(tmp_path):
    from tiptop.camera_geometry import fixed_camera_envelopes

    path = synthetic_profile(tmp_path, mount="fixed")
    profile = load_hardware_profile(path)
    camera = profile["camera_profile"]
    boxes = fixed_camera_envelopes(camera)
    assert [row["name"] for row in boxes] == ["fixed_camera_body", "fixed_camera_bracket"]
    rig = np.asarray(camera["parent_from_rig"])
    for box, dims, offset in zip(
        boxes, [camera["body_dims_m"], camera["bracket_dims_m"]],
        [[0, 0, -camera["body_dims_m"][2] / 2], camera["rig_from_bracket_xyz_m"]],
    ):
        transform = np.asarray(box["base_from_obstacle"])
        np.testing.assert_allclose(transform[:3, :3], rig[:3, :3])
        np.testing.assert_allclose(transform[:3, 3], rig[:3, 3] + rig[:3, :3] @ offset)
        np.testing.assert_allclose(box["dims_m"], np.asarray(dims) + 2 * camera["collision_margin_m"])
    # Optical pose is deliberately different from the body rig in this fixture.
    calibration = profile["_camera_calibration"]
    assert not np.allclose(rig, calibration["parent_from_camera"])
    calibration["image_rotation_deg"] = 0
    assert fixed_camera_envelopes(camera) == boxes


@pytest.mark.parametrize("change", ["parent", "mount", "roi"])
def test_fixed_profile_rejects_inconsistent_mount_or_missing_work_area(tmp_path, change):
    path = synthetic_profile(tmp_path, mount="fixed")
    data = json.loads(path.read_text())
    if change == "parent":
        calibration = json.loads((tmp_path / "hand-eye.json").read_text())
        calibration["parent_link"] = "link_6"
        (tmp_path / "hand-eye.json").write_text(json.dumps(calibration))
    elif change == "mount":
        data["camera_profile"]["mount"] = "wrist"
    else:
        data["workspace"]["pick_place_bounds_xy_m"] = None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_hardware_profile(path)


def test_observation_rotation_and_pose_must_match_measured_fixed_optics(tmp_path):
    profile = load_hardware_profile(synthetic_profile(tmp_path, mount="fixed"))
    calibration = profile["_camera_calibration"]
    expected = np.asarray(calibration["parent_from_camera"]) @ np.diag([-1, -1, 1, 1])
    packet = dict(
        camera_serial=calibration["camera_serial"], world_frame="base_link", camera_frame=calibration["camera_frame"],
        camera_parent_link="base_link", image_rotation_deg=180, robot_system=0, world_from_cam=expected,
    )
    validate_hardware_observation(packet, profile)
    for change in [
        dict(camera_parent_link="link_6"), dict(image_rotation_deg=0), dict(camera_serial="other"),
        dict(world_from_cam=np.eye(4)), dict(robot_system=1), dict(image_rotation_deg=180.5),
    ]:
        with pytest.raises(ValueError):
            validate_hardware_observation(packet | change, profile)


def test_fixed_model_has_world_boxes_and_no_wrist_camera_links(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Planning GPU required for model preparation")
    from tiptop.doosan.model import m1013_uon_curobo_cfg

    profile = load_hardware_profile(synthetic_profile(tmp_path, mount="fixed"))
    prepared = prepare_hardware_model(profile, tmp_path / "derived")
    names = prepared["robot_config"]["robot_cfg"]["kinematics"]["collision_link_names"]
    assert names == m1013_uon_curobo_cfg()["robot_cfg"]["kinematics"]["collision_link_names"]
    assert [box.name for box in prepared["fixed_obstacles"]] == [
        "configured_full_table", "configured_fixed_camera_body", "configured_fixed_camera_bracket",
    ]
    assert prepared["fixed_base_world_contact"] is False
    from tiptop.camera_geometry import fixed_camera_envelopes

    for box, row in zip(prepared["fixed_obstacles"][1:], fixed_camera_envelopes(profile["camera_profile"])):
        np.testing.assert_allclose(box.pose[:3], np.asarray(row["base_from_obstacle"])[:3, 3])
        np.testing.assert_allclose(box.dims, row["dims_m"])
        quat = Rotation.from_matrix(np.asarray(row["base_from_obstacle"])[:3, :3]).as_quat()[[3, 0, 1, 2]]
        np.testing.assert_allclose(box.pose[3:], quat)
    assert not (tmp_path / "derived/camera").exists()
    # Changing arm posture only changes robot FK; the camera stays in the world list.
    from curobo.types.base import TensorDeviceType

    prepared["kinematics"].get_state(TensorDeviceType().to_device([[0.3] * 6]))
    np.testing.assert_allclose(
        prepared["fixed_obstacles"][1].pose[:3],
        np.asarray(fixed_camera_envelopes(profile["camera_profile"])[0]["base_from_obstacle"])[:3, 3],
    )


def test_assumed_camera_can_be_reviewed_but_saved_plan_cannot_execute(tmp_path):
    from tiptop.doosan.client import validate_plan_backend
    from tiptop.doosan.hardware import hardware_profile_fingerprint

    path = synthetic_profile(tmp_path, mount="fixed")
    calibration_path = tmp_path / "hand-eye.json"
    calibration = json.loads(calibration_path.read_text())
    calibration.update(calibrated=False, consistent=False, pose_source="assumed")
    calibration_path.write_text(json.dumps(calibration))
    data = json.loads(path.read_text())
    data["camera_profile"]["mount_calibrated"] = False
    path.write_text(json.dumps(data))
    profile = load_hardware_profile(path, allow_assumed_camera=True)
    assert profile["_camera_calibration"]["calibrated"] is False
    packet = dict(
        camera_serial=calibration["camera_serial"], world_frame="base_link", camera_frame=calibration["camera_frame"],
        camera_parent_link="base_link", camera_pose_source="assumed", image_rotation_deg=180, robot_system=0,
        world_from_cam=np.asarray(calibration["parent_from_camera"]) @ np.diag([-1, -1, 1, 1]),
    )
    validate_hardware_observation(packet, profile)
    with pytest.raises(ValueError, match="pose source"):
        validate_hardware_observation(packet | dict(camera_pose_source="measured"), profile)
    plan = dict(
        robot="m1013_uon", robot_system=0, hardware_profile=str(path),
        hardware_profile_sha256=hardware_profile_fingerprint(profile),
    )
    with pytest.raises(ValueError, match="physical execution"):
        validate_plan_backend(plan, dict(success=True, robot_system=0, gripper=dict(backend="uon_native")))


def test_cached_hardware_model_is_built_once_per_profile_and_records_a_copy(tmp_path, monkeypatch):
    built = []

    def fake_prepare(profile, output):
        built.append(Path(output))
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "robot.yml").write_text("robot")
        cspace = dict(max_acceleration=[1.0] * 7, max_jerk=[2.0] * 7)
        return dict(robot_config={"robot_cfg": {"kinematics": {"cspace": cspace}}}, fixed_obstacles=[[1]],
                    gripper_spheres=np.zeros((2, 4)))

    monkeypatch.setattr(hardware_module, "HARDWARE_CACHE", tmp_path / "cache")
    monkeypatch.setattr(hardware_module, "prepare_hardware_model", fake_prepare)
    monkeypatch.setattr(hardware_module, "_PREPARED", {})
    profile = dict(version=1, tool=dict(a=1), _camera_calibration=dict(x=1))
    first = cached_hardware_model(profile, tmp_path / "task-1/hardware-model")
    first["fixed_obstacles"][0].append(2)  # a caller's edits must not reach the next task
    first["gripper_spheres"][:] = 1
    second = cached_hardware_model(profile)
    assert len(built) == 1 and second["fixed_obstacles"] == [[1]] and not second["gripper_spheres"].any()
    # Limits saved after the model was built still reach the next task, keeping the locked gripper entry.
    save_robot_limits(dict(DEFAULTS, max_acceleration=[9.0] * 6))
    cspace = cached_hardware_model(profile)["robot_config"]["robot_cfg"]["kinematics"]["cspace"]
    assert cspace["max_acceleration"] == [9.0] * 6 + [1.0] and cspace["max_jerk"][:6] == DEFAULTS["max_jerk"]
    assert (tmp_path / "task-1/hardware-model/robot.yml").read_text() == "robot"
    assert not (tmp_path / "task-2").exists()
    cached_hardware_model(dict(profile, tool=dict(a=2)))
    assert len(built) == 2 and built[0] != built[1] and built[0].parent == tmp_path / "cache"
