"""Real-adapter contracts with synthetic cameras/feedback, never hardware commands."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from tiptop.doosan import observation as module
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.orbbec_camera import OrbbecFrame, OrbbecIntrinsics
from tiptop.perception.cameras.stereo import decode_camera_request


def fixture_camera():
    k = np.diag([100.0, 100.0, 1.0]).astype(np.float32)
    intr = OrbbecIntrinsics(
        K_color=k, K_ir=k, baseline_ir=0.095, T_color_from_ir=np.eye(4), distortion_color=np.zeros(8)
    )
    frame = OrbbecFrame(
        serial="test-camera",
        timestamp=1.0,
        rgb=np.zeros((4, 6, 3), np.uint8),
        intrinsics=k,
        ir_left=np.zeros((3, 5), np.uint8),
        ir_right=np.zeros((3, 5), np.uint8),
    )
    return SimpleNamespace(serial="test-camera", read_camera=lambda: frame, get_intrinsics=lambda: intr)


class FeedbackOnly:
    def __init__(self, moved=False):
        self.reads = 0
        self.moved = moved

    def get_state(self):
        self.reads += 1
        return dict(
            success=True,
            joint_positions=[0.01 if self.moved and self.reads > 1 else 0.0] * 6,
            motion_status=0,
            robot_system=0,
        )


def test_separate_ir_pair_is_registered_to_rgb_after_shared_packet_decode(monkeypatch):
    camera = fixture_camera()
    world = np.eye(4)
    world[0, 3] = 0.5
    monkeypatch.setattr(module, "world_from_parent", lambda *a: world)
    camera_mount = np.eye(4)
    camera_mount[2, 3] = 0.1
    packet = module.capture_stationary(camera, FeedbackOnly(), parent_from_camera=camera_mount)
    np.testing.assert_allclose(packet["world_from_cam"], world @ camera_mount)
    assert packet["camera_frame"] == "undistorted_color_optical" and packet["world_frame"] == "base_link"
    from tiptop.perception.cameras import orbbec_camera

    infer = AsyncMock(return_value=np.ones((4, 6), np.float32))
    monkeypatch.setattr(orbbec_camera, "orbbec_infer_depth_async", infer)
    frame, estimate = decode_camera_request(packet, depth_trunc_m=5.0)
    assert isinstance(frame, OrbbecFrame) and frame.depth is None
    assert frame.ir_left.shape == (3, 5) and frame.rgb.shape == (4, 6, 3)
    assert asyncio.run(estimate(None, frame)).shape == (4, 6)
    assert infer.await_count == 1


def test_moving_read_and_missing_sensor_depth_are_rejected():
    with pytest.raises(ValueError, match="changed during"):
        module.capture_stationary(fixture_camera(), FeedbackOnly(moved=True), parent_from_camera=np.eye(4))
    with pytest.raises(ValueError, match="disabled"):
        module.capture_stationary(fixture_camera(), FeedbackOnly(), parent_from_camera=np.eye(4), source="sensor")


def test_sensor_depth_packet_keeps_explicit_choice_even_with_ir_fields():
    camera = fixture_camera()
    frame = camera.read_camera()
    packet = dict(
        rgb=frame.rgb,
        intrinsics=frame.intrinsics,
        depth=np.ones((4, 6), np.float32),
        ir_left=frame.ir_left,
        ir_right=frame.ir_right,
    )
    decoded, estimate = decode_camera_request(packet, depth_trunc_m=5.0)
    assert isinstance(decoded, Frame) and not isinstance(decoded, OrbbecFrame) and estimate is None


def test_read_only_backend_requires_measured_calibration_and_rejects_every_execution(tmp_path, monkeypatch):
    from tiptop.scripts.serve_orbbec_observation import ReadOnlyObservationBackend

    calibration = dict(
        calibrated=True,
        consistent=True,
        world_frame="base_link",
        parent_link="link_6",
        camera_frame="undistorted_color_optical",
        camera_serial="test-camera",
        parent_from_camera=np.eye(4).tolist(),
    )
    monkeypatch.setattr(module, "world_from_parent", lambda *a: np.eye(4))
    feedback = FeedbackOnly()
    backend = ReadOnlyObservationBackend(fixture_camera(), feedback, calibration, tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        backend.dispatch("/execute", b"{}")
    assert feedback.reads == 0
    import io

    captured = np.load(io.BytesIO(backend.dispatch("/observe", b"")), allow_pickle=False)
    assert captured["rgb"].shape == (4, 6, 3) and str(captured["observation_id"])
    with pytest.raises(ValueError, match="candidates"):
        ReadOnlyObservationBackend(fixture_camera(), feedback, dict(calibration, calibrated=False), tmp_path)
    with pytest.raises(ValueError, match="serial"):
        ReadOnlyObservationBackend(fixture_camera(), feedback, dict(calibration, camera_serial="other"), tmp_path)


def test_physical_sampling_rejects_virtual_robot_feedback():
    virtual = SimpleNamespace(
        get_state=lambda: dict(success=True, joint_positions=[0.0] * 6, motion_status=0, robot_system=1)
    )
    with pytest.raises(ValueError, match="physical/virtual"):
        module.stationary_state(virtual, expected_system=0)


def test_fixed_parent_does_not_depend_on_joints_or_load_the_wrist_model(monkeypatch):
    def unexpected_fk():
        raise AssertionError("Fixed camera must not evaluate wrist FK")

    monkeypatch.setattr(module, "_robot_urdf", unexpected_fk)
    for q in [np.zeros(6), np.ones(6)]:
        np.testing.assert_array_equal(module.world_from_parent(q, "base_link"), np.eye(4))


def test_sensor_rotation_preserves_pixel_depth_and_world_point_correspondence():
    camera = fixture_camera()
    original = camera.read_camera()
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    k = np.array([[100.0, 0, 1.2], [0, 110.0, 0.8], [0, 0, 1]])
    camera.read_camera = lambda: replace(original, rgb=rgb, depth=depth, intrinsics=k)
    mount = np.eye(4)
    mount[:3, 3] = [0.1, 0.7, 1.02]
    packet = module.capture_stationary(
        camera, FeedbackOnly(), parent_from_camera=mount, parent_link="base_link",
        source="sensor", image_rotation_deg=180,
    )
    np.testing.assert_array_equal(packet["rgb"], rgb[::-1, ::-1])
    np.testing.assert_array_equal(packet["depth"], depth[::-1, ::-1])
    pixel = np.array([2, 1, 1])
    rotated_pixel = np.array([5 - pixel[0], 3 - pixel[1], 1])
    native_point = np.r_[np.linalg.inv(k) @ pixel * depth[1, 2], 1]
    rotated_point = np.r_[np.linalg.inv(packet["intrinsics"]) @ rotated_pixel * packet["depth"][2, 3], 1]
    np.testing.assert_allclose(packet["world_from_cam"] @ rotated_point, mount @ native_point)
    np.testing.assert_array_equal(original.rgb, np.zeros((4, 6, 3), np.uint8))
    np.testing.assert_array_equal(camera.read_camera().rgb, rgb)
    assert packet["camera_mount"] == "fixed" and packet["image_rotation_deg"] == 180


def test_rotated_colour_keeps_native_ir_and_updates_registration(monkeypatch):
    camera = fixture_camera()
    native = camera.read_camera()
    left = np.arange(15, dtype=np.uint8).reshape(3, 5)
    right = left + 20
    camera.read_camera = lambda: replace(native, ir_left=left, ir_right=right)
    intr = camera.get_intrinsics()
    intr.T_color_from_ir[0, 3] = 0.024
    packet = module.capture_stationary(
        camera, FeedbackOnly(), parent_from_camera=np.eye(4), parent_link="base_link", image_rotation_deg=180,
    )
    np.testing.assert_array_equal(packet["ir_left"], left)
    np.testing.assert_array_equal(packet["ir_right"], right)
    np.testing.assert_array_equal(packet["K_ir"], intr.K_ir)
    expected = np.diag([-1, -1, 1, 1]) @ intr.T_color_from_ir
    np.testing.assert_allclose(packet["T_color_from_ir"], expected)
    from tiptop.perception.cameras import orbbec_camera

    infer = AsyncMock(return_value=np.ones((4, 6), np.float32))
    monkeypatch.setattr(orbbec_camera, "orbbec_infer_depth_async", infer)
    decoded, estimate = decode_camera_request(packet, depth_trunc_m=5.0)
    asyncio.run(estimate(None, decoded))
    np.testing.assert_allclose(infer.call_args.args[2].T_color_from_ir, expected)


def test_fixed_backend_passes_measured_parent_and_rotation_without_wrist_fk(tmp_path, monkeypatch):
    import io
    import json

    from tiptop.scripts.serve_orbbec_observation import ReadOnlyObservationBackend

    calibration = dict(
        calibrated=True, consistent=True, world_frame="base_link", parent_link="base_link",
        camera_frame="undistorted_color_optical", camera_serial="test-camera",
        parent_from_camera=np.eye(4).tolist(), image_rotation_deg=180,
    )
    monkeypatch.setattr(module, "_robot_urdf", lambda: pytest.fail("Fixed backend evaluated wrist FK"))
    feedback = FeedbackOnly()
    backend = ReadOnlyObservationBackend(fixture_camera(), feedback, calibration, tmp_path)
    health = json.loads(backend.dispatch("/health", b""))
    assert health["read_only"] and health["camera_parent_link"] == "base_link"
    assert feedback.reads == 0
    packet = np.load(io.BytesIO(backend.dispatch("/observe", b"")), allow_pickle=False)
    np.testing.assert_allclose(packet["world_from_cam"], np.diag([-1, -1, 1, 1]))
    assert str(packet["camera_mount"]) == "fixed" and int(packet["image_rotation_deg"]) == 180
    for updates in [dict(parent_link="world"), dict(image_rotation_deg=90), dict(calibrated=False)]:
        with pytest.raises(ValueError):
            module.validate_camera_calibration(dict(calibration, **updates))


def test_declared_top_view_matches_existing_lab_camera_and_factory_optics():
    import re

    from simulation.isaac_sim.lab_camera import camera_specs

    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / "tiptop/config/m1013_fixed_camera.declared.json").read_text())
    factory = json.loads((root / document["optics_source"]).read_text())
    scene = (root / document["scene_source"]).read_text()
    xyz = [float(v) for v in re.search(r'xformOp:translate = \(([^)]+)\)', scene).group(1).split(",")]
    yaw = float(re.search(r'xformOp:rotateZ = ([\d.]+)', scene).group(1))
    assert yaw == 180
    body = np.asarray(document["assumed_base_from_body"])
    np.testing.assert_allclose(body[:3, 3], xyz)
    np.testing.assert_allclose(body[:3, 3], [-0.01, 0.76, 1.02])
    np.testing.assert_allclose(body[:3, :3], np.diag([-1, -1, 1]))
    expected = body @ camera_specs(factory)["RGBRaw"]["body_from_optical"]
    np.testing.assert_allclose(module.validate_camera_calibration(document), expected)
    assert document["pose_source"] == "user_declared_isaac_lab" and document["image_rotation_deg"] == 180


def test_assumed_pose_backend_is_read_only_and_labels_its_observations(tmp_path):
    import io

    from tiptop.scripts.serve_orbbec_observation import ReadOnlyObservationBackend

    calibration = dict(
        calibrated=False, consistent=False, pose_source="assumed", world_frame="base_link", parent_link="base_link",
        camera_frame="undistorted_color_optical", camera_serial="test-camera", parent_from_camera=np.eye(4).tolist(),
        image_rotation_deg=180,
    )
    feedback = FeedbackOnly()
    backend = ReadOnlyObservationBackend(fixture_camera(), feedback, calibration, tmp_path)
    assert json.loads(backend.dispatch("/health", b""))["camera_pose_source"] == "assumed"
    with pytest.raises(ValueError, match="read-only"):
        backend.dispatch("/execute", b"{}")
    assert feedback.reads == 0
    packet = np.load(io.BytesIO(backend.dispatch("/observe", b"")), allow_pickle=False)
    assert str(packet["camera_pose_source"]) == "assumed"
    with pytest.raises(ValueError, match="observation/planning"):
        module.validate_camera_calibration(dict(calibration, parent_link="link_6"), allow_assumed=True)
