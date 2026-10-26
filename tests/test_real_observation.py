"""Real-adapter contracts with synthetic cameras/feedback, never hardware commands."""

import asyncio
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
