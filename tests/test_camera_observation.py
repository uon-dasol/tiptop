"""Camera source boundaries without devices, GPU planning or model API calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from omegaconf import OmegaConf
import pytest

from tiptop.config import default_cfg_path
from tiptop.perception.cameras.frame import Frame
from tiptop import tiptop_run as live


@pytest.mark.parametrize("mount", ["wrist", "fixed"])
def test_current_robot_motion_affects_only_wrist_extrinsics(monkeypatch, mount):
    calibration = np.eye(4)
    calibration[:3, 3] = [.02, -.03, .08]
    frame = Frame("fixture", 1., np.zeros((2, 2, 3), np.uint8), np.eye(3), np.ones((2, 2)))
    q = np.zeros(6)
    fk_calls = []

    def fk(joints):
        fk_calls.append(joints.copy())
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [.5 + joints[0], .1, .8]
        return SimpleNamespace(ee_pose=SimpleNamespace(get_numpy_matrix=lambda: pose[None]))

    monkeypatch.setattr(live, "tensor_args", SimpleNamespace(to_device=np.asarray))
    container = SimpleNamespace(
        robot=SimpleNamespace(get_joint_positions=lambda: q.copy()),
        cam=SimpleNamespace(read_camera=lambda: frame),
        motion_gen=SimpleNamespace(kinematics=SimpleNamespace(get_state=fk)),
        camera_mount=mount, camera_parent_from_cam=calibration,
    )
    first = live.capture_live_observation(container)
    q[0] = .1
    second = live.capture_live_observation(container)
    assert second.frame is frame
    assert second.q_init[0] == pytest.approx(.1)
    if mount == "wrist":
        np.testing.assert_allclose(first.world_from_cam[:3, 3], [.53, .12, .88])
        np.testing.assert_allclose(second.world_from_cam[:3, 3] - first.world_from_cam[:3, 3], [.1, 0, 0])
        assert len(fk_calls) == 2
    else:
        np.testing.assert_array_equal(first.world_from_cam, calibration)
        np.testing.assert_array_equal(second.world_from_cam, calibration)
        assert not fk_calls


def test_sensor_input_container_needs_no_recording_camera_or_wrist_mask(monkeypatch):
    cfg = OmegaConf.load(default_cfg_path)
    cfg.cameras.hand.mount = "fixed"
    cfg.perception.depth_source = "sensor"
    monkeypatch.setattr(live, "tiptop_cfg", lambda: cfg)
    monkeypatch.setattr(live, "get_robot_client", lambda: object())
    requested_depth = []
    camera = SimpleNamespace(serial="fixture")

    def create_camera(*, depth):
        requested_depth.append(depth)
        return camera

    monkeypatch.setattr(live, "get_hand_camera", create_camera)
    monkeypatch.setattr(live, "get_external_camera", lambda: pytest.fail("Recording camera must be optional"))
    monkeypatch.setattr(live, "load_gripper_mask", lambda: pytest.fail("Fixed input must not use wrist pixel mask"))
    monkeypatch.setattr(live, "load_calibration", lambda serial: np.eye(4))
    monkeypatch.setattr(live, "sam2_client", lambda: None)
    monkeypatch.setattr(live, "check_server_health", AsyncMock())
    monkeypatch.setattr(live, "build_curobo_solvers", lambda *args: (object(), object(), object()))
    container = live.get_demo_container(8, 16, .0)
    assert requested_depth == [True]
    assert container.external_cam is None
    assert container.depth_estimator is None
    assert container.gripper_mask is None
    assert container.camera_mount == "fixed"


@pytest.mark.parametrize("source, depth_server_calls", [("sensor", 0), ("foundation_stereo", 1)])
def test_health_checks_only_selected_depth_service(monkeypatch, source, depth_server_calls):
    from tiptop.perception import foundation_stereo, m2t2

    cfg = OmegaConf.load(default_cfg_path)
    cfg.perception.depth_source = source
    monkeypatch.setattr(live, "tiptop_cfg", lambda: cfg)
    depth_health, grasp_health = AsyncMock(), AsyncMock()
    monkeypatch.setattr(foundation_stereo, "check_health_status", depth_health)
    monkeypatch.setattr(m2t2, "check_health_status", grasp_health)
    asyncio.run(live.check_server_health(object()))
    assert depth_health.await_count == depth_server_calls
    assert grasp_health.await_count == 1


def test_live_stereo_observation_uses_foundation_stereo_not_supplied_depth(monkeypatch):
    from tiptop.perception import foundation_stereo
    from tiptop.perception.cameras.stereo import decode_camera_request

    rgb = np.zeros((2, 3, 3), np.uint8)
    request = dict(rgb=rgb, stereo_right_rgb=rgb + 1, stereo_baseline_m=.095,
                   intrinsics=np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]]),
                   depth=np.full((2, 3), 99.))
    frame, estimator = decode_camera_request(request, depth_trunc_m=5.)
    assert frame.depth is None  # Evaluation depth cannot replace the required stereo inference.
    infer = AsyncMock(return_value=np.full((2, 3), .6))
    monkeypatch.setattr(foundation_stereo, 'infer_depth_async', infer)
    actual = asyncio.run(estimator(object(), frame))
    np.testing.assert_array_equal(actual, np.full((2, 3), .6))
    assert infer.await_count == 1
    assert infer.call_args.kwargs['baseline'] == .095
    np.testing.assert_array_equal(infer.call_args.kwargs['right_rgb'], rgb + 1)
    # Existing depth-request clients retain their behaviour.
    original = np.array([[np.nan, -1, 1], [6, 2, 3]], np.float32)
    plain, estimator = decode_camera_request(dict(rgb=rgb, depth=original, intrinsics=np.eye(3)), depth_trunc_m=5.)
    assert estimator is None
    np.testing.assert_array_equal(plain.depth, [[0, 0, 1], [0, 2, 3]])
    assert np.isnan(original[0, 0])
