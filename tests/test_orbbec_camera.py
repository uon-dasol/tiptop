"""Orbbec adapter conversions that need no camera."""

from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock

import numpy as np

from tiptop.perception.cameras.orbbec_camera import (
    _depth_to_metres,
    _distortion_vector,
    _k_matrix,
    _transform_from_extrinsic,
)


def test_depth_marks_zero_and_saturated_pixels_invalid():
    depth = _depth_to_metres(np.array([[0, 1000, 65535, 1500]], np.uint16), scale_mm=1.0)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.0, 0.0, 1.5]])


def test_depth_scale_converts_sensor_units_to_metres():
    depth = _depth_to_metres(np.array([[2000]], np.uint16), scale_mm=0.5)
    np.testing.assert_allclose(depth, [[1.0]])


def test_extrinsic_is_row_major_rotation_with_translation_in_metres():
    extrinsic = SimpleNamespace(rot=[0, -1, 0, 1, 0, 0, 0, 0, 1], transform=[-23.7, 0.1, 0.3])
    transform = _transform_from_extrinsic(extrinsic)
    np.testing.assert_allclose(transform[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(transform[:3, 3], [-0.0237, 0.0001, 0.0003], atol=1e-7)
    np.testing.assert_allclose(transform @ [1, 0, 0, 1], [-0.0237, 1.0001, 0.0003, 1], atol=1e-6)


def test_intrinsics_and_distortion_follow_opencv_layout():
    K = _k_matrix(SimpleNamespace(fx=610.0, fy=611.0, cx=640.0, cy=358.0))
    np.testing.assert_allclose(K, [[610, 0, 640], [0, 611, 358], [0, 0, 1]])
    distortion = SimpleNamespace(k1=1, k2=2, k3=3, k4=4, k5=5, k6=6, p1=7, p2=8)
    assert _distortion_vector(distortion).tolist() == [1, 2, 7, 8, 3, 4, 5, 6]


def test_factory_selects_orbbec_registered_depth_path_without_opening_device(monkeypatch):
    from tiptop.perception import cameras

    created = []

    class FakeOrbbec:
        def __init__(self, serial, **kwargs):
            created.append((serial, kwargs))

        def get_intrinsics(self):
            return 'colour/IR calibration'

    config = SimpleNamespace(cameras=SimpleNamespace(hand=dict(type='orbbec', serial='camera-id')))
    config.cameras.hand = SimpleNamespace(type='orbbec', get=config.cameras.hand.get)
    monkeypatch.setattr(cameras, 'tiptop_cfg', lambda: config)
    monkeypatch.setattr(cameras, 'OrbbecCamera', FakeOrbbec)
    infer = AsyncMock(return_value=np.ones((2, 3), np.float32))
    monkeypatch.setattr(cameras, 'orbbec_infer_depth_async', infer)
    cam = cameras.get_hand_camera(depth=False)
    assert created == [('camera-id', dict(enable_depth=False))]
    frame = object()
    estimate = cameras.get_depth_estimator(cam)
    result = asyncio.run(estimate(None, frame))
    infer.assert_awaited_once_with(None, frame, 'colour/IR calibration')
    assert result.shape == (2, 3)
    assert cameras.get_depth_estimator(cam, source='sensor') is None
