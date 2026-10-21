from typing import Awaitable, Callable, Protocol

import aiohttp
import numpy as np

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.cameras.orbbec_camera import OrbbecCamera, orbbec_infer_depth_async
from tiptop.perception.cameras.rs_camera import (
    RealsenseCamera,
    RealsenseFrame,
    rs_infer_depth_async,
)
from tiptop.perception.cameras.zed_camera import (
    ZedCamera,
    ZedFrame,
    zed_infer_depth_async,
)

# Callable that takes an aiohttp session and a camera frame and returns a float depth map (H, W) in metres.
DepthEstimator = Callable[[aiohttp.ClientSession, Frame], Awaitable[np.ndarray]]


class Camera(Protocol):
    serial: str

    def read_camera(self) -> Frame: ...
    def close(self) -> None: ...


def get_depth_estimator(cam: Camera, source: str = "foundation_stereo") -> DepthEstimator | None:
    """Select FoundationStereo or the aligned metre depth provided by Frame."""
    if source == "sensor":
        return None
    if source != "foundation_stereo":
        raise ValueError(f"Unknown depth source {source!r}; choose 'sensor' or 'foundation_stereo'")
    if callable(getattr(cam, "get_rectified_stereo_baseline", None)):
        from tiptop.perception.cameras.stereo import infer_rectified_depth

        baseline = cam.get_rectified_stereo_baseline()

        async def _rectified_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await infer_rectified_depth(session, f, baseline)

        return _rectified_estimate
    if isinstance(cam, ZedCamera):
        intrinsics = cam.get_intrinsics()

        async def _zed_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await zed_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _zed_estimate
    elif isinstance(cam, OrbbecCamera):
        intrinsics = cam.get_intrinsics()

        async def _orbbec_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await orbbec_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _orbbec_estimate
    elif isinstance(cam, RealsenseCamera):
        intrinsics = cam.get_intrinsics()

        async def _rs_estimate(session: aiohttp.ClientSession, f: Frame) -> np.ndarray:
            return await rs_infer_depth_async(session, f, intrinsics)  # type: ignore[arg-type]

        return _rs_estimate
    else:
        raise ValueError(f"No depth estimator available for camera type: {type(cam).__name__}")


def _get_zed_camera(cam_cfg, depth: bool = False, pointcloud: bool = False) -> ZedCamera:
    """Create a ZedCamera from config."""
    serial = str(cam_cfg.serial)
    flip = cam_cfg.get("flip", False)
    resolution = cam_cfg.get("resolution", "HD720")
    fps = cam_cfg.get("fps", 60)
    return ZedCamera(serial, resolution=resolution, fps=fps, flip=flip, depth=depth, pointcloud=pointcloud)


def get_hand_camera(depth: bool = False) -> Camera:
    """Get the hand camera by serial number."""
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.hand
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg, depth=depth)
    elif cam_type == "realsense":
        return RealsenseCamera(str(cam_cfg.serial), enable_depth=depth)
    elif cam_type == "orbbec":
        serial = cam_cfg.get("serial")
        return OrbbecCamera(None if serial is None else str(serial), enable_depth=depth)
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")


def get_external_camera() -> Camera:
    """Get the external camera by serial number."""
    cfg = tiptop_cfg()
    cam_cfg = cfg.cameras.external
    cam_type = cam_cfg.type
    if cam_type == "zed":
        return _get_zed_camera(cam_cfg)
    elif cam_type == "realsense":
        return RealsenseCamera(str(cam_cfg.serial))
    elif cam_type == "orbbec":
        serial = cam_cfg.get("serial")
        return OrbbecCamera(None if serial is None else str(serial))
    else:
        raise ValueError(f"Unknown camera type: {cam_type}")
