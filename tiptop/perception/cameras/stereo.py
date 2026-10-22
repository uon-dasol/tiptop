"""Rectified stereo input independent of a particular camera vendor SDK.

The RGB image used by perception is the left image. A physical device with a
separate RGB optical frame must register the stereo depth to RGB before using
that RGB frame downstream; merely replacing its intrinsics is insufficient.
"""

from dataclasses import dataclass

import numpy as np

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.frame import Frame


@dataclass(frozen=True, kw_only=True)
class RectifiedStereoFrame(Frame):
    right_rgb: np.ndarray


def decode_camera_request(request, *, depth_trunc_m):
    """Decode live observation packets before common TiPToP perception.

    The Isaac HTTP backend sends stereo_right_rgb and stereo_baseline_m instead
    of depth. Already-computed depth packets remain supported here. The TiPToP
    websocket server uses its original RGB+depth input independently of this
    decoder. Stereo never silently
    substitutes simulator ground-truth depth when its estimator fails.
    """
    rgb = np.asarray(request["rgb"], dtype=np.uint8)
    k = np.asarray(request["intrinsics"], dtype=np.float32)
    metadata = dict(
        serial=str(request.get("camera_model", "static")),
        timestamp=float(request.get("timestamp", 0.0)),
        rgb=rgb,
        intrinsics=k,
    )
    if "stereo_right_rgb" in request:
        right = np.asarray(request["stereo_right_rgb"], dtype=np.uint8)
        baseline = float(request["stereo_baseline_m"])
        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or right.shape != rgb.shape
            or not np.isfinite(baseline)
            or baseline <= 0
        ):
            raise ValueError("Invalid stereo observation")
        frame = RectifiedStereoFrame(**metadata, right_rgb=right)

        async def estimate(session, current_frame):
            return await infer_rectified_depth(session, current_frame, baseline)

        return frame, estimate
    depth = np.asarray(request["depth"], dtype=np.float32).copy()
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Depth must be aligned to the RGB image")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[(depth < 0) | (depth > depth_trunc_m)] = 0.0
    return Frame(**metadata, depth=depth), None


async def infer_rectified_depth(session, frame: RectifiedStereoFrame, baseline_m: float):
    from tiptop.perception.foundation_stereo import infer_depth_async

    if not isinstance(frame, RectifiedStereoFrame):
        raise ValueError("Rectified stereo inference requires left and right images")
    if (
        frame.rgb.ndim != 3
        or frame.rgb.shape[-1] != 3
        or frame.rgb.shape != frame.right_rgb.shape
        or frame.rgb.dtype != np.uint8
        or frame.right_rgb.dtype != np.uint8
        or not np.isfinite(baseline_m)
        or baseline_m <= 0
    ):
        raise ValueError("Expected matching RGB uint8 images and positive metre baseline")
    k = np.asarray(frame.intrinsics)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid rectified left-camera intrinsics")
    return await infer_depth_async(
        session,
        tiptop_cfg().perception.foundation_stereo.url,
        left_rgb=frame.rgb,
        right_rgb=frame.right_rgb,
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        baseline=float(baseline_m),
    )
