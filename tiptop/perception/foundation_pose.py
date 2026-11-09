"""FoundationPose client: current RGB, aligned metre depth and SAM mask to CAD pose."""

import io
import logging
import time

import aiohttp
import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform
from tiptop.serialization import savez_fast

_log = logging.getLogger(__name__)


async def estimate_pose(session, server_url, frame, depth, mask, *, mesh_path, mesh_scale, class_id, world_from_cam):
    started = time.perf_counter()
    depth = np.asarray(depth, np.float32)
    mask = np.asarray(mask, bool)
    world_cam = _rigid_transform(world_from_cam, "world_from_cam")
    k = np.asarray(frame.intrinsics)
    if (
        frame.rgb.dtype != np.uint8
        or frame.rgb.ndim != 3
        or frame.rgb.shape[-1] != 3
        or k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
    ):
        raise ValueError("Expected RGB uint8 and valid pinhole intrinsics")
    if depth.shape != frame.rgb.shape[:2] or mask.shape != depth.shape:
        raise ValueError("Pose inputs must share the same RGB pixel grid")
    if np.count_nonzero(mask & np.isfinite(depth) & (depth > 0)) < 10:
        raise ValueError("Insufficient observed object depth for CAD pose estimation")
    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError("Mesh scale must be positive metres per mesh unit")
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    # Preserve native pixel resolution while avoiding full-image batch buffers.
    # The object remains in the same optical frame: shift K with the crop.
    ys, xs = np.where(mask)
    x0, x1 = max(0, int(xs.min()) - 100), min(depth.shape[1], int(xs.max()) + 101)
    y0, y1 = max(0, int(ys.min()) - 100), min(depth.shape[0], int(ys.max()) + 101)
    rgb = frame.rgb[y0:y1, x0:x1].copy()
    depth, mask = depth[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    k = k.copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    buffer = io.BytesIO()
    savez_fast(
        buffer,
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=k,
        mesh_path=str(mesh_path),
        mesh_scale=float(mesh_scale),
        class_id=str(class_id),
    )
    async with session.post(
        server_url.rstrip("/") + "/estimate", data=buffer.getvalue(), timeout=aiohttp.ClientTimeout(total=120.0)
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"FoundationPose: {await response.text()}")
        result = await response.json()
    pose = _rigid_transform(result["camera_from_cad"], "FoundationPose CAD pose")
    if result["class_id"] != class_id:
        raise ValueError("FoundationPose returned a different CAD identity")
    if pose[2, 3] <= 0:
        raise ValueError("FoundationPose returned a CAD origin behind the camera")
    result["camera_from_cad"] = pose
    result["world_from_cad"] = world_cam @ pose
    result["input_crop_xyxy"] = [x0, y0, x1, y1]
    result["pose_source"] = "rgbd_pose_estimator"
    _log.info("Pose estimation client total took %.2fs", time.perf_counter() - started)
    return result
