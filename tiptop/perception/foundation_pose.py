"""FoundationPose client: current RGB, aligned metre depth and SAM mask to CAD pose."""

import io

import aiohttp
import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform


async def estimate_pose(session, server_url, frame, depth, mask, *, mesh_path, mesh_scale, class_id, world_from_cam):
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
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        rgb=frame.rgb,
        depth=depth,
        mask=mask,
        intrinsics=frame.intrinsics,
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
    result["pose_source"] = "rgbd_pose_estimator"
    return result
