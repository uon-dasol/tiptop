import asyncio
import logging
import time

import aiohttp
import numpy as np
from jaxtyping import Bool, Float, UInt8
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras import (
    DepthEstimator,
    Frame,
)
from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd

_log = logging.getLogger(__name__)


async def detect_and_segment(rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str, *, client=None) -> dict:
    """Run VLM detection and SAM2 segmentation pipeline."""
    rgb_pil = Image.fromarray(rgb)
    rgb_pil_resized = rgb_pil.resize((800, int(800 * rgb_pil.size[1] / rgb_pil.size[0])), Image.Resampling.LANCZOS)
    _log.info(
        f"Resized image from {rgb_pil.size[1]}x{rgb_pil.size[0]} to {rgb_pil_resized.size[1]}x{rgb_pil_resized.size[0]}"
    )

    async def _detect():
        from tiptop.perception.gemini import detect_and_translate_async

        _log.info(f"Starting Gemini object detection")
        _st = time.perf_counter()
        _bboxes, _grounded_atoms = await detect_and_translate_async(rgb_pil_resized, task_instruction, client=client)
        _dur = time.perf_counter() - _st
        _log.info(f"Gemini detection took {_dur:.2f}s ({len(_bboxes)} objects, {len(_grounded_atoms)} atoms)")
        return _bboxes, _grounded_atoms

    bboxes, grounded_atoms = await _detect()
    return await segment_detections(rgb, bboxes, grounded_atoms)


async def segment_detections(rgb, bboxes, grounded_atoms) -> dict:
    """Segment THIS image using detector boxes or the active task's ROI seeds.

    Callers continuing an already identified task can supply its ROI seeds
    without spending another VLM request. This is not visual re-identification
    if an object leaves its ROI, and it never reuses old segmentation masks.
    """
    from tiptop.perception.sam2 import sam2_segment_objects

    # Sanitize labels: replace spaces with underscores for downstream compatibility
    bboxes = [dict(bbox, label=bbox["label"].strip().replace(" ", "_")) for bbox in bboxes]
    labels = [bbox["label"] for bbox in bboxes]
    if not all(labels) or len(set(labels)) != len(labels):
        raise ValueError("Detections must have unique nonempty labels")
    grounded_atoms = [dict(atom, args=[arg.replace(" ", "_") for arg in atom["args"]]) for atom in grounded_atoms]

    masks = await asyncio.to_thread(sam2_segment_objects, Image.fromarray(rgb), bboxes)

    return {"bboxes": bboxes, "masks": masks, "grounded_atoms": grounded_atoms}


async def predict_scene_depth(
    session: aiohttp.ClientSession,
    frame: Frame,
    world_from_cam: Float[np.ndarray, "4 4"],
    downsample_voxel_size: float,
    depth_estimator: DepthEstimator | None = None,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
) -> dict:
    """Build scene geometry from this frame, independently of the grasp provider."""
    cfg = tiptop_cfg()

    # Get depth map — use estimator (e.g. FoundationStereo) or fall back to onboard sensor depth
    if depth_estimator is not None:
        depth_map = await depth_estimator(session, frame)
    else:
        if frame.depth is None:
            raise RuntimeError(
                "No depth available: depth_estimator is None and frame.depth is not set. "
                "Either provide a depth_estimator or ensure the camera captures hardware depth."
            )
        _log.warning("No depth_estimator provided, falling back to hardware depth")
        depth_map = frame.depth

    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.shape != frame.rgb.shape[:2]:
        raise ValueError("Depth must be registered to the RGB frame and have the same image size")
    valid = np.isfinite(depth_map) & (depth_map > 0) & (depth_map <= cfg.perception.depth_trunc_m)
    if gripper_mask is not None:
        if gripper_mask.shape != valid.shape:
            raise ValueError("Gripper mask must match the current RGB image size")
    if not valid.any():
        raise ValueError("No valid scene depth")
    depth_map = np.where(valid, depth_map, 0.0)
    # Existing scene segmentation excludes NaN points, including missing depth.
    K = frame.intrinsics
    xyz_map = depth_to_xyz(depth_map, K)
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    xyz_map[~valid] = np.nan
    if gripper_mask is not None:
        xyz_map[gripper_mask] = 0.0
    rgb_map = frame.rgb.astype(np.float32) / 255.0  # make it float with [0, 1]

    # Create open3d point cloud and downsample
    pcd = await asyncio.to_thread(
        get_o3d_pcd,
        xyz_map[valid],
        rgb_map[valid],
        downsample_voxel_size,
    )
    xyz_downsampled = np.asarray(pcd.points)
    rgb_downsampled = np.asarray(pcd.colors)

    return {
        "depth_map": depth_map,
        "valid_depth_mask": valid,
        # (h, w, 3) for xyz, rgb, and valid mask map
        "xyz_map": xyz_map,
        "rgb_map": rgb_map,
        # (n, 3) for downsampled point cloud
        "xyz_downsampled": xyz_downsampled,
        "rgb_downsampled": rgb_downsampled,
        "pcd_downsampled": pcd,
    }


async def predict_depth_and_grasps(
    session, frame, world_from_cam, downsample_voxel_size, depth_estimator=None, gripper_mask=None
) -> dict:
    """Compatibility entry point for the existing M2T2 path."""
    from tiptop.perception.m2t2 import generate_grasps_async

    result = await predict_scene_depth(
        session, frame, world_from_cam, downsample_voxel_size, depth_estimator, gripper_mask
    )
    cfg = tiptop_cfg()
    result["grasps"] = await generate_grasps_async(
        session,
        cfg.perception.m2t2.url,
        scene_xyz=result["xyz_downsampled"],
        scene_rgb=result["rgb_downsampled"],
        apply_bounds=cfg.perception.m2t2.apply_bounds,
    )
    return result


async def run_visual_perception(
    session,
    frame,
    world_from_cam,
    task_instruction,
    *,
    depth_estimator=None,
    gripper_mask=None,
    client=None,
    active_task=None,
) -> dict:
    """One RGB/depth observation through Gemini, SAM2 and per-object point clouds.

    This common front end issues no grasp requests or robot commands. Object
    points remain in the world frame; their centroid is not a CAD 6D pose.
    """
    cfg = tiptop_cfg()
    if active_task is not None:
        if client is not None or active_task["instruction"] != task_instruction:
            raise ValueError("Active task must match this instruction and cannot request a new VLM call")
        detection_job = segment_detections(frame.rgb, active_task["bboxes"], active_task["grounded_atoms"])
    else:
        detection_job = detect_and_segment(frame.rgb, task_instruction, client=client)
    depth, detection = await asyncio.gather(
        predict_scene_depth(
            session, frame, world_from_cam, cfg.perception.voxel_downsample_size, depth_estimator, gripper_mask
        ),
        detection_job,
    )
    objects = {}
    for bbox, mask in zip(detection["bboxes"], detection["masks"][:, 0]):
        selected = mask & depth["valid_depth_mask"]
        objects[bbox["label"]] = dict(
            world_xyz=depth["xyz_map"][selected],
            rgb=depth["rgb_map"][selected],
            valid_pixels=int(selected.sum()),
            mask_pixels=int(mask.sum()),
        )
    return {
        **depth,
        **detection,
        "objects": objects,
        "semantic_source": "active_task_roi" if active_task is not None else "gemini",
    }
