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
from tiptop.perception.sam3_reference import merge_scene_objects
from tiptop.perception.utils import depth_to_xyz, get_o3d_pcd

_log = logging.getLogger(__name__)


async def detect_and_segment(
    rgb: UInt8[np.ndarray, "h w 3"], task_instruction: str, *, client=None, object_context=None
) -> dict:
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
        options = {} if object_context is None else dict(object_context=object_context)
        _bboxes, _grounded_atoms = await detect_and_translate_async(
            rgb_pil_resized, task_instruction, client=client, **options
        )
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

    segment_started = time.perf_counter()
    masks = await asyncio.to_thread(sam2_segment_objects, Image.fromarray(rgb), bboxes)
    _log.info(f"SAM2 segmentation took {time.perf_counter() - segment_started:.2f}s ({len(masks)} masks)")

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


def object_points(detection, depth):
    """World points and colours of each detected mask on the valid depth pixels."""
    objects = {}
    for bbox, mask in zip(detection["bboxes"], detection["masks"][:, 0]):
        selected = mask & depth["valid_depth_mask"]
        objects[bbox["label"]] = dict(
            world_xyz=depth["xyz_map"][selected],
            rgb=depth["rgb_map"][selected],
            valid_pixels=int(selected.sum()),
            mask_pixels=int(mask.sum()),
        )
    return objects


async def attach_scene_objects(results, scene_objects_job, *, record_scores=False):
    """Wait for the `object` query and add its new instances to the detections, masks and object points."""
    scene = await scene_objects_job
    count = len(results["bboxes"])
    merge_scene_objects(results, scene, record_scores=record_scores)
    added = dict(bboxes=results["bboxes"][count:], masks=results["masks"][count:])
    results["objects"] = {**results["objects"], **object_points(added, results)}
    return results


def select_reference_candidate(results, candidate):
    """Select another grasp target while retaining every instance's collision mask."""
    label = candidate["bboxes"][0]["label"]
    switched = dict(results, grounded_atoms=[dict(predicate="on", args=[label, "table"])])
    switched["objects"] = object_points(switched, results)
    switched["reference_diagnostics"] = dict(results["reference_diagnostics"], selected_index=candidate["index"])
    return switched


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
    object_context=None,
    perception_mode=None,
    sam3_url=None,
    reference_scene=None,
    sam3_confidence_threshold=None,
    appearance_weight=None,
    scene_objects=False,
    on_update=None,
) -> dict:
    """One RGB/depth observation through the selected detector and common geometry.

    This common front end issues no grasp requests or robot commands. Object
    points remain in the world frame; their centroid is not a CAD 6D pose.
    """
    cfg = tiptop_cfg()
    mode = perception_mode or cfg.perception.get("mode", "gemini_er")
    if mode not in {"gemini_er", "sam3_reference"}:
        raise ValueError(f"Unknown perception mode: {mode}")
    if mode == "sam3_reference":
        if client is not None or active_task is not None:
            raise ValueError("Reference mode cannot use a Gemini client or an active-task ROI")
        from tiptop.perception.sam3_reference import DEFAULT_CONFIDENCE_THRESHOLD, detect_reference

        detection_job = detect_reference(
            session,
            frame.rgb,
            task_instruction,
            server_url=sam3_url or cfg.perception.sam3.url,
            reference_scene=reference_scene or cfg.perception.sam3.reference_scene,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
            if sam3_confidence_threshold is None else sam3_confidence_threshold,
            appearance_weight=appearance_weight,
            scene_objects=scene_objects,
        )
    elif active_task is not None:
        if client is not None or active_task["instruction"] != task_instruction:
            raise ValueError("Active task must match this instruction and cannot request a new VLM call")
        detection_job = segment_detections(frame.rgb, active_task["bboxes"], active_task["grounded_atoms"])
    else:
        detection_job = detect_and_segment(frame.rgb, task_instruction, client=client, object_context=object_context)
    async def completed(job, event):
        result = await job
        if on_update is not None:
            await asyncio.to_thread(on_update, event, result)
        return result

    depth, detection = await asyncio.gather(
        completed(predict_scene_depth(
            session, frame, world_from_cam, cfg.perception.voxel_downsample_size, depth_estimator, gripper_mask
        ), "depth"),
        completed(detection_job, "detection"),
    )
    return {
        **depth,
        **detection,
        "objects": object_points(detection, depth),
        "semantic_source": detection.get("semantic_source", "active_task_roi" if active_task is not None else "gemini"),
    }
