"""Continue an identified task from a fresh Isaac camera request, without a Gemini call.

The active-task JSON supplies semantic labels and ROI seeds only. RGB, masks,
depth, object poses and robot q are obtained/calculated anew. No saved image or
saved joint trajectory can be supplied as this command's observation input.
"""

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image, ImageDraw

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.stereo import decode_camera_request
from tiptop.perception.known_objects import estimate_registered_grasps, registered_object, registered_target
from tiptop.perception_wrapper import run_visual_perception


def task_target(task, entry):
    label, identified = registered_target(task["bboxes"], task["grounded_atoms"])
    if identified["class_id"] != entry["class_id"]:
        raise ValueError("Task target does not match --class-id")
    return label


def make_overlay(rgb, mask, camera_from_cad, k, vertices):
    pixels = rgb.copy()
    pixels[mask] = (pixels[mask] * 0.6 + np.array([0, 220, 80]) * 0.4).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    lo, hi = vertices.min(0), vertices.max(0)
    corners = np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
    camera = corners @ camera_from_cad[:3, :3].T + camera_from_cad[:3, 3]
    projected = camera @ k.T
    projected = projected[:, :2] / projected[:, 2:3]
    for first in range(8):
        for bit in [1, 2, 4]:
            second = first ^ bit
            if first < second and min(camera[first, 2], camera[second, 2]) > 0:
                draw.line([tuple(projected[first]), tuple(projected[second])], fill=(0, 240, 255), width=2)
    return np.asarray(image)


async def run(args, task, entry, label):
    if args.depth_url:
        tiptop_cfg().perception.foundation_stereo.url = args.depth_url
    # Warm-up occurs only in live mode, never --check-inputs.
    from tiptop.perception.sam2 import sam2_client

    sam2_client()
    async with aiohttp.ClientSession() as session:
        async with session.get(args.pose_url.rstrip("/") + "/health") as response:
            response.raise_for_status()
        async with session.get(
            args.camera_url.rstrip("/") + "/observe", timeout=aiohttp.ClientTimeout(total=35.0)
        ) as response:
            response.raise_for_status()
            packet = np.load(io.BytesIO(await response.read()), allow_pickle=False)
            observation = {key: packet[key] for key in packet.files}
        observation_id = str(observation["observation_id"])
        frame, depth_estimator = decode_camera_request(observation, depth_trunc_m=tiptop_cfg().perception.depth_trunc_m)
        if depth_estimator is None:
            raise ValueError("Live sim verification requires FoundationStereo stereo input")
        started = time.perf_counter()
        visual = await run_visual_perception(
            session,
            frame,
            observation["world_from_cam"],
            task["instruction"],
            depth_estimator=depth_estimator,
            active_task=task,
        )
        points = visual["objects"][label]["world_xyz"]
        if len(points) < 10:
            raise ValueError("Target no longer has sufficient currently observed depth; re-identify it")
        # Planning point-cloud frame only; this centroid is NOT a CAD pose.
        planner = np.eye(4)
        planner[:3, 3] = points.mean(0)
        registered = await estimate_registered_grasps(
            session,
            frame,
            visual,
            observation["world_from_cam"],
            entry=entry,
            label=label,
            server_url=args.pose_url,
            world_from_planner_object=planner,
            grasp_from_tcp=np.eye(4),
        )
        pose, grasps = registered["pose"], registered["grasps"]
        vertices = (
            np.array(
                [
                    list(map(float, line.split()[1:4]))
                    for line in entry["cad"].read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            * entry["cad_scale_to_m"]
        )
        index = next(i for i, bbox in enumerate(visual["bboxes"]) if bbox["label"] == label)
        overlay = make_overlay(
            frame.rgb, visual["masks"][index, 0], pose["camera_from_cad"], frame.intrinsics, vertices
        )
        world_vertices = vertices @ pose["world_from_cad"][:3, :3].T + pose["world_from_cad"][:3, 3]
        text = (
            f"Current frame -> SAM2 + FoundationStereo -> FoundationPose\n"
            f"{label}: {len(points)} observed points, {len(grasps['scores'])} DB candidates\n"
            "Cyan: estimated CAD. Candidates not collision-filtered; no motion executed."
        )
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, observation_id=observation_id, overlay=overlay, status=text, cad_points_world=world_vertices
        )
        async with session.post(args.camera_url.rstrip("/") + "/display", data=buffer.getvalue()) as response:
            response.raise_for_status()
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(args.output / "pose-overlay.png")
        np.savez_compressed(
            args.output / "current-observation-result.npz",
            observation_id=observation_id,
            rgb=frame.rgb,
            depth=visual["depth_map"],
            masks=visual["masks"],
            intrinsics=frame.intrinsics,
            world_from_cam=observation["world_from_cam"],
            q_init=observation["q_init"],
            world_from_cad=pose["world_from_cad"],
            world_from_planner_object=planner,
            world_from_tcp=grasps["world_from_tcp"],
            grasps_obj=grasps["grasps_obj"],
            candidate_indices=grasps["candidate_indices"],
        )
        report = dict(
            success=True,
            scope="live observation -> CAD pose -> DB candidates, not planning/execution",
            observation_id=observation_id,
            class_id=entry["class_id"],
            label=label,
            pose_source=pose["pose_source"],
            candidate_count=len(grasps["scores"]),
            elapsed_s=time.perf_counter() - started,
            camera_from_cad=pose["camera_from_cad"].tolist(),
            world_from_cad=pose["world_from_cad"].tolist(),
            semantic_source=visual["semantic_source"],
            gemini_api_calls=0,
            real_tcp_calibrated=False,
            collision_filtered=False,
            physical_grasp_verified=False,
        )
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-task", type=Path, required=True)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--camera-url", default="http://127.0.0.1:13345")
    parser.add_argument("--pose-url", default="http://127.0.0.1:13346")
    parser.add_argument("--depth-url", default=None)
    parser.add_argument("--output", type=Path, default=Path("tiptop_outputs/m1013/live-perception"))
    parser.add_argument("--check-inputs", action="store_true", help="CPU only: validate task identity and local assets")
    args = parser.parse_args()
    task = json.loads(args.active_task.read_text())
    entry = registered_object(args.class_id)
    label = task_target(task, entry)
    if args.check_inputs:
        print(
            json.dumps(
                dict(success=True, class_id=entry["class_id"], label=label, gpu_inference=False, gemini_api_calls=0)
            )
        )
        return
    asyncio.run(run(args, task, entry, label))


if __name__ == "__main__":
    main()
