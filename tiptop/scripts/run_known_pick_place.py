"""Run a fresh known-object task through TiPToP and a live observation backend.

Gemini requires --allow-gemini-once. A consumed call marker prevents retries.
--active-task continues the same identified instruction using a NEW image.
"""

import argparse
import asyncio
import io
import json
import logging
import time
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.stereo import decode_camera_request
from tiptop.utils import NumpyEncoder


async def run(args):
    from curobo.types.base import TensorDeviceType

    from tiptop.doosan.known_task import plan_known_task
    from tiptop.doosan.model import get_m1013_uon_kinematics_model
    from tiptop.perception.sam2 import sam2_client
    from tiptop.tiptop_run import Observation, run_perception

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = tiptop_cfg()
    cfg.perception.foundation_stereo.url = args.depth_url
    cfg.perception.foundation_pose_url = args.pose_url
    if cfg.perception.recgen.enabled:
        raise ValueError("Known-DB baseline uses the existing observed hull path; disable RecGen")
    sam2_client()
    async with aiohttp.ClientSession() as session:
        async with session.get(args.camera_url + "/observe", timeout=aiohttp.ClientTimeout(total=35)) as response:
            response.raise_for_status()
            packet = np.load(io.BytesIO(await response.read()), allow_pickle=False)
            observed = {key: packet[key] for key in packet.files}
        observation_id = str(observed["observation_id"])
        frame, estimate = decode_camera_request(observed, depth_trunc_m=cfg.perception.depth_trunc_m)
        kin = get_m1013_uon_kinematics_model()
        spheres = (
            kin.get_state(TensorDeviceType().to_device(observed["q_init"][None])).link_spheres_tensor[0].cpu().numpy()
        )
        spheres = spheres[spheres[:, 3] > 0]
        removed = 0

        async def depth_without_robot(current_session, current_frame):
            nonlocal removed
            from tiptop.perception.utils import depth_to_xyz

            depth = (await estimate(current_session, current_frame)) if estimate else current_frame.depth.copy()
            xyz = depth_to_xyz(depth, current_frame.intrinsics)
            camera = observed["world_from_cam"]
            xyz = xyz @ camera[:3, :3].T + camera[:3, 3]
            # Chunking avoids an H*W*spheres temporary allocation.
            points = xyz.reshape(-1, 3)
            inside = np.zeros(len(points), bool)
            for begin in range(0, len(points), 4096):
                distance = np.linalg.norm(points[begin : begin + 4096, None] - spheres[None, :, :3], axis=-1)
                inside[begin : begin + 4096] = np.any(distance <= spheres[None, :, 3] + 0.003, axis=1)
            removed = int(inside.sum())
            return np.where(inside.reshape(depth.shape), 0, depth)

        active = json.loads(args.active_task.read_text()) if args.active_task else None
        client = None
        if active is None:
            from google import genai
            from google.genai import types

            if not args.allow_gemini_once:
                raise ValueError("Gemini permission required: use --allow-gemini-once or --active-task")
            marker = args.output / "gemini-call-consumed.json"
            with marker.open("x") as stream:
                json.dump(dict(instruction=args.task, observation_id=observation_id, maximum_calls=1), stream)
            client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
        try:
            observation = Observation(frame, observed["world_from_cam"], observed["q_init"])
            env, _, scene, atoms = await run_perception(
                session,
                observation,
                args.task,
                args.output,
                depth_estimator=depth_without_robot,
                include_workspace=False,
                log_to_rerun=False,
                grasp_provider="known_db",
                client=client,
                active_task=active,
                empty_place=True,
            )
        finally:
            if client is not None:
                await client.aio.aclose()
                client.close()
        visual = scene.perception_results
        semantic = json.loads((args.output / "semantic-task.json").read_text())
        name = next(iter(scene.grasps))
        index = next(i for i, box in enumerate(visual["bboxes"]) if box["label"] == name)
        overlay = frame.rgb.copy()
        mask = visual["masks"][index, 0]
        overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 220, 80]) * 0.4).astype(np.uint8)
        Image.fromarray(overlay).save(args.output / "target-overlay.png")
        status = (
            f"Gemini -> SAM2 -> FoundationStereo -> FoundationPose -> UON DB\n"
            f"Target: {name}. Observed empty region selected. Planning pick and place..."
        )
        display = io.BytesIO()
        np.savez_compressed(display, observation_id=observation_id, overlay=overlay, status=status)
        async with session.post(args.camera_url + "/display", data=display.getvalue()) as response:
            response.raise_for_status()
        # Preserve current inference, so planning fixes need no additional API call.
        import pickle
        from dataclasses import replace

        cached_scene = replace(
            scene,
            object_pcds={},
            perception_results={
                key: value for key, value in scene.perception_results.items() if key != "pcd_downsampled"
            },
        )
        with (args.output / "planning-input.pkl").open("wb") as stream:
            pickle.dump((env, cached_scene, observation), stream)
        profile = json.loads(args.camera_preparation.read_text())["profile"]
        plan, report = await asyncio.to_thread(
            plan_known_task, env, scene, observation, camera_profile=profile, output=args.output
        )
        plan.update(
            observation_id=observation_id,
            task=args.task,
            gemini_api_calls=int((args.output / "gemini-call-consumed.json").exists()),
            gemini_api_calls_this_process=int(active is None),
            semantic_source=semantic["semantic_source"],
        )
        (args.output / "plan.json").write_text(json.dumps(plan, cls=NumpyEncoder, indent=2) + "\n")
        if not args.execute:
            print("Complete plan saved; no motion requested.")
            return
        async with session.post(
            args.camera_url + "/execute", json=json.loads(json.dumps(plan, cls=NumpyEncoder))
        ) as response:
            response.raise_for_status()
            print("EXECUTION_ACCEPTED", await response.text(), flush=True)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            async with session.get(args.camera_url + "/result") as response:
                response.raise_for_status()
                result = await response.json()
            if result.get("finished"):
                (args.output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in result.items() if k not in {"trace", "calls"}}, ensure_ascii=False),
                    flush=True,
                )
                if not result.get("success"):
                    raise RuntimeError("Physical pick/place verification failed; see execution.json")
                return
        raise TimeoutError("Execution result timed out; inspect the live backend before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="후추통 집고 빈 곳에 놓아")
    parser.add_argument("--camera-url", default="http://127.0.0.1:13345")
    parser.add_argument("--depth-url", default="http://127.0.0.1:13344")
    parser.add_argument("--pose-url", default="http://127.0.0.1:13346")
    parser.add_argument(
        "--camera-preparation", type=Path, default=Path("simulation/isaac_sim/config/wrist_camera.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-task", type=Path)
    parser.add_argument("--allow-gemini-once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.active_task and args.allow_gemini_once:
        parser.error("Choose a new Gemini call OR continuation, not both")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
