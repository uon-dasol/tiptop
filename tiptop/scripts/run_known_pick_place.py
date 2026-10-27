"""Run a fresh known-object task through TiPToP and a live observation backend.

Gemini requires --allow-gemini-once. A consumed call marker prevents retries.
--active-task continues the same identified instruction using a NEW image.
--perception-mode sam3_reference uses a fresh reference-based SAM3 detection,
without Gemini permission, calls, or SAM2 initialization.
"""

import argparse
import asyncio
import gc
import io
import json
import logging
import re
import threading
import time
import uuid
from copy import copy
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import numpy as np
from google.genai.errors import APIError
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
    mode = getattr(args, "perception_mode", None) or cfg.perception.get("mode", "gemini_er")
    if mode == "gemini_er":
        sam2_client()
    elif mode != "sam3_reference":
        raise ValueError(f"Unknown perception mode: {mode}")
    async with aiohttp.ClientSession() as session:
        if mode == "sam3_reference":
            from tiptop.perception.sam3_reference import check_reference_health, parse_reference_task

            if args.active_task:
                raise ValueError("Reference mode requires a fresh detection, not --active-task")
            parse_reference_task(args.task)
            await check_reference_health(session, args.sam3_url or cfg.perception.sam3.url)
        args._stage = "observation"
        async with session.get(args.camera_url + "/observe", timeout=aiohttp.ClientTimeout(total=35)) as response:
            response.raise_for_status()
            packet = np.load(io.BytesIO(await response.read()), allow_pickle=False)
            observed = {key: packet[key] for key in packet.files}
        observation_id = str(observed["observation_id"])
        args._observation_id = observation_id
        # Keep the exact camera/FK packet even if identification or placement
        # fails before planning-input.pkl can be written. Never use it as a live
        # execution observation without the backend's freshness checks.
        np.savez_compressed(args.output / "observation.npz", **observed)
        frame, estimate = decode_camera_request(observed, depth_trunc_m=cfg.perception.depth_trunc_m)
        # Fail before any paid identification if a required local model is down.
        from tiptop.doosan.readiness import check_model_health

        args._stage = "connections"
        health_checks = [check_model_health(session, args.pose_url)]
        if estimate is not None:
            health_checks.append(check_model_health(session, args.depth_url))
        await asyncio.gather(*health_checks)
        hardware = None
        profile_path = getattr(args, "hardware_profile", None)
        physical = str(observed.get("camera_model", "")).startswith("orbbec/")
        if physical:
            from tiptop.doosan.hardware import load_hardware_profile, prepare_hardware_model

            if profile_path is None:
                raise ValueError(
                    "Physical known-object planning requires --hardware-profile with measured mount/TCP/camera/workspace data"
                )
            measured_profile = load_hardware_profile(profile_path)
            if (
                str(observed.get("camera_serial", "")) != measured_profile["_camera_calibration"]["camera_serial"]
                or str(observed.get("world_frame", "")) != "base_link"
                or int(observed.get("robot_system", -1)) != 0
            ):
                raise ValueError(
                    "Physical observation camera/robot/base frame differs from the measured hardware profile"
                )
            hardware = prepare_hardware_model(measured_profile, args.output / "hardware-model")
            kin = hardware["kinematics"]
            if args.execute:
                from tiptop.doosan.readiness import check_gateway_state

                async with session.get(
                    args.gateway_url.rstrip("/") + "/state", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response.raise_for_status()
                    check_gateway_state(
                        await response.json(content_type=None), execute=True, observed_q=observed["q_init"]
                    )
        else:
            if profile_path is not None:
                raise ValueError("A hardware profile requires the physical Orbbec observation backend")
            from tiptop.doosan.virtual_camera import resolve_camera_preparation

            camera_preparation = resolve_camera_preparation(observed, getattr(args, "camera_preparation", None))
            kin = get_m1013_uon_kinematics_model()
        known_fixed = []
        if hardware is None and "workspace_table_bounds_m" in observed:
            from curobo.geom.types import Cuboid

            bounds = np.asarray(observed["workspace_table_bounds_m"], float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
                raise ValueError("Invalid declared fixed-table geometry")
            known_fixed = [
                Cuboid(
                    name="configured_full_table",
                    dims=(bounds[1] - bounds[0]).tolist(),
                    pose=[*((bounds[0] + bounds[1]) / 2).tolist(), 1, 0, 0, 0],
                )
            ]
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
        args._stage = "perception"
        client = None
        use_gemini = mode == "gemini_er" and active is None
        if use_gemini:
            from google import genai
            from google.genai import types

            if not args.allow_gemini_once:
                raise ValueError("Gemini permission required: use --allow-gemini-once or --active-task")
        try:
            if use_gemini:
                # Missing credentials fail before reserving the one-call budget.
                client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
                marker = args.output / "gemini-call-consumed.json"
                with marker.open("x") as stream:
                    json.dump(dict(instruction=args.task, observation_id=observation_id, maximum_calls=1), stream)
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
                registered_grasp_from_tcp=None if hardware is None else hardware["grasp_from_tcp"],
                perception_mode=mode,
                sam3_url=getattr(args, "sam3_url", None),
                reference_scene=getattr(args, "reference_scene", None),
            )
        except APIError as error:
            from tiptop.perception.gemini import DEFAULT_GEMINI_MODEL

            headers = getattr(getattr(error, "response", None), "headers", {}) or {}
            allowed = {"date", "server", "retry-after", "x-request-id", "x-goog-request-id", "x-cloud-trace-context"}
            record = dict(
                model=DEFAULT_GEMINI_MODEL,
                observation_id=observation_id,
                code=error.code,
                status=error.status,
                message=error.message,
                response_headers={key: str(value) for key, value in headers.items() if key.lower() in allowed},
                automatic_retries=0,
            )
            (args.output / "gemini-error.json").write_text(json.dumps(record, indent=2) + "\n")
            raise
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
            f"{'Gemini -> SAM2' if mode == 'gemini_er' else 'Reference -> SAM3'} -> Depth -> FoundationPose -> UON DB\n"
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
        if hardware is None:
            from tiptop.doosan.virtual_camera import camera_planning_options

            model_options = camera_planning_options(camera_preparation)
            model_options["fixed_obstacles"] = [*known_fixed, *model_options["fixed_obstacles"]]
        else:
            model_options = {key: hardware[key] for key in ["robot_config", "gripper_spheres", "fixed_obstacles"]}
            model_options["camera_profile"] = None
        args._stage = "planning"
        plan, report = await asyncio.to_thread(
            plan_known_task,
            env,
            scene,
            observation,
            output=args.output,
            max_approach_tilt_deg=args.max_approach_tilt_deg,
            planning_settings=getattr(args, "planning_settings", None),
            **model_options,
        )
        fingerprint = None
        if hardware is not None:
            from tiptop.doosan.hardware import hardware_profile_fingerprint

            fingerprint = hardware_profile_fingerprint(measured_profile)
        plan.update(
            observation_id=observation_id,
            task=args.task,
            gemini_api_calls=int((args.output / "gemini-call-consumed.json").exists()),
            gemini_api_calls_this_process=int(use_gemini),
            perception_mode=mode,
            semantic_source=semantic["semantic_source"],
            hardware_profile=None if profile_path is None else str(Path(profile_path).resolve()),
            hardware_profile_sha256=fingerprint,
            robot_system=0 if hardware is not None else 1,
        )
        if hardware is None:
            from tiptop.doosan.virtual_camera import camera_preparation_id

            plan.update(
                camera_mount=camera_preparation["profile"]["mount"],
                camera_preparation_sha256=camera_preparation_id(camera_preparation),
            )
        (args.output / "plan.json").write_text(json.dumps(plan, cls=NumpyEncoder, indent=2) + "\n")
        if not args.execute:
            print("Complete plan saved; no motion requested.")
            return dict(status="planned", success=None, executed=False)
        args._stage = "execution"
        if hardware is not None:
            result = await execute_physical(plan, args.gateway_url)
        else:
            result = await execute_live_plan(session, args.camera_url, plan)
        (args.output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"trace", "calls"}}, ensure_ascii=False), flush=True
        )
        if result.get("success") is False:
            raise RuntimeError("Pick/place verification failed; see execution.json")
        return dict(status=result.get("status", "completed"), success=result.get("success"), executed=True)


async def execute_live_plan(session, camera_url, plan, *, timeout=240, poll_interval=2):
    async with session.post(camera_url + "/execute", json=json.loads(json.dumps(plan, cls=NumpyEncoder))) as response:
        response.raise_for_status()
        # Existing camera backends return JSON bytes as application/octet-stream.
        accepted = await response.json(content_type=None)
        if not accepted.get("accepted") or accepted.get("observation_id") != plan["observation_id"]:
            raise ValueError("Backend did not acknowledge this observation's execution")
    deadline = time.monotonic() + timeout
    query = urlencode(dict(observation_id=plan["observation_id"]))
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        async with session.get(camera_url + "/result?" + query) as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("observation_id") != plan["observation_id"]:
            raise ValueError("Backend returned another task's result; inspect before another execution")
        if result.get("finished"):
            return result
    raise TimeoutError("Execution result timed out; inspect the live backend before retrying")


async def execute_physical(plan, gateway_url):
    from tiptop.doosan.client import DoosanUONClient
    from tiptop.doosan.execution import execute_measured_plan

    client = DoosanUONClient(gateway_url)
    cancelled = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(execute_measured_plan, plan, client, cancelled=cancelled))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancelled.set()
        await asyncio.to_thread(client.stop)
        try:
            await worker
        except Exception:
            pass
        raise


def reserve_output_directory(requested):
    """Reserve a fresh output; preserve the common numbered-run/task layout."""
    requested = Path(requested).expanduser()
    if not requested.name:
        requested = requested / "task-001"
    parts = list(requested.parts)
    indices = [len(parts) - 1]
    if requested.name == "task" and len(parts) > 1:
        indices.append(len(parts) - 2)
    numbered = None
    for index in indices:
        match = re.fullmatch(r"(.*[-_])(\d+)", parts[index])
        if match is not None:
            numbered = index, match
            break
    attempt = 0
    while True:
        if attempt == 0:
            candidate = requested
        elif numbered is not None:
            index, match = numbered
            updated = parts.copy()
            updated[index] = match[1] + str(int(match[2]) + attempt).zfill(len(match[2]))
            candidate = Path(*updated)
        else:
            candidate = requested.with_name(f"{requested.name}-{attempt:03d}")
        attempt += 1
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            if not candidate.is_dir() or any(candidate.iterdir()):
                continue
        try:
            # Exclusive creation prevents concurrent commands sharing an empty
            # output directory, and preserves interrupted attempts as well.
            with (candidate / ".task-reserved").open("x"):
                pass
        except FileExistsError:
            continue
        return candidate


async def run_task(args):
    """A failure ends this instruction; the next instruction starts fresh."""
    args.output = reserve_output_directory(args.output)
    print(f"TASK_OUTPUT {args.output}", flush=True)
    args._stage, args._observation_id = "setup", None
    try:
        result = await run(args)
    except Exception as error:
        result = dict(
            status="failed", success=False, failed_stage=args._stage, error_type=type(error).__name__, error=str(error)
        )
        if hasattr(error, "perception_diagnostics"):
            result["perception_diagnostics"] = error.perception_diagnostics
        print("\033[31mTASK_FAILED " + str(error) + "\033[0m", flush=True)
    result.update(
        task=args.task,
        observation_id=args._observation_id,
        output=str(args.output),
        perception_mode=getattr(args, "perception_mode", None) or tiptop_cfg().perception.get("mode", "gemini_er"),
        gemini_api_calls=int((args.output / "gemini-call-consumed.json").exists()),
    )
    (args.output / "task-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


async def run_interactive(args, *, read_instruction=input):
    """Explicit user instructions only: no automatic target selection or retries."""
    args.output = reserve_output_directory(args.output)
    print(f"SESSION_OUTPUT {args.output}", flush=True)
    mode = getattr(args, "perception_mode", None) or tiptop_cfg().perception.get("mode", "gemini_er")
    cost = "Gemini 최대 1회" if mode == "gemini_er" else "SAM3 reference 추론 1회 (Gemini 0회)"
    print(f"작업 지시를 입력하세요. 각 지시는 새 관측과 {cost}를 사용합니다. 종료: /quit", flush=True)
    while True:
        try:
            # No task is running while waiting for input. Keeping terminal input
            # on the main thread also lets Ctrl-C interrupt it immediately.
            instruction = read_instruction("작업> ").strip()
        except EOFError:
            return
        if instruction in {"/quit", "/exit"}:
            return
        if not instruction:
            continue
        task = copy(args)
        task.task = instruction
        task.output = args.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        await run_task(task)
        gc.collect()


def main():
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="후추통 집고 빈 곳에 놓아")
    parser.add_argument("--camera-url", default="http://127.0.0.1:13345")
    parser.add_argument("--depth-url", default="http://127.0.0.1:13344")
    parser.add_argument("--pose-url", default="http://127.0.0.1:13346")
    parser.add_argument("--perception-mode", choices=["gemini_er", "sam3_reference"], default=None)
    parser.add_argument("--sam3-url", help="Override the SAM3 reference server URL")
    parser.add_argument("--reference-scene", help="Reference scene relative to the SAM3 server reference root")
    parser.add_argument(
        "--camera-preparation", type=Path,
        help="Optional virtual camera config assertion; otherwise use the live backend configuration (legacy: wrist)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tiptop_outputs/m1013/tasks/task-001"),
        help="Output directory; automatically increments when used (default: tiptop_outputs/m1013/tasks/task-001)",
    )
    parser.add_argument("--active-task", type=Path)
    parser.add_argument("--hardware-profile", type=Path, help="Measured physical mount/TCP/camera/workspace profile")
    parser.add_argument("--allow-gemini-once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connections and a fresh camera frame only; no inference or motion, even with --execute",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Keep accepting explicit instructions after each result"
    )
    parser.add_argument("--gateway-url", help="Physical gateway; requires --hardware-profile with --execute or --check")
    parser.add_argument(
        "--max-approach-tilt-deg",
        type=float,
        default=3.0,
        help="Hard maximum gripper approach tilt from world down in degrees (default: 3); never widened automatically",
    )
    defaults = KnownTaskPlanningSettings()
    parser.add_argument(
        "--motion-timeout-s",
        type=float,
        default=defaults.motion_timeout_s,
        help="Per-request MotionGen search budget in seconds (default: 1); no automatic increase",
    )
    parser.add_argument(
        "--motion-position-tolerance-m",
        type=float,
        default=defaults.motion_position_tolerance_m,
        help="MotionGen position convergence tolerance in metres (default: 0.01)",
    )
    parser.add_argument(
        "--motion-rotation-tolerance",
        type=float,
        default=defaults.motion_rotation_tolerance,
        help="MotionGen quaternion rotation convergence threshold (default: 0.1)",
    )
    parser.add_argument(
        "--placement-support-tolerance-m",
        type=float,
        default=defaults.placement_support_tolerance_m,
        help="Placement support-height tolerance in metres (default: 0.01)",
    )
    parser.add_argument(
        "--placement-xy-tolerance-m",
        type=float,
        default=defaults.placement_xy_tolerance_m,
        help="Placement summed XY boundary violation tolerance in metres (default: 0.01)",
    )
    parser.add_argument(
        "--release-clearance-m",
        type=float,
        default=None,
        help="Nominal CAD bottom clearance above support; default: support + motion position tolerances + 0.005 m",
    )
    args = parser.parse_args()
    try:
        args.planning_settings = KnownTaskPlanningSettings(
            **{name: getattr(args, name) for name in KnownTaskPlanningSettings.__dataclass_fields__}
        )
    except ValueError as error:
        parser.error(str(error))
    args.perception_mode = args.perception_mode or tiptop_cfg().perception.get("mode", "gemini_er")
    if args.perception_mode == "sam3_reference" and (args.active_task or args.allow_gemini_once):
        parser.error("Reference mode uses neither --active-task nor --allow-gemini-once")
    if not np.isfinite(args.max_approach_tilt_deg) or not 0 <= args.max_approach_tilt_deg <= 90:
        parser.error("--max-approach-tilt-deg must be finite degrees in [0, 90]")
    if args.active_task and args.allow_gemini_once:
        parser.error("Choose a new Gemini call OR continuation, not both")
    if args.interactive and (args.active_task or (args.perception_mode == "gemini_er" and not args.allow_gemini_once)):
        parser.error("Interactive tasks require --allow-gemini-once and cannot reuse --active-task ROIs")
    if args.hardware_profile and args.execute and not args.gateway_url:
        parser.error("Physical execution requires an explicit --gateway-url and measured --hardware-profile")
    if args.gateway_url and not (args.hardware_profile and (args.execute or args.check)):
        parser.error("--gateway-url requires --hardware-profile and either --execute or --check")
    logging.basicConfig(level=logging.INFO)
    try:
        if args.check:
            from tiptop.doosan.readiness import check_readiness

            result = asyncio.run(check_readiness(args))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = asyncio.run(run_interactive(args) if args.interactive else run_task(args))
    except KeyboardInterrupt:
        return
    if result and result.get("success") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
