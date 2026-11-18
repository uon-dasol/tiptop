"""Run a fresh known-object task through TiPToP and a live observation backend.

Gemini requires --allow-gemini-once. A consumed call marker prevents retries.
--active-task continues the same identified instruction using a NEW image.
--perception-mode sam3_reference uses a fresh reference-based SAM3 detection,
without Gemini permission, calls, or SAM2 initialization.
"""

# ruff: noqa: E402
# Capture startup before the heavyweight imports so cold CLI cost stays visible.
import time

_CLI_STARTED_AT = time.perf_counter()

if __name__ == '__main__':
    import sys
    from tiptop.task_worker import forward_if_requested

    try:
        _worker_exit = forward_if_requested(sys.argv[1:])
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f'Task worker: {error}') from error
    if _worker_exit is not None:
        raise SystemExit(_worker_exit)

import argparse
import asyncio
import gc
import io
import json
import logging
import os
import re
import threading
import uuid
from copy import copy
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
import numpy as np
from google.genai.errors import APIError
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception.cameras.stereo import decode_camera_request
from tiptop.perception.robot_filter import points_inside_spheres
from tiptop.perception.sam3_reference import DEFAULT_CONFIDENCE_THRESHOLD
from tiptop.task_logging import LOG_OWNER_ENV, capture_terminal_log
from tiptop.task_timing import TaskTimings
from tiptop.console import timing_line
from tiptop.serialization import savez_fast
from tiptop.task_result_log import append_task_log
from tiptop.task_presets import TASK_PRESETS, TaskSettings, prepare_task_environment
from tiptop.utils import NumpyEncoder

_TASK_SEQUENCE = count(1)


_ELAPSED = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|seconds?)\b", re.IGNORECASE)


class _TiptopProgressFilter(logging.Filter):
    """Hide tiptop's INFO progress messages from the console; warnings and the elapsed-time lines stay."""

    def filter(self, record):
        return not (
            record.name.startswith("tiptop.") and record.levelno == logging.INFO and not _ELAPSED.search(record.getMessage())
        )


class _TimingFormatter(logging.Formatter):
    """Highlight TiPToP elapsed-time logs in the console."""

    module_labels = {
        "sam3_reference": "SAM3",
        "foundation_stereo": "FoundationStereo",
        "foundation_pose": "FoundationPose",
        "m2t2": "M2T2",
        "shape_completion": "RecGen",
        "perception_wrapper": "Perception",
        "tiptop_run": "Pipeline",
        "planning": "cuTAMP",
        "motion_planning": "cuRobo",
        "in_place": "cuRobo",
        "execute_plan": "Execution",
        "orbbec_camera": "Orbbec",
        "rs_camera": "RealSense",
        "zed_camera": "ZED",
    }

    def format(self, record):
        text = super().format(record)
        if (record.name.startswith("tiptop.") and record.levelno <= logging.INFO
                and _ELAPSED.search(record.getMessage())):
            module = record.name.rsplit(".", 1)[-1]
            label = self.module_labels.get(module, module)
            return f"\033[32m[{label}] {text}\033[0m"
        return text


def quiet_noisy_loggers():
    """Hide cuRobo's solver chatter (errors only) and cuTAMP's progress info (warnings and errors stay)."""
    logging.getLogger("curobo").setLevel(logging.ERROR)
    logging.getLogger("cutamp").setLevel(logging.WARNING)


def configured_full_table(fixed_obstacles):
    """The single fixed obstacle named like the profile's full table."""
    tables = [box for box in fixed_obstacles if box.name == "configured_full_table"]
    if len(tables) != 1:
        raise ValueError("--table-support declared needs exactly one configured full_table in the workspace")
    return tables[0]


async def run(args):
    timing = getattr(args, "_timing_trace", None)
    if timing is None:
        timing = args._timing_trace = TaskTimings()
        timing.phase("setup")
    from curobo.types.base import TensorDeviceType

    from tiptop.doosan.known_task import plan_known_task
    from tiptop.doosan.model import get_m1013_uon_kinematics_model
    from tiptop.perception.sam2 import sam2_client
    from tiptop.tiptop_run import Observation, run_perception

    args._timings = {}
    args._background = []
    args._preview = None
    args.output.mkdir(parents=True, exist_ok=True)
    task_settings = getattr(args, "task_settings", TaskSettings())
    perception_instruction = task_settings.perception_instruction(args.task)
    cfg = tiptop_cfg()
    cfg.perception.foundation_stereo.url = args.depth_url
    cfg.perception.foundation_pose_url = args.pose_url
    if cfg.perception.recgen.enabled:
        raise ValueError("Known-DB baseline uses the existing observed hull path; disable RecGen")
    mode = getattr(args, "perception_mode", None) or cfg.perception.get("mode", "gemini_er")
    if mode == "gemini_er":
        cfg.experimental.pick_place_next_to = False
        sam2_client()
    elif mode != "sam3_reference":
        raise ValueError(f"Unknown perception mode: {mode}")
    task_settings.validate_perception_mode(mode)
    async with aiohttp.ClientSession() as session:
        timing.phase("reference_readiness")
        if mode == "sam3_reference":
            from tiptop.perception.known_objects import prefetch_registered_assets
            from tiptop.perception.sam3_reference import check_reference_health, parse_reference_task

            if args.active_task:
                raise ValueError("Reference mode requires a fresh detection, not --active-task")
            class_id = parse_reference_task(perception_instruction)
            args._background.append(asyncio.create_task(asyncio.to_thread(prefetch_registered_assets, class_id)))
            await check_reference_health(session, args.sam3_url or cfg.perception.sam3.url)
        args._stage = "observation"
        timing.phase("observation_request")
        async with session.get(args.camera_url + "/observe", timeout=aiohttp.ClientTimeout(total=35)) as response:
            response.raise_for_status()
            observation_packet = await response.read()
            packet = np.load(io.BytesIO(observation_packet), allow_pickle=False)
            observed = {key: packet[key] for key in packet.files}
        observation_id = str(observed["observation_id"])
        args._observation_id = observation_id
        timing.phase("observation_save_decode")
        # Keep the exact camera/FK packet even if identification or placement
        # fails before planning-input.pkl can be written. Never use it as a live
        # execution observation without the backend's freshness checks.
        if not getattr(args, "minimal_records", False):
            (args.output / "observation.npz").write_bytes(observation_packet)
        frame, estimate = decode_camera_request(observed, depth_trunc_m=cfg.perception.depth_trunc_m)
        if getattr(args, "viewer", False):
            timing.phase("observation_viewer")
            from tiptop.perception_preview import TaskPreview

            try:
                args._preview = TaskPreview(
                    args.output, observation_id=observation_id, task=args.task,
                    depth_source=str(observed.get("depth_source", "unknown")),
                    max_tilt_deg=args.max_approach_tilt_deg,
                )
                args._preview.update("observation", dict(rgb=frame.rgb, depth=frame.depth))
            except OSError as error:
                logging.getLogger(__name__).warning("Perception viewer unavailable: %s", error)
        # Fail before any paid identification if a required local model is down.
        from tiptop.doosan.readiness import check_model_health

        args._stage = "connections"
        timing.phase("model_server_readiness")
        health_checks = [check_model_health(session, args.pose_url)]
        if estimate is not None:
            health_checks.append(check_model_health(session, args.depth_url))
        await asyncio.gather(*health_checks)
        timing.phase("hardware_preparation")
        hardware = None
        profile_path = getattr(args, "hardware_profile", None)
        physical = str(observed.get("camera_model", "")).startswith("orbbec/")
        if physical:
            from tiptop.doosan.hardware import cached_hardware_model, load_hardware_profile, validate_hardware_observation

            if profile_path is None:
                raise ValueError(
                    "Physical known-object planning requires --hardware-profile with measured mount/TCP/camera/workspace data"
                )
            measured_profile = load_hardware_profile(
                profile_path, allow_assumed_camera=not args.execute, allow_assumed_model=not args.execute
            )
            validate_hardware_observation(observed, measured_profile)
            pick_place_bounds = measured_profile["workspace"].get("pick_place_bounds_xy_m")
            hardware = cached_hardware_model(
                measured_profile, None if getattr(args, "minimal_records", False) else args.output / "hardware-model"
            )
            if args._preview is not None:
                args._preview.spheres = hardware["gripper_spheres"]
                args._preview.urdf_path = Path(hardware["robot_config"]["robot_cfg"]["kinematics"]["urdf_path"])
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
            pick_place_bounds = observed.get("pick_place_bounds_xy_m")
            kin = get_m1013_uon_kinematics_model()
        known_fixed = []
        timing.phase("robot_geometry_preparation")
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

        def robot_pixels(depth, intrinsics):
            from tiptop.perception.utils import depth_to_xyz

            xyz = depth_to_xyz(depth, intrinsics)
            camera = observed["world_from_cam"]
            xyz = xyz @ camera[:3, :3].T + camera[:3, 3]
            return points_inside_spheres(xyz.reshape(-1, 3), spheres, 0.003).reshape(depth.shape)

        async def depth_without_robot(current_session, current_frame):
            nonlocal removed

            with timing.measure("depth_estimation"):
                depth = (await estimate(current_session, current_frame)) if estimate else current_frame.depth.copy()
            with timing.measure("robot_depth_mask"):
                inside = await asyncio.to_thread(robot_pixels, depth, current_frame.intrinsics)
            removed = int(inside.sum())
            return np.where(inside, 0, depth)

        active = json.loads(args.active_task.read_text()) if args.active_task else None
        grasp_from_tcp = (
            np.asarray(camera_preparation.get("db_grasp_from_tcp", np.eye(4)), float)
            if hardware is None else hardware["grasp_from_tcp"]
        )
        args._stage = "perception"
        timing.phase("perception")
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
            table_support = getattr(args, "table_support", "ransac-cad")
            declared_table = None
            if table_support == "declared":
                declared_table = configured_full_table(hardware["fixed_obstacles"] if hardware else known_fixed)
            perception_started = time.perf_counter()
            env, _, scene, atoms = await run_perception(
                session,
                observation,
                perception_instruction,
                args.output,
                depth_estimator=depth_without_robot,
                include_workspace=False,
                log_to_rerun=False,
                grasp_provider="known_db",
                client=client,
                active_task=active,
                pick_only=task_settings.pick_only,
                registered_grasp_from_tcp=grasp_from_tcp,
                cad_mask_min_iou=getattr(args, "cad_mask_min_iou", 0.75),
                table_support=table_support,
                declared_table=declared_table,
                perception_mode=mode,
                sam3_url=getattr(args, "sam3_url", None),
                reference_scene=getattr(args, "reference_scene", None),
                sam3_confidence_threshold=getattr(args, "sam3_confidence_threshold", None),
                appearance_weight=args.appearance_weight if getattr(args, "appearance_ranking", False) else None,
                pick_place_bounds_xy=pick_place_bounds,
                preview=args._preview,
                background_tasks=args._background,
                surrounding_obstacles=getattr(args, "surrounding_obstacles", True),
                minimal_records=getattr(args, "minimal_records", False),
            )
            args._timings["perception"] = time.perf_counter() - perception_started
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
        timing.phase("perception_output_preparation")
        visual = scene.perception_results
        task_fixed = hardware["fixed_obstacles"] if hardware else known_fixed
        env, region = prepare_task_environment(env, scene, task_settings, fixed_obstacles=task_fixed)
        if region is not None:
            (args.output / "placement-region.json").write_text(json.dumps(region, cls=NumpyEncoder, indent=2) + "\n")
        (args.output / "task-settings.json").write_text(
            json.dumps(dict(task_preset=task_settings.preset, instruction=args.task,
                            perception_instruction=perception_instruction, placement_region=region),
                       cls=NumpyEncoder, ensure_ascii=False, indent=2) + "\n")
        if args._preview is not None and not task_settings.holding_goal:
            args._preview.placement(region, observation.world_from_cam, frame.intrinsics)
        elif args._preview is not None and task_settings.fixed_box_present:
            args._preview.placement(task_settings.box_region(), observation.world_from_cam, frame.intrinsics,
                                    is_destination=task_settings.preset == "home_drop")
        semantic = json.loads((args.output / "semantic-task.json").read_text())
        name = next(iter(scene.grasps))
        index = next(i for i, box in enumerate(visual["bboxes"]) if box["label"] == name)
        overlay = frame.rgb.copy()
        mask = visual["masks"][index, 0]
        overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 220, 80]) * 0.4).astype(np.uint8)
        if not getattr(args, "minimal_records", False):
            Image.fromarray(overlay).save(args.output / "target-overlay.png", compress_level=1)
        status = (
            f"{'Gemini -> SAM2' if mode == 'gemini_er' else 'Reference -> SAM3'} -> Depth -> FoundationPose -> UON DB\n"
            f"Target: {name}. Task: {TASK_PRESETS[task_settings.preset]['label']}. Planning complete path and home..."
        )
        timing.phase("camera_display")
        display = io.BytesIO()
        savez_fast(display, observation_id=observation_id, overlay=overlay, status=status)
        async with session.post(args.camera_url + "/display", data=display.getvalue()) as response:
            response.raise_for_status()
        timing.phase("planning_input_snapshot")
        # Preserve inference before planning mutates it; disk IO overlaps planning.
        if not getattr(args, "minimal_records", False):
            from tiptop.planning_input import save_planning_input_in_background

            args._planning_input_save = save_planning_input_in_background(
                env, scene, observation, args.output / "planning-input.pkl",
                background_tasks=args._background, timing=timing,
            )
        if hardware is None:
            from tiptop.doosan.virtual_camera import camera_planning_options

            model_options = camera_planning_options(camera_preparation)
            model_options["fixed_obstacles"] = [*known_fixed, *model_options["fixed_obstacles"]]
        else:
            model_options = {
                key: hardware[key]
                for key in ["robot_config", "gripper_spheres", "fixed_obstacles", "fixed_base_world_contact"]
            }
            model_options["camera_profile"] = None
        args._stage = "planning"
        timing.phase("planning")
        planning_job = asyncio.create_task(asyncio.to_thread(
            plan_known_task,
            env,
            scene,
            observation,
            output=args.output,
            max_approach_tilt_deg=args.max_approach_tilt_deg,
            planning_settings=getattr(args, "planning_settings", None),
            task_settings=task_settings,
            **model_options,
        ))
        try:
            plan, report = await asyncio.shield(planning_job)
        except asyncio.CancelledError:
            # A cancelled worker request must finish its CPU/GPU planning
            # thread before another request can change process-global adapters.
            await planning_job
            raise
        args._timings["planning"] = report["planning_total_seconds"]
        if args._preview is not None:
            timing.phase("plan_viewer")
            args._preview.update("planning", report)
            await asyncio.to_thread(args._preview.publish_plan, plan, observation.world_from_cam, frame.intrinsics)
        timing.phase("plan_export")
        fingerprint = None
        if hardware is not None:
            from tiptop.doosan.hardware import hardware_profile_fingerprint

            fingerprint = hardware_profile_fingerprint(measured_profile)
            plan["planning_only"] = (
                measured_profile.get("calibrated") is not True
                or measured_profile["_camera_calibration"].get("pose_source") == "assumed"
            )
        plan.update(
            observation_id=observation_id,
            task=args.task,
            gemini_api_calls=int((args.output / "gemini-call-consumed.json").exists()),
            gemini_api_calls_this_process=int(use_gemini),
            perception_mode=mode,
            semantic_source=semantic["semantic_source"],
            hardware_profile=None if profile_path is None else str(Path(profile_path).resolve()),
            hardware_profile_sha256=fingerprint,
            db_grasp_from_tcp=np.asarray(grasp_from_tcp).tolist(),
            robot_system=0 if hardware is not None else 1,
        )
        if hardware is None:
            from tiptop.doosan.virtual_camera import camera_preparation_id

            plan.update(
                camera_mount=camera_preparation["profile"]["mount"],
                camera_preparation_sha256=camera_preparation_id(camera_preparation),
            )
        else:
            plan["camera_pose_source"] = measured_profile["_camera_calibration"].get("pose_source", "measured")
        (args.output / "plan.json").write_text(json.dumps(plan, cls=NumpyEncoder, indent=2) + "\n")
        if not args.execute:
            print("Complete plan saved; no motion requested.")
            return dict(status="planned", success=None, executed=False)
        args._stage = "execution"
        timing.phase("execution_request")
        execution_started = time.perf_counter()
        if hardware is not None:
            result = await execute_physical(plan, args.gateway_url, report_path=args.output / "physical-execution.json")
        else:
            result = await execute_live_plan(session, args.camera_url, plan)
        args._timings["execution"] = result.get("execution_duration_s", time.perf_counter() - execution_started)
        args._execution_timings = result.get("execution_timings", {})
        timing.phase("execution_result_save")
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


async def execute_physical(plan, gateway_url, *, report_path=None):
    from tiptop.doosan.client import DoosanUONClient
    from tiptop.doosan.execution import execute_measured_plan

    client = DoosanUONClient(gateway_url)
    cancelled = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(
        execute_measured_plan, plan, client, cancelled=cancelled, report_path=report_path))
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


async def prepare_before_tasks(args, output):
    """No observation/gateway/inference requests; retain local caches for this process."""
    if not getattr(args, "preload_objects", None) or getattr(args, "_sequence_prepared", False):
        return None
    from tiptop.perception.sequence import SequencePreparationError, prepare_sequence

    cfg = tiptop_cfg()
    mode = getattr(args, "perception_mode", None) or cfg.perception.get("mode", "gemini_er")
    try:
        async with aiohttp.ClientSession() as session:
            report = await prepare_sequence(
                session, args.preload_objects, pose_url=args.pose_url,
                sam3_url=(args.sam3_url or cfg.perception.sam3.url) if mode == "sam3_reference" else None,
                reference_scene=(args.reference_scene or cfg.perception.sam3.reference_scene)
                if mode == "sam3_reference" else None,
            )
    except SequencePreparationError as error:
        (output / "sequence-preparation.json").write_text(json.dumps(error.report, ensure_ascii=False, indent=2)+"\n")
        raise
    (output / "sequence-preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    args._sequence_prepared = True
    print(f"SEQUENCE_READY objects={len(report['class_ids'])} elapsed={report['elapsed_s']:.2f}s", flush=True)
    return report


async def run_preparation_only(args):
    args.output = reserve_output_directory(args.output)
    try:
        report = await prepare_before_tasks(args, args.output)
    except Exception as error:
        print(f"SEQUENCE_PREPARATION_FAILED {error}", flush=True)
        return dict(status="preparation_failed", success=False, executed=False, error=str(error))
    return dict(status="prepared", success=True, executed=False, preparation=report)


async def run_task(args):
    """Reserve a fresh task directory and persist logs through completion/cancellation."""
    args.output = reserve_output_directory(args.output)
    external = getattr(args, "_terminal_log_external", False) or os.environ.get(LOG_OWNER_ENV) == "launcher"
    with nullcontext() if external else capture_terminal_log(args.output / "terminal.log"):
        return await _run_task(args)


async def _run_task(args):
    """A failure ends this instruction; the next instruction starts fresh."""
    timing = args._timing_trace = TaskTimings()
    task_index = next(_TASK_SEQUENCE)
    timing.phase("setup")
    args._background = []
    args._planning_input_save = None
    args._execution_timings = {}
    args._preview = None
    print(f"TASK_OUTPUT {args.output}", flush=True)
    args._stage, args._observation_id = "setup", None
    args._timings = {}
    try:
        if getattr(args, "preload_objects", None) and not getattr(args, "_sequence_prepared", False):
            args._stage = "sequence_preparation"
            timing.phase("sequence_preparation")
            await prepare_before_tasks(args, args.output)
            args._stage = "setup"
            timing.phase("setup")
        result = await run(args)
    except Exception as error:
        timing.finish(status="failed")
        result = dict(
            status="failed", success=False, failed_stage=args._stage, error_type=type(error).__name__, error=str(error)
        )
        if hasattr(error, "perception_diagnostics"):
            result["perception_diagnostics"] = error.perception_diagnostics
        print("\033[31mTASK_FAILED " + str(error) + "\033[0m", flush=True)
        if getattr(args, "_preview", None) is not None:
            args._preview.update("error", dict(message="작업 중단: " + str(error)))
    finally:
        timing.phase("background_save_wait")
        background = list(getattr(args, "_background", []))
        drain = asyncio.gather(*background, return_exceptions=True)
        try:
            outcomes = await asyncio.shield(drain)
        except asyncio.CancelledError:
            # Cancelling to_thread does not stop its disk writer. Drain it before
            # the worker accepts another task or the process shuts down.
            await drain
            raise
    failures = [str(outcome) for task, outcome in zip(background, outcomes)
                if isinstance(outcome, Exception) and task is not args._planning_input_save]
    if failures:
        logging.getLogger(__name__).error("Saving perception outputs failed: %s", failures)
        result["perception_save_errors"] = failures
    planning_failures = [str(outcome) for task, outcome in zip(background, outcomes)
                         if isinstance(outcome, Exception) and task is args._planning_input_save]
    if planning_failures:
        logging.getLogger(__name__).error("Saving planning input failed: %s", planning_failures)
        result["planning_input_save_errors"] = planning_failures
    timing.phase("task_result_save")
    result.update(
        task=args.task,
        observation_id=args._observation_id,
        output=str(args.output),
        perception_mode=getattr(args, "perception_mode", None) or tiptop_cfg().perception.get("mode", "gemini_er"),
        gemini_api_calls=int((args.output / "gemini-call-consumed.json").exists()),
    )
    (args.output / "task-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    timing.finish()
    timings = timing.snapshot()
    timings.update(
        process_id=os.getpid(), task_index_in_process=task_index,
        observation_id=args._observation_id, status=result["status"],
        max_approach_tilt_deg=getattr(args, "max_approach_tilt_deg", None),
        viewer=bool(getattr(args, "viewer", False)),
    )
    timings["legacy_totals_seconds"] = dict(args._timings)
    timings["execution_details_seconds"] = args._execution_timings
    cli_started = getattr(args, "_cli_started_at", None)
    if cli_started is not None:
        timings["cli_wall_seconds"] = time.perf_counter() - cli_started
        timings["cli_scope"] = "CLI module entry through task-result write; excludes pixi/Python startup"
        if getattr(args, '_worker_request', False):
            timings['cli_scope'] = 'Worker request through task-result write; client reports Worker Wall separately'
        timings["before_task_seconds"] = timing.started - cli_started
    try:
        (args.output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    except OSError as error:
        logging.getLogger(__name__).warning("Could not save timings.json: %s", error)
    append_task_log(args.output, result, timings["task_wall_seconds"], demo=getattr(args, "minimal_records", False))
    if all(key in args._timings for key in ("perception", "planning", "execution")):
        perception, planning, execution = (
            round(args._timings[key], 2) for key in ("perception", "planning", "execution")
        )
        print(
            timing_line(f"[Total] 인식 {perception:.2f}s + 전체 계획 {planning:.2f}s + "
                        f"동작 {execution:.2f}s = {perception + planning + execution:.2f}s"),
            flush=True,
        )
    print(timing_line(f"[Wall] 작업 전체 {timings['task_wall_seconds']:.2f}s (저장 완료 대기 포함)"), flush=True)
    if cli_started is not None:
        scope = '서버 요청 처리 시작부터' if getattr(args, '_worker_request', False) else '모듈 시작부터'
        suffix = '' if getattr(args, '_worker_request', False) else ' (pixi/Python 시작 제외)'
        print(timing_line(f"[CLI Wall] {scope} {timings['cli_wall_seconds']:.2f}s{suffix}"), flush=True)
    return result


async def run_interactive(args, *, read_instruction=input):
    """Explicit user instructions only: no automatic target selection or retries."""
    args.output = reserve_output_directory(args.output)
    print(f"SESSION_OUTPUT {args.output}", flush=True)
    mode = getattr(args, "perception_mode", None) or tiptop_cfg().perception.get("mode", "gemini_er")
    try:
        await prepare_before_tasks(args, args.output)
    except Exception as error:
        print(f"SEQUENCE_PREPARATION_FAILED {error}", flush=True)
        return dict(status="preparation_failed", success=False, executed=False, error=str(error))
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


FIXED_PLANNING_SETTINGS = {"motion_timeout_s", "robot_world_tolerance_m"}  # no CLI or UI option; set in KnownTaskPlanningSettings


def parse_args(argv=None):
    from tiptop.doosan.known_task import KnownTaskPlanningSettings
    from tiptop.perception.placement import TABLE_SUPPORT_MODES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="후추통 집고 박스에 놓아")
    parser.add_argument("--task-preset", choices=list(TASK_PRESETS), default="home_drop",
                        help="Placement task independent of --perception-mode")
    parser.add_argument("--fixed-box-present", action="store_true", default=TaskSettings().fixed_box_present,
                        help=argparse.SUPPRESS)  # Legacy positive flag; installed geometry cannot be disabled.
    parser.add_argument("--release-height-m", type=float,
                        help="Object bottom above destination top, metres (default: 0); "
                             "cuTAMP collision clearance also applies; ignored for in-place return")
    box_defaults = TaskSettings()
    for name in TaskSettings.__dataclass_fields__:
        if name.startswith("box_"):
            parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(box_defaults, name),
                                help="Fixed box geometry in base_link metres")
    parser.add_argument('--worker-socket', help='Submit one task to a persistent local worker (or TIPTOP_TASK_WORKER_SOCKET)')
    parser.add_argument("--camera-url", default="http://127.0.0.1:13345")
    parser.add_argument("--depth-url", default="http://127.0.0.1:13344")
    parser.add_argument("--pose-url", default="http://127.0.0.1:13346")
    parser.add_argument("--perception-mode", choices=["gemini_er", "sam3_reference"], default=None)
    parser.add_argument("--sam3-url", help="Override the SAM3 reference server URL")
    parser.add_argument("--reference-scene", help="Reference scene relative to the SAM3 server reference root")
    parser.add_argument(
        "--appearance-ranking", action="store_true",
        help="SAM3 mode: rank candidates by SAM3 score plus --appearance-weight times their colour similarity "
             "to the reference boxes, before verifying them",
    )
    parser.add_argument(
        "--appearance-weight", type=float, default=1.0,
        help="Weight of the colour similarity (0-1) added to the SAM3 score when --appearance-ranking is set (default: 1)",
    )
    parser.add_argument(
        "--sam3-confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"SAM3 mode: keep only candidates scoring above this (default: {DEFAULT_CONFIDENCE_THRESHOLD}); "
             "needs a SAM3 server that accepts the per-request value",
    )
    parser.add_argument("--preload-objects", nargs="+", metavar="CLASS_ID",
                        help="Prepare this sequence's catalog objects, or all, before starting tasks")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare caches/references only; no camera, inference, gateway or robot requests")
    parser.add_argument("--minimal-records", action="store_true",
                        help="Demo mode: skip the optional files (camera packet, perception outputs, candidate crops, target "
                             "overlay, planning-input pickle); the plan, results, timings and live viewer are kept")
    parser.add_argument("--viewer", action="store_true", help="Publish RGB/depth, masks and the final planned grasp to the local viewer")
    parser.add_argument(
        "--surrounding-obstacles", action=argparse.BooleanOptionalAction, default=True,
        help="Register all perceived objects for collision checking (default: on); "
             "--no-surrounding-obstacles keeps only the selected grasp target. Table and fixed structures remain checked",
    )
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
        "--cad-mask-min-iou", type=float, default=0.75,
        help="Minimum CAD silhouette/mask IoU in [0,1] (default: 0.75); mask pixels outside the CAD that are as dark as the table (shadow) are ignored",
    )
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
    parser.add_argument("--max-wrist-travel-deg", type=float, default=defaults.max_wrist_travel_deg,
                        help="Reject a planned leg whose J4 or J5 moves farther than this from its start, degrees "
                             "(default: 90)")
    parser.add_argument("--min-table-clearance-m", type=float, default=defaults.min_table_clearance_m,
                        help="Minimum distance of the open gripper spheres above the support plane (estimated "
                             "tabletop + 20 mm), metres (default: 0.01)")
    parser.add_argument("--min-engagement-depth-m", type=float, default=defaults.min_engagement_depth_m,
                        help="Minimum depth of the fingertip plane below the estimated object top, metres "
                             "(default: 0.02); 0 only requires the fingertips at or below the top")
    parser.add_argument("--num-particles", type=int, default=defaults.num_particles,
                        help="Most IK-feasible DB grasps, best score first, handed to cuTAMP, one particle each "
                             f"(default: {defaults.num_particles}); more grasps take longer to plan")
    parser.add_argument("--table-support", choices=TABLE_SUPPORT_MODES, default="ransac-cad",
                        help="How the table support height is obtained: RANSAC plane scored by the posed-CAD bottoms "
                             "(ransac-cad, default), RANSAC scored by the mask-bottom points (ransac-mask), or the "
                             "configured full table without RANSAC (declared). Non-default modes reject a CAD bottom "
                             "more than 30 mm from the support plane")
    args = parser.parse_args(argv)
    if args.prepare_only and (not args.preload_objects or args.execute or args.interactive or args.check or args.active_task):
        parser.error("--prepare-only requires --preload-objects and cannot combine with execute/interactive/check/active-task")
    if args.preload_objects:
        if args.check:
            parser.error("--check does not prepare caches; use --prepare-only --preload-objects instead")
        from tiptop.object_catalog import sequence_objects

        try:
            args.preload_objects = [item["class_id"] for item in sequence_objects(args.preload_objects)]
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    if not args.interactive:
        args._cli_started_at = _CLI_STARTED_AT
    try:
        args.planning_settings = KnownTaskPlanningSettings(
            **{name: getattr(args, name) for name in KnownTaskPlanningSettings.__dataclass_fields__
               if name not in FIXED_PLANNING_SETTINGS}
        )
    except ValueError as error:
        parser.error(str(error))
    args.perception_mode = args.perception_mode or tiptop_cfg().perception.get("mode", "gemini_er")
    try:
        args.task_settings = TaskSettings(preset=args.task_preset, release_height_m=args.release_height_m,
                                         fixed_box_present=args.fixed_box_present,
                                         **{name: getattr(args, name) for name in TaskSettings.__dataclass_fields__
                                            if name.startswith("box_")})
        args.task_settings.validate_perception_mode(args.perception_mode)
        args.task_settings.perception_instruction(args.task)
    except ValueError as error:
        parser.error(str(error))
    if args.perception_mode == "sam3_reference" and (args.active_task or args.allow_gemini_once):
        parser.error("Reference mode uses neither --active-task nor --allow-gemini-once")
    if not np.isfinite(args.max_approach_tilt_deg) or not 0 <= args.max_approach_tilt_deg <= 90:
        parser.error("--max-approach-tilt-deg must be finite degrees in [0, 90]")
    if not np.isfinite(args.cad_mask_min_iou) or not 0 <= args.cad_mask_min_iou <= 1:
        parser.error("--cad-mask-min-iou must be finite and between 0 and 1")
    if not np.isfinite(args.appearance_weight) or args.appearance_weight < 0:
        parser.error("--appearance-weight must be finite and not negative")
    if not np.isfinite(args.sam3_confidence_threshold) or not 0 <= args.sam3_confidence_threshold <= 1:
        parser.error("--sam3-confidence-threshold must be finite and between 0 and 1")
    if args.active_task and args.allow_gemini_once:
        parser.error("Choose a new Gemini call OR continuation, not both")
    if args.interactive and (args.active_task or (args.perception_mode == "gemini_er" and not args.allow_gemini_once)):
        parser.error("Interactive tasks require --allow-gemini-once and cannot reuse --active-task ROIs")
    if args.hardware_profile and args.execute and not args.gateway_url:
        parser.error("Physical execution requires an explicit --gateway-url and measured --hardware-profile")
    if args.gateway_url and not (args.hardware_profile and (args.execute or args.check)):
        parser.error("--gateway-url requires --hardware-profile and either --execute or --check")
    return args


def main():
    args = parse_args()
    console = logging.StreamHandler()
    console.setFormatter(_TimingFormatter("%(levelname)s:%(name)s:%(message)s"))
    console.addFilter(_TiptopProgressFilter())
    logging.basicConfig(level=logging.INFO, handlers=[console])
    quiet_noisy_loggers()
    try:
        if args.prepare_only:
            result = asyncio.run(run_preparation_only(args))
        elif args.check:
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
