"""Read-only connection checks; no model prediction, robot command or download."""

import asyncio
import importlib.util
import io
import os
from pathlib import Path

import aiohttp
import numpy as np

from tiptop.config import tiptop_cfg
from tiptop.object_catalog import load_catalog, registered_object
from tiptop.perception.cameras.geometry import _rigid_transform


async def _json(session, url, *, params=None):
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
        response.raise_for_status()
        # Isaac's existing observation backend labels JSON as octet-stream.
        return await response.json(content_type=None)


async def check_model_health(session, url):
    health = await _json(session, url.rstrip("/") + "/health")
    if health.get("status") != "healthy" or health.get("model_loaded") is not True:
        raise ValueError("Endpoint does not report a healthy loaded model")
    return health


def _observation(packet, profile, camera_preparation=None):
    from tiptop.perception.cameras.stereo import decode_camera_request

    rgb = np.asarray(packet["rgb"])
    k = np.asarray(packet["intrinsics"])
    q = np.asarray(packet["q_init"], float)
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[-1] != 3
        or min(rgb.shape[:2]) < 2
        or k.shape != (3, 3)
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or q.shape != (6,)
        or not np.isfinite(q).all()
        or not str(packet["observation_id"])
    ):
        raise ValueError("Camera observation requires RGB uint8, valid intrinsics, ID and six finite robot joints")
    _rigid_transform(packet["world_from_cam"], "world_from_cam")
    frame, estimator = decode_camera_request(packet, depth_trunc_m=tiptop_cfg().perception.depth_trunc_m)
    if estimator is None and not np.any(frame.depth > 0):
        raise ValueError("Camera returned no valid aligned sensor depth")
    physical = str(packet.get("camera_model", "")).startswith("orbbec/")
    if physical != (profile is not None):
        raise ValueError("Physical camera and measured --hardware-profile must be used together")
    if physical:
        from tiptop.doosan.hardware import validate_hardware_observation

        validate_hardware_observation(packet, profile)
    details = dict(
        camera_model=str(packet.get("camera_model", "unknown")),
        observation_id=str(packet["observation_id"]),
        rgb_size=[rgb.shape[1], rgb.shape[0]],
        physical=physical,
        depth_source="foundation_stereo" if estimator else str(packet.get("depth_source", "sensor")),
        q_init=q.tolist(),
    )
    if physical:
        details["camera_mount"] = "fixed" if profile["_camera_calibration"]["parent_link"] == "base_link" else "wrist"
        details["camera_pose_source"] = profile["_camera_calibration"].get("pose_source", "measured")
    else:
        from tiptop.doosan.virtual_camera import resolve_camera_preparation

        preparation = resolve_camera_preparation(packet, camera_preparation)
        details["camera_mount"] = preparation["profile"]["mount"]
    return details


def check_gateway_state(state, *, execute, observed_q=None, require_open=False):
    """State-only gateways suffice for planning, native feedback is needed to move."""
    q = np.asarray(state.get("joint_positions"), float)
    if (
        state.get("success") is not True
        or state.get("robot_system") != 0
        or state.get("motion_status") != 0
        or q.shape != (6,)
        or not np.isfinite(q).all()
    ):
        raise ValueError("Gateway requires current six-joint feedback from an idle physical robot")
    if observed_q is not None:
        observed_q = np.asarray(observed_q, float)
        if observed_q.shape != (6,) or not np.isfinite(observed_q).all() or np.max(np.abs(q - observed_q)) > 0.01:
            raise ValueError("Robot changed since observation; acquire and check again")
    gripper_open = None
    if execute:
        if state.get("read_only"):
            raise ValueError("Execution gateway is read-only")
        gripper = state.get("gripper", {})
        if gripper.get("backend") != "uon_native" or gripper.get("feedback_fresh") is not True:
            detail = gripper.get("error")
            raise ValueError("Execution requires fresh native UON gripper feedback" + (f": {detail}" if detail else ""))
        try:
            position, target, tolerance = (
                float(gripper[key])
                for key in ("position_counts", "open_position_counts", "position_tolerance_counts")
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("Gripper opening feedback/configuration missing; rebuild and restart the updated gateway") from None
        if not np.isfinite([position, target, tolerance]).all() or tolerance <= 0:
            raise ValueError("Gripper opening feedback/configuration is invalid")
        gripper_open = abs(position - target) <= tolerance
        if require_open and not gripper_open:
            raise ValueError("Gripper did not reach its open position; arm approach is blocked")
        if state.get("execution", {}).get("status") in {"failed", "cancelled"}:
            raise ValueError("Inspect and recover the previous gateway failure before another task")
        connections = state.get("connections", {})
        if (
            connections.get("trajectory_action_ready") is not True
            or connections.get("stop_service_ready") is not True
            or connections.get("gripper_command_subscribers") != 1
            or connections.get("gripper_command_publishers") != 1
            or connections.get("gripper_state_publishers") != 1
        ):
            raise ValueError(
                "Execution requires ready trajectory/stop interfaces and one native gripper driver; rebuild the gateway if connection status is missing"
            )
    return dict(robot_system=0, idle=True, read_only=bool(state.get("read_only")),
                gripper=state.get("gripper"), gripper_open=gripper_open)


async def check_readiness(args):
    """Check selected services and a fresh camera packet without running perception."""
    cfg = tiptop_cfg()
    checks = []

    async def collect(name, operation):
        try:
            details = await operation()
            checks.append(dict(name=name, success=True, details=details))
            return details
        except Exception as error:
            checks.append(dict(name=name, success=False, error=str(error)))
            return None

    profile = None
    profile_path = getattr(args, "hardware_profile", None)
    if profile_path is not None:

        async def hardware():
            nonlocal profile
            from tiptop.doosan.hardware import load_hardware_profile

            profile = load_hardware_profile(
                profile_path, allow_assumed_camera=not args.execute, allow_assumed_model=not args.execute
            )
            pose_source = profile["_camera_calibration"].get("pose_source", "measured")
            return dict(
                profile=str(Path(profile_path).resolve()), inputs_valid=True,
                camera_pose_source=pose_source,
                measured_inputs_valid=profile.get("calibrated") is True and pose_source != "assumed",
                planning_only=profile.get("calibrated") is not True or pose_source == "assumed",
            )

        await collect("hardware_profile", hardware)

    mode = args.perception_mode

    async def catalog():
        from tiptop.perception.sam3_reference import parse_reference_task

        class_ids = (
            [parse_reference_task(args.task)]
            if mode == "sam3_reference"
            else [row["class_id"] for row in load_catalog()["objects"]]
        )
        for class_id in class_ids:
            registered_object(class_id)
        return dict(classes=len(class_ids), cad_and_db_files_present=True)

    await collect("catalog", catalog)
    if cfg.perception.recgen.enabled:
        checks.append(dict(name="configuration", success=False, error="Known-DB mode requires RecGen disabled"))

    async with aiohttp.ClientSession() as session:

        async def camera():
            health = await _json(session, args.camera_url.rstrip("/") + "/health")
            if health.get("success") is not True or health.get("motion_stopped") or health.get("active_observation_id"):
                raise ValueError("Observation backend is unavailable or stopped after a motion failure")
            async with session.get(
                args.camera_url.rstrip("/") + "/observe", timeout=aiohttp.ClientTimeout(total=35)
            ) as response:
                response.raise_for_status()
                with np.load(io.BytesIO(await response.read()), allow_pickle=False) as packet:
                    details = _observation(packet, profile, getattr(args, "camera_preparation", None))
            return details

        # Capturing a new packet supersedes earlier unexecuted observation IDs.
        observation = await collect("camera_observation", camera)

        operations = [collect("foundation_pose", lambda: check_model_health(session, args.pose_url))]
        if observation is None or observation["depth_source"] == "foundation_stereo":
            operations.append(collect("foundation_stereo", lambda: check_model_health(session, args.depth_url)))

        if mode == "sam3_reference":

            async def reference():
                from tiptop.perception.sam3_reference import check_reference_health, parse_reference_task

                url = args.sam3_url or cfg.perception.sam3.url
                await check_reference_health(session, url)
                result = await _json(
                    session,
                    url.rstrip("/") + "/reference",
                    params=dict(
                        class_id=parse_reference_task(args.task),
                        scene=args.reference_scene or cfg.perception.sam3.reference_scene,
                    ),
                )
                if result.get("success") is not True or result.get("inference_performed") is not False:
                    raise ValueError("SAM3 reference validation endpoint is unavailable; restart the updated server")
                return result

            operations.append(collect("sam3_reference", reference))
        else:

            async def gemini_sam2():
                from tiptop.utils import get_tiptop_cache_dir

                if not args.active_task and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                    raise ValueError("Gemini API key is not configured; no remote authentication was attempted")
                if args.active_task and not args.active_task.is_file():
                    raise FileNotFoundError(args.active_task)
                if cfg.perception.sam.mode == "remote":
                    await _json(session, cfg.perception.sam.url.rstrip("/") + "/health")
                elif cfg.perception.sam.mode == "local":
                    if importlib.util.find_spec("sam2") is None:
                        raise ValueError("SAM2 is not installed")
                    if not (get_tiptop_cache_dir() / "sam2.1_hiera_large.pt").is_file():
                        raise ValueError("Local SAM2 checkpoint is missing; check does not download weights")
                else:
                    raise ValueError("Invalid SAM2 mode")
                return dict(sam2_mode=cfg.perception.sam.mode, gemini_authentication_verified=False)

            operations.append(collect("gemini_sam2", gemini_sam2))

        if args.gateway_url:

            async def gateway():
                state = await _json(session, args.gateway_url.rstrip("/") + "/state")
                return check_gateway_state(
                    state,
                    execute=args.execute,
                    observed_q=None if observation is None else np.asarray(observation["q_init"]),
                )

            operations.append(collect("robot_gateway", gateway))
        await asyncio.gather(*operations)

    return dict(
        success=all(item["success"] for item in checks),
        check_only=True,
        scope="execution_connections" if args.execute else "planning_connections",
        checks=checks,
        model_inference_calls=0,
        gemini_api_calls=0,
        motion_commands=0,
        note="Checks acquire a new observation. Object detection, pose accuracy, collision-free planning and grasp success remain unverified.",
    )
