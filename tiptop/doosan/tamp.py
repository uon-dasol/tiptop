"""Minimal M1013/UON cuTAMP connection check, without editing external repositories.

Run in a dedicated process: the two cuTAMP entry-point substitutions are temporary
process globals and are not intended for a concurrent planning server.
"""

import argparse
import json
import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tiptop.object_catalog import registered_object

ROOT = Path(__file__).resolve().parents[2]
OBJECT = "black_pepper_shaker"
_PEPPER = registered_object(OBJECT, check_files=False)
DATA = _PEPPER["db"].parent
CAD = _PEPPER["cad"]
DEFAULT_TARGET_XYZ = (0.65, 0.0, 0.38)
DEFAULT_TARGET_DIMS = (0.8, 0.8, 0.04)


def motion_world_cfg(env):
    """Keep upstream world conversion from appending statics to env.movables."""
    from cutamp.utils.common import get_world_cfg

    return get_world_cfg(deepcopy(env), include_movables=True)


@contextmanager
def m1013_entrypoint(container, *, motion_timeout_s=None):
    """Supply the existing container and extend only the robot-name validation.

    Keep the real config.robot=m1013_uon throughout planning. The upstream validator
    checks other options on a copy with an accepted name; no UR5 model is loaded.
    IK and MotionGen must be passed explicitly. GUI gripper animation is unsupported.
    """
    from contextlib import ExitStack

    import cutamp.algorithm as algorithm
    import cutamp.motion_solver as motion_solver

    validate = algorithm.validate_tamp_config
    load = algorithm.load_robot_container

    def validate_config(config):
        if config.robot == "m1013_uon":
            if config.enable_visualizer:
                raise ValueError("This minimal adapter requires enable_visualizer=False")
            validate(replace(config, robot="ur5"))
        else:
            validate(config)

    def load_container(robot, tensor_args):
        return container if robot == "m1013_uon" else load(robot, tensor_args)

    original_plan_config = motion_solver.MotionGenPlanConfig

    def plan_config(*args, **kwargs):
        # The upstream 250 ms budget can expire while Isaac shares the GPU.
        # Give the same collision-constrained solver time; do not change its
        # geometry, constraints, or candidate count.
        kwargs["timeout"] = motion_timeout_s
        return original_plan_config(*args, **kwargs)

    if motion_timeout_s is not None and (not np.isfinite(motion_timeout_s) or motion_timeout_s <= 0):
        raise ValueError("Motion planning timeout must be finite and positive")
    with ExitStack() as stack:
        stack.enter_context(patch.object(algorithm, "validate_tamp_config", validate_config))
        stack.enter_context(patch.object(algorithm, "load_robot_container", load_container))
        if motion_timeout_s is not None:
            stack.enter_context(patch.object(motion_solver, "MotionGenPlanConfig", plan_config))
        yield


def load_case(
    data,
    cad,
    candidate_index,
    object_xyz,
    target_xyz,
    grasp_from_tcp=None,
    object_rpy=(0.0, 0.0, 0.0),
    target_dims=DEFAULT_TARGET_DIMS,
):
    """Read source assets only; create a small convex planning mesh in memory."""
    import trimesh
    from curobo.geom.types import Cuboid, Mesh
    from cutamp.envs import TAMPEnvironment
    from cutamp.tamp_domain import On
    from scipy.spatial.transform import Rotation

    merged = json.loads((data / f"{OBJECT}_merged_grasp_zero_pose.json").read_text())
    if merged["metadata"]["coordinate_frame"] != "object_zero_pose":
        raise ValueError("Expected object_zero_pose grasp records")
    row = merged["data"][candidate_index]
    vertices = (
        np.array(
            [list(map(float, line.split()[1:4])) for line in cad.read_text().splitlines() if line.startswith("v ")]
        )
        * 0.001
    )
    hull = trimesh.convex.convex_hull(vertices)
    xyzw = Rotation.from_euler("xyz", object_rpy).as_quat()
    object_pose = [*object_xyz, float(xyzw[3]), *xyzw[:3].tolist()]
    obj = Mesh(name=OBJECT, pose=object_pose, vertices=hull.vertices.tolist(), faces=hull.faces.tolist())
    if len(target_dims) != 3 or not np.isfinite(target_dims).all() or np.any(np.asarray(target_dims) <= 0):
        raise ValueError("target_dims must contain three positive finite dimensions")
    target = Cuboid(name="target", dims=list(target_dims), pose=[*target_xyz, 1, 0, 0, 0])
    env = TAMPEnvironment(
        "m1013_one_pepper",
        [obj],
        [target],
        {"Movable": [obj], "Surface": [target]},
        frozenset({On.ground(OBJECT, "target")}),
    )

    # Recorded contacts fit the model fingers without the former Rx(pi) flip.
    # Zero translation remains provisional; this is not a calibrated TCP offset.
    transform = np.eye(4) if grasp_from_tcp is None else np.asarray(grasp_from_tcp)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("grasp_from_tcp must be a finite rigid 4x4 transform")
    obj_from_tcp = np.asarray(row["grasp_mat"]) @ transform
    inputs = {
        "candidate_index": candidate_index,
        "scene_id": row["scene_id"],
        "source_pregrasp_index": row["source_pregrasp_index"],
        "score": row["score"],
        "data": str(data.resolve()),
        "cad": str(cad.resolve()),
        "obj_to_m_scale": 0.001,
        "object_xyz": list(object_xyz),
        "target_xyz": list(target_xyz),
        "target_dims": list(target_dims),
        "object_rpy": list(object_rpy),
        "object_pose": object_pose,
        "goal": f"On({OBJECT}, target)",
        "goal_is_region_not_exact_pose": True,
        "object_extent_m": np.ptp(vertices, axis=0).tolist(),
        "db_grasp_from_model_tcp": transform.tolist(),
        "frame_calibrated": False,
        "frame_source": "contact_checked_identity_zero_translation" if grasp_from_tcp is None else "user_supplied",
        "model_gripper_locked_open_during_planning": True,
        "physical_grasp_verified": False,
    }
    return env, obj_from_tcp, inputs


class RecordingClient:
    """Exercise the real executor contract without sending commands to a gateway."""

    def __init__(self, q_init):
        self.q = np.array(q_init, dtype=float)
        self.calls = []

    def open_gripper(self, **kwargs):
        self.calls.append({"type": "gripper", "action": "open"})
        return {"success": True}

    def close_gripper(self, **kwargs):
        self.calls.append({"type": "gripper", "action": "close"})
        return {"success": True}

    def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
        q, v, dt = np.asarray(joint_confs), np.asarray(joint_vels), np.asarray(durations)
        if (
            q.ndim != 2
            or q.shape[1] != 6
            or len(q) == 0
            or q.shape != v.shape
            or dt.shape != (len(q),)
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or not np.isfinite(dt).all()
            or np.any(dt <= 0)
        ):
            raise ValueError("Expected finite six-axis positions/velocities and positive durations")
        if np.max(np.abs(q[0] - self.q)) > np.radians(1):
            raise ValueError("Discontinuous trajectory start")
        self.q = q[-1].copy()
        self.calls.append({"type": "trajectory", "waypoints": len(q), "dof": 6})
        return {"success": True}


def check(args, output, report):
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from cutamp.algorithm import setup_cutamp
    from cutamp.config import TAMPConfiguration
    from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place
    from cutamp.task_planning import task_plan_generator
    from cutamp.utils.common import get_world_cfg

    from tiptop.doosan.model import get_m1013_uon_ik_solver, load_m1013_uon_container, m1013_uon_curobo_cfg
    from tiptop.doosan.planning import prepare_joint_path
    from tiptop.planning import run_planning, save_tiptop_plan, serialize_plan

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    transform = json.loads(args.grasp_from_tcp.read_text()) if args.grasp_from_tcp else None
    env, obj_from_tcp, inputs = load_case(
        args.data,
        args.cad,
        args.candidate,
        args.object_xyz,
        args.target_xyz,
        transform,
        args.object_rpy,
        args.target_dims,
    )
    report.update(inputs)
    report["stage"] = "initialization"
    tensor_args = TensorDeviceType()
    container = load_m1013_uon_container(tensor_args)
    ik = get_m1013_uon_ik_solver(get_world_cfg(env, include_movables=False))
    config = TAMPConfiguration(
        robot="m1013_uon",
        num_particles=args.particles,
        num_opt_steps=100,
        max_loop_dur=args.seconds,
        m2t2_grasps=True,
        grasp_dof=6,
        num_initial_plans=1,
        curobo_plan=True,
        max_motion_refine_attempts=4,
        enable_visualizer=False,
        enable_experiment_logging=False,
        warmup_ik=False,
        warmup_motion_gen=False,
        coll_n_spheres=32,
        coll_sphere_radius=0.003,
        placement_check="obb",
        time_dilation_factor=0.2,
    )
    with m1013_entrypoint(container):
        _, _, _, world = setup_cutamp(env, config, args.q_init, ik_solver=ik)
        skeleton = next(
            iter(task_plan_generator(world.initial_state, world.goal_state, [MoveFree, MoveHolding, Pick, Place]))
        )
        report["robot"] = world.robot_name
        report["joint_limits_shape"] = list(container.joint_limits.shape)
        report["task_skeleton"] = [op.name for op in skeleton]
        report["q_init"] = list(args.q_init)
        report["fk_tcp"] = container.kin_model.get_state(world.q_init[None]).ee_pose.get_matrix()[0].cpu().tolist()
        robot_cfg = deepcopy(m1013_uon_curobo_cfg())
        robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = config.coll_n_spheres
        mg = MotionGen(
            MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=motion_world_cfg(env),
                use_cuda_graph=False,
                trajopt_particle_opt=False,
                num_trajopt_seeds=8,
                num_graph_seeds=8,
                position_threshold=0.01,
                rotation_threshold=0.1,
            )
        )
        report["motion_gen_dof"] = mg.kinematics.get_dof()
        if args.init_only:
            report.update(success=True, stage="initialized", task_plan_only=True)
            return
        grasps = {
            OBJECT: {
                "grasps_obj": tensor_args.to_device(obj_from_tcp[None]),
                "confidences_pt": tensor_args.to_device([1.0]),
            }
        }
        report["stage"] = "run_cutamp"
        plan, elapsed, failure = run_planning(env, config, np.array(args.q_init), ik, grasps, mg, env.statics)
    report.update(planning_seconds=elapsed, planning_failure=failure)
    if plan is None:
        raise RuntimeError(f"cuTAMP returned no executable plan: {failure}")
    report["stage"] = "serialize_plan"
    serialized = serialize_plan(plan, np.array(args.q_init))
    serialized.update(robot="m1013_uon", frame_calibrated=False, physical_grasp_verified=False)
    for step in serialized["steps"]:
        if step["type"] == "trajectory":
            positions, velocities, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
            step.update(positions=positions, velocities=velocities)
    report["stage"] = "gateway_contract"
    gateway_calls = verify_gateway_contract(serialized)
    save_tiptop_plan(serialized, output / "plan.json")
    report.update(
        success=True,
        stage="executor_verified",
        executor_calls=gateway_calls,
        gateway_contract_verified=True,
        terminal_velocity_helper="prepare_joint_path",
        steps=len(plan),
        sequence=[s["label"] for s in plan],
        plan_file=str(output / "plan.json"),
    )


def verify_gateway_contract(plan):
    """Run the real client and gateway payload validator with in-process transport."""
    import importlib.util

    from tiptop.doosan.client import DoosanUONClient
    from tiptop.execute_plan import execute_cutamp_plan

    path = ROOT / "ros2_ws/src/execution/tiptop_execution/tiptop_execution/trajectory.py"
    spec = importlib.util.spec_from_file_location("m1013_gateway_trajectory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = RecordingClient(plan["q_init"])

    def request(path, payload=None, timeout=10.0):
        # Exercise JSON conversion as the actual HTTP transport does.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        if path == "/trajectory":
            q, v, _ = module.validate_trajectory(payload)
            return recorder.execute_joint_impedance_path(q, v, payload["durations"])
        if path == "/gripper":
            return getattr(recorder, f"{payload['action']}_gripper")()
        raise ValueError(f"Unexpected gateway request: {path}")

    client = DoosanUONClient()
    client._request = request
    execute_cutamp_plan(plan["steps"], client)
    return recorder.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--cad", type=Path, default=CAD)
    parser.add_argument("--output", type=Path, default=ROOT / "tiptop_outputs/m1013/tamp")
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--object-xyz", type=float, nargs=3, default=[0.55, -0.12, 0.5])
    parser.add_argument(
        "--object-rpy",
        type=float,
        nargs=3,
        default=[0.0] * 3,
        help="Object rotation in world coordinates, XYZ Euler radians",
    )
    parser.add_argument("--target-xyz", type=float, nargs=3, default=DEFAULT_TARGET_XYZ)
    parser.add_argument("--target-dims", type=float, nargs=3, default=DEFAULT_TARGET_DIMS)
    parser.add_argument("--q-init", type=float, nargs=6, default=[0.0] * 6)
    parser.add_argument("--grasp-from-tcp", type=Path, help="JSON 4x4 DB-grasp-from-model-TCP transform")
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    # Outputs must stay in the project; /media assets are read-only inputs.
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error("--output must be a subdirectory of the TiPToP repository")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    report = {"success": False, "stage": "inputs", "sent_to_robot": False, "external_cutamp_files_modified": False}
    try:
        check(args, output, report)
    except Exception as exc:
        logging.exception("M1013 cuTAMP check failed")
        report.update(error_type=type(exc).__name__, error=str(exc))
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
