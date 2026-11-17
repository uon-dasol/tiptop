"""M1013 adapter for existing TiPToP perception, cuTAMP and plan serialization.

The observation and ProcessedScene are device independent. No simulator object
state, recorded trajectory, or external source modification is used here.
"""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tiptop.console import log_tag
from tiptop.doosan.planning_runtime import serialized_planning

IK_CANDIDATE_FACTOR = 1.5  # CAD-clear candidates IK-solved per particle; IK failures are dropped


@dataclass(frozen=True)
class KnownTaskPlanningSettings:
    """Numerical planning limits, independent of collision and approach checks."""

    motion_timeout_s: float = 1.0
    motion_position_tolerance_m: float = 0.01
    motion_rotation_tolerance: float = 0.1
    max_wrist_travel_deg: float = 90.0
    min_table_clearance_m: float = 0.010
    min_engagement_depth_m: float = 0.020
    num_particles: int = 32  # most IK-feasible grasps handed to cuTAMP; each gets one particle
    robot_world_tolerance_m: float = 0.01  # summed overlap depth of robot spheres into obstacles that cuTAMP accepts

    def __post_init__(self):
        values = [value for name, value in asdict(self).items() if name != "min_engagement_depth_m"]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Planning time and tolerances must be finite and positive")
        if not np.isfinite(self.min_engagement_depth_m) or self.min_engagement_depth_m < 0:
            raise ValueError("Minimum engagement depth must be finite and not negative")
        if not isinstance(self.num_particles, int) or self.num_particles < 1:
            raise ValueError("num_particles must be a positive integer")


def log_candidate_stage(tag, label, before, after, seconds=None, note="", *, enough):
    """One funnel line: candidates kept and dropped by a stage, red when none remain, yellow when fewer than `enough`."""
    kind = "fail" if after == 0 else "warn" if after < enough else "ok"
    timing = "" if seconds is None else f" · {seconds:.2f}s"
    log_tag(tag, f"{label}  {before:,} → {after:,} (−{before - after:,}){note}{timing}", kind)


FINGER_SYMMETRY_RAD = 2 * math.pi / 3  # Fingers repeat every 120 degrees about the TCP z axis


def z_rotations(angles):
    """Homogeneous rotations about z for each angle in radians."""
    angles = np.asarray(angles, float)
    matrices = np.tile(np.eye(4), (len(angles), 1, 1))
    cos, sin = np.cos(angles), np.sin(angles)
    matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 0], matrices[:, 1, 1] = cos, -sin, sin, cos
    return matrices


def j6_symmetry_shifts(j6, j6_limits):
    """Multiples of the finger symmetry that move each J6 value closest to the middle of its limits."""
    neutral = float(np.mean(j6_limits))
    return -np.round((np.asarray(j6, float) - neutral) / FINGER_SYMMETRY_RAD) * FINGER_SYMMETRY_RAD


def turn_grasps(grasps, angles):
    """Turn each grasp about its own z axis by the matching angle in radians, on the grasps' device and dtype."""
    import torch

    grasps = torch.as_tensor(grasps)
    return grasps @ torch.as_tensor(z_rotations(angles), dtype=grasps.dtype, device=grasps.device)


def first_successful_solutions(solved):
    """First successful seed's joint solution per goal, as the screened-IK reuse selects it."""
    import torch

    rows = torch.arange(len(solved.solution), device=solved.solution.device)
    return solved.solution[rows, solved.success.to(torch.int32).argmax(dim=1)]


def fold_j6_by_symmetry(ik, solved, j6_limits):
    """Shift each goal's J6 by multiples of the finger symmetry toward the middle of `j6_limits`.

    J6 is the TCP z axis, so turning a grasp by 120 degrees only changes J6. The J6 window spans 360 degrees, so IK
    always lands inside it. A shift is kept only if the turned gripper body is still feasible. The first successful
    seed's solutions in `solved` are shifted in place; returns the applied shifts in radians.
    """
    import torch
    from curobo.types.state import JointState

    q = first_successful_solutions(solved)
    shifts = j6_symmetry_shifts(q[:, 5].cpu().numpy(), j6_limits)
    shifts_tensor = torch.as_tensor(shifts, dtype=q.dtype, device=q.device)
    folded = q.clone()
    folded[:, 5] += shifts_tensor
    state = JointState.from_position(folded, joint_names=list(ik.joint_names))
    free = ik.check_constraints(state).feasible.reshape(len(q), -1).all(dim=1)
    keep = free & solved.success.any(dim=1)
    shifts = np.where(keep.cpu().numpy(), shifts, 0.0)
    solved.solution[:, :, 5] += torch.as_tensor(shifts, dtype=q.dtype, device=q.device)[:, None]
    return shifts


def approach_tilt_mask(world_from_tcp, max_tilt_deg=3.0):
    """Hard bound on the DB grasp +Z approach axis relative to world down.

    Rotation about that axis is unrestricted. This supplements the geometry
    collision checks; it is not a clearance guarantee or an object tilt test.
    """
    limit = float(max_tilt_deg)
    if not np.isfinite(limit) or not 0 <= limit <= 90:
        raise ValueError("Maximum approach tilt must be finite degrees in [0, 90]")
    matrices = np.asarray(world_from_tcp, float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
        raise ValueError("Expected finite [N, 4, 4] grasp transforms")
    return -matrices[:, 2, 2] >= np.cos(np.deg2rad(limit)) - 1e-12


def engagement_depth_m(world_from_tcp, world_from_cad, cad_vertices):
    """Height of the highest posed CAD vertex above each TCP origin (the fingertip plane), in metres."""
    pose = np.asarray(world_from_cad, float)
    top_z = float((np.asarray(cad_vertices, float) @ pose[2, :3]).max() + pose[2, 3])
    return top_z - np.asarray(world_from_tcp, float)[:, 2, 3]


@serialized_planning
def plan_known_task(
    env,
    scene,
    observation,
    *,
    camera_profile,
    output,
    robot_config=None,
    gripper_spheres=None,
    fixed_obstacles=(),
    fixed_base_world_contact=False,
    max_approach_tilt_deg=3.0,
    planning_settings: KnownTaskPlanningSettings | None = None,
    reuse_screened_ik: bool = True,
    try_feasible_first: bool = True,
    reuse_planners: bool = True,
    unique_goals_first: bool = True,
    task_settings=None,
):
    import cutamp.algorithm as algorithm
    import cutamp.tamp_world as tamp_world
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from cutamp.config import TAMPConfiguration
    from cutamp.motion_solver import MotionPlanningError
    from cutamp.robots import RobotContainer
    from cutamp.utils.common import get_world_cfg
    from cutamp.utils.shapes import sample_greedy_surface_spheres

    from tiptop.doosan.camera import with_camera_envelope
    from tiptop.doosan.grasp_screening import screen_cad_grasps
    from tiptop.doosan.ik_reuse import ScreenedIKReuse
    from tiptop.doosan.known_planning import run_known_planning
    from tiptop.doosan.planning_runtime import cached_planner, configuration_key, reset_ik, reset_motion_gen, unique_path_goals_first
    from tiptop.doosan.model import ASSETS, m1013_uon_curobo_cfg
    from tiptop.doosan.planning import prepare_joint_path
    from tiptop.doosan.tamp import m1013_entrypoint, verify_gateway_contract
    from tiptop.perception.cad_cache import load_cad_mesh
    from tiptop.perception.known_objects import registered_object
    from tiptop.perception.placement import table_support_z
    from tiptop.doosan.in_place import HOME_DROP_OFFSET_M, SPEED_FACTOR, make_in_place_solver, source_region
    from tiptop.doosan.home_drop import validate_home_drop_plan
    from tiptop.planning import run_planning, serialize_plan
    from tiptop.task_presets import FIXED_BOX_NAME, TaskSettings, prepare_task_environment
    from tiptop.doosan.place_task import cad_bottom_correction, placement_height_context, run_standard_place, validate_place_plan
    from tiptop.utils import NumpyEncoder

    settings = KnownTaskPlanningSettings() if planning_settings is None else planning_settings
    task = TaskSettings() if task_settings is None else task_settings
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    if len(scene.grasps) != 1:
        raise ValueError("Known-task adapter expects exactly one catalog target")
    name, supplied = next(iter(scene.grasps.items()))
    result = supplied["registered"]
    data, pose = result["grasps"], result["pose"]["world_from_cad"]
    entry = registered_object(result["pose"]["class_id"])
    tcp = data["world_from_tcp"]
    tool = (
        np.load(ASSETS / "gripper_spheres_open.npy") if gripper_spheres is None else np.asarray(gripper_spheres, float)
    )
    if tool.ndim != 2 or tool.shape[1] != 4 or not len(tool) or not np.isfinite(tool).all() or np.any(tool[:, 3] <= 0):
        raise ValueError("Expected positive finite open-tool collision spheres at this model TCP")
    filter_started = time.perf_counter()
    centers = np.einsum("nij,kj->nki", tcp[:, :3, :3], tool[:, :3]) + tcp[:, None, :3, 3]
    support_z = table_support_z(scene.table_cuboid.pose[2] + scene.table_cuboid.dims[2] / 2)
    table_clearance = (centers[:, :, 2] - tool[None, :, 3] - support_z).min(1)
    # Reject candidates whose open gripper spheres come closer than the configured clearance to the
    # support plane, which lies 20 mm above the estimated tabletop.
    tilt_ok = approach_tilt_mask(tcp, max_approach_tilt_deg)
    valid = np.flatnonzero(tilt_ok & (table_clearance >= settings.min_table_clearance_m))
    table_filter_seconds = time.perf_counter() - filter_started
    cad_started = time.perf_counter()
    mesh = load_cad_mesh(entry["cad"], entry["cad_scale_to_m"])
    hull = mesh.convex_hull
    cad_model_seconds = time.perf_counter() - cad_started
    engagement_depth = engagement_depth_m(tcp, pose, mesh.vertices)
    deep_enough = valid[engagement_depth[valid] >= settings.min_engagement_depth_m]
    cad_started = time.perf_counter()
    # cuTAMP receives at most `num_particles` IK-feasible grasps, so IK only needs a margin of clear
    # candidates over that count.
    cad_candidate_limit = math.ceil(IK_CANDIDATE_FACTOR * settings.num_particles)
    cad_screening = screen_cad_grasps(tcp, pose, tool, deep_enough, hull, limit=cad_candidate_limit)
    screened = cad_screening.indices
    cad_check_seconds = time.perf_counter() - cad_started
    enough = settings.num_particles
    log_tag("Grasp", f"DB 파지 {len(tcp):,}개를 단계별로 거릅니다 · 탈락 수와 남은 수 · 노랑: 입자 수 {enough}개보다 적음", "info")
    log_candidate_stage("Grasp", f"① 접근각 ≤{max_approach_tilt_deg:g}°", len(tcp), int(tilt_ok.sum()), enough=enough)
    log_candidate_stage("Grasp", f"② 테이블 여유 ≥{settings.min_table_clearance_m * 1000:g}mm", int(tilt_ok.sum()), len(valid),
                        table_filter_seconds, enough=enough)
    log_candidate_stage("Grasp", f"③ 파지 깊이 ≥{settings.min_engagement_depth_m * 1000:g}mm", len(valid), len(deep_enough),
                        enough=enough)
    stopped = "" if cad_screening.complete else f", 통과 {cad_candidate_limit}개에서 중단"
    log_candidate_stage("Grasp", "④ CAD 충돌 없음", len(deep_enough), len(screened), cad_check_seconds,
                        f" · {cad_screening.checked}/{cad_screening.available}개 검사{stopped}", enough=enough)
    if not screened:
        raise ValueError(
            f"No DB grasp within {max_approach_tilt_deg:g} degrees clears observed table, "
            f"{settings.min_engagement_depth_m * 1000:g} mm minimum engagement depth and full CAD hull "
            f"(tilt and table: {len(valid)}, then depth: {len(deep_enough)}, then CAD hull: 0)"
        )
    screening_seconds = time.perf_counter() - started

    visual = scene.perception_results
    table = scene.table_cuboid
    env, region = prepare_task_environment(env, scene, task, fixed_obstacles=fixed_obstacles)
    obj = next(item for item in env.movables if item.name == name)
    sampled = sample_greedy_surface_spheres(obj, n_spheres=32, sphere_radius=0.003)
    cad_world = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
    table_bounds = (np.asarray(table.pose[:3]) + np.array([-.5, .5])[:, None]*np.asarray(table.dims))
    if task.holding_goal and (not region or region.get("source") != "pick_source"):
        region = source_region(table_bounds, cad_world)
    height_by_surface = {}
    if not task.holding_goal:
        from scipy.spatial.transform import Rotation

        object_pose = np.asarray(obj.pose)
        world_from_object = np.eye(4)
        world_from_object[:3, :3] = Rotation.from_quat(object_pose[[4, 5, 6, 3]]).as_matrix()
        world_from_object[:3, 3] = object_pose[:3]
        correction = cad_bottom_correction(sampled.cpu().numpy(), world_from_object, pose, hull.vertices)
        # If a sampled sphere extends below the CAD, preserve upstream's
        # collision clearance for zero-height placing rather than forcing it
        # into the box. Positive release heights use the complete CAD bottom.
        height_by_surface[region["destination"]] = max(0., task.release_height_m + correction)
        region["cad_bottom_correction_m"] = correction
        region["sampled_support_height_offset_m"] = height_by_surface[region["destination"]]
    visual["placement_region"] = region

    # Keep the original partial-hull Mesh in both the IK and motion worlds.
    # Only memoize its sphere sampling; substituting a sphere mesh here would
    # change the approach collision geometry.
    def cached_sample(current, n_spheres, sphere_radius, **kwargs):
        if current.name == name:
            if n_spheres != 32 or sphere_radius != 0.003:
                raise ValueError("Placement and collision sampling parameters differ")
            return sampled.clone()
        return sample_greedy_surface_spheres(current, n_spheres, sphere_radius, **kwargs)

    (output / "placement-region.json").write_text(json.dumps(region, cls=NumpyEncoder, indent=2) + "\n")
    env.statics.extend(fixed_obstacles)
    env.type_to_objects["Movable"] = env.movables
    cfg = copy.deepcopy(m1013_uon_curobo_cfg() if robot_config is None else robot_config)
    # Stay inside the wire limits after float32 planning/interpolation. Keep the
    # gateway's strict limits unchanged; never clamp a generated trajectory.
    cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
    cspace["position_limit_clip"] = np.maximum(cspace.get("position_limit_clip", 0), 0.001).tolist()
    if camera_profile is not None:
        cfg = with_camera_envelope(
            cfg,
            camera_profile,
            urdf_path=cfg["robot_cfg"]["kinematics"]["urdf_path"]
            if robot_config is not None
            else ASSETS / "m1013_uon.urdf",
            output_dir=output / "model",
        )
    # Include observed non-target hulls as well as the table and configured structures.
    static_world = get_world_cfg(copy.deepcopy(env), include_movables=False)
    model_key = configuration_key(cfg, tool, fixed_base_world_contact)
    def build_ik():
        ik = IKSolver(
            IKSolverConfig.load_from_robot_config(
                copy.deepcopy(cfg["robot_cfg"]),
                static_world,
                collision_cache={'obb': max(1, len(static_world.cuboid)), 'mesh': max(1, len(static_world.mesh))},
                num_seeds=32,
                self_collision_check=True,
                self_collision_opt=True,
                position_threshold=0.001,
                rotation_threshold=0.01,
                use_cuda_graph=False,
            )
        )
        if fixed_base_world_contact:
            from tiptop.doosan.fixed_base_contact import allow_fixed_base_mounting_contact
            allow_fixed_base_mounting_contact(ik)
        return ik
    ik, ik_runtime = cached_planner('ik', model_key, build_ik,
        lambda value: reset_ik(value, copy.deepcopy(static_world)), enabled=reuse_planners)
    joint_limits = ik.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
    if fixed_base_world_contact:
        from tiptop.doosan.fixed_base_contact import allow_fixed_base_mounting_contact, check_fixed_base_clearance

        # Only tabletop mounting contact is intentional. Other static geometry
        # must clear the stationary base before omitting its constant world cost.
        check_fixed_base_clearance(ik.kinematics, observation.q_init,
            [box for box in fixed_obstacles if box.name != "configured_full_table"]
            + [box for box in env.statics if box.name.startswith(FIXED_BOX_NAME)])
    batch = np.asarray(screened)
    ik_started = time.perf_counter()
    tcp_batch = tcp[batch]
    solved = ik.solve_batch(Pose.from_matrix(ik.tensor_args.to_device(tcp_batch)), return_seeds=4)
    ik_success = solved.success.any(dim=1)
    q_first = first_successful_solutions(solved)
    shifts = fold_j6_by_symmetry(ik, solved, joint_limits[:, 5])
    tcp_batch = tcp_batch @ z_rotations(shifts)
    screened_goals = Pose.from_matrix(ik.tensor_args.to_device(tcp_batch))
    reusable_ik = ScreenedIKReuse(ik, screened_goals, solved) if reuse_screened_ik else None
    solved_rows = ik_success.cpu().numpy()
    feasible = batch[solved_rows]
    ik_seconds = time.perf_counter() - ik_started
    if solved_rows.any():
        neutral = float(np.mean(joint_limits[:, 5]))
        before = np.degrees(np.abs(q_first[:, 5].cpu().numpy()[solved_rows] - neutral)).max()
        after = np.degrees(np.abs(q_first[:, 5].cpu().numpy()[solved_rows] + shifts[solved_rows] - neutral)).max()
        log_tag("IK", f"J6 120° 접기: {int(solved_rows.sum())}개 중 {int((shifts != 0).sum())}개 조정 · "
                f"중립 대비 최대 편차 {before:.0f}° → {after:.0f}°", "info")
    grasp_shifts = np.zeros(len(tcp))
    grasp_shifts[batch] = shifts
    # Angle, geometry and IK screening preserve the DB's descending score order.
    selected = feasible[:settings.num_particles]
    log_candidate_stage("IK", "⑤ 충돌 없는 IK 해", len(batch), len(feasible), ik_seconds, enough=enough)
    log_candidate_stage("IK", f"⑥ cuTAMP에 전달 (상한 {enough})", len(feasible), len(selected), enough=enough)
    if not len(selected):
        raise ValueError("No camera-aware IK candidate in the screened batch")
    tensor = TensorDeviceType()
    def build_container():
        kin = CudaRobotModel(RobotConfig.from_dict(copy.deepcopy(cfg["robot_cfg"]), tensor).kinematics)
        return RobotContainer(
            "m1013_uon",
            kin,
            kin.kinematics_config.joint_limits.position,
            tensor.to_device(tool),
            torch.eye(4, device=tensor.device),
        )
    container, container_runtime = cached_planner('container', model_key, build_container,
        lambda value: None, enabled=reuse_planners)
    kin = container.kin_model
    grasps = {
        name: dict(
            grasps_obj=turn_grasps(supplied["grasps_obj"][selected], grasp_shifts[selected]),
            confidences_pt=supplied["confidences_pt"][selected],
        )
    }
    config = TAMPConfiguration(
        robot="m1013_uon",
        num_particles=len(selected),
        ordered_grasps=True,
        num_opt_steps=100,
        max_loop_dur=30.0,
        m2t2_grasps=True,
        grasp_dof=6,
        num_initial_plans=1,
        curobo_plan=True,
        max_motion_refine_attempts=None,
        enable_visualizer=False,
        enable_experiment_logging=False,
        warmup_ik=False,
        warmup_motion_gen=False,
        coll_n_spheres=32,
        coll_sphere_radius=0.003,
        placement_check="obb",
        time_dilation_factor=SPEED_FACTOR,
    )
    robot_cfg = copy.deepcopy(cfg)
    robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 32

    def safe_world(environment, include_movables=False):
        return get_world_cfg(copy.deepcopy(environment), include_movables=include_movables)

    def build_motion_gen():
        initial_world = safe_world(env, include_movables=True)
        mg = MotionGen(
            MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=initial_world,
                collision_cache={'obb': max(1, len(initial_world.cuboid)),
                                 'mesh': max(1, len(initial_world.mesh))},
                use_cuda_graph=False,
                trajopt_particle_opt=False,
                num_trajopt_seeds=4,
                num_graph_seeds=4,
                position_threshold=settings.motion_position_tolerance_m,
                rotation_threshold=settings.motion_rotation_tolerance,
            )
        )
        if fixed_base_world_contact:
            allow_fixed_base_mounting_contact(mg)
        return mg
    mg, motion_runtime = cached_planner('motion', configuration_key(model_key, asdict(settings)), build_motion_gen,
        lambda value: reset_motion_gen(value, safe_world(env, include_movables=True)), enabled=reuse_planners)

    original_rank = algorithm.get_ranked_satisfying_particles
    particle_order = []
    motion_trials = dict(count=0, total=0)
    def rank_particles(plan_info, *args, **kwargs):
        particles = original_rank(plan_info, *args, **kwargs)
        distinct = ""
        if unique_goals_first:
            grasp_keys = [op.values[1] for op in plan_info['plan_skeleton'] if op.operator.name == 'Pick']
            particles, audit = unique_path_goals_first(particles, grasp_keys)
            particle_order.append(audit)
            distinct = f" (서로 다른 파지 {audit['unique_goals']}개)"
        motion_trials.update(count=0, total=len(particles["q0"]))
        log_tag("cuTAMP", f"입자 {config.num_particles}개 중 제약을 만족한 입자 {motion_trials['total']}개{distinct} · "
                          "이 순서로 모션 계획을 시도해 처음 성공하면 멈춥니다", "ok")
        return particles

    if task.preset == "home_drop":
        solver = make_in_place_solver(settings)
    else:
        solver = partial(run_standard_place, algorithm.solve_curobo, settings, region, pose, hull.vertices)
    def logged_solver(*args, **kwargs):
        motion_trials["count"] += 1
        label = f"{motion_trials['count']}/{motion_trials['total']}번째 입자"
        started = time.perf_counter()
        try:
            result = solver(*args, **kwargs)
        except MotionPlanningError as error:
            log_tag("Motion", f"{label} 실패 · {error} · {time.perf_counter() - started:.2f}s", "warn")
            raise
        log_tag("Motion", f"{label} 성공 · {time.perf_counter() - started:.2f}s", "ok")
        return result

    original_world_cost = tamp_world.get_world_collision_cost
    def world_cost(*args, **kwargs):
        from tiptop.doosan.fixed_base_contact import filter_fixed_base_world_cost

        return filter_fixed_base_world_cost(original_world_cost(*args, **kwargs), kin)

    surfaces = env.type_to_objects["Surface"]
    with (
        m1013_entrypoint(container, motion_timeout_s=settings.motion_timeout_s),
        patch.object(algorithm, "get_world_cfg", safe_world),
        patch.object(tamp_world, "get_world_cfg", safe_world),
        patch.object(algorithm, "solve_curobo", logged_solver),
        patch.object(algorithm, "get_ranked_satisfying_particles", rank_particles),
        patch.object(tamp_world, "sample_greedy_surface_spheres", cached_sample),
        patch.object(tamp_world, "get_world_collision_cost", world_cost if fixed_base_world_contact else original_world_cost),
        placement_height_context(height_by_surface),
    ):
        plan, elapsed, failure, search = run_known_planning(
            partial(run_planning, robot_world_tolerance=settings.robot_world_tolerance_m),
            env,
            config,
            observation.q_init,
            ik,
            grasps,
            mg,
            surfaces,
            screened_ik=reusable_ik,
            try_feasible_first=try_feasible_first,
        )
    if plan is None:
        raise RuntimeError(f"Existing cuTAMP found no complete {task.preset} plan: {failure}")
    serialized = serialize_plan(plan, observation.q_init)
    for step in plan:
        if step.get("label") == "Pick.descend" and "grasp_target_world_from_tcp" in step:
            serialized["selected_grasp"] = dict(world_from_tcp=step["grasp_target_world_from_tcp"],
                                                source="completed_motion_plan")
    serialized.update(
        robot="m1013_uon",
        placement_region=region,
        task_mode=task.preset,
        task_settings=asdict(task),
        returns_to_start=True,
        class_id=entry["class_id"],
        max_approach_tilt_deg=float(max_approach_tilt_deg),
        max_motion_refine_attempts=None,
        planning_settings=asdict(settings),
        label=name,
        physical_grasp_verified=False,
    )
    for step in serialized["steps"]:
        if step["type"] == "trajectory":
            positions, velocities, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
            step.update(positions=positions, velocities=velocities)

    def tcp_at(q):
        state = kin.get_state(tensor.to_device(np.asarray(q)[None]))
        return state.ee_pose.get_matrix().cpu().numpy()[0]

    if task.preset == "home_drop":
        check_name = "home_drop_check"
        return_check = validate_home_drop_plan(serialized, tcp_at, pose, hull.vertices, task.drop_region(),
            lift_distance_m=HOME_DROP_OFFSET_M, position_tolerance_m=settings.motion_position_tolerance_m)
    else:
        check_name = "placement_check"
        return_check = validate_place_plan(serialized, tcp_at, pose, hull.vertices, region,
                                          position_tolerance_m=settings.motion_position_tolerance_m)
    serialized[check_name] = return_check
    collision_registration = dict(
        surrounding_obstacles=visual.get("surrounding_obstacles", True),
        perceived_collision_labels=sorted(
            set(scene.object_meshes) & {obj.name for obj in [*env.movables, *env.statics]}
        ),
    )
    serialized.update(collision_registration)
    report = dict(
        **collision_registration,
        **{check_name: return_check},
        task_mode=task.preset,
        task_settings=asdict(task),
        planning_seconds=elapsed,
        planning_total_seconds=time.perf_counter() - started,
        candidate_screening_seconds=screening_seconds,
        candidate_table_filter_seconds=table_filter_seconds,
        cad_model_seconds=cad_model_seconds,
        cad_collision_check_seconds=cad_check_seconds,
        ik_screening_seconds=ik_seconds,
        ik_reuse=dict(
            {} if reusable_ik is None else reusable_ik.stats,
            enabled=reuse_screened_ik, complete_plan_fallback=(
                reusable_ik is not None and reusable_ik.stats["reused_batches"] > 0 and len(search["attempts"]) > 1
            ),
        ),
        planning_search=search,
        planning_settings=asdict(settings),
        planner_runtime=dict(enabled=reuse_planners, ik=ik_runtime, container=container_runtime, motion=motion_runtime),
        particle_order=dict(enabled=unique_goals_first, passes=particle_order),
        supplied_candidate_ids=data["candidate_indices"][selected].tolist(),
        tilt_and_table_candidates=len(valid),
        engagement_depth_candidates=len(deep_enough),
        supplied_engagement_depth_mm=(engagement_depth[selected] * 1000).tolist(),
        hull_clear_candidates=len(screened),
        hull_clear_candidates_complete=cad_screening.complete,
        cad_candidate_limit=cad_candidate_limit,
        cad_candidates_available=cad_screening.available,
        cad_candidates_checked=cad_screening.checked,
        cad_distance_queries=cad_screening.distance_queries,
        cad_distance_points=cad_screening.distance_points,
        gravity_preference_applied=False,
        supplied_approach_tilt_deg=np.rad2deg(np.arccos(np.clip(-tcp[selected, 2, 2], -1, 1))).tolist(),
        max_motion_refine_attempts=None,
        max_approach_tilt_deg=float(max_approach_tilt_deg),
        configured_fixed_obstacles=len(fixed_obstacles) + int(task.fixed_box_present),
        motion_planning_timeout_s=settings.motion_timeout_s,
        sequence=[step["label"] for step in plan],
        executor_calls=verify_gateway_contract(serialized),
        geometry_source="existing TiPToP observed partial hull",
        m2t2_api_calls=0,
    )
    (output / "planning-result.json").write_text(json.dumps(report, cls=NumpyEncoder, indent=2) + "\n")
    return serialized, report
