"""M1013 adapter for existing TiPToP perception, cuTAMP and plan serialization.

The observation and ProcessedScene are device independent. No simulator object
state, recorded trajectory, or external source modification is used here.
"""

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


@dataclass(frozen=True)
class KnownTaskPlanningSettings:
    """Numerical planning limits, independent of collision and approach checks."""

    placement_support_tolerance_m: float = 0.01
    placement_xy_tolerance_m: float = 0.01
    motion_timeout_s: float = 1.0
    motion_position_tolerance_m: float = 0.01
    motion_rotation_tolerance: float = 0.1
    release_clearance_m: float | None = None

    def __post_init__(self):
        values = [
            self.placement_support_tolerance_m,
            self.placement_xy_tolerance_m,
            self.motion_timeout_s,
            self.motion_position_tolerance_m,
            self.motion_rotation_tolerance,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Planning time and tolerances must be finite and positive")
        # Leave 5 mm beyond the two translational error budgets. This is a
        # nominal model allowance, not a bound on calibration/rotation errors.
        budget = self.placement_support_tolerance_m + self.motion_position_tolerance_m
        clearance = budget + 0.005 if self.release_clearance_m is None else self.release_clearance_m
        if not np.isfinite(clearance) or clearance <= budget:
            raise ValueError("Release clearance must exceed support plus motion position tolerances")
        object.__setattr__(self, "release_clearance_m", float(clearance))


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
    max_approach_tilt_deg=3.0,
    planning_settings: KnownTaskPlanningSettings | None = None,
):
    import cutamp.algorithm as algorithm
    import cutamp.tamp_world as tamp_world
    import open3d as o3d
    import torch
    import trimesh
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from cutamp.config import TAMPConfiguration
    from cutamp.robots import RobotContainer
    from cutamp.utils.common import get_world_cfg
    from cutamp.utils.shapes import sample_greedy_surface_spheres

    from tiptop.doosan.camera import with_camera_envelope
    from tiptop.doosan.model import ASSETS, m1013_uon_curobo_cfg
    from tiptop.doosan.planning import prepare_joint_path
    from tiptop.doosan.tamp import m1013_entrypoint, verify_gateway_contract
    from tiptop.perception.known_objects import registered_object
    from tiptop.perception.placement import placement_surface_height
    from tiptop.planning import run_planning, serialize_plan
    from tiptop.utils import NumpyEncoder

    settings = KnownTaskPlanningSettings() if planning_settings is None else planning_settings
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
    centers = np.einsum("nij,kj->nki", tcp[:, :3, :3], tool[:, :3]) + tcp[:, None, :3, 3]
    support_z = scene.perception_results["empty_region"]["support_z_m"]
    table_clearance = (centers[:, :, 2] - tool[None, :, 3] - support_z).min(1)
    valid = np.flatnonzero(approach_tilt_mask(tcp, max_approach_tilt_deg) & (table_clearance >= 0.002))
    mesh = trimesh.load(entry["cad"], force="mesh", process=False)
    mesh.vertices *= entry["cad_scale_to_m"]
    hull = mesh.convex_hull
    inverse = np.linalg.inv(pose)
    screened, gaps = [], []
    for index in valid:
        transform = inverse @ tcp[index]
        local = tool[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        _, distances, _ = trimesh.proximity.closest_point_naive(hull, local)
        inside = (np.einsum("sfj,fj->sf", local[:, None] - hull.triangles_center[None], hull.face_normals) <= 1e-9).all(
            1
        )
        gap = float((np.where(inside, -distances, distances) - tool[:, 3]).min())
        if gap >= 0.0005:
            screened.append(int(index))
            gaps.append(gap)
    if not screened:
        raise ValueError(
            f"No DB grasp within {max_approach_tilt_deg:g} degrees clears observed table and full CAD hull"
        )
    screening_seconds = time.perf_counter() - started

    # All above-table scene points count, including objects the VLM did not name.
    # The depth estimator already removed visible robot points using current FK.
    visual = scene.perception_results
    xyz = visual["xyz_map"]
    target_index = next(i for i, box in enumerate(visual["bboxes"]) if box["label"] == name)
    good = np.isfinite(xyz).all(-1) & ~visual["masks"][target_index, 0]
    table = scene.table_cuboid
    low = np.asarray(table.pose[:3]) - np.asarray(table.dims) / 2
    high = np.asarray(table.pose[:3]) + np.asarray(table.dims) / 2
    good &= (xyz[..., 2] > support_z + 0.012) & (xyz[..., 2] < support_z + 0.5)
    good &= (xyz[..., 0] > low[0]) & (xyz[..., 0] < high[0])
    good &= (xyz[..., 1] > low[1]) & (xyz[..., 1] < high[1])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[good])).voxel_down_sample(0.008)
    points = np.asarray(cloud.points)
    labels = np.asarray(cloud.cluster_dbscan(eps=0.035, min_points=12)) if len(points) else np.array([])
    obstacles = []
    for index in sorted(set(labels.tolist()) - {-1}):
        group = points[labels == index]
        low, high = group.min(0), group.max(0)
        obstacles.append(
            Cuboid(
                name=f"observed_obstacle_{index}",
                dims=(high - low + 0.012).tolist(),
                pose=[*((low + high) / 2).tolist(), 1, 0, 0, 0],
            )
        )
    # Other objects remain static obstacles for this single-object task.
    env = copy.deepcopy(env)
    env.movables = [obj for obj in env.movables if obj.name == name]
    # cuTAMP already uses sampled partial-hull spheres for movable collision
    # checks. Cache that same representation so its real lower envelope can
    # also define the release height, with no full-CAD collision replacement.
    obj = env.movables[0]
    if not np.allclose(obj.pose[3:], [1, 0, 0, 0]):
        raise ValueError("Observed placement expects a world-aligned partial hull object frame")
    sampled = sample_greedy_surface_spheres(obj, n_spheres=32, sphere_radius=0.003)
    sphere_bottom = float((sampled[:, 2] - sampled[:, 3]).min())
    cad_world = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
    cad_bottom = float(cad_world[:, 2].min() - obj.pose[2])
    region = visual["empty_region"]
    surface = next(surface for surface in env.type_to_objects["Surface"] if surface.name == "observed_empty_region")
    top = placement_surface_height(
        support_z=support_z,
        cad_bottom=cad_bottom,
        sphere_bottom=sphere_bottom,
        release_clearance=settings.release_clearance_m,
    )
    surface.pose[2] = top - surface.dims[2] / 2
    region.update(
        planning_surface_top_z_m=top,
        release_clearance_m=settings.release_clearance_m,
        sampled_sphere_bottom_local_m=sphere_bottom,
        estimated_cad_bottom_local_m=cad_bottom,
        planner_support_buffer_m=0.002,
        support_tolerance_m=settings.placement_support_tolerance_m,
    )

    # Keep the original partial-hull Mesh in both the IK and motion worlds.
    # Only memoize its sphere sampling; substituting a sphere mesh here would
    # change the approach collision geometry.
    def cached_sample(current, n_spheres, sphere_radius, **kwargs):
        if current.name == name:
            if n_spheres != 32 or sphere_radius != 0.003:
                raise ValueError("Placement and collision sampling parameters differ")
            return sampled.clone()
        return sample_greedy_surface_spheres(current, n_spheres, sphere_radius, **kwargs)

    (output / "empty-region.json").write_text(json.dumps(region, cls=NumpyEncoder, indent=2) + "\n")
    env.statics.extend([*obstacles, *fixed_obstacles])
    env.type_to_objects["Movable"] = env.movables
    cfg = copy.deepcopy(m1013_uon_curobo_cfg() if robot_config is None else robot_config)
    if camera_profile is not None:
        cfg = with_camera_envelope(
            cfg,
            camera_profile,
            urdf_path=cfg["robot_cfg"]["kinematics"]["urdf_path"]
            if robot_config is not None
            else ASSETS / "m1013_uon.urdf",
            output_dir=output / "model",
        )
    static_world = WorldConfig(cuboid=[table, *obstacles, *fixed_obstacles])
    ik = IKSolver(
        IKSolverConfig.load_from_robot_config(
            copy.deepcopy(cfg["robot_cfg"]),
            static_world,
            num_seeds=32,
            self_collision_check=True,
            self_collision_opt=True,
            position_threshold=0.001,
            rotation_threshold=0.01,
            use_cuda_graph=False,
        )
    )
    batch = np.asarray(screened[:48])
    ik_started = time.perf_counter()
    solved = ik.solve_batch(Pose.from_matrix(ik.tensor_args.to_device(tcp[batch])), return_seeds=4)
    feasible = batch[solved.success.any(dim=1).cpu().numpy()]
    ik_seconds = time.perf_counter() - ik_started
    # Angle, geometry and IK screening preserve the DB's descending score order.
    selected = feasible[:16]
    if not len(selected):
        raise ValueError("No camera-aware IK candidate in the screened batch")
    tensor = TensorDeviceType()
    kin = CudaRobotModel(RobotConfig.from_dict(copy.deepcopy(cfg["robot_cfg"]), tensor).kinematics)
    container = RobotContainer(
        "m1013_uon",
        kin,
        kin.kinematics_config.joint_limits.position,
        tensor.to_device(tool),
        torch.eye(4, device=tensor.device),
    )
    grasps = {
        name: dict(grasps_obj=supplied["grasps_obj"][selected], confidences_pt=supplied["confidences_pt"][selected])
    }
    config = TAMPConfiguration(
        robot="m1013_uon",
        num_particles=16,
        num_opt_steps=100,
        max_loop_dur=30.0,
        m2t2_grasps=True,
        grasp_dof=6,
        num_initial_plans=1,
        curobo_plan=True,
        max_motion_refine_attempts=3,
        enable_visualizer=False,
        enable_experiment_logging=False,
        warmup_ik=False,
        warmup_motion_gen=False,
        coll_n_spheres=32,
        coll_sphere_radius=0.003,
        placement_check="obb",
        time_dilation_factor=0.2,
    )
    robot_cfg = copy.deepcopy(cfg)
    robot_cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 32
    mg = MotionGen(
        MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=get_world_cfg(copy.deepcopy(env), include_movables=True),
            use_cuda_graph=False,
            trajopt_particle_opt=False,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            position_threshold=settings.motion_position_tolerance_m,
            rotation_threshold=settings.motion_rotation_tolerance,
        )
    )

    def safe_world(environment, include_movables=False):
        return get_world_cfg(copy.deepcopy(environment), include_movables=include_movables)

    surfaces = env.type_to_objects["Surface"]
    with (
        m1013_entrypoint(container, motion_timeout_s=settings.motion_timeout_s),
        patch.object(algorithm, "get_world_cfg", safe_world),
        patch.object(tamp_world, "sample_greedy_surface_spheres", cached_sample),
    ):
        plan, elapsed, failure = run_planning(
            env,
            config,
            observation.q_init,
            ik,
            grasps,
            mg,
            surfaces,
            placement_support_tolerance=settings.placement_support_tolerance_m,
            placement_xy_tolerance=settings.placement_xy_tolerance_m,
        )
    if plan is None:
        raise RuntimeError(f"Existing cuTAMP found no complete pick/place plan: {failure}")
    serialized = serialize_plan(plan, observation.q_init)
    serialized.update(
        robot="m1013_uon",
        empty_region=visual["empty_region"],
        class_id=entry["class_id"],
        max_approach_tilt_deg=float(max_approach_tilt_deg),
        planning_settings=asdict(settings),
        label=name,
        physical_grasp_verified=False,
    )
    for step in serialized["steps"]:
        if step["type"] == "trajectory":
            positions, velocities, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
            step.update(positions=positions, velocities=velocities)
    report = dict(
        planning_seconds=elapsed,
        planning_total_seconds=time.perf_counter() - started,
        candidate_screening_seconds=screening_seconds,
        ik_screening_seconds=ik_seconds,
        planning_settings=asdict(settings),
        supplied_candidate_ids=data["candidate_indices"][selected].tolist(),
        hull_clear_candidates=len(screened),
        gravity_preference_applied=False,
        supplied_approach_tilt_deg=np.rad2deg(np.arccos(np.clip(-tcp[selected, 2, 2], -1, 1))).tolist(),
        observed_obstacles=len(obstacles),
        max_approach_tilt_deg=float(max_approach_tilt_deg),
        configured_fixed_obstacles=len(fixed_obstacles),
        placement_support_tolerance_m=settings.placement_support_tolerance_m,
        placement_xy_tolerance_m=settings.placement_xy_tolerance_m,
        motion_planning_timeout_s=settings.motion_timeout_s,
        sequence=[step["label"] for step in plan],
        executor_calls=verify_gateway_contract(serialized),
        geometry_source="existing TiPToP observed partial hull",
        m2t2_api_calls=0,
    )
    (output / "planning-result.json").write_text(json.dumps(report, cls=NumpyEncoder, indent=2) + "\n")
    return serialized, report
