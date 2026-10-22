"""M1013 adapter for existing TiPToP perception, cuTAMP and plan serialization.

The observation and ProcessedScene are device independent. No simulator object
state, recorded trajectory, or external source modification is used here.
"""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def plan_known_task(env, scene, observation, *, camera_profile, output):
    import cutamp.algorithm as algorithm
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

    from tiptop.doosan.camera import with_camera_envelope
    from tiptop.doosan.model import ASSETS, get_m1013_uon_gripper_spheres, m1013_uon_curobo_cfg
    from tiptop.doosan.planning import prepare_joint_path
    from tiptop.doosan.tamp import m1013_entrypoint, verify_gateway_contract
    from tiptop.perception.known_objects import registered_object
    from tiptop.planning import run_planning, serialize_plan
    from tiptop.utils import NumpyEncoder

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
    tool = np.load(ASSETS / "gripper_spheres_open.npy")
    centers = np.einsum("nij,kj->nki", tcp[:, :3, :3], tool[:, :3]) + tcp[:, None, :3, 3]
    support_z = scene.perception_results["empty_region"]["support_z_m"]
    table_clearance = (centers[:, :, 2] - tool[None, :, 3] - support_z).min(1)
    valid = np.flatnonzero((-tcp[:, 2, 2] >= np.cos(np.deg2rad(15))) & (table_clearance >= 0.002))
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
        raise ValueError("No downward DB grasp clears observed table and full CAD hull")

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
    env.statics.extend(obstacles)
    env.type_to_objects["Movable"] = env.movables
    cfg = with_camera_envelope(
        copy.deepcopy(m1013_uon_curobo_cfg()),
        camera_profile,
        urdf_path=ASSETS / "m1013_uon.urdf",
        output_dir=output / "model",
    )
    static_world = WorldConfig(cuboid=[table, *obstacles])
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
    solved = ik.solve_batch(Pose.from_matrix(ik.tensor_args.to_device(tcp[batch])), return_seeds=4)
    selected = batch[solved.success.any(dim=1).cpu().numpy()][:16]
    if not len(selected):
        raise ValueError("No camera-aware IK candidate in the screened batch")
    tensor = TensorDeviceType()
    kin = CudaRobotModel(RobotConfig.from_dict(copy.deepcopy(cfg["robot_cfg"]), tensor).kinematics)
    container = RobotContainer(
        "m1013_uon",
        kin,
        kin.kinematics_config.joint_limits.position,
        get_m1013_uon_gripper_spheres(tensor),
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
            position_threshold=0.01,
            rotation_threshold=0.1,
        )
    )

    def safe_world(environment, include_movables=False):
        return get_world_cfg(copy.deepcopy(environment), include_movables=include_movables)

    surfaces = env.type_to_objects["Surface"]
    with m1013_entrypoint(container), patch.object(algorithm, "get_world_cfg", safe_world):
        plan, elapsed, failure = run_planning(env, config, observation.q_init, ik, grasps, mg, surfaces)
    if plan is None:
        raise RuntimeError(f"Existing cuTAMP found no complete pick/place plan: {failure}")
    serialized = serialize_plan(plan, observation.q_init)
    serialized.update(
        robot="m1013_uon",
        empty_region=visual["empty_region"],
        class_id=entry["class_id"],
        label=name,
        physical_grasp_verified=False,
    )
    for step in serialized["steps"]:
        if step["type"] == "trajectory":
            positions, velocities, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
            step.update(positions=positions, velocities=velocities)
    report = dict(
        planning_seconds=elapsed,
        supplied_candidate_ids=data["candidate_indices"][selected].tolist(),
        hull_clear_candidates=len(screened),
        observed_obstacles=len(obstacles),
        sequence=[step["label"] for step in plan],
        executor_calls=verify_gateway_contract(serialized),
        geometry_source="existing TiPToP observed partial hull",
        m2t2_api_calls=0,
    )
    (output / "planning-result.json").write_text(json.dumps(report, cls=NumpyEncoder, indent=2) + "\n")
    return serialized, report
