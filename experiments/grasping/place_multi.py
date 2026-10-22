"""Prepare and verify a short table placement using the existing IK and executor.

Object attachment here is only a planning collision model, never a PhysX joint.
"""

import math
from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from experiments.grasping.paths import read_json
from simulation.isaac_sim.place_pepper import (
    body_relative_position as body_relative_position,
)
from simulation.isaac_sim.place_pepper import (
    execute_placement as execute_placement,
)
from simulation.isaac_sim.place_pepper import (
    placement_error as placement_error,
)
from simulation.isaac_sim.place_pepper import (
    placement_poses,
)


def prepare_placement(record, offset, candidates_path):
    """Keep the verified pick prefix and check carrying/release/retreat curves."""
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from cutamp.costs import sphere_to_sphere_overlap

    from tiptop.doosan.model import m1013_uon_curobo_cfg

    source = read_json(candidates_path)
    selected = next(c for c in source["successes"] if c["candidate_index"] == record["candidate_index"])
    grasp = np.asarray(selected["world_from_tcp"])
    above, shifted, release, retreat = placement_poses(grasp, offset)
    obj = np.asarray(record["world_from_object"])
    destination = obj.copy()
    destination[:2, 3] += offset
    world = WorldConfig(
        cuboid=[Cuboid(name="target", dims=record["target_dims"], pose=[*record["target_xyz"], 1, 0, 0, 0])]
    )
    for row in record.get("distractor_collision_cuboids", []):
        world.cuboid.append(Cuboid(name=row["name"], dims=row["dims"], pose=row["pose"]))
    local = np.load(candidates_path.parent / "initial.npz")["local_object_spheres"].copy()
    if record.get("observed_tcp_from_object") is not None:
        # Re-use the observed held transform, including grasp settling/reorientation.
        tcp_from_object = np.asarray(record["observed_tcp_from_object"])
        cad = (
            np.asarray(
                [
                    [float(x) for x in line.split()[1:4]]
                    for line in __import__("pathlib").Path(record["target_object"]["cad"]).read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            * 0.001
        )
        destination = np.asarray(record["observed_settled_object_pose"]).copy()
        destination[:2, 3] += offset
        # Preserve the held tool/object orientation; release just above the support plane.
        destination[:3, :3] = grasp[:3, :3] @ tcp_from_object[:3, :3]
        table_top = record["target_xyz"][2] + record["target_dims"][2] / 2
        destination[2, 3] = table_top - float((cad @ destination[:3, :3].T)[:, 2].min()) + 0.004
        release = destination @ np.linalg.inv(tcp_from_object)
        shifted = release.copy()
        shifted[2, 3] = max(above[2, 3], release[2, 3] + 0.10)
        retreat = release.copy()
        retreat[:3, 3] -= 0.12 * release[:3, 2]

    else:
        tcp_from_object = np.linalg.inv(grasp) @ obj

    attached = local.copy()
    attached[:, :3] = local[:, :3] @ tcp_from_object[:3, :3].T + tcp_from_object[:3, 3]
    placed = local.copy()
    placed[:, :3] = local[:, :3] @ destination[:3, :3].T + destination[:3, 3]
    table_xy = np.asarray(record["target_xyz"])[:2]
    half = np.asarray(record["target_dims"])[:2] / 2
    if np.any(placed[:, :2] - placed[:, 3:4] < table_xy - half + 0.005) or np.any(
        placed[:, :2] + placed[:, 3:4] > table_xy + half - 0.005
    ):
        raise RuntimeError("The destination object would extend beyond the tabletop")
    # Use the observed successful closure for the closed collision model, not a full close.
    closed_angle = record.get("observed_gripper_angle_rad", record["grasp_joint_rad"])
    reference = np.asarray(record["steps"][-1]["positions"][-1])
    steps, checks = [], []
    for label, start, end, carrying in [
        ("transfer", above, shifted, True),
        ("lower to place", shifted, release, True),
        ("retreat after release", release, retreat, False),
    ]:
        cfg = deepcopy(m1013_uon_curobo_cfg(closed_angle if carrying else 0.0)["robot_cfg"])
        cfg["kinematics"]["extra_collision_spheres"]["attached_object"] = len(attached)
        solver = IKSolver(
            IKSolverConfig.load_from_robot_config(
                cfg,
                world,
                num_seeds=64,
                self_collision_check=True,
                self_collision_opt=True,
                position_threshold=0.0005,
                rotation_threshold=0.003,
                high_precision=True,
                use_cuda_graph=False,
            )
        )
        if carrying:
            solver.attach_object_to_robot(sphere_radius=None, sphere_tensor=solver.tensor_args.to_device(attached))
        matrices = np.repeat(start[None], 25, axis=0)
        matrices[:, :3, 3] = np.linspace(start[:3, 3], end[:3, 3], 25)
        seed = solver.tensor_args.to_device(np.repeat(reference[None, None], 25, axis=0))
        result = solver.solve_batch(
            Pose.from_matrix(solver.tensor_args.to_device(matrices)), seed_config=seed, return_seeds=8
        )
        if not bool(result.success.any(dim=1).all()):
            raise RuntimeError(f"{label}: no collision-free IK along Cartesian path")
        positions = []
        limits = solver.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
        for i in range(25):
            choices = result.solution[i][result.success[i]].cpu().numpy()
            wrapped = choices + 2 * np.pi * np.round((reference - choices) / (2 * np.pi))
            choices = np.where((wrapped >= limits[0]) & (wrapped <= limits[1]), wrapped, choices)
            q = choices[np.linalg.norm(choices - reference, axis=1).argmin()]
            if np.max(np.abs(q - reference)) > 0.5:
                retry = solver.solve_single(
                    Pose.from_matrix(solver.tensor_args.to_device(matrices[i : i + 1])),
                    seed_config=solver.tensor_args.to_device(reference[None, None]),
                    return_seeds=32,
                )
                choices = retry.solution[retry.success].cpu().numpy()
                if len(choices):
                    wrapped = choices + 2 * np.pi * np.round((reference - choices) / (2 * np.pi))
                    choices = np.where((wrapped >= limits[0]) & (wrapped <= limits[1]), wrapped, choices)
                    q = choices[np.linalg.norm(choices - reference, axis=1).argmin()]
                if np.max(np.abs(q - reference)) > 0.5:
                    raise RuntimeError(
                        f"{label} sample{i}: discontinuous IK branch; max jump={float(np.max(np.abs(q - reference))):.3f}rad"
                    )
            if i == 0 and np.max(np.abs(q - reference)) > math.radians(1):
                raise RuntimeError(f"{label}: start differs by more than executor tolerance")
            positions.append(q)
            reference = q
        positions = np.asarray(positions)
        curve = CubicHermiteSpline(np.arange(26) * 0.2, np.vstack([positions[:1], positions]), np.zeros((26, 6)))
        samples = curve(np.linspace(0, 5.0, 1201))  # Check the commanded curve at 240 Hz.
        feasible = solver.check_constraints(JointState.from_position(solver.tensor_args.to_device(samples)))
        if not bool(feasible.feasible.all()):
            raise RuntimeError(f"{label}: cubic curve collides with table/self/held object")
        if not carrying:
            spheres = solver.fk(solver.tensor_args.to_device(samples)).get_link_spheres()
            overlap = sphere_to_sphere_overlap(spheres, solver.tensor_args.to_device(placed), 0.0)
            if float(overlap.max()) > 0.001:
                raise RuntimeError(f"Retreat intersects placed object: {float(overlap.max())} m")
        steps.append(
            dict(
                type="trajectory",
                label=label,
                positions=positions.tolist(),
                velocities=np.zeros_like(positions).tolist(),
                dt=0.2,
            )
        )
        checks.append(
            dict(label=label, samples=len(samples), held_object_checked=carrying, self_table_limits_checked=True)
        )
    record["post_pick_steps"] = [
        steps[0],
        steps[1],
        dict(type="gripper", label="release on table", action="open"),
        steps[2],
    ]
    record["placement"] = dict(
        preparation_mode=(
            "observed_held_transform_and_CAD_support_height"
            if record.get("observed_tcp_from_object") is not None
            else "nominal_DB_transform"
        ),
        orientation_goal="allow_settling_after_release",
        offset_xy_m=list(offset),
        destination_object_pose=destination.tolist(),
        release_clearance_m=0.004,
        collision_checks=checks,
        physical_placement_verified=False,
        collision_gripper_angle_rad=float(closed_angle),
    )
    return record
