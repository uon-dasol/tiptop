"""Read-only source integration probe: 3 known CAD/DB instances -> existing cuTAMP API.

No real robot transport or Isaac execution. Any returned plan needs separate physics validation.
"""

import copy
import json
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import cutamp.algorithm as cutamp_algorithm
import numpy as np
import torch
import trimesh
from curobo.geom.types import Cuboid, Mesh
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from cutamp.algorithm import setup_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import Holding, MoveFree, Pick
from cutamp.task_planning import task_plan_generator
from cutamp.utils.common import get_world_cfg
from scipy.spatial.transform import Rotation

from experiments.grasping.known_scene_adapter import Instance, db_to_planner
from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json
from tiptop.doosan.model import get_m1013_uon_ik_solver, load_m1013_uon_container, m1013_uon_curobo_cfg
from tiptop.doosan.planning import prepare_joint_path
from tiptop.doosan.tamp import m1013_entrypoint, motion_world_cfg, verify_gateway_contract
from tiptop.planning import run_planning, serialize_plan

ensure_workdirs()
output = WORK_ROOT / "audit/cutamp-three-object-safe-world-probe.json"
report = dict(
    success=False,
    physical_grasp_verified=False,
    scope="initialization_and_bounded_planning_only",
    goal="Holding(car_leather_conditioner_bottle-01)",
)
started = time.monotonic()
try:
    torch.manual_seed(0)
    np.random.seed(0)
    tensor = TensorDeviceType()
    objects = []
    grasps = {}
    provenance = []
    entries = read_json(W / "plans/scene-wide.json")
    for entry in entries:
        cls = entry["class"]
        name = entry["instance_id"]
        t = np.array(entry["world_from_object"])
        vertices = (
            np.asarray(
                [
                    [float(x) for x in line.split()[1:4]]
                    for line in Path(entry["cad"]).read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            * 0.001
        )
        hull = trimesh.convex.convex_hull(vertices)
        q = Rotation.from_matrix(t[:3, :3]).as_quat()
        mesh = Mesh(
            name=name,
            pose=[*t[:3, 3].tolist(), float(q[3]), *q[:3].tolist()],
            vertices=hull.vertices.tolist(),
            faces=hull.faces.tolist(),
        )
        objects.append(mesh)
        source = read_json(W / "plans" / f"multi-wide-{cls}/candidate-ik.json")
        db = read_json(Path(entry["db_directory"]) / f"{cls}_merged_grasp_zero_pose.json")["data"]
        selected = [
            dict(candidate_index=c["candidate_index"], **db[c["candidate_index"]]) for c in source["successes"][:16]
        ]
        inp = db_to_planner(selected, Instance(name, cls, t, "Isaac_GT_scene_definition"), t)
        grasps[name] = {
            "grasps_obj": tensor.to_device(inp["grasps_obj"]),
            "confidences_pt": tensor.to_device(inp["confidences_pt"]),
        }
        provenance.append(dict(instance=name, candidate_count=len(selected), metadata=inp["metadata"]))
    target = entries[1]["instance_id"]
    table = Cuboid(name="table", dims=[0.8, 0.8, 0.04], pose=[0.65, 0, 0.38, 1, 0, 0, 0])
    env = TAMPEnvironment(
        "temporary_three_known_classes",
        objects,
        [table],
        {"Movable": objects, "Surface": [table]},
        frozenset({Holding.ground(target)}),
    )
    q_init = np.array(read_json(W / "plans/multi-wide-car_leather_conditioner_bottle/pick.json")["q_init"])
    before = [x.name for x in env.movables]
    container = load_m1013_uon_container(tensor)
    ik = get_m1013_uon_ik_solver(get_world_cfg(env, include_movables=False), num_seeds=32)
    config = TAMPConfiguration(
        robot="m1013_uon",
        num_particles=8,
        num_opt_steps=50,
        max_loop_dur=30.0,
        m2t2_grasps=True,
        grasp_dof=6,
        num_initial_plans=1,
        curobo_plan=True,
        max_motion_refine_attempts=2,
        enable_visualizer=False,
        enable_experiment_logging=False,
        warmup_ik=False,
        warmup_motion_gen=False,
        coll_n_spheres=32,
        coll_sphere_radius=0.003,
        placement_check="obb",
        time_dilation_factor=0.2,
    )

    def safe_world_cfg(environment, include_movables=False):
        return get_world_cfg(copy.deepcopy(environment), include_movables=include_movables)

    with m1013_entrypoint(container), patch.object(cutamp_algorithm, "get_world_cfg", safe_world_cfg):
        _, _, _, world = setup_cutamp(env, config, q_init, ik_solver=ik)
        skeleton = next(iter(task_plan_generator(world.initial_state, world.goal_state, [MoveFree, Pick])))
        report.update(
            initialized=True,
            robot=world.robot_name,
            movables=before,
            task_skeleton=[x.name for x in skeleton],
            grasps={k: list(v["grasps_obj"].shape) for k, v in grasps.items()},
            candidate_provenance=provenance,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        cfg = copy.deepcopy(m1013_uon_curobo_cfg())
        cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 32
        mg = MotionGen(
            MotionGenConfig.load_from_robot_config(
                robot_cfg=cfg,
                world_model=motion_world_cfg(env),
                use_cuda_graph=False,
                trajopt_particle_opt=False,
                num_trajopt_seeds=4,
                num_graph_seeds=4,
                position_threshold=0.01,
                rotation_threshold=0.1,
            )
        )
        assert before == [x.name for x in env.movables]
        plan, elapsed, failure = run_planning(env, config, q_init, ik, grasps, mg, [table])
    report["temporary_fix_scope"] = (
        "get_world_cfg protected only inside this dedicated Python process; source library unchanged"
    )
    report.update(
        planning_seconds=elapsed,
        planning_failure=failure,
        planning_returned_plan=plan is not None,
        environment_movables_preserved=before == [x.name for x in env.movables],
    )
    if plan is not None:
        serialized = serialize_plan(plan, q_init)
        serialized.update(robot="m1013_uon", physical_grasp_verified=False, frame_calibrated=False)
        for step in serialized["steps"]:
            if step["type"] == "trajectory":
                positions, velocities, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
                step.update(positions=positions, velocities=velocities)
        calls = verify_gateway_contract(serialized)
        (WORK_ROOT / "plans/cutamp-three-object-safe-world-plan.json").write_text(
            json.dumps(serialized, default=lambda x: x.tolist(), indent=2) + "\n"
        )
        report.update(success=True, executor_contract_verified=True, executor_calls=calls, steps=len(plan))
except Exception as exc:
    report.update(error=str(exc), traceback=traceback.format_exc())
finally:
    report["wall_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["candidate_provenance", "traceback", "executor_calls"]},
            indent=2,
        ),
        flush=True,
    )
