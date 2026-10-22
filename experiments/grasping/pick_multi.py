"""Contact pick-and-lift test on the table, using screened UON DB candidates.

Prepare with TiPToP Python --prepare; execute with Isaac's python.sh.
The object remains a free rigid body. No grasp attachment or object teleport is used.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "simulation/isaac_sim"))
from experiments.grasping.paths import WORK_ROOT as W  # noqa: E402
from experiments.grasping.paths import read_json  # noqa: E402

OUTPUT = W / "runs/pick-multi"
CANDIDATES = ROOT / "tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json"


def prepare(output, candidate_rank, *, candidates_path=None, place_offset=None, save=True):
    import numpy as np
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from cutamp.costs import sphere_to_sphere_overlap
    from scipy.interpolate import CubicHermiteSpline

    from tiptop.doosan.model import m1013_uon_curobo_cfg
    from tiptop.doosan.tamp import DEFAULT_TARGET_DIMS, DEFAULT_TARGET_XYZ

    candidates_path = Path(candidates_path or CANDIDATES).resolve()
    source = read_json(candidates_path)
    DATA = Path(source["target_object"]["db_directory"])
    object_class = source["target_object"]["class"]
    candidate = source["successes"][candidate_rank]
    row = read_json(DATA / f"{object_class}_merged_grasp_zero_pose.json")["data"][candidate["candidate_index"]]
    raw = next(
        r
        for r in read_json(DATA / f"output_grasp/{row['scene_id']}.json")
        if r["source_pregrasp_index"] == row["source_pregrasp_index"]
    )
    goal = np.array(candidate["world_from_tcp"])
    hover = goal.copy()
    hover[:3, 3] -= 0.12 * goal[:3, 2]
    lifted = goal.copy()
    lifted[2, 3] += 0.10
    world = WorldConfig(
        cuboid=[Cuboid(name="target", dims=list(DEFAULT_TARGET_DIMS), pose=[*DEFAULT_TARGET_XYZ, 1, 0, 0, 0])]
    )
    for box in source["distractor_collision_cuboids"]:
        world.cuboid.append(Cuboid(name=box["name"], dims=box["dims"], pose=box["pose"]))
    angle = raw["quality"]["grasp_complete_joint_pos_rad"]["spinner_joint"]
    paths, reference = [], np.array(candidate["q"])
    object_spheres = np.load(candidates_path.parent / "initial.npz")["local_object_spheres"]
    world_from_object = np.array(source["world_from_object"])
    object_spheres[:, :3] = object_spheres[:, :3] @ world_from_object[:3, :3].T + world_from_object[:3, 3]
    for label, start, end, gripper in [("approach", hover, goal, 0.0), ("lift", goal, lifted, angle)]:
        solver = IKSolver(
            IKSolverConfig.load_from_robot_config(
                m1013_uon_curobo_cfg(gripper)["robot_cfg"],
                world,
                num_seeds=64,
                self_collision_check=True,
                self_collision_opt=True,
                position_threshold=0.0005,
                rotation_threshold=0.003,
                high_precision=True,
            )
        )
        matrices = np.repeat(goal[None], 25, axis=0)
        matrices[:, :3, 3] = np.linspace(start[:3, 3], end[:3, 3], len(matrices))
        seed = solver.tensor_args.to_device(np.repeat(reference[None, None], len(matrices), axis=0))
        result = solver.solve_batch(
            Pose.from_matrix(solver.tensor_args.to_device(matrices)), seed_config=seed, return_seeds=8
        )
        if not bool(result.success.any(dim=1).all()):
            raise RuntimeError(f"{label}: IK failed along Cartesian path")
        order = list(range(len(matrices) - 1, -1, -1)) if label == "approach" else list(range(len(matrices)))
        positions = np.zeros((len(matrices), 6))
        for i in order:
            choices = result.solution[i][result.success[i]].cpu().numpy()
            limits = solver.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
            wrapped = choices + 2 * np.pi * np.round((reference - choices) / (2 * np.pi))
            choices = np.where((wrapped >= limits[0]) & (wrapped <= limits[1]), wrapped, choices)
            q = choices[np.linalg.norm(choices - reference, axis=1).argmin()]
            if np.max(np.abs(q - reference)) > 0.5:
                raise RuntimeError(f"{label}: discontinuous IK branch")
            positions[i] = reference = q
        reference = positions[-1]
        # Check the actual cubic command curve at 240 Hz.
        dt = 0.20
        stamps = np.arange(len(positions) + 1) * dt
        curve = CubicHermiteSpline(stamps, np.vstack([positions[:1], positions]), np.zeros((len(stamps), 6)))
        samples = curve(np.linspace(0, stamps[-1], int(round(stamps[-1] * 240)) + 1))
        metrics = solver.check_constraints(JointState.from_position(solver.tensor_args.to_device(samples)))
        if not bool(metrics.feasible.all()):
            raise RuntimeError(f"{label}: interpolated path violates table/self-collision/limits")
        if label == "approach":
            spheres = solver.kinematics.get_state(solver.tensor_args.to_device(samples)).get_link_spheres()
            overlap = sphere_to_sphere_overlap(spheres, solver.tensor_args.to_device(object_spheres), 0.0)
            if float(overlap.max()) > 0.001:
                raise RuntimeError(f"Approach intersects target object in collision model: {float(overlap.max())} m")
        paths.append(
            {
                "type": "trajectory",
                "label": label,
                "positions": positions.tolist(),
                "velocities": np.zeros_like(positions).tolist(),
                "dt": dt,
            }
        )
    steps = [paths[0], {"type": "gripper", "label": "contact close", "action": "close"}, paths[1]]
    record = {
        "candidate_index": candidate["candidate_index"],
        "candidate_rank": candidate_rank,
        "screened_candidates": source["screened"],
        "scene_id": row["scene_id"],
        "source_pregrasp_index": row["source_pregrasp_index"],
        "world_from_object": source["world_from_object"],
        "grasp_joint_rad": angle,
        "contacts_object": [],
        "target_xyz": list(DEFAULT_TARGET_XYZ),
        "target_dims": list(DEFAULT_TARGET_DIMS),
        "q_init": paths[0]["positions"][0],
        "steps": steps,
        "approach_lift_curve_collision_checked": True,
        "approach_object_collision_checked": True,
        "physical_grasp_verified": False,
    }
    record.update(
        {k: source[k] for k in ["target_object", "distractors", "distractor_collision_cuboids", "scene_layout"]}
    )
    record["candidate_source_path"] = str(candidates_path)
    record["raw_source_path"] = str(DATA / f"output_grasp/{row['scene_id']}.json")
    record["raw_source_sha256"] = hashlib.sha256(Path(record["raw_source_path"]).read_bytes()).hexdigest()
    if place_offset is not None:
        from experiments.grasping.audit_multiscene_geometry import audit
        from experiments.grasping.place_multi import prepare_placement

        record = prepare_placement(record, place_offset, candidates_path)
        geometry = audit(record, candidates_path)
        if not geometry["full_object_box_audit_passed"]:
            raise RuntimeError("Full CAD carry geometry rejected")
        record["full_object_box_audit"] = geometry
    if save:
        output.mkdir(parents=True, exist_ok=True)
        (output / "pick.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Prepared candidate {candidate['candidate_index']}", flush=True)
    return record


def run(args):
    import numpy as np
    from isaacsim import SimulationApp
    from m1013_uon import ARM, ARM_ASSET, GRIPPER, GRIPPER_ASSET, build_stage

    from experiments.grasping.scene_multi import decorate

    record = read_json(args.output / "pick.json")
    OBJECT = Path(record["target_object"]["asset"])
    DATA = Path(record["target_object"]["db_directory"])
    raw = next(
        r
        for r in read_json(DATA / f"output_grasp/{record['scene_id']}.json")
        if r["source_pregrasp_index"] == record["source_pregrasp_index"]
    )
    if hashlib.sha256(Path(record["raw_source_path"]).read_bytes()).hexdigest() != record["raw_source_sha256"]:
        raise ValueError("Raw DB record file changed since preparation")
    recorded_force = raw["quality"]["contact_force_n"]
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ARM_ASSET, GRIPPER_ASSET, OBJECT, *[Path(x["asset"]) for x in record["distractors"]]]
    }
    if record.get("placement_source_sha256") and hashes != record["placement_source_sha256"]:
        raise ValueError("Source assets changed since the successful pick used to prepare placement")
    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 900,
            "extra_args": ["--/metricsAssembler/operationMode=0", "--/metricsAssembler/changeListenerEnabled=false"],
        }
    )
    report = dict(
        candidate_index=record["candidate_index"],
        success=False,
        physical_grasp_verified=False,
        object_attached=False,
        phase="initializing",
        arm_gravity_compensation=True,
        object_gravity_m_s2=9.81,
        trace=[],
        contact_parameters="source USD / PhysX defaults",
        arm_drive_stiffness_scale=1.0,
        source_sha256=hashes,
        probe=args.probe,
    )
    suffix = "-source-drive"
    suffix += f"-{args.close_target}"
    if args.solver_iterations:
        suffix += f"-solver{args.solver_iterations}"
    if args.physics_hz != 100:
        suffix += f"-hz{args.physics_hz}"
    report_path = args.output / (f"probe-{args.probe}{suffix}.json" if args.probe else "check.json")
    client = None
    try:
        import omni.kit.app
        import omni.timeline
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.core.experimental.prims", True
        )
        from isaacsim.core.experimental.prims import RigidPrim
        from omni.physx import get_physx_simulation_interface
        from plan_client import IsaacPlanClient
        from pxr import PhysicsSchemaTools, UsdUtils

        from tiptop.execute_plan import execute_cutamp_plan

        stage, _, _ = build_stage(args.output)
        decorate(stage, record)
        if args.gripper_under_arm_root:
            editor = Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath("/World/Gripper", ARM + "/Gripper")
            if not editor.ApplyEdits():
                raise RuntimeError("Could not move gripper under the arm namespace")
            GRIPPER = ARM + "/Gripper/UON_3finger_gripper"
            report["gripper_namespace"] = GRIPPER
            stage.GetRootLayer().Export(str(args.output / "m1013_uon.usda"))
        if args.probe == "isolated":
            import yaml
            from m1013_uon import MOUNT_CONFIG, pose

            mount = yaml.safe_load(MOUNT_CONFIG.read_text())
            # Use the DB target pose, not the arm's observed pose.
            candidates = read_json(CANDIDATES)
            selected = next(c for c in candidates["successes"] if c["candidate_index"] == record["candidate_index"])
            desired_body = (
                pose(mount["tcp"]["xyz"], mount["tcp"]["rpy"]).GetInverse()
                * Gf.Matrix4d(selected["world_from_tcp"]).GetTranspose()
            )
            cache = UsdGeom.XformCache()
            root = stage.GetPrimAtPath("/World/Gripper")
            relative = (
                cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER + "/body"))
                * cache.GetLocalToWorldTransform(root).GetInverse()
            )
            UsdGeom.Xformable(root).GetOrderedXformOps()[0].Set(relative.GetInverse() * desired_body)
            fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(GRIPPER + "/joints/FixedJoint"))
            fixed.GetBody0Rel().ClearTargets(True)
            fixed.GetLocalPos0Attr().Set(Gf.Vec3f(desired_body.ExtractTranslation()))
            fixed.GetLocalRot0Attr().Set(Gf.Quatf(desired_body.ExtractRotationQuat()))
            stage.RemovePrim("/World/Arm")
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(GRIPPER))
        root_path = GRIPPER if args.probe == "isolated" else ARM + "/root_joint"
        articulation_settings = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path))
        report["solver_position_iterations_original"] = (
            articulation_settings.GetSolverPositionIterationCountAttr().Get()
        )
        if args.solver_iterations:
            articulation_settings.CreateSolverPositionIterationCountAttr(args.solver_iterations)
        report["solver_position_iterations"] = articulation_settings.GetSolverPositionIterationCountAttr().Get()
        report["solver_velocity_iterations_original"] = (
            articulation_settings.GetSolverVelocityIterationCountAttr().Get()
        )
        if args.solver_velocity_iterations is not None:
            articulation_settings.CreateSolverVelocityIterationCountAttr(args.solver_velocity_iterations)
        report["solver_velocity_iterations"] = articulation_settings.GetSolverVelocityIterationCountAttr().Get()
        report["mimic_constraints"] = []
        for i in range(1, 4):
            mimic = PhysxSchema.PhysxMimicJointAPI.Get(stage.GetPrimAtPath(GRIPPER + f"/joints/f{i}_joint"), "rotX")
            before = [mimic.GetNaturalFrequencyAttr().Get(), mimic.GetDampingRatioAttr().Get()]
            if args.reference_mimic_compliance:
                # Values in the local DROID Franka/Robotiq reference USD.
                mimic.CreateNaturalFrequencyAttr(1000.0)
                mimic.CreateDampingRatioAttr(0.05)
            report["mimic_constraints"].append(
                {
                    "finger": i,
                    "original_frequency_damping": before,
                    "frequency_damping": [mimic.GetNaturalFrequencyAttr().Get(), mimic.GetDampingRatioAttr().Get()],
                }
            )
        if args.probe == "empty":
            separated = np.array(record["world_from_object"])
            separated[1, 3] += 0.4
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/Pepper")).GetOrderedXformOps()[0].Set(
                Gf.Matrix4d(separated.T.tolist())
            )
        gravity = 0.0 if args.probe == "isolated" else 9.81
        UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene")).GetGravityMagnitudeAttr().Set(gravity)
        report.update(object_gravity_m_s2=gravity, robot_arm_present=args.probe != "isolated")
        object_path = record["target_object"]["prim_path"]
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(object_path)).CreateKinematicEnabledAttr(False)
        # A kinematic table stays fixed while exposing its object-contact sensor.
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Table")).CreateKinematicEnabledAttr(True)
        for path in [object_path, "/World/Table", *[GRIPPER + f"/finger{i}" for i in range(1, 4)]]:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(GRIPPER + "/joints/spinner_joint"), "angular")
        report["spinner_max_effort_source"] = drive.GetMaxForceAttr().Get()
        if args.spinner_max_torque is not None:
            drive.CreateMaxForceAttr(args.spinner_max_torque)
        report["spinner_max_effort_effective_nm"] = drive.GetMaxForceAttr().Get()
        prepared_limit = record.get("placement_spinner_torque_limit_nm")
        if prepared_limit is not None and not math.isclose(prepared_limit, report["spinner_max_effort_effective_nm"]):
            raise ValueError("Spinner torque cap differs from the successful pick used to prepare this placement")
        obj = RigidPrim(object_path)
        gripper_body = RigidPrim(GRIPPER + "/body")
        fingers = RigidPrim([GRIPPER + f"/finger{i}" for i in range(1, 4)], contact_filter_paths=[object_path])
        wrist = RigidPrim(ARM + "/link_6") if args.probe != "isolated" else None
        table_contact = RigidPrim("/World/Table", contact_filter_paths=[object_path])
        report.update(
            target_class=record["target_object"]["class"],
            target_instance=record["target_object"]["instance_id"],
            distractor_classes=[x["class"] for x in record["distractors"]],
            multi_object_verified=False,
            max_robot_distractor_force_n=0.0,
            distractor_trace=[],
        )
        others = [(row, RigidPrim(row["prim_path"])) for row in record["distractors"]]
        other_ids = [PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(x["prim_path"]).GetPath()) for x, _ in others]
        robot_paths = [ARM + f"/link_{i}" for i in range(1, 7)] + [
            GRIPPER + "/" + n for n in ["body", "spinner", "finger1", "finger2", "finger3"]
        ]
        for path in robot_paths:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
        robot_other = RigidPrim(robot_paths, contact_filter_paths=[x["prim_path"] for x, _ in others])
        other_table = RigidPrim("/World/Table", contact_filter_paths=[x["prim_path"] for x, _ in others])
        other_baselines = {}

        class ContactClient(IsaacPlanClient):
            def _tick(self):
                super()._tick()
                if self.ticks % max(1, round(0.05 / self.dt)) == 0:
                    if report["phase"] == "transfer_and_place" and report.get("carry_body_offset") is not None:
                        from experiments.grasping.place_multi import body_relative_position

                        bp, bq = gripper_body.get_world_poses()
                        local = body_relative_position(
                            obj.get_world_poses()[0].numpy()[0], bp.numpy()[0], bq.numpy()[0]
                        )
                        error = float(np.linalg.norm(local - report["carry_body_offset"]))
                        report["max_carry_slip_m"] = max(error, report.get("max_carry_slip_m", 0.0))
                        if error > 0.02:
                            raise RuntimeError(f"Object slipped during carrying: {error:.4f}m")
                    if report["phase"] in ("approach_close_lift", "hold", "transfer_and_place", "release_and_retreat"):
                        sleeping = bool(physics.is_sleeping(stage_id, object_id))
                        p, oq = obj.get_world_poses()
                        bp, bq = gripper_body.get_world_poses()
                        ff = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                        report.setdefault("passive_trace", []).append(
                            {
                                "time_s": self.sim.get_simulation_time(),
                                "phase": report["phase"],
                                "object_sleeping": sleeping,
                                "completed_actions": len(self.calls),
                                "finger_forces_n": ff.tolist(),
                                "object_position": p.numpy()[0].tolist(),
                                "object_orientation": oq.numpy()[0].tolist(),
                                "body_position": bp.numpy()[0].tolist(),
                                "body_orientation": bq.numpy()[0].tolist(),
                                "joint_positions": self.robot.get_dof_positions(dof_indices=self.arm)
                                .numpy()[0]
                                .tolist(),
                                "joint_targets": self.robot.get_dof_position_targets(dof_indices=self.arm)
                                .numpy()[0]
                                .tolist(),
                            }
                        )
                    if other_baselines:
                        contact = float(
                            np.linalg.norm(robot_other.get_contact_force_matrix(dt=self.dt).numpy(), axis=-1).max()
                        )
                        report["max_robot_distractor_force_n"] = max(report["max_robot_distractor_force_n"], contact)
                        for row, prim in others:
                            position, orientation = prim.get_world_poses()
                            p = position.numpy()[0]
                            displacement = float(np.linalg.norm(p - other_baselines[row["instance_id"]]["position"]))
                            report["distractor_trace"].append(
                                dict(
                                    time_s=self.sim.get_simulation_time(),
                                    phase=report["phase"],
                                    instance_id=row["instance_id"],
                                    displacement_m=displacement,
                                    position=p.tolist(),
                                    orientation=orientation.numpy()[0].tolist(),
                                )
                            )
                            if displacement > 0.005:
                                raise RuntimeError("Non-target object moved more than 5 mm: " + row["instance_id"])
                    q = self.robot.get_dof_positions().numpy()[0]
                    v = self.robot.get_dof_velocities().numpy()[0]
                    target = self.robot.get_dof_position_targets().numpy()[0]
                    projected = self.robot.get_dof_projected_joint_forces().numpy()[0]
                    k = self.robot.get_dof_gains()[0].numpy()[0]
                    limit = self.robot.get_dof_max_efforts().numpy()[0]
                    report.setdefault("spinner_diagnostics", []).append(
                        {
                            "time_s": self.sim.get_simulation_time(),
                            "phase": report["phase"],
                            "position_rad": float(q[self.spinner]),
                            "velocity_rad_s": float(v[self.spinner]),
                            "target_rad": float(target[self.spinner]),
                            "torque_limit_nm": float(limit[self.spinner]),
                            "position_error_term_nm": float(k[self.spinner] * (target[self.spinner] - q[self.spinner])),
                            "projected_joint_reaction_nm": float(projected[self.spinner]),
                            "actual_motor_torque_nm": None,
                        }
                    )

            def close_gripper(self, speed=1.0, force=0.1):
                def close():
                    start = self.get_state()["gripper"]["position_rad"]
                    peak, stopped = np.zeros(3), False
                    hold_target = start
                    arm_target = np.array(record["steps"][0]["positions"][-1])
                    report["closing_trace"] = []
                    for tick in range(1, math.ceil(8 / self.dt) + 1):
                        if not stopped:
                            hold_target = min(self.close_angle, start + 0.35 * tick * self.dt)
                        self.robot.set_dof_position_targets(hold_target, dof_indices=[self.spinner])
                        self.robot.set_dof_velocity_targets(0.0, dof_indices=[self.spinner])
                        self._tick()
                        forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                        peak = np.maximum(peak, forces)
                        if tick % max(1, round(0.05 / self.dt)) == 0:
                            report["closing_trace"].append(
                                {
                                    "time_s": self.sim.get_simulation_time(),
                                    "positions": self.robot.get_dof_positions().numpy()[0].tolist(),
                                    "position_targets": self.robot.get_dof_position_targets().numpy()[0].tolist(),
                                    "velocity_targets": self.robot.get_dof_velocity_targets().numpy()[0].tolist(),
                                    "projected_forces": self.robot.get_dof_projected_joint_forces().numpy()[0].tolist(),
                                    "finger_forces_n": forces.tolist(),
                                }
                            )
                        if not stopped and np.all(forces > 0.1) and forces.sum() >= recorded_force:
                            stopped = True
                            report["contact_stop_target_rad"] = hold_target
                    physics.wake_up(stage_id, object_id)
                    self._tick()
                    forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                    state = self.get_state()
                    if np.count_nonzero(forces > 0.1) < 2:
                        raise RuntimeError(
                            f"No opposing-finger contact after waking object; forces={forces.tolist()}, peak={peak.tolist()}"
                        )
                    sample("closed")
                    return {
                        "action": "close",
                        "contact_fingers": int(np.count_nonzero(forces > 0.1)),
                        "contact_force_n": forces.tolist(),
                        "peak_contact_force_n": peak.tolist(),
                        "position_rad": state["gripper"]["position_rad"],
                        "target_rad": hold_target,
                        "contact_stop_threshold_reached": stopped,
                        "target_at_close_limit": math.isclose(hold_target, self.close_angle),
                        "arm_error_deg": np.degrees(np.array(state["joint_positions"]) - arm_target).tolist(),
                    }

                return self._run("gripper", close)

        client = ContactClient(
            app,
            realtime=not args.headless,
            physics_dt=1.0 / args.physics_hz,
            articulation_path=root_path,
            arm_joint_names=[] if args.probe == "isolated" else [f"joint_{i}" for i in range(1, 7)],
        )
        report["physics_dt_s"] = client.dt
        physics = get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        object_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(object_path).GetPath())
        client.robot.set_link_enabled_gravities(False)
        initial = dict(zip([f"joint_{i}" for i in range(1, 7)], record["q_init"]))
        initial.update(spinner_joint=0.0, f1_joint=0.0, f2_joint=0.0, f3_joint=0.0)
        client.robot.set_dof_positions([initial[n] for n in client.names])
        gains = client.robot.get_dof_gains()
        report["robot_parameters"] = {
            "link_names": client.robot.link_names,
            "masses_kg": client.robot.get_link_masses().numpy()[0].tolist(),
            "inertias": client.robot.get_link_inertias().numpy()[0].tolist(),
            "dof_names": client.names,
            "armatures": client.robot.get_dof_armatures().numpy()[0].tolist(),
            "max_efforts_nm": client.robot.get_dof_max_efforts().numpy()[0].tolist(),
            "stiffness": gains[0].numpy()[0].tolist(),
            "damping": gains[1].numpy()[0].tolist(),
            "object_mass_kg": float(obj.get_masses().numpy().reshape(-1)[0]),
        }
        stage.GetRootLayer().Export(str(args.output / "effective_scene.usda"))
        viewport = get_active_viewport()
        viewport.camera_path = "/World/Camera"

        def sample(label):
            position, quat = obj.get_world_poses()
            position = position.numpy()[0].astype(float)
            forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1)
            row = {
                "label": label,
                "time_s": client.sim.get_simulation_time(),
                "object_position": position.tolist(),
                "object_orientation": quat.numpy()[0].tolist(),
                "finger_forces_n": forces.tolist(),
                "table_object_force_n": float(
                    np.linalg.norm(table_contact.get_contact_force_matrix(dt=client.dt).numpy())
                ),
            }
            row["object_sleeping"] = physics.is_sleeping(stage_id, object_id)
            row["body_position"] = gripper_body.get_world_poses()[0].numpy()[0].tolist()
            row["body_orientation"] = gripper_body.get_world_poses()[1].numpy()[0].tolist()
            if wrist is not None:
                row["wrist_position"] = wrist.get_world_poses()[0].numpy()[0].tolist()
                row["wrist_orientation"] = wrist.get_world_poses()[1].numpy()[0].tolist()
            report["trace"].append(row)
            print("PICK_STATE", json.dumps(row), flush=True)
            return position

        report["phase"] = "settle_on_table"
        for _ in range(math.ceil(2 / client.dt)):
            client._tick()
        baseline = sample("settled")
        for row, prim in others:
            p, q = prim.get_world_poses()
            p = p.numpy()[0].astype(float)
            q = q.numpy()[0].astype(float)
            if np.linalg.norm(p - np.asarray(row["world_from_object"])[:3, 3]) > 0.01:
                raise RuntimeError("Distractor did not settle near planned pose: " + row["instance_id"])
            other_baselines[row["instance_id"]] = dict(position=p.tolist(), orientation=q.tolist())
        report["distractor_baselines"] = other_baselines

        nominal = np.array(record["world_from_object"])[:3, 3]
        if args.probe != "empty" and np.linalg.norm(baseline - nominal) > 0.01:
            raise RuntimeError("Object did not settle near prepared pose; replan from observed pose")
        report["phase"] = "approach_close_lift"
        report["calls"] = client.calls
        if args.probe:
            if args.probe != "isolated":
                execute_cutamp_plan(record["steps"][:1], client)
            report["probe_trace"] = []
            close_target = record["grasp_joint_rad"] if args.close_target == "recorded" else client.close_angle
            report["closing_target_rad"] = close_target
            body_position, body_orientation = gripper_body.get_world_poses()
            report["body_pose_before_close"] = [body_position.numpy()[0].tolist(), body_orientation.numpy()[0].tolist()]
            for tick in range(1, math.ceil(8 / client.dt) + 1):
                client.robot.set_dof_position_targets(
                    min(close_target, 0.35 * tick * client.dt), dof_indices=[client.spinner]
                )
                client.robot.set_dof_velocity_targets(0.0, dof_indices=[client.spinner])
                client._tick()
                if tick % max(1, round(0.05 / client.dt)) == 0:
                    row = client.get_state()
                    row["all_joint_positions"] = client.robot.get_dof_positions().numpy()[0].tolist()
                    row["object_position"] = obj.get_world_poses()[0].numpy()[0].tolist()
                    row["arm_target"] = (
                        client.robot.get_dof_position_targets(dof_indices=client.arm).numpy()[0].tolist()
                    )
                    row["arm_velocity_target"] = (
                        client.robot.get_dof_velocity_targets(dof_indices=client.arm).numpy()[0].tolist()
                    )
                    row["forces_n"] = np.linalg.norm(
                        fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1
                    ).tolist()
                    row["joint_force"] = client.robot.get_dof_projected_joint_forces().numpy()[0].tolist()
                    report["probe_trace"].append(row)
            report.update(success=True, phase="diagnostic_complete")
            body_position, body_orientation = gripper_body.get_world_poses()
            finger_position, finger_orientation = fingers.get_world_poses()
            report["body_pose_after_close"] = [body_position.numpy()[0].tolist(), body_orientation.numpy()[0].tolist()]
            report["finger_poses_after_close"] = [finger_position.numpy().tolist(), finger_orientation.numpy().tolist()]
            print("CONTACT_DIAGNOSTIC_COMPLETE", args.probe, flush=True)
            return
        execute_cutamp_plan(record["steps"], client)
        sample("lift_end")
        report["phase"] = "hold"
        heights = []
        for i in range(math.ceil(2 / client.dt)):
            client._tick()
            heights.append(float(obj.get_world_poses()[0].numpy()[0, 2] - baseline[2]))
            if i % 25 == 0:
                sample("hold")
        minimum = min(heights)
        if minimum < 0.04:
            raise RuntimeError(f"Object was not held 4 cm above table: min lift={minimum:.4f} m")
        physics.wake_up(stage_id, object_id)
        client._tick()
        sample("awake_held")
        if np.count_nonzero(np.array(report["trace"][-1]["finger_forces_n"]) > 0.1) < 2:
            raise RuntimeError("Lifted object has no opposing contact after wake-up")
        report.update(
            success=True,
            physical_grasp_verified=True,
            min_hold_lift_m=minimum,
            final_state=client.get_state(),
            phase="held",
            source_sha256=hashes,
        )
        report.update(
            force_adequacy_verified=False,
            actual_motor_torque_measured=False,
            recorded_contact_force_n=recorded_force,
            torque_diagnostic_note="Projected joint reaction and PD position term are not actual implicit motor torque.",
        )
        if record.get("post_pick_steps"):
            from experiments.grasping.place_multi import body_relative_position, execute_placement

            report["pick_hold_verified"] = True
            last = report["trace"][-1]
            report["carry_body_offset"] = body_relative_position(
                last["object_position"], last["body_position"], last["body_orientation"]
            ).tolist()
            if record.get("observed_tcp_from_object"):
                from scipy.spatial.transform import Rotation

                def matrix(position, orientation):
                    t = np.eye(4)
                    t[:3, :3] = Rotation.from_quat(np.asarray(orientation)[[1, 2, 3, 0]]).as_matrix()
                    t[:3, 3] = position
                    return t

                current_tcp = matrix(last["body_position"], last["body_orientation"]) @ np.asarray(
                    record["body_from_tcp"]
                )
                current_relative = np.linalg.inv(current_tcp) @ matrix(
                    last["object_position"], last["object_orientation"]
                )
                prepared_relative = np.asarray(record["observed_tcp_from_object"])
                shift = float(np.linalg.norm(current_relative[:3, 3] - prepared_relative[:3, 3]))
                angle = float(
                    np.degrees(Rotation.from_matrix(prepared_relative[:3, :3].T @ current_relative[:3, :3]).magnitude())
                )
                report["held_pose_replay_error"] = dict(translation_m=shift, rotation_deg=angle)
                if shift > 0.003 or angle > 2:
                    raise RuntimeError("Held pose differs from placement preparation; observe and replan")
            execute_placement(
                client, record, obj, baseline, sample, lambda: physics.wake_up(stage_id, object_id), report
            )
        from scipy.spatial.transform import Rotation

        for identity in other_ids:
            physics.wake_up(stage_id, identity)
        client._tick()
        final_forces = np.linalg.norm(other_table.get_contact_force_matrix(dt=client.dt).numpy()[0], axis=-1)
        report["distractor_final"] = []
        for index, (row, prim) in enumerate(others):
            pp, qq = prim.get_world_poses()
            p = pp.numpy()[0].astype(float)
            q = qq.numpy()[0].astype(float)
            b = other_baselines[row["instance_id"]]
            dist = float(np.linalg.norm(p - b["position"]))
            angular = float(
                np.degrees(
                    (
                        Rotation.from_quat(np.asarray(b["orientation"])[[1, 2, 3, 0]]).inv()
                        * Rotation.from_quat(q[[1, 2, 3, 0]])
                    ).magnitude()
                )
            )
            linear, _ = prim.get_velocities()
            final = dict(
                instance_id=row["instance_id"],
                class_name=row["class"],
                displacement_m=dist,
                rotation_deg=angular,
                table_support_n=float(final_forces[index]),
                linear_speed_m_s=float(np.linalg.norm(linear.numpy()[0])),
            )
            report["distractor_final"].append(final)
            if dist > 0.002 or angular > 2 or final_forces[index] < 0.05 or final["linear_speed_m_s"] > 0.01:
                raise RuntimeError("Non-target object was disturbed or unsupported: " + str(final))
        if report["max_robot_distractor_force_n"] > 0.1:
            raise RuntimeError("Robot contacted non-target object: " + str(report["max_robot_distractor_force_n"]))
        report["multi_object_verified"] = True
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
            raise RuntimeError("Source asset changed")
        event = "CONTACT_PICK_PLACE_SUCCESS" if report.get("physical_placement_verified") else "CONTACT_PICK_SUCCESS"
        print(event, json.dumps({"lift_m": minimum, "phase": report["phase"]}), flush=True)
        if args.snapshot:
            import asyncio

            path = args.output / ("placed.png" if report.get("physical_placement_verified") else "held.png")
            task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(path)).wait_for_result())
            for _ in range(300):
                app.update()
                if task.done():
                    task.result()
                    break
            else:
                raise RuntimeError("Snapshot timed out")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not args.check:
            while app.is_running():
                app.update()
    except Exception as exc:
        report.update(success=False, physical_grasp_verified=False, error=str(exc))
        if client is not None:
            client.stop()
            try:
                sample("failure")
            except Exception:
                pass
        if not args.check:
            import omni.timeline

            omni.timeline.get_timeline_interface().pause()
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            while app.is_running():
                app.update()
            return
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare", action="store_true")
    preparation.add_argument(
        "--prepare-placement",
        action="store_true",
        help="Extend an existing successful pick with transfer/release/retreat; TiPToP Python only",
    )
    parser.add_argument("--pick-source", type=Path, help="Directory containing successful pick.json and check.json")
    parser.add_argument("--place-offset", type=float, nargs=2, default=[0.0, 0.12], metavar=("X", "Y"))
    parser.add_argument("--candidate-rank", type=int)
    parser.add_argument("--candidates", type=Path, help="Screened scene candidate-ik.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument(
        "--probe", choices=["empty", "contact", "isolated"], help="Controlled closing diagnostic without lifting"
    )
    parser.add_argument("--solver-iterations", type=int, help="Explicit contact solver convergence experiment")
    parser.add_argument("--solver-velocity-iterations", type=int, help="Explicit velocity solver comparison")
    parser.add_argument(
        "--reference-mimic-compliance",
        action="store_true",
        help="Diagnostic: Franka/Robotiq reference mimic frequency 1000, damping ratio .05",
    )
    parser.add_argument(
        "--gripper-under-arm-root",
        action="store_true",
        help="Diagnostic: put gripper under the arm namespace as in the Franka reference",
    )
    parser.add_argument("--physics-hz", type=int, default=100)
    parser.add_argument("--close-target", choices=["recorded", "limit"], default="recorded")
    parser.add_argument(
        "--spinner-max-torque",
        type=float,
        help="Diagnostic stage-only spinner torque cap in Nm; omit to retain the USD value",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.spinner_max_torque is not None and (
        not math.isfinite(args.spinner_max_torque) or args.spinner_max_torque <= 0
    ):
        parser.error("--spinner-max-torque must be finite and positive")
    if args.solver_iterations is not None and not 1 <= args.solver_iterations <= 255:
        parser.error("--solver-iterations must be in [1, 255]")
    if args.solver_velocity_iterations is not None and not 0 <= args.solver_velocity_iterations <= 255:
        parser.error("--solver-velocity-iterations must be in [0, 255]")
    if not args.output.is_relative_to(ROOT / "tiptop_outputs") or args.output == ROOT / "tiptop_outputs":
        parser.error("Use a new directory under tiptop_outputs")
    if args.headless and not args.check:
        parser.error("--headless requires --check")
    if args.prepare_placement:
        if args.pick_source is None:
            parser.error("--prepare-placement requires --pick-source")
        from experiments.grasping.place_multi import prepare_placement

        record = read_json(args.pick_source / "pick.json")
        result = read_json(args.pick_source / "check.json")
        if not result.get("success") or not result.get("physical_grasp_verified"):
            parser.error("Placement preparation needs a physically verified successful pick")
        if result["candidate_index"] != record["candidate_index"] or record.get("post_pick_steps"):
            parser.error("Use a matching pick-only input and result")
        close = next(c for c in result["calls"] if c.get("action") == "close")
        record["observed_gripper_angle_rad"] = close["position_rad"]
        record["placement_source_sha256"] = result["source_sha256"]
        record["placement_spinner_torque_limit_nm"] = result.get(
            "spinner_max_effort_effective_nm", result["spinner_max_effort_source"]
        )
        record = prepare_placement(
            record, args.place_offset, Path(record.get("candidate_source_path") or args.candidates or CANDIDATES)
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "pick.json").write_text(json.dumps(record, indent=2) + "\n")
        print("Prepared pick / transfer / lower / release / retreat; physical placement not yet verified", flush=True)
    elif args.prepare:
        prepare(args.output, args.candidate_rank or 0, candidates_path=args.candidates)
    elif not (args.output / "pick.json").is_file():
        parser.error("Run --prepare with TiPToP Python first")
    else:
        run(args)


if __name__ == "__main__":
    main()
