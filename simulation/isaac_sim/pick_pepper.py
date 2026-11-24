"""Contact pick-and-lift test on the table, using screened UON DB candidates.

Prepare with TiPToP Python --prepare; execute with Isaac's python.sh.
The object remains a free rigid body. No grasp attachment or object teleport is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'tiptop_outputs/m1013/isaac/pick'
CANDIDATES = ROOT / 'tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json'


def prepare(output, candidate_rank):
    import numpy as np
    from scipy.interpolate import CubicHermiteSpline
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from cutamp.costs import sphere_to_sphere_overlap
    from tiptop.doosan.model import m1013_uon_curobo_cfg
    from tiptop.doosan.tamp import DATA, DEFAULT_TARGET_DIMS, DEFAULT_TARGET_XYZ

    source = json.loads(CANDIDATES.read_text())
    candidate = source['successes'][candidate_rank]
    row = json.loads((DATA / 'black_pepper_shaker_merged_grasp_zero_pose.json').read_text())['data'][candidate['candidate_index']]
    raw = next(r for r in json.loads((DATA / f'output_grasp/{row["scene_id"]}.json').read_text())
               if r['source_pregrasp_index'] == row['source_pregrasp_index'])
    goal = np.array(candidate['world_from_tcp'])
    hover = goal.copy(); hover[:3, 3] -= .12 * goal[:3, 2]
    lifted = goal.copy(); lifted[2, 3] += .10
    world = WorldConfig(cuboid=[Cuboid(name='target', dims=list(DEFAULT_TARGET_DIMS),
                                      pose=[*DEFAULT_TARGET_XYZ, 1, 0, 0, 0])])
    angle = raw['quality']['grasp_complete_joint_pos_rad']['spinner_joint']
    paths, reference = [], np.array(candidate['q'])
    object_spheres = np.load(CANDIDATES.parent / 'initial.npz')['local_object_spheres']
    world_from_object = np.array(source['world_from_object'])
    object_spheres[:, :3] = object_spheres[:, :3] @ world_from_object[:3, :3].T + world_from_object[:3, 3]
    for label, start, end, gripper in [('approach', hover, goal, 0.), ('lift', goal, lifted, angle)]:
        solver = IKSolver(IKSolverConfig.load_from_robot_config(
            m1013_uon_curobo_cfg(gripper)['robot_cfg'], world, num_seeds=64,
            self_collision_check=True, self_collision_opt=True,
            position_threshold=.0005, rotation_threshold=.003, high_precision=True))
        matrices = np.repeat(goal[None], 25, axis=0)
        matrices[:, :3, 3] = np.linspace(start[:3, 3], end[:3, 3], len(matrices))
        seed = solver.tensor_args.to_device(np.repeat(reference[None, None], len(matrices), axis=0))
        result = solver.solve_batch(Pose.from_matrix(solver.tensor_args.to_device(matrices)),
                                    seed_config=seed, return_seeds=8)
        if not bool(result.success.any(dim=1).all()):
            raise RuntimeError(f'{label}: IK failed along Cartesian path')
        order = list(range(len(matrices) - 1, -1, -1)) if label == 'approach' else list(range(len(matrices)))
        positions = np.zeros((len(matrices), 6))
        for i in order:
            choices = result.solution[i][result.success[i]].cpu().numpy()
            limits = solver.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
            wrapped = choices + 2 * np.pi * np.round((reference - choices) / (2 * np.pi))
            choices = np.where((wrapped >= limits[0]) & (wrapped <= limits[1]), wrapped, choices)
            q = choices[np.linalg.norm(choices - reference, axis=1).argmin()]
            if np.max(np.abs(q - reference)) > .5:
                raise RuntimeError(f'{label}: discontinuous IK branch')
            positions[i] = reference = q
        reference = positions[-1]
        # Check the actual cubic command curve at the execution's 100 Hz rate.
        dt = .20
        stamps = np.arange(len(positions) + 1) * dt
        curve = CubicHermiteSpline(stamps, np.vstack([positions[:1], positions]),
                                  np.zeros((len(stamps), 6)))
        samples = curve(np.arange(0, stamps[-1] + .005, .01))
        metrics = solver.check_constraints(JointState.from_position(solver.tensor_args.to_device(samples)))
        if not bool(metrics.feasible.all()):
            raise RuntimeError(f'{label}: interpolated path violates table/self-collision/limits')
        if label == 'approach':
            spheres = solver.kinematics.get_state(solver.tensor_args.to_device(samples)).get_link_spheres()
            overlap = sphere_to_sphere_overlap(spheres, solver.tensor_args.to_device(object_spheres), 0.)
            if float(overlap.max()) > .001:
                raise RuntimeError(f'Approach intersects target object in collision model: {float(overlap.max())} m')
        paths.append({'type': 'trajectory', 'label': label, 'positions': positions.tolist(),
                      'velocities': np.zeros_like(positions).tolist(), 'dt': dt})
    steps = [paths[0], {'type': 'gripper', 'label': 'contact close', 'action': 'close'}, paths[1]]
    record = {
        'candidate_index': candidate['candidate_index'], 'candidate_rank': candidate_rank,
        'screened_candidates': source['screened'], 'scene_id': row['scene_id'],
        'source_pregrasp_index': row['source_pregrasp_index'],
        'world_from_object': source['world_from_object'], 'grasp_joint_rad': angle,
        'contacts_object': [], 'target_xyz': list(DEFAULT_TARGET_XYZ),
        'target_dims': list(DEFAULT_TARGET_DIMS), 'q_init': paths[0]['positions'][0], 'steps': steps,
        'approach_lift_curve_collision_checked': True, 'approach_object_collision_checked': True,
        'physical_grasp_verified': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'pick.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Prepared candidate {candidate["candidate_index"]}: approach / contact close / 10 cm lift', flush=True)


def run(args):
    import numpy as np
    from m1013_uon import ARM_ASSET, GRIPPER_ASSET, GRIPPER, ARM, build_stage
    from grasp_scene import OBJECT, decorate
    from isaacsim import SimulationApp

    record = json.loads((args.output / 'pick.json').read_text())
    from tiptop.doosan.tamp import DATA
    raw = next(r for r in json.loads((DATA / f'output_grasp/{record["scene_id"]}.json').read_text())
               if r['source_pregrasp_index'] == record['source_pregrasp_index'])
    recorded_force = raw['quality']['contact_force_n']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ARM_ASSET, GRIPPER_ASSET, OBJECT]}
    if record.get('placement_source_sha256') and hashes != record['placement_source_sha256']:
        raise ValueError('Source assets changed since the successful pick used to prepare placement')
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 900,
                         'extra_args': ['--/metricsAssembler/operationMode=0',
                                        '--/metricsAssembler/changeListenerEnabled=false']})
    report = dict(candidate_index=record['candidate_index'], success=False,
                  physical_grasp_verified=False, object_attached=False, phase='initializing',
                  arm_gravity_compensation=True, object_gravity_m_s2=9.81, trace=[],
                  contact_parameters='source USD / PhysX defaults',
                  arm_drive_stiffness_scale=1., source_sha256=hashes, probe=args.probe)
    suffix = '-source-drive'
    suffix += f'-{args.close_target}'
    if args.solver_iterations: suffix += f'-solver{args.solver_iterations}'
    if args.physics_hz != 100: suffix += f'-hz{args.physics_hz}'
    report_path = args.output / (f'probe-{args.probe}{suffix}.json' if args.probe else 'check.json')
    client = None
    try:
        import omni.kit.app
        import omni.timeline
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            'isaacsim.core.experimental.prims', True)
        from isaacsim.core.experimental.prims import RigidPrim
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, UsdUtils
        from plan_client import IsaacPlanClient
        from tiptop.execute_plan import execute_cutamp_plan

        stage, _, _ = build_stage(args.output)
        decorate(stage, record)
        if args.gripper_under_arm_root:
            editor = Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath('/World/Gripper', ARM + '/Gripper')
            if not editor.ApplyEdits():
                raise RuntimeError('Could not move gripper under the arm namespace')
            GRIPPER = ARM + '/Gripper/UON_3finger_gripper'
            report['gripper_namespace'] = GRIPPER
            stage.GetRootLayer().Export(str(args.output / 'm1013_uon.usda'))
        if args.probe == 'isolated':
            from m1013_uon import pose
            import yaml
            from m1013_uon import MOUNT_CONFIG

            mount = yaml.safe_load(MOUNT_CONFIG.read_text())
            # Use the DB target pose, not the arm's observed pose.
            candidates = json.loads(CANDIDATES.read_text())
            selected = next(c for c in candidates['successes'] if c['candidate_index'] == record['candidate_index'])
            desired_body = pose(mount['tcp']['xyz'], mount['tcp']['rpy']).GetInverse() * Gf.Matrix4d(selected['world_from_tcp']).GetTranspose()
            cache = UsdGeom.XformCache()
            root = stage.GetPrimAtPath('/World/Gripper')
            relative = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER + '/body')) * cache.GetLocalToWorldTransform(root).GetInverse()
            UsdGeom.Xformable(root).GetOrderedXformOps()[0].Set(relative.GetInverse() * desired_body)
            fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(GRIPPER + '/joints/FixedJoint'))
            fixed.GetBody0Rel().ClearTargets(True)
            fixed.GetLocalPos0Attr().Set(Gf.Vec3f(desired_body.ExtractTranslation()))
            fixed.GetLocalRot0Attr().Set(Gf.Quatf(desired_body.ExtractRotationQuat()))
            stage.RemovePrim('/World/Arm')
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(GRIPPER))
        root_path = GRIPPER if args.probe == 'isolated' else ARM + '/root_joint'
        articulation_settings = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path))
        report['solver_position_iterations_original'] = articulation_settings.GetSolverPositionIterationCountAttr().Get()
        if args.solver_iterations:
            articulation_settings.CreateSolverPositionIterationCountAttr(args.solver_iterations)
        report['solver_position_iterations'] = articulation_settings.GetSolverPositionIterationCountAttr().Get()
        report['solver_velocity_iterations_original'] = articulation_settings.GetSolverVelocityIterationCountAttr().Get()
        if args.solver_velocity_iterations is not None:
            articulation_settings.CreateSolverVelocityIterationCountAttr(args.solver_velocity_iterations)
        report['solver_velocity_iterations'] = articulation_settings.GetSolverVelocityIterationCountAttr().Get()
        report['mimic_constraints'] = []
        for i in range(1, 4):
            mimic = PhysxSchema.PhysxMimicJointAPI.Get(stage.GetPrimAtPath(GRIPPER + f'/joints/f{i}_joint'), 'rotX')
            before = [mimic.GetNaturalFrequencyAttr().Get(), mimic.GetDampingRatioAttr().Get()]
            if args.reference_mimic_compliance:
                # Values in the local DROID Franka/Robotiq reference USD.
                mimic.CreateNaturalFrequencyAttr(1000.)
                mimic.CreateDampingRatioAttr(.05)
            report['mimic_constraints'].append({'finger': i, 'original_frequency_damping': before,
                                              'frequency_damping': [mimic.GetNaturalFrequencyAttr().Get(),
                                                                    mimic.GetDampingRatioAttr().Get()]})
        if args.probe == 'empty':
            separated = np.array(record['world_from_object'])
            separated[1, 3] += .4
            UsdGeom.Xformable(stage.GetPrimAtPath('/World/Pepper')).GetOrderedXformOps()[0].Set(
                Gf.Matrix4d(separated.T.tolist()))
        gravity = 0. if args.probe == 'isolated' else 9.81
        UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene')).GetGravityMagnitudeAttr().Set(gravity)
        report.update(object_gravity_m_s2=gravity, robot_arm_present=args.probe != 'isolated')
        object_path = '/World/Pepper/Asset/black_pepper_shaker'
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(object_path)).CreateKinematicEnabledAttr(False)
        # A kinematic table stays fixed while exposing its object-contact sensor.
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Table')).CreateKinematicEnabledAttr(True)
        for path in [object_path, '/World/Table', *[GRIPPER + f'/finger{i}' for i in range(1, 4)]]:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(GRIPPER + '/joints/spinner_joint'), 'angular')
        report['spinner_max_effort_source'] = drive.GetMaxForceAttr().Get()
        if args.spinner_max_torque is not None:
            drive.CreateMaxForceAttr(args.spinner_max_torque)
        report['spinner_max_effort_effective_nm'] = drive.GetMaxForceAttr().Get()
        prepared_limit = record.get('placement_spinner_torque_limit_nm')
        if prepared_limit is not None and not math.isclose(prepared_limit, report['spinner_max_effort_effective_nm']):
            raise ValueError('Spinner torque cap differs from the successful pick used to prepare this placement')
        obj = RigidPrim(object_path)
        gripper_body = RigidPrim(GRIPPER + '/body')
        fingers = RigidPrim([GRIPPER + f'/finger{i}' for i in range(1, 4)],
                            contact_filter_paths=[object_path])
        wrist = RigidPrim(ARM + '/link_6') if args.probe != 'isolated' else None
        table_contact = RigidPrim('/World/Table', contact_filter_paths=[object_path])

        class ContactClient(IsaacPlanClient):
            def _tick(self):
                super()._tick()
                if self.ticks % max(1, round(.05 / self.dt)) == 0:
                    if report['phase'] == 'transfer_and_place' and report.get('carry_body_offset') is not None:
                        from place_pepper import body_relative_position

                        bp, bq = gripper_body.get_world_poses()
                        local = body_relative_position(obj.get_world_poses()[0].numpy()[0],
                                                       bp.numpy()[0], bq.numpy()[0])
                        error = float(np.linalg.norm(local - report['carry_body_offset']))
                        report['max_carry_slip_m'] = max(error, report.get('max_carry_slip_m', 0.))
                        if error > .02:
                            raise RuntimeError(f'Object slipped during carrying: {error:.4f}m')
                    q = self.robot.get_dof_positions().numpy()[0]
                    v = self.robot.get_dof_velocities().numpy()[0]
                    target = self.robot.get_dof_position_targets().numpy()[0]
                    projected = self.robot.get_dof_projected_joint_forces().numpy()[0]
                    k = self.robot.get_dof_gains()[0].numpy()[0]
                    limit = self.robot.get_dof_max_efforts().numpy()[0]
                    report.setdefault('spinner_diagnostics', []).append({
                        'time_s': self.sim.get_simulation_time(), 'phase': report['phase'],
                        'position_rad': float(q[self.spinner]), 'velocity_rad_s': float(v[self.spinner]),
                        'target_rad': float(target[self.spinner]),
                        'torque_limit_nm': float(limit[self.spinner]),
                        'position_error_term_nm': float(k[self.spinner] * (target[self.spinner] - q[self.spinner])),
                        'projected_joint_reaction_nm': float(projected[self.spinner]),
                        'actual_motor_torque_nm': None})

            def close_gripper(self, speed=1.0, force=.1, effort_counts=None):
                def close():
                    start = self.get_state()['gripper']['position_rad']
                    peak, stopped = np.zeros(3), False
                    hold_target = start
                    arm_target = np.array(record['steps'][0]['positions'][-1])
                    report['closing_trace'] = []
                    for tick in range(1, math.ceil(8 / self.dt) + 1):
                        if not stopped:
                            hold_target = min(self.close_angle, start + .35 * tick * self.dt)
                        self.robot.set_dof_position_targets(hold_target, dof_indices=[self.spinner])
                        self.robot.set_dof_velocity_targets(0., dof_indices=[self.spinner])
                        self._tick()
                        forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                        peak = np.maximum(peak, forces)
                        if tick % max(1, round(.05 / self.dt)) == 0:
                            report['closing_trace'].append({
                                'time_s': self.sim.get_simulation_time(),
                                'positions': self.robot.get_dof_positions().numpy()[0].tolist(),
                                'position_targets': self.robot.get_dof_position_targets().numpy()[0].tolist(),
                                'velocity_targets': self.robot.get_dof_velocity_targets().numpy()[0].tolist(),
                                'projected_forces': self.robot.get_dof_projected_joint_forces().numpy()[0].tolist(),
                                'finger_forces_n': forces.tolist()})
                        if not stopped and np.all(forces > .1) and forces.sum() >= recorded_force:
                            stopped = True
                            report['contact_stop_target_rad'] = hold_target
                    physics.wake_up(stage_id, object_id)
                    self._tick()
                    forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                    state = self.get_state()
                    if np.count_nonzero(forces > .1) < 2:
                        raise RuntimeError(f'No opposing-finger contact after waking object; forces={forces.tolist()}, peak={peak.tolist()}')
                    sample('closed')
                    return {'action': 'close', 'contact_fingers': int(np.count_nonzero(forces > .1)),
                            'contact_force_n': forces.tolist(), 'peak_contact_force_n': peak.tolist(),
                            'position_rad': state['gripper']['position_rad'], 'target_rad': hold_target,
                            'contact_stop_threshold_reached': stopped,
                            'target_at_close_limit': math.isclose(hold_target, self.close_angle),
                            'arm_error_deg': np.degrees(np.array(state['joint_positions']) - arm_target).tolist()}
                return self._run('gripper', close)

        client = ContactClient(app, realtime=not args.headless, physics_dt=1. / args.physics_hz,
                               articulation_path=root_path,
                               arm_joint_names=[] if args.probe == 'isolated' else [f'joint_{i}' for i in range(1, 7)])
        report['physics_dt_s'] = client.dt
        physics = get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        object_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(object_path).GetPath())
        client.robot.set_link_enabled_gravities(False)
        initial = dict(zip([f'joint_{i}' for i in range(1, 7)], record['q_init']))
        initial.update(spinner_joint=0., f1_joint=0., f2_joint=0., f3_joint=0.)
        client.robot.set_dof_positions([initial[n] for n in client.names])
        gains = client.robot.get_dof_gains()
        report['robot_parameters'] = {
            'link_names': client.robot.link_names,
            'masses_kg': client.robot.get_link_masses().numpy()[0].tolist(),
            'inertias': client.robot.get_link_inertias().numpy()[0].tolist(),
            'dof_names': client.names,
            'armatures': client.robot.get_dof_armatures().numpy()[0].tolist(),
            'max_efforts_nm': client.robot.get_dof_max_efforts().numpy()[0].tolist(),
            'stiffness': gains[0].numpy()[0].tolist(), 'damping': gains[1].numpy()[0].tolist(),
            'object_mass_kg': float(obj.get_masses().numpy().reshape(-1)[0]),
        }
        viewport = get_active_viewport(); viewport.camera_path = '/World/GraspCamera'

        def sample(label):
            position, quat = obj.get_world_poses()
            position = position.numpy()[0].astype(float)
            forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1)
            row = {'label': label, 'time_s': client.sim.get_simulation_time(),
                   'object_position': position.tolist(), 'object_orientation': quat.numpy()[0].tolist(),
                   'finger_forces_n': forces.tolist(),
                   'table_object_force_n': float(np.linalg.norm(table_contact.get_contact_force_matrix(dt=client.dt).numpy()))}
            row['object_sleeping'] = physics.is_sleeping(stage_id, object_id)
            row['body_position'] = gripper_body.get_world_poses()[0].numpy()[0].tolist()
            row['body_orientation'] = gripper_body.get_world_poses()[1].numpy()[0].tolist()
            if wrist is not None:
                row['wrist_position'] = wrist.get_world_poses()[0].numpy()[0].tolist()
                row['wrist_orientation'] = wrist.get_world_poses()[1].numpy()[0].tolist()
            report['trace'].append(row); print('PICK_STATE', json.dumps(row), flush=True)
            return position

        report['phase'] = 'settle_on_table'
        for _ in range(math.ceil(2 / client.dt)): client._tick()
        baseline = sample('settled')
        nominal = np.array(record['world_from_object'])[:3, 3]
        if args.probe != 'empty' and np.linalg.norm(baseline - nominal) > .01:
            raise RuntimeError('Object did not settle near prepared pose; replan from observed pose')
        report['phase'] = 'approach_close_lift'
        report['calls'] = client.calls
        if args.probe:
            if args.probe != 'isolated': execute_cutamp_plan(record['steps'][:1], client)
            report['probe_trace'] = []
            close_target = record['grasp_joint_rad'] if args.close_target == 'recorded' else client.close_angle
            report['closing_target_rad'] = close_target
            body_position, body_orientation = gripper_body.get_world_poses()
            report['body_pose_before_close'] = [body_position.numpy()[0].tolist(), body_orientation.numpy()[0].tolist()]
            for tick in range(1, math.ceil(8 / client.dt) + 1):
                client.robot.set_dof_position_targets(min(close_target, .35 * tick * client.dt),
                                                     dof_indices=[client.spinner])
                client.robot.set_dof_velocity_targets(0., dof_indices=[client.spinner])
                client._tick()
                if tick % max(1, round(.05 / client.dt)) == 0:
                    row = client.get_state()
                    row['all_joint_positions'] = client.robot.get_dof_positions().numpy()[0].tolist()
                    row['object_position'] = obj.get_world_poses()[0].numpy()[0].tolist()
                    row['arm_target'] = client.robot.get_dof_position_targets(dof_indices=client.arm).numpy()[0].tolist()
                    row['arm_velocity_target'] = client.robot.get_dof_velocity_targets(dof_indices=client.arm).numpy()[0].tolist()
                    row['forces_n'] = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1).tolist()
                    row['joint_force'] = client.robot.get_dof_projected_joint_forces().numpy()[0].tolist()
                    report['probe_trace'].append(row)
            report.update(success=True, phase='diagnostic_complete')
            body_position, body_orientation = gripper_body.get_world_poses()
            finger_position, finger_orientation = fingers.get_world_poses()
            report['body_pose_after_close'] = [body_position.numpy()[0].tolist(), body_orientation.numpy()[0].tolist()]
            report['finger_poses_after_close'] = [finger_position.numpy().tolist(), finger_orientation.numpy().tolist()]
            print('CONTACT_DIAGNOSTIC_COMPLETE', args.probe, flush=True)
            return
        execute_cutamp_plan(record['steps'], client)
        sample('lift_end')
        report['phase'] = 'hold'
        heights = []
        for i in range(math.ceil(2 / client.dt)):
            client._tick()
            heights.append(float(obj.get_world_poses()[0].numpy()[0, 2] - baseline[2]))
            if i % 25 == 0: sample('hold')
        minimum = min(heights)
        if minimum < .04:
            raise RuntimeError(f'Object was not held 4 cm above table: min lift={minimum:.4f} m')
        physics.wake_up(stage_id, object_id)
        client._tick()
        sample('awake_held')
        if np.count_nonzero(np.array(report['trace'][-1]['finger_forces_n']) > .1) < 2:
            raise RuntimeError('Lifted object has no opposing contact after wake-up')
        report.update(success=True, physical_grasp_verified=True, min_hold_lift_m=minimum,
                      final_state=client.get_state(), phase='held', source_sha256=hashes)
        report.update(force_adequacy_verified=False, actual_motor_torque_measured=False,
                      recorded_contact_force_n=recorded_force,
                      torque_diagnostic_note='Projected joint reaction and PD position term are not actual implicit motor torque.')
        if record.get('post_pick_steps'):
            from place_pepper import body_relative_position, execute_placement

            report['pick_hold_verified'] = True
            last = report['trace'][-1]
            report['carry_body_offset'] = body_relative_position(
                last['object_position'], last['body_position'], last['body_orientation']).tolist()
            execute_placement(client, record, obj, baseline, sample,
                              lambda: physics.wake_up(stage_id, object_id), report)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
            raise RuntimeError('Source asset changed')
        event = 'CONTACT_PICK_PLACE_SUCCESS' if report.get('physical_placement_verified') else 'CONTACT_PICK_SUCCESS'
        print(event, json.dumps({'lift_m': minimum, 'phase': report['phase']}), flush=True)
        if args.snapshot:
            import asyncio
            path = args.output / ('placed.png' if report.get('physical_placement_verified') else 'held.png')
            task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(path)).wait_for_result())
            for _ in range(300):
                app.update()
                if task.done(): task.result(); break
            else: raise RuntimeError('Snapshot timed out')
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if not args.check:
            while app.is_running(): app.update()
    except Exception as exc:
        report.update(success=False, physical_grasp_verified=False, error=str(exc))
        if client is not None:
            client.stop()
            try: sample('failure')
            except Exception: pass
        if not args.check:
            import omni.timeline
            omni.timeline.get_timeline_interface().pause()
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            while app.is_running(): app.update()
            return
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        import omni.timeline
        omni.timeline.get_timeline_interface().stop(); app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prepare-placement', action='store_true',
                        help='Extend an existing successful pick with transfer/release/retreat; TiPToP Python only')
    parser.add_argument('--pick-source', type=Path, help='Directory containing successful pick.json and check.json')
    parser.add_argument('--place-offset', type=float, nargs=2, default=[0., .12], metavar=('X', 'Y'))
    parser.add_argument('--candidate-rank', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--probe', choices=['empty', 'contact', 'isolated'], help='Controlled closing diagnostic without lifting')
    parser.add_argument('--solver-iterations', type=int, help='Explicit contact solver convergence experiment')
    parser.add_argument('--solver-velocity-iterations', type=int, help='Explicit velocity solver comparison')
    parser.add_argument('--reference-mimic-compliance', action='store_true',
                        help='Diagnostic: Franka/Robotiq reference mimic frequency 1000, damping ratio .05')
    parser.add_argument('--gripper-under-arm-root', action='store_true',
                        help='Diagnostic: put gripper under the arm namespace as in the Franka reference')
    parser.add_argument('--physics-hz', type=int, default=100)
    parser.add_argument('--close-target', choices=['recorded', 'limit'], default='recorded')
    parser.add_argument('--spinner-max-torque', type=float,
                        help='Diagnostic stage-only spinner torque cap in Nm; omit to retain the USD value')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); args.output = args.output.resolve()
    if args.spinner_max_torque is not None and (not math.isfinite(args.spinner_max_torque) or args.spinner_max_torque <= 0):
        parser.error('--spinner-max-torque must be finite and positive')
    if args.solver_iterations is not None and not 1 <= args.solver_iterations <= 255:
        parser.error('--solver-iterations must be in [1, 255]')
    if args.solver_velocity_iterations is not None and not 0 <= args.solver_velocity_iterations <= 255:
        parser.error('--solver-velocity-iterations must be in [0, 255]')
    if not args.output.is_relative_to(ROOT) or args.output == ROOT: parser.error('Use a project output directory')
    if args.headless and not args.check: parser.error('--headless requires --check')
    if args.prepare and args.prepare_placement: parser.error('Choose one preparation mode')
    if args.prepare_placement:
        if args.pick_source is None: parser.error('--prepare-placement requires --pick-source')
        from place_pepper import prepare_placement

        record = json.loads((args.pick_source / 'pick.json').read_text())
        result = json.loads((args.pick_source / 'check.json').read_text())
        if not result.get('success') or not result.get('physical_grasp_verified'):
            parser.error('Placement preparation needs a physically verified successful pick')
        if result['candidate_index'] != record['candidate_index'] or record.get('post_pick_steps'):
            parser.error('Use a matching pick-only input and result')
        close = next(c for c in result['calls'] if c.get('action') == 'close')
        record['observed_gripper_angle_rad'] = close['position_rad']
        record['placement_source_sha256'] = result['source_sha256']
        record['placement_spinner_torque_limit_nm'] = result.get(
            'spinner_max_effort_effective_nm', result['spinner_max_effort_source'])
        record = prepare_placement(record, args.place_offset, CANDIDATES)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'pick.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Prepared pick / transfer / lower / release / retreat; physical placement not yet verified', flush=True)
    elif args.prepare: prepare(args.output, args.candidate_rank)
    elif not (args.output / 'pick.json').is_file(): parser.error('Run --prepare with TiPToP Python first')
    else: run(args)


if __name__ == '__main__': main()
