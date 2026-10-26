"""Replay a successful UON record with only the gripper and a free object.

DB geometry is restored without the arm, table, cuTAMP, or a new IK solution.
A driven prismatic fixture moves the gripper; the object is never attached.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tiptop.object_catalog import registered_object

_PEPPER = registered_object('black_pepper_shaker', check_files=False)
DATA = _PEPPER['db'].parent


def run(args):
    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation
    from m1013_uon import GRIPPER_ASSET, GRIPPER, MOUNT_CONFIG, pose
    from grasp_scene import OBJECT, decorate
    from isaacsim import SimulationApp

    row = json.loads((DATA / 'black_pepper_shaker_merged_grasp_zero_pose.json').read_text())['data'][args.candidate]
    raw_path = DATA / f'output_grasp/{row["scene_id"]}.json'
    raw = next(r for r in json.loads(raw_path.read_text())
               if r['source_pregrasp_index'] == row['source_pregrasp_index'])
    pre = json.loads((DATA / f'pre_grasp/{row["scene_id"]}.json').read_text())[0]['data'][row['source_pregrasp_index']]
    quality = raw['quality']
    world_object = np.array(quality['object_initial_mat'])
    # Saved grasps are transformed after closing to restore the object pose.
    # Undo that transform to recover the actual collection attempt pose.
    restored_grasp = np.array(raw['grasp_mat'])
    world_grasp = np.linalg.inv(np.array(quality['object_motion_restore_mat'])) @ restored_grasp
    pregrasp_matrix = np.eye(4)
    pregrasp_matrix[:3, :3] = Rotation.from_euler('xyz', pre['target_orientation'], degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    pregrasp_matrix[:3, 3] = pre['target_points']
    attempt_pose_error = float(np.max(np.abs(world_grasp - pregrasp_matrix)))
    if attempt_pose_error > 1e-5:
        raise ValueError('Restored grasp does not recover the original pregrasp attempt')
    # Common translation only: preserve the collection's relative poses/rotation.
    world_object[2, 3] += .5
    world_grasp[2, 3] += .5
    if args.table:
        world_object = np.eye(4)
        world_object[:3, :3] = Rotation.from_euler('x', math.pi / 2).as_matrix()
        cad = _PEPPER['cad']
        vertices = np.array([[float(x) for x in line.split()[1:4]] for line in cad.read_text().splitlines()
                             if line.startswith('v ')]) * .001
        world_object[2, 3] = .4 - float((vertices @ world_object[:3, :3].T)[:, 2].min()) + .002
        world_grasp = world_object @ np.array(row['grasp_mat'])
    clearance = pre['pregrasp_clearance']
    angle = quality['grasp_complete_joint_pos_rad']['spinner_joint']
    output = args.output / f'candidate-{args.candidate}-{args.close_mode}'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(candidate_index=args.candidate, scene_id=row['scene_id'],
                  source_pregrasp_index=row['source_pregrasp_index'], success=False,
                  physical_grasp_verified=False, object_attached=False, gravity_m_s2=args.gravity,
                  gravity_start=args.gravity_start, robot_gravity_compensation=True,
                  table_enabled=args.table,
                  physics_hz=args.physics_hz,
                  world_from_object=world_object.tolist(), world_from_grasp=world_grasp.tolist(),
                  approach_vector=world_grasp[:3, 2].tolist(), recorded_approach_vector=raw['approach_vector'],
                  original_attempt_pose_max_error=attempt_pose_error, recorded_joint_rad=angle,
                  recorded_quality=quality, pregrasp=pre, close_mode=args.close_mode,
                  known_differences=['Collection controller and physics timestep are not recorded.',
                                    'DB grasp -> model TCP identity/zero translation is under test.',
                                    'Existing rigid USD adds physics to the same object CAD.',
                                    'A prismatic fixture replaces the collection base controller.',
                                    'Stress directions/duration: six world axes, 0.25 s each; original sequence unavailable.',
                                    'Open angle 0; target_width -> spinner mapping is not recorded.'],
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (GRIPPER_ASSET, OBJECT, raw_path)}, samples=[])
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 900,
                         'extra_args': ['--/metricsAssembler/operationMode=0',
                                        '--/metricsAssembler/changeListenerEnabled=false']})
    try:
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdLux
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            'isaacsim.core.experimental.prims', True)
        from isaacsim.core.experimental.prims import RigidPrim
        from plan_client import IsaacPlanClient
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, UsdUtils

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        scene.CreateGravityMagnitudeAttr(0.)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        light = UsdLux.DomeLight.Define(stage, '/World/Light')
        light.CreateIntensityAttr(1000.)
        root = UsdGeom.Xform.Define(stage, '/World/Gripper')
        root.GetPrim().GetReferences().AddReference(str(GRIPPER_ASSET))
        mount = yaml.safe_load(MOUNT_CONFIG.read_text())['tcp']
        body_goal = pose(mount['xyz'], mount['rpy']).GetInverse() * Gf.Matrix4d(world_grasp.T.tolist())
        body_start = Gf.Matrix4d(body_goal)
        body_start.SetTranslateOnly(body_goal.ExtractTranslation() - Gf.Vec3d(*world_grasp[:3, 2]) * clearance)
        cache = UsdGeom.XformCache()
        relative = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER + '/body'))
        root.AddTransformOp().Set(relative.GetInverse() * body_start)
        # A real driven joint, so approach/lift transfer forces through contact.
        joint_path = GRIPPER + '/joints/FixedJoint'
        stage.RemovePrim(joint_path)
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        fixture = UsdGeom.Xform.Define(stage, '/World/Fixture').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(fixture)
        UsdPhysics.MassAPI.Apply(fixture).CreateMassAttr(1.)
        anchor = UsdPhysics.FixedJoint.Define(stage, '/World/FixtureRoot')
        anchor.CreateBody1Rel().SetTargets([fixture.GetPath()])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(0))
        anchor.CreateLocalPos1Attr(Gf.Vec3f(0))
        stage.GetPrimAtPath(GRIPPER).RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        settings = PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim())
        settings.CreateEnabledSelfCollisionsAttr(False)
        report['position_solver_iterations'] = settings.GetSolverPositionIterationCountAttr().Get()
        joint.CreateBody0Rel().SetTargets([fixture.GetPath()])
        joint.CreateBody1Rel().SetTargets([GRIPPER + '/body'])
        joint.CreateExcludeFromArticulationAttr(False)
        joint.CreateAxisAttr('Z')
        joint.CreateLocalPos0Attr(Gf.Vec3f(body_start.ExtractTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(body_start.ExtractRotationQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(1000.)
        drive.CreateDampingAttr(130.)
        drive.CreateMaxForceAttr(200.)
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        record = dict(world_from_object=world_object.tolist(), contacts_object=[],
                      target_xyz=[0., 0., .38], target_dims=[.8, .8, .04])
        decorate(stage, record)
        scene.GetGravityMagnitudeAttr().Set(args.gravity if args.gravity_start == 'initial' else 0.)
        # No table/ground contacts in this collection-geometry baseline.
        if not args.table:
            for prim in list(stage.GetPrimAtPath('/World').GetChildren()):
                if prim.GetName().startswith('Table'): stage.RemovePrim(prim.GetPath())
        else:
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Table')).CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath('/World/Table')).CreateThresholdAttr(0.)
        object_path = '/World/Pepper/Asset/black_pepper_shaker'
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(object_path)).CreateKinematicEnabledAttr(False)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
        for path in (object_path, *[GRIPPER + f'/finger{i}' for i in range(1, 4)]):
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        obj = RigidPrim(object_path)
        fingers = RigidPrim([GRIPPER + f'/finger{i}' for i in range(1, 4)],
                            contact_filter_paths=[object_path] + (['/World/Table'] if args.table else []))
        table = RigidPrim('/World/Table', contact_filter_paths=[object_path]) if args.table else None
        body = RigidPrim(GRIPPER + '/body')
        stage.GetRootLayer().Export(str(output / 'scene.usda'))
        client = IsaacPlanClient(app, realtime=not args.headless, physics_dt=1. / args.physics_hz,
                                 articulation_path='/World/FixtureRoot', arm_joint_names=[])
        client.robot.set_link_enabled_gravities(False)
        report['dof_names'] = client.names
        axis = client.names.index('FixedJoint')
        client.robot.set_dof_positions(np.zeros(len(client.names)))
        client.robot.set_dof_position_targets(np.zeros(len(client.names)))
        client.robot.set_dof_velocity_targets(0.)
        physics = get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        object_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(object_path).GetPath())
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        viewport = get_active_viewport()
        viewport.camera_path = '/World/GraspCamera'
        UsdGeom.Xformable(stage.GetPrimAtPath(viewport.camera_path)).GetOrderedXformOps()[0].Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(.4, -.5, .9), Gf.Vec3d(0, 0, .65), Gf.Vec3d(0, 0, 1)).GetInverse())

        def observe(phase):
            pos, quat = obj.get_world_poses()
            bpos, bquat = body.get_world_poses()
            forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1)
            q, v = client.robot.get_dof_positions().numpy()[0], client.robot.get_dof_velocities().numpy()[0]
            state = dict(phase=phase, time_s=client.sim.get_simulation_time(),
                         object_position=pos.numpy()[0].tolist(), object_quaternion=quat.numpy()[0].tolist(),
                         body_position=bpos.numpy()[0].tolist(), body_quaternion=bquat.numpy()[0].tolist(),
                         forces_n=forces.tolist(), spinner_rad=float(q[client.spinner]),
                         spinner_velocity=float(v[client.spinner]), fixture_m=float(q[axis]))
            state['joint_positions'] = q.tolist()
            state['object_sleeping'] = physics.is_sleeping(stage_id, object_id)
            if table is not None:
                state['table_object_force_n'] = float(np.linalg.norm(table.get_contact_force_matrix(dt=client.dt).numpy()))
                state['finger_table_forces_n'] = np.linalg.norm(
                    fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 1], axis=1).tolist()
            state['mimic_error_rad'] = [float(q[client.names.index(f'f{i}_joint')] + 2 * q[client.spinner])
                                        for i in range(1, 4)]
            report['samples'].append(state)
            return state

        def step_axis(start, end, phase):
            count = max(1, math.ceil(abs(end - start) / .02 / client.dt))
            for i in range(1, count + 1):
                client.robot.set_dof_position_targets(start + (end - start) * i / count, dof_indices=[axis])
                client._tick()
                if i % max(1, round(.05 / client.dt)) == 0: observe(phase)
            for _ in range(math.ceil(1. / client.dt)): client._tick()
            return observe(phase + '_settled')

        for _ in range(math.ceil((2. if args.table else .5) / client.dt)): client._tick()
        if args.table:
            # Use the observed settled object pose. Change only the fixture's
            # reference frame; do not reposition the free object.
            position, quat = obj.get_world_poses()
            observed_object = np.eye(4)
            q = quat.numpy()[0]
            observed_object[:3, :3] = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
            observed_object[:3, 3] = position.numpy()[0]
            world_grasp = observed_object @ np.array(row['grasp_mat'])
            body_goal = pose(mount['xyz'], mount['rpy']).GetInverse() * Gf.Matrix4d(world_grasp.T.tolist())
            body_start = Gf.Matrix4d(body_goal)
            body_start.SetTranslateOnly(body_goal.ExtractTranslation() - Gf.Vec3d(*world_grasp[:3, 2]) * clearance)
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(body_start.ExtractTranslation()))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(body_start.ExtractRotationQuat()))
            app.update()
            report['settled_world_from_object'] = observed_object.tolist()
            report['settled_world_from_grasp'] = world_grasp.tolist()
            for _ in range(math.ceil(.5 / client.dt)): client._tick()
        observe('initial')
        approached = step_axis(0., clearance, 'approach')
        print('RECORD_REPLAY_APPROACHED', json.dumps(approached), flush=True)
        target, contact_stop = 0., False
        for i in range(math.ceil(8. / client.dt)):
            if not contact_stop:
                target = min(angle if args.close_mode == 'recorded' else client.close_angle,
                             target + .35 * client.dt)
            client.robot.set_dof_position_targets(target, dof_indices=[client.spinner])
            client._tick()
            if i % max(1, round(.05 / client.dt)) == 0: observe('close')
            if not contact_stop and args.close_mode == 'contact':
                forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1)
                if np.all(forces > .1) and forces.sum() >= quality['contact_force_n']:
                    contact_stop = True
                    report['contact_stop_target_rad'] = target
                    report['contact_stop_time_s'] = client.sim.get_simulation_time()
        closed = observe('closed')
        print('RECORD_REPLAY_CLOSED', json.dumps(closed), flush=True)
        if np.count_nonzero(np.array(closed['forces_n']) > .1) < 2 and not closed['object_sleeping']:
            raise RuntimeError('No maintained opposing-finger contact after closing')
        if args.gravity_start == 'after-close':
            scene.GetGravityMagnitudeAttr().Set(args.gravity)
            # Process the live USD scene property change before stepping.
            app.update()
            physics.wake_up(stage_id, object_id)
            report['gravity_enabled_at_s'] = client.sim.get_simulation_time()
            print('RECORD_REPLAY_GRAVITY_ENABLED', args.gravity, flush=True)
        start_index = len(report['samples'])
        step_axis(clearance, clearance - .10, 'translate')
        for i in range(math.ceil(2. / client.dt)):
            client._tick()
            if i % max(1, round(.05 / client.dt)) == 0: observe('hold')
        # At 0 g, displacement alone could be inertia. Challenge the grasp with
        # the force magnitude in this very record, then check relative pose.
        report['replay_stress_force_n'] = quality['max_test_force_n']
        for direction in np.concatenate([np.eye(3), -np.eye(3)]):
            for i in range(math.ceil(.25 / client.dt)):
                obj.apply_forces([direction * quality['max_test_force_n']])
                client._tick()
                if i % max(1, round(.025 / client.dt)) == 0: observe('stress')
        obj.apply_forces([[0., 0., 0.]])
        for i in range(math.ceil(1. / client.dt)):
            client._tick()
            if i % max(1, round(.05 / client.dt)) == 0: observe('after_stress')
        report['before_final_wake'] = observe('before_final_wake')
        physics.wake_up(stage_id, object_id)
        client._tick()
        observe('awake_final_contact_check')
        held = report['samples'][start_index:]
        initial_relative = np.array(closed['object_position']) - np.array(closed['body_position'])
        errors = [float(np.linalg.norm(np.array(s['object_position']) - s['body_position'] - initial_relative)) for s in held]
        q0 = closed['object_quaternion']
        initial_rotation = Rotation.from_quat(q0[1:] + q0[:1])
        rotation_errors = [(initial_rotation.inv() * Rotation.from_quat(
            s['object_quaternion'][1:] + s['object_quaternion'][:1])).magnitude() for s in held]
        contact_ratio = float(np.mean([np.count_nonzero(np.array(s['forces_n']) > .1) >= 2 for s in held]))
        displacement = float(np.dot(np.array(held[-1]['object_position']) - closed['object_position'], -world_grasp[:3, 2]))
        report.update(max_relative_translation_m=max(errors), contact_hold_ratio=contact_ratio,
                      max_relative_rotation_deg=float(np.degrees(max(rotation_errors))),
                      object_displacement_m=displacement)
        final_contacts = int(np.count_nonzero(np.array(held[-1]['forces_n']) > .1))
        report['final_contact_fingers'] = final_contacts
        if max(errors) > .005 or max(rotation_errors) > math.radians(10) or final_contacts < 2 or displacement < .08:
            raise RuntimeError('Object did not remain in contact and follow the gripper for 10 cm')
        report.update(success=True, physical_grasp_verified=True,
                      verification=f'{args.gravity} m/s2 ({args.gravity_start}) contact transport + recorded-force stress')
        print('RECORD_REPLAY_SUCCESS', json.dumps({k: report[k] for k in (
            'max_relative_translation_m', 'contact_hold_ratio', 'object_displacement_m')}), flush=True)
        if args.snapshot:
            import asyncio
            task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(output / 'held.png')).wait_for_result())
            for _ in range(200):
                app.update()
                if task.done(): task.result(); break
        if not args.headless:
            (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            while app.is_running(): app.update()
    except Exception as exc:
        report['error'] = str(exc)
        try: observe('failure')
        except Exception: pass
        print('RECORD_REPLAY_FAILED', str(exc), flush=True)
        if not args.headless:
            import omni.timeline
            omni.timeline.get_timeline_interface().pause()
            (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            while app.is_running(): app.update()
    finally:
        report['source_assets_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                              for p, h in report['source_sha256'].items())
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        import omni.timeline
        omni.timeline.get_timeline_interface().stop()
        app.close()
    return report['success']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=int, default=0)
    parser.add_argument('--close-mode', choices=['recorded', 'contact'], default='contact')
    parser.add_argument('--physics-hz', type=int, default=100)
    parser.add_argument('--gravity', type=float, default=0., help='World gravity magnitude in m/s2')
    parser.add_argument('--gravity-start', choices=['initial', 'after-close'], default='initial')
    parser.add_argument('--table', action='store_true', help='Stable upright object on an 80 cm tabletop; use a suitable DB candidate')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'tiptop_outputs/m1013/isaac/grasp-replay')
    args = parser.parse_args()
    if args.physics_hz <= 0: parser.error('--physics-hz must be positive')
    if not math.isfinite(args.gravity) or args.gravity < 0: parser.error('--gravity must be finite and nonnegative')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT) or args.output == ROOT: parser.error('Use a project output directory')
    sys.exit(0 if run(args) else 1)
