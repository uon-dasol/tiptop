"""Reference the existing M1013 and UON assets into a metre-based test stage.

Run with Isaac Sim's python.sh, independently of TiPToP's pixi environment.
With --plan, replay stored joint paths through TiPToP's existing executor.
Contact picking is not verified by either mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
ARM_ASSET = ROOT / 'ros2_ws/src/drivers/manipulator/doosan-robot2/dsr_description2/usd/m1013.usd'
GRIPPER_ASSET = ROOT / 'ros2_ws/src/description/uon_3f_description/usd/UON_3finger_gripper.usd'
MOUNT_CONFIG = ROOT / 'ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml'
MODEL = ROOT / 'tiptop/doosan/assets/m1013_uon/m1013_uon.urdf'
ARM = '/World/Arm/m1013'
GRIPPER = '/World/Gripper/UON_3finger_gripper'


def pose(xyz, rpy):
    """URDF parent-to-child transform in USD's row-vector convention."""
    from pxr import Gf

    rotation = Gf.Matrix4d(1)
    for axis, angle in zip((Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()), rpy):
        rotation *= Gf.Matrix4d().SetRotate(Gf.Rotation(axis, math.degrees(angle)))
    return rotation * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*xyz))


def model_fk(positions, target='link_6'):
    """Read FK directly from the existing planning URDF; no second robot model."""
    from pxr import Gf

    transforms = {'base_link': Gf.Matrix4d(1)}
    for joint in ET.parse(MODEL).getroot().findall('joint'):
        parent = joint.find('parent').get('link')
        if parent not in transforms:
            continue
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0] * 3
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0] * 3
        local = pose(xyz, rpy)
        if joint.get('name') in positions:
            axis = Gf.Vec3d(*[float(v) for v in joint.find('axis').get('xyz').split()])
            local = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, math.degrees(positions[joint.get('name')]))) * local
        child = joint.find('child').get('link')
        transforms[child] = local * transforms[parent]
        if child == target:
            return transforms[child]
    raise ValueError(f'Cannot resolve {target} in {MODEL}')


def build_stage(output):
    import omni.usd
    import yaml
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

    mount = yaml.safe_load(MOUNT_CONFIG.read_text())
    if mount['mount']['parent'] != 'link_6':
        raise ValueError('Recheck the USD mount path after changing mount.parent')
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)

    # The official arm says centimetres, but its joint offsets and mesh transforms
    # already match the metre-based URDF (e.g. 0.62 m for joint_3).
    # Deliberately preserve those values, including physical attributes.
    UsdGeom.Xform.Define(stage, '/World/Arm').GetPrim().GetReferences().AddReference(str(ARM_ASSET))
    UsdGeom.Xform.Define(stage, '/World/Gripper').GetPrim().GetReferences().AddReference(str(GRIPPER_ASSET))
    wrist = stage.GetPrimAtPath(ARM + '/link_6')
    body = stage.GetPrimAtPath(GRIPPER + '/body')
    if not (wrist.HasAPI(UsdPhysics.RigidBodyAPI) and body.HasAPI(UsdPhysics.RigidBodyAPI)):
        raise ValueError('The selected mount frames must be rigid bodies')
    cache = UsdGeom.XformCache()
    mount_pose = pose(mount['mount']['xyz'], mount['mount']['rpy'])
    desired_body = mount_pose * cache.GetLocalToWorldTransform(wrist)
    body_in_gripper = cache.GetLocalToWorldTransform(body)
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Gripper')).AddTransformOp().Set(
        body_in_gripper.GetInverse() * desired_body
    )

    # Fold the gripper into the arm's articulation. Reuse its existing fixed joint
    # as the flange joint instead of retaining its former attachment to world.
    stage.GetPrimAtPath(GRIPPER).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(GRIPPER + '/joints/FixedJoint'))
    joint.GetBody0Rel().SetTargets([wrist.GetPath()])
    joint.GetBody1Rel().SetTargets([body.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*mount['mount']['xyz']))
    joint.CreateLocalRot0Attr(Gf.Quatf(mount_pose.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(False)
    tcp = UsdGeom.Xform.Define(stage, GRIPPER + '/body/tcp')
    tcp.AddTransformOp().Set(pose(mount['tcp']['xyz'], mount['tcp']['rpy']))

    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr(60)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(200)
    key = UsdLux.DistantLight.Define(stage, '/World/KeyLight')
    key.CreateIntensityAttr(1000)
    key.AddRotateXYZOp().Set(Gf.Vec3f(35, -30, -30))
    ground = UsdGeom.Cube.Define(stage, '/World/Ground')
    ground.CreateSizeAttr(1)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01))
    ground.AddScaleOp().Set(Gf.Vec3d(6, 6, 0.02))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.14, 0.18)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2.1, -2.1, 1.8), Gf.Vec3d(0, 0, 0.9), Gf.Vec3d(0, 0, 1)
    ).GetInverse())
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    camera.CreateFocalLengthAttr(24)

    cache.Clear()
    actual = cache.GetLocalToWorldTransform(wrist)
    expected = model_fk({})
    translation_error = (actual.ExtractTranslation() - expected.ExtractTranslation()).GetLength()
    if translation_error > 0.001:
        raise ValueError(f'Zero-pose wrist differs from URDF by {translation_error} m')
    articulations = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if articulations != [ARM + '/root_joint']:
        raise ValueError(f'Expected a single arm articulation: {articulations}')
    output.mkdir(parents=True, exist_ok=True)
    stage_path = output / 'm1013_uon.usda'
    stage.GetRootLayer().Export(str(stage_path))
    return stage, mount, {'stage': str(stage_path), 'articulations': articulations,
                          'zero_pose_wrist_error_m': translation_error,
                          'mount_calibrated': mount['mount']['calibrated'],
                          'tcp_calibrated': mount['tcp']['calibrated']}


def check_motion(app, mount):
    import numpy as np
    import omni.timeline
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf

    robot = Articulation(ARM + '/root_joint')
    wrist = RigidPrim(ARM + '/link_6')
    body = RigidPrim(GRIPPER + '/body')
    base = RigidPrim(ARM + '/base_link')
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app.update()
    names = robot.dof_names
    arm_names = [f'joint_{i}' for i in range(1, 7)]
    required = arm_names + ['spinner_joint', 'f1_joint', 'f2_joint', 'f3_joint']
    if set(names) != set(required):
        raise ValueError(f'Unexpected merged DOFs: {names}')
    gravities = robot.get_link_enabled_gravities().numpy()[0]
    masses = robot.get_link_masses().numpy()[0]
    if not gravities.all() or not np.isfinite(masses).all() or not (masses > 0).all():
        raise ValueError('Every robot link must have gravity enabled and positive finite mass')
    indices = {name: names.index(name) for name in required}
    rows = []

    def settle(label, arm_degrees, spinner_degrees):
        targets = np.deg2rad(arm_degrees + [spinner_degrees])
        command_indices = [indices[name] for name in arm_names + ['spinner_joint']]
        robot.set_dof_position_targets(targets, dof_indices=command_indices)
        deadline = SimulationManager.get_simulation_time() + 3.0
        # Step by simulation time, not wall time or an assumed render frequency.
        for _ in range(1800):
            app.update()
            if SimulationManager.get_simulation_time() >= deadline:
                break
        else:
            raise RuntimeError('Physics did not advance by three seconds')
        q = robot.get_dof_positions().numpy()[0]
        if not np.isfinite(q).all():
            raise ValueError('Non-finite joint positions')
        error = float(np.max(np.abs(q[command_indices] - targets)))
        wpos, wrot = wrist.get_world_poses()
        bpos, brot = body.get_world_poses()
        wm = Gf.Matrix4d().SetRotate(Gf.Quatd(float(wrot.numpy()[0, 0]), Gf.Vec3d(*wrot.numpy()[0, 1:].astype(float))))
        wm.SetTranslateOnly(Gf.Vec3d(*wpos.numpy()[0].astype(float)))
        bm = Gf.Matrix4d().SetRotate(Gf.Quatd(float(brot.numpy()[0, 0]), Gf.Vec3d(*brot.numpy()[0, 1:].astype(float))))
        bm.SetTranslateOnly(Gf.Vec3d(*bpos.numpy()[0].astype(float)))
        expected_body = pose(mount['mount']['xyz'], mount['mount']['rpy']) * wm
        mount_error = (bm.ExtractTranslation() - expected_body.ExtractTranslation()).GetLength()
        relative_rotation = bm.ExtractRotationQuat().GetInverse() * expected_body.ExtractRotationQuat()
        mount_angle = abs(Gf.Rotation(relative_rotation).GetAngle())
        expected_wrist = model_fk({name: float(q[indices[name]]) for name in arm_names})
        fk_error = (wm.ExtractTranslation() - expected_wrist.ExtractTranslation()).GetLength()
        fk_angle = abs(Gf.Rotation(wm.ExtractRotationQuat().GetInverse() * expected_wrist.ExtractRotationQuat()).GetAngle())
        actual_tcp = pose(mount['tcp']['xyz'], mount['tcp']['rpy']) * bm
        model_tcp = model_fk({name: float(q[indices[name]]) for name in arm_names}, 'gripper_tcp')
        tcp_error = (actual_tcp.ExtractTranslation() - model_tcp.ExtractTranslation()).GetLength()
        mimic_error = max(abs(float(q[indices[name]] + 2 * q[indices['spinner_joint']]))
                          for name in ('f1_joint', 'f2_joint', 'f3_joint'))
        base_drift = float(np.linalg.norm(base.get_world_poses()[0].numpy()[0]))
        row = {'name': label, 'positions_deg': dict(zip(names, np.rad2deg(q).tolist())),
               'max_target_error_deg': math.degrees(error), 'mount_error_m': mount_error,
               'mount_error_deg': mount_angle, 'urdf_wrist_error_m': fk_error,
               'urdf_wrist_error_deg': fk_angle, 'urdf_tcp_error_m': tcp_error,
               'mimic_error_deg': math.degrees(mimic_error),
               'base_drift_m': base_drift}
        rows.append(row)
        print('ASSEMBLY_CHECK', json.dumps(row), flush=True)
        if (error > math.radians(2) or mount_error > 0.001 or mount_angle > 0.2
                or fk_error > 0.002 or fk_angle > 0.2 or tcp_error > 0.002 or mimic_error > math.radians(1)
                or base_drift > 0.001):
            raise ValueError(f'Assembly motion check failed: {row}')

    settle('home', [0] * 6, 0)
    settle('arm_and_gripper', [5, 5, -5, 5, 5, 5], 30)
    settle('gripper_60', [5, 5, -5, 5, 5, 5], 60)
    settle('return', [0] * 6, 0)
    return {'dof_names': names, 'samples': rows, 'success': True,
            'gravity_enabled': dict(zip(robot.link_names, gravities.tolist())),
            'link_masses_kg': dict(zip(robot.link_names, masses.tolist())),
            'contact_grasp_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--check', action='store_true', help='Run the selected check, write result, and exit')
    parser.add_argument('--plan', type=Path, help='Replay an existing TiPToP M1013 plan with the existing executor')
    parser.add_argument('--snapshot', action='store_true', help='Save a preview PNG after the motion check')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.headless and not args.check:
        parser.error('--headless requires --check')
    args.output = (args.output or ROOT / 'tiptop_outputs/m1013/isaac' /
                   ('replay' if args.plan else 'assembly')).resolve()
    if not args.output.is_relative_to(ROOT) or args.output == ROOT:
        parser.error('--output must be a subdirectory of the TiPToP repository')
    plan = None
    if args.plan:
        from plan_client import load_plan

        try:
            plan = load_plan(args.plan)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            parser.error(str(exc))
    for path in (ARM_ASSET, GRIPPER_ASSET, MOUNT_CONFIG, MODEL):
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARM_ASSET, GRIPPER_ASSET)}

    from isaacsim import SimulationApp

    # This process alone disables automatic unit conversion: the source arm's
    # metadata disagrees with its coordinates. Never change global preferences.
    app = SimulationApp({'headless': args.headless, 'width': 960, 'height': 720,
                         'extra_args': ['--/metricsAssembler/operationMode=0',
                                        '--/metricsAssembler/changeListenerEnabled=false']})
    report = {'success': False, 'contact_grasp_verified': False}
    try:
        import omni.kit.app
        import omni.timeline

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            'isaacsim.core.experimental.prims', True)
        _, mount, report = build_stage(args.output)
        report.update(success=False, contact_grasp_verified=False)
        # Reopen the exported composition to test the same artifact the user gets.
        if not omni.usd.get_context().open_stage(report['stage']):
            raise RuntimeError('Could not reopen the saved assembly stage')
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        viewport.camera_path = '/World/Camera'
        if plan is None:
            report.update(check_motion(app, mount))
        else:
            import logging
            from plan_client import replay_plan

            logging.basicConfig(level=logging.INFO)
            report['plan'] = str(args.plan.resolve())
            replay_plan(app, plan, realtime=not args.headless, report=report)
        if args.snapshot:
            import asyncio
            from omni.kit.viewport.utility import capture_viewport_to_file

            for _ in range(30):
                app.update()
            snapshot = args.output / 'assembly.png'
            capture = capture_viewport_to_file(viewport, str(snapshot))
            task = asyncio.ensure_future(capture.wait_for_result())
            for _ in range(300):
                app.update()
                if task.done():
                    task.result()
                    break
            else:
                task.cancel()
                raise RuntimeError('Preview capture timed out')
            report['snapshot'] = str(snapshot)
        report['source_sha256'] = source_hashes
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in source_hashes.items()):
            raise RuntimeError('A source USD changed during the check')
        print(('Plan replay completed' if plan else 'Assembly checks passed') +
              '; contact picking has not been tested.', flush=True)
        (args.output / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
        if not args.check:
            while app.is_running():
                app.update()
    except Exception as exc:
        report.update(success=False, error=str(exc))
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
        # Record the final state before stop resets the articulation.
        omni.timeline.get_timeline_interface().stop()
        app.close()


if __name__ == '__main__':
    main()
