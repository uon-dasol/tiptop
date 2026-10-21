"""Inspect one recorded grasp with the arm pointing down and the object present.

Prepare IK with TiPToP's Python (--prepare), then view with Isaac's python.sh.
The object is fixed and contact is disabled: this checks frames, not grasp physics.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/black_pepper_shaker/UON_3finger_gripper'
OBJECT = ROOT / 'data/black_pepper_shaker/3d_model/edited/black_pepper_shaker_rigid.usd'
OUTPUT = ROOT / 'tiptop_outputs/m1013/isaac/grasp-scene'


def prepare(output):
    import numpy as np
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from tiptop.doosan.model import m1013_uon_curobo_cfg
    from tiptop.doosan.tamp import DEFAULT_TARGET_DIMS, DEFAULT_TARGET_XYZ

    merged = json.loads((DATA / 'black_pepper_shaker_merged_grasp_zero_pose.json').read_text())
    row = merged['data'][0]
    if merged['metadata']['coordinate_frame'] != 'object_zero_pose':
        raise ValueError('Expected object-local grasp records')
    raw_file = DATA / f'output_grasp/{row["scene_id"]}.json'
    raw = next(r for r in json.loads(raw_file.read_text())
               if r['source_pregrasp_index'] == row['source_pregrasp_index'])
    world_from_object = np.array(raw['quality']['object_initial_mat'])
    world_from_object[:3, 3] = [.55, -.12, .50]
    world_from_tcp = world_from_object @ np.array(row['grasp_mat'])
    hover = world_from_tcp.copy()
    hover[:3, 3] -= .08 * world_from_tcp[:3, 2]
    world = WorldConfig(cuboid=[Cuboid(
        name='target', dims=list(DEFAULT_TARGET_DIMS), pose=[*DEFAULT_TARGET_XYZ, 1, 0, 0, 0])])
    solver = IKSolver(IKSolverConfig.load_from_robot_config(
        m1013_uon_curobo_cfg()['robot_cfg'], world, num_seeds=64,
        self_collision_check=True, self_collision_opt=True,
        position_threshold=.0001, rotation_threshold=.001, high_precision=True))
    poses = {}
    reference = np.zeros(6)
    for label, matrix in [('hover', hover), ('grasp', world_from_tcp)]:
        result = solver.solve_single(Pose.from_matrix(solver.tensor_args.to_device(matrix[None])),
                                     return_seeds=8)
        if not bool(result.success.any()):
            raise RuntimeError(f'No self-collision-free arm IK for {label}')
        solutions = result.solution[result.success].cpu().numpy()
        q = solutions[np.linalg.norm(solutions - reference, axis=1).argmin()]
        reference = q
        poses[label] = {'q': q.tolist(), 'world_from_tcp': matrix.tolist()}
    record = {
        'candidate_index': 0, 'scene_id': row['scene_id'],
        'source_pregrasp_index': row['source_pregrasp_index'],
        'world_from_object': world_from_object.tolist(), 'poses': poses,
        'grasp_joint_rad': raw['quality']['grasp_complete_joint_pos_rad']['spinner_joint'],
        'approach_world': world_from_tcp[:3, 2].tolist(),
        'approach_angle_from_down_deg': float(np.degrees(np.arccos(np.clip(-world_from_tcp[2, 2], -1, 1)))),
        'contacts_object': [c['position'] for c in raw['contact_points_object_local']],
        'db_grasp_from_model_tcp': np.eye(4).tolist(),
        'object_contact_enabled': False, 'physical_grasp_verified': False,
        'gravity_magnitude': 0.0,
        'target_xyz': list(DEFAULT_TARGET_XYZ), 'target_dims': list(DEFAULT_TARGET_DIMS),
        'ik_checks': 'arm limits, self collision and table; object contact is not checked',
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'scene.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def decorate(stage, record):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    from tiptop.doosan.tamp import DEFAULT_TARGET_DIMS, DEFAULT_TARGET_XYZ

    # Match the DB's zero-gravity collection for this geometry inspection.
    UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene')).GetGravityMagnitudeAttr().Set(0.0)
    world_from_object = np.array(record['world_from_object'])
    obj = UsdGeom.Xform.Define(stage, '/World/Pepper')
    obj.AddTransformOp().Set(Gf.Matrix4d(world_from_object.T.tolist()))
    # The source USD's nested scale is 10x the CAD used by the DB; its config
    # explicitly uses scale 0.1. Apply that outside the referenced asset.
    asset = UsdGeom.Xform.Define(stage, '/World/Pepper/Asset')
    asset.AddScaleOp().Set(Gf.Vec3f(.1))
    asset.GetPrim().GetReferences().AddReference(str(OBJECT))
    for prim in Usd.PrimRange(asset.GetPrim()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
        obj.GetPrim()).ComputeAlignedBox()
    center = record.get('target_xyz', DEFAULT_TARGET_XYZ)
    size = record.get('target_dims', DEFAULT_TARGET_DIMS)
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    table.CreateSizeAttr(1)
    table.AddTranslateOp().Set(Gf.Vec3d(*center))
    table.AddScaleOp().Set(Gf.Vec3d(*size))
    table.CreateDisplayColorAttr([Gf.Vec3f(.25, .31, .35)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    # Legs are visual; the tabletop is the same collision cuboid as the planner.
    height = center[2] - size[2] / 2
    for i, (x, y) in enumerate([(x, y) for x in (-1, 1) for y in (-1, 1)]):
        leg = UsdGeom.Cube.Define(stage, f'/World/TableLeg_{i}')
        leg.CreateSizeAttr(1)
        leg.AddTranslateOp().Set(Gf.Vec3d(center[0] + x * (size[0] / 2 - .04),
                                         center[1] + y * (size[1] / 2 - .04), height / 2))
        leg.AddScaleOp().Set(Gf.Vec3d(.04, .04, height))
        leg.CreateDisplayColorAttr([Gf.Vec3f(.16, .18, .20)])
    for i, contact in enumerate(record['contacts_object']):
        point = world_from_object @ np.r_[contact, 1]
        marker = UsdGeom.Sphere.Define(stage, f'/World/Contact_{i}')
        marker.CreateRadiusAttr(.0015)
        marker.AddTranslateOp().Set(Gf.Vec3d(*point[:3]))
        marker.CreateDisplayColorAttr([Gf.Vec3f(1, .1, .1)])
    # A close view of the contact region is also available in the camera menu.
    for name, eye, target in [
        ('Camera', (1.65, -1.65, 1.35), (.35, -.08, .55)),
        ('GraspCamera', (1.0, -.75, .85), (.54, -.12, .52)),
    ]:
        camera = UsdGeom.Camera.Define(stage, '/World/' + name)
        camera.ClearXformOpOrder()
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(28)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100))
    return bounds


def view(args):
    import numpy as np
    from m1013_uon import ARM, ARM_ASSET, GRIPPER, GRIPPER_ASSET, build_stage, pose
    from isaacsim import SimulationApp

    record = json.loads((args.output / 'scene.json').read_text())
    sources = [ARM_ASSET, GRIPPER_ASSET, OBJECT]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 900,
                         'extra_args': ['--/metricsAssembler/operationMode=0',
                                        '--/metricsAssembler/changeListenerEnabled=false']})
    report = dict(record, success=False, selected_pose=args.pose, source_sha256=hashes,
                  gravity_magnitude=0.0)
    try:
        import omni.kit.app
        import omni.timeline
        from pxr import Gf, UsdGeom
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            'isaacsim.core.experimental.prims', True)
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager

        stage, mount, assembly = build_stage(args.output)
        bounds = decorate(stage, record)
        stage.GetRootLayer().Export(assembly['stage'])
        viewport = get_active_viewport()
        viewport.camera_path = '/World/GraspCamera' if args.closeup else '/World/Camera'
        robot = Articulation(ARM + '/root_joint')
        body = RigidPrim(GRIPPER + '/body')
        omni.timeline.get_timeline_interface().play()
        app.update()
        names = robot.dof_names
        spinner = 0.0 if args.pose == 'hover' else record['grasp_joint_rad']
        joint_values = dict(zip([f'joint_{i}' for i in range(1, 7)], record['poses'][args.pose]['q']))
        joint_values.update(spinner_joint=spinner, f1_joint=-2 * spinner,
                            f2_joint=-2 * spinner, f3_joint=-2 * spinner)
        q = np.array([joint_values[name] for name in names])
        # This is scene initialization, not a planned motion from the old home.
        robot.set_dof_positions(q)
        robot.set_dof_velocity_targets(0.0)
        for _ in range(90):
            SimulationManager.step(update_fabric=SimulationManager.is_fabric_enabled())
        positions, orientations = body.get_world_poses()
        xyz, quat = positions.numpy()[0].astype(float), orientations.numpy()[0].astype(float)
        world_from_body = Gf.Matrix4d().SetRotate(Gf.Quatd(float(quat[0]), Gf.Vec3d(*quat[1:])))
        world_from_body.SetTranslateOnly(Gf.Vec3d(*xyz))
        body_from_tcp = pose(mount['tcp']['xyz'], mount['tcp']['rpy'])
        actual_tcp = np.array(body_from_tcp * world_from_body).T
        expected = np.array(record['poses'][args.pose]['world_from_tcp'])
        report['joint_error_rad'] = float(np.max(np.abs(robot.get_dof_positions().numpy()[0] - q)))
        error = float(np.linalg.norm(actual_tcp[:3, 3] - expected[:3, 3]))
        angular = float(np.degrees(np.arccos(np.clip(
            (np.trace(actual_tcp[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1))))
        if error > .002 or angular > .5:
            raise RuntimeError(f'Pose mismatch: {error * 1000:.3f} mm, {angular:.3f} deg')
        report.update(success=True, tcp_position_error_m=error, tcp_rotation_error_deg=angular,
                      actual_tcp=actual_tcp.tolist(), q_init=record['poses'][args.pose]['q'],
                      object_world_bounds=[list(bounds.GetMin()), list(bounds.GetMax())],
                      stage=assembly['stage'])
        for _ in range(30):
            app.update()
        if args.snapshot:
            import asyncio

            path = args.output / f'{args.pose}.png'
            task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(path)).wait_for_result())
            for _ in range(300):
                app.update()
                if task.done():
                    task.result()
                    break
            else:
                raise RuntimeError('Snapshot timed out')
            report['snapshot'] = str(path)
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in sources):
            raise RuntimeError('Source asset changed')
        print('Grasp scene ready; object fixed, contact disabled.', flush=True)
        (args.output / f'{args.pose}-check.json').write_text(json.dumps(report, indent=2) + '\n')
        if not args.check:
            while app.is_running():
                app.update()
    except Exception as exc:
        report.update(success=False, error=str(exc))
        raise
    finally:
        (args.output / f'{args.pose}-check.json').write_text(json.dumps(report, indent=2) + '\n')
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--pose', choices=['hover', 'grasp'], default='hover')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--closeup', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT) or args.output == ROOT:
        parser.error('--output must stay inside the project')
    if args.headless and not args.check:
        parser.error('--headless requires --check')
    if args.prepare:
        prepare(args.output)
    elif not (args.output / 'scene.json').is_file():
        parser.error('Run --prepare with TiPToP Python first')
    else:
        view(args)


if __name__ == '__main__':
    main()
