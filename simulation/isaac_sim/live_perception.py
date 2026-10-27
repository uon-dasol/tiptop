"""Live Isaac sensor endpoint for the shared TiPToP perception front end.

GET /observe renders the current stage and reads joint feedback. POST /execute
uses the existing plan executor; simulator object poses/contact forces are read
only for evaluation, never supplied as perception or controller pose inputs.
Run with Isaac's Python, independently of the inference environments.
"""

import argparse
import io
import math
import traceback
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'simulation/isaac_sim'))
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=ROOT / "simulation/isaac_sim/config/three_objects.json")
parser.add_argument("--asset-root", type=Path, help="Optional CAD root containing <class>/edited/<class>_rigid.usd")
parser.add_argument("--camera-preparation", type=Path, default=ROOT / "simulation/isaac_sim/config/wrist_camera.json")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--port", type=int, default=13345)
args, _ = parser.parse_known_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
if any((args.output / name).exists() for name in ["execution.json", "accepted-plan.json"]):
    parser.error("This directory belongs to an earlier runtime; use a new session output directory")
(args.output / "close").unlink(missing_ok=True)
from tiptop.doosan.virtual_camera import load_camera_preparation, camera_preparation_id, validate_virtual_camera_plan

prepared = load_camera_preparation(args.camera_preparation)
from observation_server import ObservationServer, fixture_target_index, require_current_result
from task_runtime import ExecutionStore, evaluate_pick_place

# Reserve the endpoint before allocating an Isaac window/GPU resources.
executions = ExecutionStore(args.output)
server = ObservationServer(port=args.port, executions=executions)
try:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1500,
            "height": 950,
            "enable_crashreporter": False,
            "extra_args": [
                "--/metricsAssembler/operationMode=0",
                "--/metricsAssembler/changeListenerEnabled=false",
                "--/log/file=" + str(args.output / "kit.log"),
            ],
        }
    )
except BaseException:
    server.close()
    raise
try:
    import numpy as np
    import omni.kit.app
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, UsdUtils
    from camera import IsaacStereoCamera
    from m1013_uon import ARM, GRIPPER, build_stage
    from plan_client import IsaacPlanClient, validate_plan
    from tiptop.execute_plan import execute_cutamp_plan

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.core.experimental.prims", True
    )
    stage, _, _ = build_stage(args.output)
    get_active_viewport().camera_path = "/World/Camera"
    table_center = np.array([0.65, 0.0, 0.38])
    table_dims = np.array([0.8, 0.8, 0.04])
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(*table_center))
    table.AddScaleOp().Set(Gf.Vec3d(*table_dims))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.35, 0.4)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    # Fixture setup, not perception input or object poses used by the controller.
    for index, row in enumerate(json.loads(args.scene.read_text())):
        parent = UsdGeom.Xform.Define(stage, f"/World/Item_{index}")
        parent.AddTransformOp().Set(Gf.Matrix4d(np.asarray(row["world_from_object"]).T.tolist()))
        asset = UsdGeom.Xform.Define(stage, f"/World/Item_{index}/Asset")
        asset.AddScaleOp().Set(Gf.Vec3f(*row["asset_scale"]))
        asset_path = (
            args.asset_root / row["class"] / "edited" / (row["class"] + "_rigid.usd")
            if args.asset_root is not None else ROOT / row["asset"]
        )
        if not asset_path.is_file():
            raise FileNotFoundError(f"Fixture USD not found: {asset_path}; check --asset-root or --scene")
        asset.GetPrim().GetReferences().AddReference(str(asset_path.resolve()))
    mount = prepared["profile"]["mount"]
    camera_parent = ARM + "/link_6" if mount == "wrist" else "/World"
    camera = IsaacStereoCamera(stage, prepared["profile"], parent_path=camera_parent)
    for row in prepared.get("fixed_obstacles", []):
        box = UsdGeom.Cube.Define(stage, "/World/FixedStructures/" + row["name"])
        box.CreateSizeAttr(1.0)
        box.AddTransformOp().Set(Gf.Matrix4d(np.asarray(row["base_from_obstacle"], float).T.tolist()))
        box.AddScaleOp().Set(Gf.Vec3d(*row["dims_m"]))
        box.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.3)])
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    # Evaluation sensors read physics only; no object pose enters the planner here.
    from isaacsim.core.experimental.prims import RigidPrim
    from omni.physx import get_physx_simulation_interface
    fixture = json.loads(args.scene.read_text())
    paths = [f"/World/Item_{i}/Asset/{entry['class']}" for i, entry in enumerate(fixture)]
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Table')).CreateKinematicEnabledAttr(True)
    for path in [*paths, '/World/Table', *[GRIPPER + f'/finger{i}' for i in range(1, 4)]]:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    UsdPhysics.DriveAPI(stage.GetPrimAtPath(GRIPPER + '/joints/spinner_joint'), 'angular').CreateMaxForceAttr(5.)
    bodies = [RigidPrim(path, contact_filter_paths=["/World/Table"]) for path in paths]
    # Isaac pairs filters per prim when their counts match (three fingers and
    # three objects). Use one view per target so all fingers measure that target.
    fingers = [RigidPrim([GRIPPER + f'/finger{i}' for i in range(1, 4)],
                        contact_filter_paths=[path]) for path in paths]
    target_index = target_body = object_id = None
    other_indices = []
    physics = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    execution_report = None
    pending_plan = None
    execution_dir = None
    baseline = None
    other_baselines = []

    def target_contact_forces():
        return np.linalg.norm(fingers[target_index].get_contact_force_matrix(dt=client.dt).numpy()[:, 0], axis=1)

    class ContactClient(IsaacPlanClient):
        def _tick(self):
            super()._tick()
            if execution_report is not None and self.ticks % 12 == 0:
                positions, orientations = target_body.get_world_poses()
                position = positions.numpy()[0]
                forces = target_contact_forces()
                carrying = getattr(self, 'carrying', False)
                support = float(np.linalg.norm(target_body.get_contact_force_matrix(dt=self.dt).numpy()))
                self.ever_lifted = self.ever_lifted or position[2] - baseline[2] > .03
                # A tilted object still held by the fingers is not a failure.
                # Sustained table contact with no finger contact BEFORE release
                # is evidence of losing the object after lifting.
                lost = carrying and self.ever_lifted and support > .01 and float(forces.max()) < .1
                self.drop_contact_time = self.drop_contact_time + 12 * self.dt if lost else 0.
                execution_report['trace'].append(dict(time_s=self.sim.get_simulation_time(),
                    completed_actions=len(self.calls), position=position.tolist(),
                    orientation_wxyz=orientations.numpy()[0].tolist(), forces_n=forces.tolist(),
                    phase='transport' if carrying else 'approach_or_release', table_support_force_n=support))
                if self.drop_contact_time >= .25:
                    execution_report['unintended_drop_detected'] = True
                    raise RuntimeError('Object lost before commanded release')
                if self.ticks % 120 == 0:
                    executions.write(execution_dir / 'execution-progress.json', execution_report)
                for index, body_index in enumerate(other_indices):
                    prim = bodies[body_index]
                    displacement = float(np.linalg.norm(prim.get_world_poses()[0].numpy()[0] - other_baselines[index]))
                    execution_report['max_other_displacement_m'] = max(execution_report['max_other_displacement_m'], displacement)
                    entry = execution_report['other_objects'][index]
                    entry['max_displacement_m'] = max(entry['max_displacement_m'], displacement)
                    if displacement > .005:
                        raise RuntimeError('A non-target object moved more than 5 mm')
        def close_gripper(self, speed=1., force=.1):
            def close():
                # Same fixed 5 Nm cap and position closing as previous contact trials.
                # Contact is checked for completion; there is no force-maintenance controller.
                start = self.get_state()['gripper']['position_rad']
                for tick in range(1, math.ceil(8 / self.dt) + 1):
                    target = min(self.close_angle, start + .35 * tick * self.dt)
                    self.robot.set_dof_position_targets(target, dof_indices=[self.spinner])
                    self.robot.set_dof_velocity_targets(0., dof_indices=[self.spinner])
                    self._tick()
                physics.wake_up(stage_id, object_id)
                self._tick()
                forces = target_contact_forces()
                if np.count_nonzero(forces > .1) < 2:
                    raise RuntimeError('No opposing-finger contact after closing: ' + str(forces.tolist()))
                return dict(action='close', forces_n=forces.tolist(), torque_cap_nm=5.,
                            position_rad=self.get_state()['gripper']['position_rad'])
            result = self._run('gripper', close)
            self.carrying = bool(result['success'])
            return result

        def open_gripper(self, speed=1., force=.1):
            self.carrying = False  # Deliberate release is not an unintended drop.
            return super().open_gripper(speed=speed, force=force)

        def prepare_task(self):
            state = self.get_state()
            if state['motion_status'] != 0 or abs(state['gripper']['position_rad'] - self.open_angle) > .03:
                raise ValueError('A new task requires an idle arm and an open gripper; inspect/release after a motion failure')
            self.cancelled = False
            self.calls = []
            self.carrying = self.ever_lifted = False
            self.drop_contact_time = 0.
            self.wall_start, self.sim_start = time.perf_counter(), self.sim.get_simulation_time()
    client = ContactClient(app, realtime=True, physics_dt=1.0 / 240.0)
    client.robot.set_link_enabled_gravities(False)
    initial = dict(zip([f"joint_{i}" for i in range(1, 7)], prepared["q_capture"]))
    initial.update(spinner_joint=0.0, f1_joint=0.0, f2_joint=0.0, f3_joint=0.0)
    client.robot.set_dof_positions([initial[name] for name in client.names])
    client.robot.set_dof_position_targets(
        [initial[client.names[i]] for i in client.controlled], dof_indices=client.controlled
    )
    for _ in range(480):
        client._tick()
    # Check every target's contact view before accepting a physical execution.
    sensor_shapes = [view.get_contact_force_matrix(dt=client.dt).numpy().shape for view in fingers]
    if any(shape != (3, 1, 3) for shape in sensor_shapes):
        raise RuntimeError(f"Expected three fingers and one target per contact view: {sensor_shapes}")
    (args.output / 'contact-sensors.json').write_text(json.dumps(
        [dict(class_id=entry['class'], force_matrix_shape=shape) for entry, shape in zip(fixture, sensor_shapes)], indent=2))
    panel = ui.Window("TiPToP LIVE - current observation / pose", width=650, height=640)
    with panel.frame:
        with ui.VStack(spacing=6):
            ui.Label("Gemini -> SAM2 -> FoundationStereo -> FoundationPose -> DB -> cuTAMP -> pick/place", height=28)
            image = ui.Image(width=640, height=400)
            status = ui.Label("Waiting for a current observation request...", height=120, word_wrap=True)
    panel.position_x, panel.position_y = 820, 60
    latest_id = None
    latest_q = None
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Camera')).GetOrderedXformOps()[0].Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.4,-1.45,1.15), Gf.Vec3d(.6,0,.48), Gf.Vec3d(0,0,1)).GetInverse())

    def dispatch(path, body):
        global latest_id, latest_q, pending_plan, target_index, target_body, object_id, other_indices, execution_dir
        if path == "/health":
            return json.dumps(dict(success=True, pid=os.getpid(), active_observation_id=executions.active_id,
                                   motion_stopped=client.cancelled, camera_mount=mount,
                                   camera_preparation_sha256=camera_preparation_id(prepared))).encode()
        if executions.active_id is not None:
            raise ValueError('An execution is active; wait for its result before requesting another task')
        if path == "/observe":
            if not client.timeline.is_playing():
                client.timeline.play()
            if not client.cancelled:
                client._tick()
            request = camera.capture_request(
                client.get_joint_positions(), "Select a known object and place it in an observed empty region", pause_timeline=False
            )
            latest_id, latest_q = uuid.uuid4().hex, request["q_init"].copy()
            request["observation_id"] = latest_id
            request["camera_preparation"] = json.dumps(prepared)
            # Declared fixed workspace geometry, not object poses read from physics.
            # The partial observed table alone can miss a near-side arm collision.
            request["workspace_table_bounds_m"] = np.stack([table_center-table_dims/2,table_center+table_dims/2])
            preview = args.output / (latest_id + ".png")
            Image.fromarray(request["rgb"]).save(preview)
            image.source_url = str(preview)
            stale = stage.GetPrimAtPath('/World/EstimatedCAD')
            if stale.IsValid():
                UsdGeom.Imageable(stale).MakeInvisible()
            status.text = "Fresh frame captured. Waiting for task inference or the next instruction."
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **request)
            return buffer.getvalue()
        if path == "/display":
            packet = np.load(io.BytesIO(body), allow_pickle=False)
            require_current_result(str(packet["observation_id"]), latest_id, client.get_joint_positions(), latest_q)
            overlay = args.output / (latest_id + "-pose.png")
            Image.fromarray(packet["overlay"]).save(overlay)
            image.source_url = str(overlay)
            status.text = str(packet["status"])
            if "cad_points_world" in packet:
                points = UsdGeom.Points.Define(stage, "/World/EstimatedCAD")
                points.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in packet["cad_points_world"][::5]])
                points.CreateWidthsAttr([0.0015])
                points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                points.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.9, 1.0)])
                points.MakeVisible()
            print("LIVE_PERCEPTION_DISPLAY " + status.text, flush=True)
            return b'{"success":true}'
        if path == '/execute':
            plan = json.loads(body)
            validate_virtual_camera_plan(plan, prepared)
            selected = fixture_target_index(fixture, plan.get('class_id'))
            require_current_result(plan['observation_id'], latest_id, client.get_joint_positions(), plan['q_init'])
            validated = validate_plan(plan)
            client.prepare_task()
            execution_dir = executions.accept(plan)
            pending_plan = validated
            latest_id = latest_q = None  # A consumed observation can never execute again.
            target_index, target_body = selected, bodies[selected]
            object_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(paths[selected]).GetPath())
            other_indices = [i for i in range(len(fixture)) if i != selected]
            status.text = f"Target: {plan['class_id']}. Executing current task."
            return json.dumps(dict(accepted=True, observation_id=plan['observation_id'])).encode()
        raise ValueError("Unknown live camera endpoint")

    (args.output / "process.json").write_text(json.dumps(dict(pid=os.getpid(), port=args.port)))
    print(f"LIVE_CAMERA_READY pid={os.getpid()} port={args.port} camera_mount={mount}", flush=True)
    while app.is_running() and not (args.output / "close").exists():
        app.update()
        server.poll(dispatch)
        if pending_plan is not None:
            plan, pending_plan = pending_plan, None
            stale = stage.GetPrimAtPath('/World/EstimatedCAD')
            if stale.IsValid():
                UsdGeom.Imageable(stale).MakeInvisible()
            baseline = target_body.get_world_poses()[0].numpy()[0].copy()
            other_baselines = [bodies[i].get_world_poses()[0].numpy()[0].copy() for i in other_indices]
            execution_report = dict(success=False, observation_id=plan['observation_id'], trace=[],
                class_id=fixture[target_index]['class'], instance_id=fixture[target_index]['instance_id'],
                target_path=paths[target_index],
                other_objects=[dict(class_id=fixture[i]['class'], instance_id=fixture[i]['instance_id'],
                    baseline_position=position.tolist(), max_displacement_m=0.)
                    for i, position in zip(other_indices, other_baselines)],
                object_attached=False, object_teleported=False, max_other_displacement_m=0.,
                spinner_torque_cap_nm=float(client.robot.get_dof_max_efforts().numpy()[0,client.spinner]),
                armatures=client.robot.get_dof_armatures().numpy()[0].tolist(),
                baseline_position=baseline.tolist(), calls=client.calls, gemini_api_calls=0,
                finished=False, unintended_drop_detected=False,
                criterion='lift > 30 mm, moved > 100 mm, released in observed region, table support, no unintended drop or non-target displacement; any final orientation')
            try:
                execute_cutamp_plan(plan['steps'], client)
                for _ in range(480):
                    client._tick()
                physics.wake_up(stage_id, object_id)
                for _ in range(24):
                    client._tick()
                final = target_body.get_world_poses()[0].numpy()[0]
                forces = target_contact_forces()
                support = float(np.linalg.norm(target_body.get_contact_force_matrix(dt=client.dt).numpy()))
                region = plan['empty_region']
                execution_report.update(evaluate_pick_place(
                    baseline=baseline, final=final, trace=execution_report['trace'], forces=forces,
                    support=support, region=region, max_other_displacement=execution_report['max_other_displacement_m'],
                    unintended_drop=execution_report['unintended_drop_detected']))
                lift, moved = execution_report['maximum_lift_m'], execution_report['horizontal_displacement_m']
                status.text = ('PICK / PLACE PASSED' if execution_report['success'] else 'PICK / PLACE FAILED') + '\n' + f'Lift: {lift*1000:.1f} mm. Move: {moved*1000:.1f} mm. Table support: {support:.2f} N'
            except Exception as error:
                execution_report.update(error=str(error), physical_grasp_verified=False)
                status.text = 'CONTACT TRIAL STOPPED: ' + str(error)
                client.stop()
            finally:
                execution_report['finished'] = True
                execution_report['gemini_api_calls'] = plan.get('gemini_api_calls', 0)
                positions, orientations = target_body.get_world_poses()
                execution_report['final_position'] = positions.numpy()[0].tolist()
                execution_report['final_orientation_wxyz'] = orientations.numpy()[0].tolist()
                for entry, i in zip(execution_report['other_objects'], other_indices):
                    entry['final_position'] = bodies[i].get_world_poses()[0].numpy()[0].tolist()
                executions.finish(execution_report)
                print('LIVE_CONTACT_RESULT ' + json.dumps({k:v for k,v in execution_report.items() if k not in ['trace','calls']}), flush=True)
                execution_report = None
                try:
                    current = camera.capture_request(client.get_joint_positions(), 'Post-execution preview', pause_timeline=False)
                    preview = execution_dir / 'final-camera.png'
                    Image.fromarray(current['rgb']).save(preview)
                    image.source_url = str(preview)
                except Exception as preview_error:
                    print('FINAL_PREVIEW_ERROR', str(preview_error), flush=True)
        time.sleep(0.015)
    camera.close()
except BaseException:
    failure = traceback.format_exc()
    (args.output / 'fatal-error.txt').write_text(failure)
    print(failure, flush=True)
    raise
finally:
    if server is not None:
        server.close()
    (args.output / "process.json").unlink(missing_ok=True)
    app.close()
