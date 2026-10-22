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
parser.add_argument("--camera-preparation", type=Path, default=ROOT / "simulation/isaac_sim/config/wrist_camera.json")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--port", type=int, default=13345)
args, _ = parser.parse_known_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
if any((args.output / name).exists() for name in ["execution.json", "accepted-plan.json"]):
    parser.error("This runtime already contains an execution; use a new output directory")
(args.output / "close").unlink(missing_ok=True)
prepared = json.loads(args.camera_preparation.read_text())
if not prepared["success"]:
    raise ValueError("A collision-checked camera preparation is required")
from observation_server import ObservationServer, require_current_result

# Reserve the endpoint before allocating an Isaac window/GPU resources.
server = ObservationServer(port=args.port, result_path=args.output / "execution.json")
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
    from plan_client import IsaacPlanClient, load_plan
    from tiptop.execute_plan import execute_cutamp_plan

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.core.experimental.prims", True
    )
    stage, _, _ = build_stage(args.output)
    get_active_viewport().camera_path = "/World/Camera"
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.38))
    table.AddScaleOp().Set(Gf.Vec3d(0.8, 0.8, 0.04))
    table.CreateDisplayColorAttr([Gf.Vec3f(0.3, 0.35, 0.4)])
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    # Fixture setup, not perception input or object poses used by the controller.
    for index, row in enumerate(json.loads(args.scene.read_text())):
        parent = UsdGeom.Xform.Define(stage, f"/World/Item_{index}")
        parent.AddTransformOp().Set(Gf.Matrix4d(np.asarray(row["world_from_object"]).T.tolist()))
        asset = UsdGeom.Xform.Define(stage, f"/World/Item_{index}/Asset")
        asset.AddScaleOp().Set(Gf.Vec3f(*row["asset_scale"]))
        asset.GetPrim().GetReferences().AddReference(str(ROOT / row["asset"]))
    camera = IsaacStereoCamera(stage, prepared["profile"], parent_path=ARM + "/link_6")
    # Evaluation sensors read physics only; no object pose enters the planner here.
    from isaacsim.core.experimental.prims import RigidPrim
    from omni.physx import get_physx_simulation_interface
    fixture = json.loads(args.scene.read_text())
    paths = [f"/World/Item_{i}/Asset/{entry['class']}" for i, entry in enumerate(fixture)]
    target_path = paths[0]
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Table')).CreateKinematicEnabledAttr(True)
    for path in [*paths, '/World/Table', *[GRIPPER + f'/finger{i}' for i in range(1, 4)]]:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    UsdPhysics.DriveAPI(stage.GetPrimAtPath(GRIPPER + '/joints/spinner_joint'), 'angular').CreateMaxForceAttr(5.)
    target_body = RigidPrim(target_path, contact_filter_paths=["/World/Table"])
    fingers = RigidPrim([GRIPPER + f'/finger{i}' for i in range(1, 4)], contact_filter_paths=[target_path])
    others = [RigidPrim(path) for path in paths[1:]]
    physics = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    object_id = PhysicsSchemaTools.sdfPathToInt(stage.GetPrimAtPath(target_path).GetPath())
    execution_report = None
    pending_plan = None
    has_executed = False
    baseline = None
    other_baselines = []

    class ContactClient(IsaacPlanClient):
        def _tick(self):
            super()._tick()
            if execution_report is not None and self.ticks % 12 == 0:
                position = target_body.get_world_poses()[0].numpy()[0]
                forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                execution_report['trace'].append(dict(time_s=self.sim.get_simulation_time(),
                    completed_actions=len(self.calls), position=position.tolist(), forces_n=forces.tolist()))
                if self.ticks % 120 == 0:
                    (args.output / 'execution-progress.json').write_text(json.dumps(execution_report, indent=2))
                for index, prim in enumerate(others):
                    displacement = float(np.linalg.norm(prim.get_world_poses()[0].numpy()[0] - other_baselines[index]))
                    execution_report['max_other_displacement_m'] = max(execution_report['max_other_displacement_m'], displacement)
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
                forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=self.dt).numpy()[:, 0], axis=1)
                if np.count_nonzero(forces > .1) < 2:
                    raise RuntimeError('No opposing-finger contact after closing: ' + str(forces.tolist()))
                return dict(action='close', forces_n=forces.tolist(), torque_cap_nm=5.,
                            position_rad=self.get_state()['gripper']['position_rad'])
            return self._run('gripper', close)
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
        global latest_id, latest_q, pending_plan, has_executed
        if path == "/health":
            return json.dumps(dict(success=True, pid=os.getpid())).encode()
        if path == "/observe":
            if not client.timeline.is_playing():
                client.timeline.play()
            client._tick()
            request = camera.capture_request(
                client.get_joint_positions(), "후추통 집고 빈 곳에 놓아", pause_timeline=False
            )
            latest_id, latest_q = uuid.uuid4().hex, request["q_init"].copy()
            request["observation_id"] = latest_id
            preview = args.output / (latest_id + ".png")
            Image.fromarray(request["rgb"]).save(preview)
            image.source_url = str(preview)
            status.text = "Fresh frame captured. Running local mask/depth/pose inference..."
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
            print("LIVE_PERCEPTION_DISPLAY " + status.text, flush=True)
            return b'{"success":true}'
        if path == '/execute':
            if has_executed or pending_plan is not None:
                raise ValueError('This runtime accepts one execution only')
            plan = json.loads(body)
            if plan.get('class_id') != fixture[0]['class']:
                raise ValueError('This fixture evaluates the pepper target only')
            require_current_result(plan['observation_id'], latest_id, client.get_joint_positions(), plan['q_init'])
            path = args.output / 'accepted-plan.json'
            path.write_text(json.dumps(plan))
            pending_plan = load_plan(path)
            status.text = 'Current-observation plan accepted. Starting one physical contact trial.'
            return b'{"accepted":true}'
        raise ValueError("Unknown live camera endpoint")

    (args.output / "process.json").write_text(json.dumps(dict(pid=os.getpid(), port=args.port)))
    print(f"LIVE_CAMERA_READY pid={os.getpid()} port={args.port}", flush=True)
    while app.is_running() and not (args.output / "close").exists():
        app.update()
        server.poll(dispatch)
        if pending_plan is not None:
            plan, pending_plan = pending_plan, None
            has_executed = True
            stale = stage.GetPrimAtPath('/World/EstimatedCAD')
            if stale.IsValid():
                UsdGeom.Imageable(stale).MakeInvisible()
            baseline = target_body.get_world_poses()[0].numpy()[0].copy()
            other_baselines = [prim.get_world_poses()[0].numpy()[0].copy() for prim in others]
            execution_report = dict(success=False, observation_id=plan['observation_id'], trace=[],
                object_attached=False, object_teleported=False, max_other_displacement_m=0.,
                spinner_torque_cap_nm=float(client.robot.get_dof_max_efforts().numpy()[0,client.spinner]),
                armatures=client.robot.get_dof_armatures().numpy()[0].tolist(),
                baseline_position=baseline.tolist(), calls=client.calls, gemini_api_calls=0,
                finished=False, criterion='lift > 30 mm, moved > 100 mm, released in observed region, table support, no non-target displacement')
            try:
                execute_cutamp_plan(plan['steps'], client)
                for _ in range(480):
                    client._tick()
                physics.wake_up(stage_id, object_id)
                for _ in range(24):
                    client._tick()
                final = target_body.get_world_poses()[0].numpy()[0]
                forces = np.linalg.norm(fingers.get_contact_force_matrix(dt=client.dt).numpy()[:,0],axis=1)
                support = float(np.linalg.norm(target_body.get_contact_force_matrix(dt=client.dt).numpy()))
                region = plan['empty_region']
                in_region = bool(np.all(np.abs(final[:2] - np.array(region['center_xy_m'])) <= region['size_m']/2 + .01))
                lift = max(row['position'][2] - baseline[2] for row in execution_report['trace'])
                moved = float(np.linalg.norm(final[:2] - baseline[:2]))
                execution_report.update(maximum_lift_m=float(lift), final_forces_n=forces.tolist(),
                    table_support_force_n=support, inside_selected_region=in_region,
                    horizontal_displacement_m=moved, empty_region=region)
                execution_report['success'] = bool(lift > .03 and moved > .1 and in_region
                    and support > .01 and float(forces.max()) < .1
                    and abs(float(final[2]-baseline[2])) < .015
                    and execution_report['max_other_displacement_m'] <= .005)
                execution_report['physical_grasp_verified'] = bool(lift > .03)
                execution_report['physical_placement_verified'] = execution_report['success']
                status.text = ('PICK / PLACE PASSED' if execution_report['success'] else 'PICK / PLACE FAILED') + '\n' + f'Lift: {lift*1000:.1f} mm. Move: {moved*1000:.1f} mm. Table support: {support:.2f} N'
            except Exception as error:
                execution_report.update(error=str(error), physical_grasp_verified=False)
                status.text = 'CONTACT TRIAL STOPPED: ' + str(error)
                client.stop()
            finally:
                execution_report['finished'] = True
                execution_report['gemini_api_calls'] = plan.get('gemini_api_calls', 0)
                execution_report['final_position'] = target_body.get_world_poses()[0].numpy()[0].tolist()
                (args.output / 'execution.json').write_text(json.dumps(execution_report, indent=2))
                print('LIVE_CONTACT_RESULT ' + json.dumps({k:v for k,v in execution_report.items() if k not in ['trace','calls']}), flush=True)
                execution_report = None
                try:
                    current = camera.capture_request(client.get_joint_positions(), 'Post-execution preview', pause_timeline=False)
                    preview = args.output / 'final-camera.png'
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
