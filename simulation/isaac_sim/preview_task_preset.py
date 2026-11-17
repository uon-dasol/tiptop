"""Display the fixed-box preset in the shared lab, without physics or robot IO.

Run using Isaac Sim's python.sh --no-ros-env. The arm uses the saved lab home,
not live hardware feedback. Geometry and optics are shared with --lab runtime.
"""

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args, _ = parser.parse_known_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "error.txt").unlink(missing_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": args.headless, "width": 1600, "height": 1000,
        "window_width": 1600, "window_height": 1000,
        "enable_crashreporter": False,
        "extra_args": ["--/metricsAssembler/operationMode=0",
                       "--/metricsAssembler/changeListenerEnabled=false",
                       "--/log/file=" + str(output / "kit.log")],
    })
    camera = None
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport
        from PIL import Image
        from pxr import Gf, UsdGeom
        from camera import IsaacLabCamera
        from lab_scene import LAB_HARDWARE_PROFILE, attach_lab_environment, validate_lab_geometry, load_lab_home
        from m1013_uon import ARM, GRIPPER, build_stage, model_fk
        from tiptop.task_presets import TaskSettings

        settings = TaskSettings(preset="home_drop")
        stage, _, robot_report = build_stage(output, align_spinner=True)
        environment = attach_lab_environment(stage)
        camera = IsaacLabCamera(stage)
        hardware = json.loads(LAB_HARDWARE_PROFILE.read_text())
        calibration_path = LAB_HARDWARE_PROFILE.parent / hardware["camera_calibration"]
        hardware["_camera_calibration"] = json.loads(calibration_path.read_text())
        validate_lab_geometry(environment, camera, hardware)
        omni.timeline.get_timeline_interface().stop()
        # This virtual exclusion volume is not a physical wall. Hide its visual
        # solid so it cannot occlude the robot; retain the saved collider.
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/FixedStructures/left_table_keepout/Collision")).MakeInvisible()

        # Author the shared URDF home directly: no articulation controller or
        # physics tick is needed for a stationary location preview.
        q = load_lab_home()
        positions = dict(zip((f"joint_{i}" for i in range(1, 7)), q))
        cache = UsdGeom.XformCache()
        old_wrist = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ARM + "/link_6"))
        new_wrist = model_fk(positions)
        gripper = stage.GetPrimAtPath("/World/Gripper")
        gripper_home = cache.GetLocalToWorldTransform(gripper) * old_wrist.GetInverse() * new_wrist
        tool = stage.GetPrimAtPath(ARM + "/tool0")
        tool_home = (cache.GetLocalToWorldTransform(tool) * old_wrist.GetInverse() * new_wrist
                     if tool.IsValid() else None)
        for index in range(1, 7):
            prim = stage.GetPrimAtPath(ARM + f"/link_{index}")
            parent = cache.GetLocalToWorldTransform(prim.GetParent())
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(model_fk(positions, prim.GetName()) * parent.GetInverse())
        UsdGeom.Xformable(gripper).MakeMatrixXform().Set(gripper_home)
        if tool_home is not None:
            UsdGeom.Xformable(tool).MakeMatrixXform().Set(tool_home * cache.GetLocalToWorldTransform(tool.GetParent()).GetInverse())

        box = settings.box_spec()
        xyz = box["pose"][:3]
        dims = box["dims"]
        box_root = UsdGeom.Cube.Define(stage, "/World/PresetFixedBox")
        box_root.GetPrim().SetDisplayName("Fixed box - shared planning envelope")
        box_root.CreateSizeAttr(1.)
        box_root.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        box_root.AddScaleOp().Set(Gf.Vec3d(*dims))
        box_root.CreateDisplayColorAttr([Gf.Vec3f(.15, .50, .95)])
        top = settings.box_bottom_z_m + settings.box_size_z_m
        release = top + settings.release_height_m

        def lines(path, points, color, width=.003):
            curve = UsdGeom.BasisCurves.Define(stage, path)
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        x, y = xyz[:2]
        dx, dy = dims[0]/2, dims[1]/2
        lines("/World/ReleaseHeightOutline", [
            (x-dx, y-dy, release), (x+dx, y-dy, release),
            (x+dx, y+dy, release), (x-dx, y+dy, release),
            (x-dx, y-dy, release)], (1., .55, .02))
        if release > top:
            lines("/World/ReleaseHeightGauge", [(x+dx+.025, y, top), (x+dx+.025, y, release)], (1., .55, .02))

        views = {
            "Overview": ((1.8, -1.8, 1.65), (0., .33, .45), (0., 0., 1.)),
            "Box close-up": ((.85, -.42, .80), (0., .36, .13), (0., 0., 1.)),
            "Top": ((x, y, 2.6), (x, y, 0.), (0., 1., 0.)),
        }
        paths = {}
        for label, (eye, target, up) in views.items():
            path = "/World/Preview" + label.replace(" ", "").replace("-", "")
            view = UsdGeom.Camera.Define(stage, path)
            view.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse())
            view.CreateFocalLengthAttr(24.)
            view.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
            paths[label] = path
        paths["Gemini RGB"] = camera.root + "/RGB180"
        viewport = get_active_viewport()
        viewport.camera_path = paths["Overview"]
        panel = ui.Window("TiPToP - fixed box location (preview only)", width=420, height=280)
        with panel.frame:
            with ui.VStack(spacing=8):
                ui.Label("M1013 + UON gripper / shared lab environment", height=24)
                ui.Label(f"Blue: fixed box {dims[0]*100:g} x {dims[1]*100:g} x {dims[2]*100:g} cm\n"
                         f"Center XY: ({x:.3f}, {y:.3f}) m in base_link\n"
                         f"Bottom Z: {settings.box_bottom_z_m:.3f} m / top Z: {top:.3f} m\n"
                         f"Orange: object bottom release Z = {release:.3f} m\n"
                         "Saved lab home; no live robot connection.", height=130, word_wrap=True)
                with ui.HStack(height=28, spacing=4):
                    for label, path in paths.items():
                        ui.Button(label, clicked_fn=lambda p=path: setattr(viewport, "camera_path", p))
        panel.position_x, panel.position_y = 1160, 60

        # Export a new layer with the original asset references intact.
        stage.GetRootLayer().Export(str(output / "fixed-box.usda"))
        rep.orchestrator.set_capture_on_play(False)
        for label in ("Overview", "Box close-up", "Top"):
            product = rep.create.render_product(paths[label], (1600, 1000))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(product)
            for _ in range(6):
                rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=True)
            rgb = np.asarray(annotator.get_data())[..., :3].copy()
            Image.fromarray(rgb).save(output / (label.lower().replace(" ", "-") + ".png"))
            annotator.detach()
            product.destroy()
        rgb = camera.preview_rgb(render=True)
        if rgb is not None:
            Image.fromarray(rgb).save(output / "gemini-rgb.png")
        cache = UsdGeom.XformCache()
        tcp = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER + "/body/tcp"))).T
        expected_tcp = np.asarray(model_fk(positions, "gripper_tcp")).T
        if not np.allclose(tcp, expected_tcp, atol=1e-6):
            raise ValueError("Preview TCP differs from the shared planning URDF at lab home")
        actual_bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(box_root.GetPrim()).ComputeAlignedRange()
        report = dict(box=box, box_bounds_m=[list(actual_bounds.GetMin()), list(actual_bounds.GetMax())],
                      visual_box_open_top=False,
                      release_bottom_z_m=release, saved_home_rad=q.tolist(), tcp_base_xyz_m=tcp[:3, 3].tolist(),
                      lab_scene=environment["source"], shared_lab_profile_validated=True,
                      hardware_profile=str(LAB_HARDWARE_PROFILE),
                      robot_assets=robot_report, physics_advanced=False, hardware_connected=False)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        viewport.camera_path = paths["Overview"]
        print("FIXED_BOX_PREVIEW_READY " + str(output), flush=True)
        while not args.headless and app.is_running():
            app.update()
            time.sleep(1/60)
    except KeyboardInterrupt:
        pass
    except BaseException:
        import traceback

        error = traceback.format_exc()
        (output / "error.txt").write_text(error)
        print(error, flush=True)
        raise
    finally:
        if camera is not None:
            camera.close()
        app.close()


if __name__ == "__main__":
    main()
