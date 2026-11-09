"""Render the saved lab scene through the SDK-derived Gemini RGB/depth views.

Uses geometric raycast depth, not a simulation of stereo matching or sensor noise.
No robot control, physics stepping, hardware connection or HTTP server is started.
"""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, default=HERE / "scenes/lab_table_gripper_view.usda")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 800,
                     "enable_crashreporter": False,
                     "extra_args": ["--/metricsAssembler/operationMode=0",
                                    "--/metricsAssembler/changeListenerEnabled=false",
                                    "--/log/file=" + str(args.output / "kit.log")]})
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Usd, UsdGeom
    from lab_camera import MOUNT, FACTORY, camera_specs, CV_FROM_USD, align_depth180_to_rgb180

    omni.usd.get_context().open_stage(str(args.scene.resolve()))
    stage = omni.usd.get_context().get_stage()
    factory = json.loads(FACTORY.read_text())
    specs = camera_specs(factory)
    rep.orchestrator.set_capture_on_play(False)
    products, annotators = [], {}
    for label, annotation in (("RGB180", "rgb"), ("Depth180", "distance_to_image_plane")):
        path = MOUNT + "/" + label
        assert stage.GetPrimAtPath(path).IsValid(), path
        product = rep.create.render_product(path, tuple(specs[label]["resolution_wh"]))
        products.append(product)
        annotator = rep.AnnotatorRegistry.get_annotator(annotation)
        annotator.attach(product)
        annotators[label] = annotator
    # Keep the saved robot/gripper posture frozen; render without advancing time.
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=True)
    rgb = np.asarray(annotators["RGB180"].get_data())[..., :3].copy()
    depth = np.asarray(annotators["Depth180"].get_data()).squeeze().copy()
    if rgb.shape != (720, 1280, 3) or depth.shape != (480, 848):
        raise RuntimeError(f"Unexpected rendered sizes: {rgb.shape}, {depth.shape}")
    depth[~np.isfinite(depth)] = 0
    Image.fromarray(rgb).save(args.output / "rgb180.png")
    import cv2

    visual = cv2.applyColorMap(np.clip((depth-.3)/1.7*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    visual[depth <= 0] = 0
    cv2.imwrite(str(args.output / "native-depth180.png"), visual)
    aligned = align_depth180_to_rgb180(depth, factory)
    visual_aligned = cv2.applyColorMap(np.clip((aligned-.3)/1.7*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    visual_aligned[aligned <= 0] = 0
    cv2.imwrite(str(args.output / "aligned-depth180.png"), visual_aligned)
    matrices = {}
    cache = UsdGeom.XformCache()
    for label in ("RGB180", "Depth180"):
        usd = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(MOUNT + "/" + label))).T
        matrices["world_from_" + label] = usd @ CV_FROM_USD
    np.savez_compressed(args.output / "camera-observation.npz", rgb=rgb, native_depth_m=depth, depth_m=aligned,
                        K_color=specs["RGB180"]["K"], K_depth=specs["Depth180"]["K"], **matrices)
    center = depth[depth.shape[0]*2//5:depth.shape[0]*3//5,
                   depth.shape[1]*2//5:depth.shape[1]*3//5]
    valid = center[center > 0]
    report = dict(rgb_shape=list(rgb.shape), native_depth_shape=list(depth.shape),
                  center_depth_median_m=float(np.median(valid)) if valid.size else None,
                  depth_source="ideal geometric raycast, native depth camera grid",
                  world_pose_calibrated=False, physics_advanced=False,
                  aligned_depth_shape=list(aligned.shape),
                  note="RGB/depth both rotated180. depth_m is geometrically D2C-aligned; native_depth_m retains the native grid. No physical sensor noise model.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LAB_CAMERA_RENDER_READY " + json.dumps(report), flush=True)
    for annotator in annotators.values():
        annotator.detach()
    for product in products:
        product.destroy()
finally:
    app.close()
