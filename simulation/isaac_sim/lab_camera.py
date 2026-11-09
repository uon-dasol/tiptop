"""Author the lab's Gemini 336L views from the connected device's SDK calibration.

Run with Isaac's python.sh. This does not start Isaac, open hardware or move robots.
The body placement is the user's approximate layout; SDK intrinsics/extrinsics
only describe the sensors relative to each other, not the robot base.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACTORY = HERE / "config/gemini_336l_factory.json"
MOUNT = "/World/FixedStructures/gemini336l_mount"
OPTICS = HERE / "scenes/lab_table_camera_optics.usda"
CV_FROM_USD = np.diag([1., -1., -1., 1.])
ROTATE_180 = np.diag([-1., -1., 1., 1.])


def rotated_intrinsics(stream):
    """Positive focal lengths in a 180-degree rotated image/optical frame."""
    width, height = stream["resolution_wh"]
    k = np.array(stream["K"], float)
    k[0, 2] = width - 1 - k[0, 2]
    k[1, 2] = height - 1 - k[1, 2]
    d = np.array(stream["distortion_opencv"], float)
    d[2:4] *= -1  # Tangential terms change sign; radial terms do not.
    return k, d


def camera_specs(factory):
    # Body +X is defined as viewed from its front. The SDK left IR is therefore
    # at +X; optical image +X points in body -X. Body -Z looks at the table.
    baseline = -float(factory["ir_right_from_ir_left"][0][3])
    body_from_left = np.diag([-1., 1., -1., 1.])
    body_from_left[:3, 3] = [baseline / 2, 0, .00478]
    # 4.78 mm: 336L depth zero behind front glass, datasheet section4.9.
    poses = {
        "color": body_from_left @ np.linalg.inv(factory["color_from_ir_left"]),
        "depth": body_from_left,
        "ir_left": body_from_left,
        "ir_right": body_from_left @ np.linalg.inv(factory["ir_right_from_ir_left"]),
    }
    specs = {}
    for label, key, rotate in [
        ("RGBRaw", "color", False), ("RGB180", "color", True),
        ("DepthRaw", "depth", False), ("Depth180", "depth", True),
        ("IRLeft", "ir_left", False), ("IRRight", "ir_right", False),
    ]:
        stream = factory["streams"][key]
        k, d = rotated_intrinsics(stream) if rotate else (
            np.array(stream["K"], float), np.array(stream["distortion_opencv"], float)
        )
        pose = poses[key] @ (ROTATE_180 if rotate else np.eye(4))
        specs[label] = dict(stream=key, resolution_wh=stream["resolution_wh"], K=k, D=d,
                            body_from_optical=pose, body_from_usd=pose @ CV_FROM_USD,
                            rotated_180=rotate)
    return specs


def align_depth180_to_rgb180(depth, factory):
    """Geometric D2C using both rotated optical frames, K/D and SDK extrinsics.

    Four-neighbour z-buffer splatting preserves unknown pixels as zero. This
    models projection/occlusion, not the device's proprietary hole filling.
    """
    import cv2

    specs = camera_specs(factory)
    native, color = specs["Depth180"], specs["RGB180"]
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    width, height = color["resolution_wh"]
    result = np.full((height, width), np.inf, dtype=np.float32)
    if not valid.any():
        return np.zeros((height, width), dtype=np.float32)
    k = native["K"]
    z = depth[valid]
    points = np.column_stack(((u[valid]-k[0, 2])/k[0, 0]*z,
                              (v[valid]-k[1, 2])/k[1, 1]*z, z))
    transform = np.linalg.inv(color["body_from_optical"]) @ native["body_from_optical"]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points = points[points[:, 2] > 0]
    uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), color["K"], color["D"])
    pixels = np.floor(uv.reshape(-1, 2)).astype(int)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = pixels[:, 0]+dx, pixels[:, 1]+dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.minimum.at(result, (y[inside], x[inside]), points[inside, 2])
    result[~np.isfinite(result)] = 0
    return result


def author_camera(stage, path, spec):
    from pxr import Gf, Sdf, UsdGeom, Vt

    camera = UsdGeom.Camera.Define(stage, path)
    # Hide the editor's camera model/frustum; this camera remains a render viewpoint.
    camera.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    camera.AddTransformOp().Set(Gf.Matrix4d(spec["body_from_usd"].T.tolist()))
    width, height = spec["resolution_wh"]
    k, d = spec["K"], spec["D"]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    # Conventional USD fallback, while the RTX OpenCV schema carries exact K/D.
    focal = 12.
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(width * focal / fx)
    camera.CreateVerticalApertureAttr(height * focal / fy)
    camera.CreateHorizontalApertureOffsetAttr((width / 2 - cx) * focal / fx)
    camera.CreateVerticalApertureOffsetAttr((cy - height / 2) * focal / fy)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
    camera.CreateFocusDistanceAttr(1.02)
    camera.CreateFStopAttr(0.)
    prim = camera.GetPrim()
    schemas = Sdf.TokenListOp()
    schemas.prependedItems = ["OmniLensDistortionOpenCvPinholeAPI"]
    prim.SetMetadata("apiSchemas", schemas)
    prim.SetMetadata("documentation", "SDK factory optics; estimated lab mounting. "
                     "Depth cameras define projection geometry, not physical sensor noise or a depth publisher.")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token, custom=False).Set("opencvPinhole")
    ns = "omni:lensdistortion:opencvPinhole:"
    values = dict(fx=fx, fy=fy, cx=cx, cy=cy)
    values.update(dict(zip(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"), d)))
    for name, value in values.items():
        prim.CreateAttribute(ns + name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
    prim.CreateAttribute(ns + "imageSize", Sdf.ValueTypeNames.Int2, custom=False).Set(Gf.Vec2i(width, height))
    prim.SetCustomDataByKey("gemini:resolution", Vt.IntArray([width, height]))
    prim.SetCustomDataByKey("gemini:rotated180", spec["rotated_180"])
    return camera


def write_optics(factory_path=FACTORY, output=OPTICS):
    from pxr import Usd, UsdGeom

    factory = json.loads(Path(factory_path).read_text())
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.OverridePrim(MOUNT)
    for label, spec in camera_specs(factory).items():
        author_camera(stage, MOUNT + "/" + label, spec)
    stage.GetRootLayer().documentation = (
        "Generated by simulation/isaac_sim/lab_camera.py from Gemini336L "
        + factory["serial"] + " factory optics. Body mount belongs to lab_table_camera.usda. "
        "RGB180 and Depth180 share the requested 180-degree image rotation. "
        "Native depth and colour have different pixel grids; alignment is a separate operation."
    )
    stage.GetRootLayer().Export(str(output))
    return stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", type=Path, default=FACTORY)
    parser.add_argument("--output", type=Path, default=OPTICS)
    args = parser.parse_args()
    write_optics(args.factory, args.output)
    print("Wrote Gemini336L camera optics:", args.output)
