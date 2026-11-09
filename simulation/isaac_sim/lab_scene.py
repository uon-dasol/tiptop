"""Reuse the saved lab environment with the existing controllable M1013 asset."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB_SCENE = HERE / "scenes/lab_table_gripper_view.usda"
LAB_HOME = HERE / "config/lab_home.json"
LAB_HARDWARE_PROFILE = ROOT / "tiptop/config/m1013_fixed_hardware.declared.json"


def load_lab_home():
    """User-selected lab arm home, in radians; does not connect to hardware."""
    profile = json.loads(LAB_HOME.read_text())
    q = np.asarray(profile["joint_positions_deg"], float)
    if profile["joint_names"] != [f"joint_{i}" for i in range(1, 7)] or q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Lab home requires joint_1 through joint_6 and six finite degree values")
    return np.deg2rad(q)


def saved_view_transform(scene_path):
    from pxr import Usd, UsdGeom

    source = Usd.Stage.Open(str(Path(scene_path).resolve()))
    for path in ("/World/Camera", "/World/LabOverview"):
        prim = source.GetPrimAtPath(path)
        if prim.IsValid():
            return UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    raise ValueError(f"No saved overview camera in {scene_path}")


def box_from_bounds(name, lower, upper, world_from_local):
    """Conservative oriented box in base_link, retaining the USD axes."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    transform = np.asarray(world_from_local, float).copy()
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    centre = transform @ np.r_[(lower + upper) / 2, 1.]
    transform[:3, :3] /= scale
    transform[:3, 3] = centre[:3]
    return dict(name=name, dims_m=((upper-lower)*scale).tolist(), base_from_obstacle=transform.tolist())


def attach_lab_environment(stage, scene_path=LAB_SCENE):
    """Reference shared geometry and derive planner boxes from those same prims."""
    from pxr import Usd, UsdGeom, UsdPhysics

    scene_path = Path(scene_path).resolve()
    source = Usd.Stage.Open(str(scene_path))
    for path in ("/World/Table", "/World/FixedStructures"):
        if not source.GetPrimAtPath(path).IsValid():
            raise ValueError(f"Lab scene lacks {path}: {scene_path}")
        stage.RemovePrim(path)
        UsdGeom.Xform.Define(stage, path).GetPrim().GetReferences().AddReference(str(scene_path), path)
    stage.RemovePrim("/World/Ground")  # Example ground would overlap the lab tabletop.
    cache = UsdGeom.XformCache()
    obstacles, table_corners = [], []
    right_bounds = None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(("/World/Table/", "/World/FixedStructures/")):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        if prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            lower, upper = np.full(3, -size/2), np.full(3, size/2)
        elif prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            lower, upper = points.min(0), points.max(0)
        else:
            continue
        if path.startswith("/World/Table/"):
            corners = np.array([[x,y,z,1] for x in (lower[0],upper[0])
                                for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
            table_corners.extend((corners @ matrix.T)[:, :3])
            if prim.GetName() == "Right":
                world = (corners @ matrix.T)[:, :3]
                right_bounds = np.stack([world.min(0), world.max(0)])
            continue
        name = "lab_" + path.removeprefix("/World/FixedStructures/").replace("/", "_")
        if prim.GetName() == "right_table_structure":
            # Four bars surround the hollow frame; leave its central opening free.
            xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
            if len(xs) != 4 or len(ys) != 4:
                raise ValueError("Expected the saved upper frame's rectangular opening")
            ranges = [([xs[0],ys[0],lower[2]], [xs[-1],ys[1],upper[2]]),
                      ([xs[0],ys[-2],lower[2]], [xs[-1],ys[-1],upper[2]]),
                      ([xs[0],ys[1],lower[2]], [xs[1],ys[-2],upper[2]]),
                      ([xs[-2],ys[1],lower[2]], [xs[-1],ys[-2],upper[2]])]
            for i, (lo, hi) in enumerate(ranges):
                obstacles.append(box_from_bounds(name+f"_{i}", lo, hi, matrix))
        else:
            obstacles.append(box_from_bounds(name, lower, upper, matrix))
    corners = np.asarray(table_corners)
    bounds = np.stack([corners.min(0), corners.max(0)])
    if right_bounds is None:
        raise ValueError("Lab scene needs the Right table section")
    work_area = add_work_area_tape(stage, right_bounds)
    # Saved view/scene drive targets are historical snapshots, not the lab home.
    q = load_lab_home()
    result = dict(fixed_obstacles=obstacles, table_bounds=bounds, q_capture=q.tolist(),
                  pick_place_bounds_xy_m=work_area.tolist(), source=str(scene_path))
    return result


def add_work_area_tape(stage, right_bounds, width=.025):
    """Visual tape only; restrict object endpoints without blocking arm transit."""
    from pxr import Gf, UsdGeom

    low, high = np.asarray(right_bounds)
    middle = (low+high)/2
    spans = high-low
    strips = [([low[0]+width/2,middle[1]], [width,spans[1]]),
              ([high[0]-width/2,middle[1]], [width,spans[1]]),
              ([middle[0],low[1]+width/2], [spans[0]-2*width,width]),
              ([middle[0],high[1]-width/2], [spans[0]-2*width,width])]
    for index, (xy,dims) in enumerate(strips):
        cube = UsdGeom.Cube.Define(stage, f"/World/RightWorkAreaTape/Edge_{index}")
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*xy, high[2]+.0002))
        cube.AddScaleOp().Set(Gf.Vec3d(*dims, .0004))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.85,.03,.04)])
    return np.stack([low[:2]+width, high[:2]-width])


def lab_preparation(environment, camera, *, hardware_profile=LAB_HARDWARE_PROFILE):
    """Build the lab configuration directly from its USD and physical profile."""
    from copy import deepcopy
    from tiptop.doosan.hardware import load_hardware_profile, hardware_profile_fingerprint
    from tiptop.doosan.virtual_camera import load_camera_preparation

    hardware = load_hardware_profile(hardware_profile)
    calibration = hardware["_camera_calibration"]
    if hardware["camera_profile"]["mount"] != "fixed":
        raise ValueError("Lab synchronization requires a fixed-camera hardware profile")
    optical_pose = np.asarray(calibration["parent_from_camera"], float)
    if calibration.get("image_rotation_deg", 0) == 180:
        optical_pose = optical_pose @ np.diag([-1., -1., 1., 1.])
    if not np.allclose(camera.world_from_cam(), optical_pose, atol=1e-6, rtol=0):
        raise ValueError("Lab optical camera pose differs from the shared hardware profile")

    profile = deepcopy(hardware["camera_profile"])
    profile.update(
        model_name="gemini_336l_lab_rgbd", sensor_model_name="isaac_geometric_rgbd",
        resolution_wh=[camera.width, camera.height],
        baseline_m=abs(float(camera.factory["ir_right_from_ir_left"][0][3])),
        horizontal_fov_deg=float(np.rad2deg(2*np.arctan(camera.width/(2*camera.intrinsics[0,0])))),
        intrinsics=camera.intrinsics.tolist(),
        distortion_opencv=camera.specs["RGB180"]["D"].tolist(),
        optics="SDK RGB180 K/D; undistorted RGB and aligned ideal geometric depth; no stereo noise model",
        collision_geometry_in_fixed_obstacles=False,
        fixed_base_world_contact=hardware["workspace"].get("fixed_base_world_contact", False),
    )
    # The USD retains detailed camera colliders. Planning uses the same padded
    # body/bracket envelopes as the physical profile, with its body-frame rig.
    obstacles = [deepcopy(row) for row in environment["fixed_obstacles"]
                 if not row["name"].startswith("lab_gemini336l_mount_")]
    expected = {row["name"]: row for row in hardware["workspace"]["fixed_obstacles"]
                if row["name"] != "full_table"}
    actual = {row["name"]: row for row in obstacles}
    if actual.keys() != expected.keys() or any(
        not np.allclose(actual[name][key], expected[name][key], atol=1e-6, rtol=0)
        for name in expected for key in ("dims_m", "base_from_obstacle")
    ):
        raise ValueError("Lab structure geometry differs from the shared hardware profile")
    table = next(row for row in hardware["workspace"]["fixed_obstacles"] if row["name"] == "full_table")
    table_pose = np.asarray(table["base_from_obstacle"], float)
    table_bounds = table_pose[:3, 3] + np.array([-.5, .5])[:, None] * np.asarray(table["dims_m"])
    if not np.allclose(table_pose[:3, :3], np.eye(3), atol=1e-6, rtol=0) or not np.allclose(
        environment["table_bounds"], table_bounds, atol=1e-6, rtol=0
    ):
        raise ValueError("Lab table geometry differs from the shared hardware profile")
    if not np.allclose(environment["pick_place_bounds_xy_m"], hardware["workspace"]["pick_place_bounds_xy_m"],
                       atol=1e-6, rtol=0):
        raise ValueError("Lab work area differs from the shared hardware profile")
    return load_camera_preparation(dict(
        profile=profile, q_capture=environment["q_capture"], fixed_obstacles=obstacles,
        lab_scene=environment["source"], pick_place_bounds_xy_m=environment["pick_place_bounds_xy_m"],
        db_grasp_from_tcp=hardware["tool"]["db_grasp_from_tcp"],
        lab_hardware_profile_sha256=hardware_profile_fingerprint(hardware),
    ))
