"""On-demand RGB stereo camera for the live Isaac HTTP observation backend.

This is an ideal rectified colour stereo sensor with a declared body envelope.
It uses a 336L-inspired body envelope, not the physical device's IR stereo and
separate colour optics. It does not emulate electronics, noise or RGB/IR offsets.
The consumer computes FoundationStereo depth before common TiPToP perception;
this observation payload is not the TiPToP websocket RGB+depth request.
Importing this module does not launch Isaac; construct it after SimulationApp.
"""

import time

import numpy as np


class IsaacStereoCamera:
    def __init__(self, stage, profile, *, parent_path, root_name="PerceptionCamera"):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, UsdPhysics

        self.stage, self.profile = stage, profile
        self.serial = "isaac/" + profile["model_name"]
        self._rep = rep
        self._baseline = float(profile["baseline_m"])
        if self._baseline <= 0 or not stage.GetPrimAtPath(parent_path).IsValid():
            raise ValueError("Expected valid attachment prim and positive stereo baseline")
        self.root = parent_path + "/" + root_name
        rig = UsdGeom.Xform.Define(stage, self.root)
        rig.AddTransformOp().Set(Gf.Matrix4d(np.asarray(profile["parent_from_rig"], float).T.tolist()))
        body_dims = np.asarray(profile["body_dims_m"], float)
        for label, dims, centre in [
            ("Body", body_dims, [0.0, 0.0, -body_dims[2] / 2]),
            ("Bracket", profile["bracket_dims_m"], profile["rig_from_bracket_xyz_m"]),
        ]:
            box = UsdGeom.Cube.Define(stage, self.root + "/" + label)
            box.CreateSizeAttr(1.0)
            box.AddTranslateOp().Set(Gf.Vec3d(*centre))
            box.AddScaleOp().Set(Gf.Vec3d(*dims))
            box.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.17, 0.2)])
            # A wrist attachment inherits the existing wrist rigid body. A fixed
            # attachment is a static collider. No extra independent rigid body.
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        self.width, self.height = profile["resolution_wh"]
        horizontal = 24.0
        focal = horizontal / (2 * np.tan(np.deg2rad(profile["horizontal_fov_deg"]) / 2))
        fx = self.width * focal / horizontal
        self.intrinsics = np.array([[fx, 0.0, self.width / 2], [0.0, fx, self.height / 2], [0.0, 0.0, 1.0]])
        self.paths, self.products, self.annotators = [], [], []
        for label, offset in [("Left", -self._baseline / 2), ("Right", self._baseline / 2)]:
            path = self.root + "/" + label
            camera = UsdGeom.Camera.Define(stage, path)
            optical_from_usd = np.diag([1.0, -1.0, -1.0, 1.0])
            optical_from_usd[0, 3] = offset
            camera.AddTransformOp().Set(Gf.Matrix4d(optical_from_usd.T.tolist()))
            camera.CreateFocalLengthAttr(float(focal))
            camera.CreateHorizontalApertureAttr(horizontal)
            camera.CreateVerticalApertureAttr(horizontal * self.height / self.width)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.17, 20.0))
            product = rep.create.render_product(path, resolution=(self.width, self.height))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(product)
            self.paths.append(path)
            self.products.append(product)
            self.annotators.append(annotator)
        rep.orchestrator.set_capture_on_play(False)

    def get_rectified_stereo_baseline(self):
        return self._baseline

    def capture_request(self, q_init, task_instruction, *, pause_timeline=True):
        """Render THIS stage state; no old frame, object pose or path is loaded."""
        from pxr import Usd, UsdGeom

        q = np.asarray(q_init, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected current M1013 joint feedback")
        self._rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=pause_timeline)
        left, right = [np.asarray(annot.get_data())[..., :3].copy() for annot in self.annotators]
        if left.shape != (self.height, self.width, 3) or right.shape != left.shape:
            raise RuntimeError("Camera render is not ready")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_from_usd = np.asarray(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.paths[0]))).T
        world_from_cam = world_from_usd @ np.diag([1.0, -1.0, -1.0, 1.0])
        return dict(
            rgb=left,
            stereo_right_rgb=right,
            stereo_baseline_m=self._baseline,
            intrinsics=self.intrinsics.copy(),
            world_from_cam=world_from_cam,
            q_init=q.copy(),
            task=task_instruction,
            camera_model=self.serial,
            timestamp=time.time(),
        )

    def close(self):
        for annotator in self.annotators:
            annotator.detach()
        for product in self.products:
            product.destroy()


class IsaacLabCamera:
    """Saved 336L optics adapted to the existing RGB-D decoder.

    Depth is ideal renderer geometry, not FoundationStereo output.
    RGB and aligned depth are undistorted together for downstream pinhole K.
    """

    def __init__(self, stage):
        import json
        import cv2
        import omni.replicator.core as rep
        from lab_camera import FACTORY, MOUNT, camera_specs

        self.stage, self._rep = stage, rep
        self.factory = json.loads(FACTORY.read_text())
        self.specs = camera_specs(self.factory)
        self.root = MOUNT
        self.width, self.height = self.specs["RGB180"]["resolution_wh"]
        self.intrinsics = self.specs["RGB180"]["K"].copy()
        self.serial = "isaac/gemini_336l_lab_rgbd"
        self.products, self.annotators = [], []
        self._attach_render_products()
        self._maps = cv2.initUndistortRectifyMap(
            self.intrinsics, self.specs["RGB180"]["D"], None, self.intrinsics,
            (self.width, self.height), cv2.CV_32FC1)
        rep.orchestrator.set_capture_on_play(False)

    def _attach_render_products(self):
        for label, kind in (("RGB180", "rgb"), ("Depth180", "distance_to_image_plane")):
            path = self.root + "/" + label
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise ValueError(f"Missing lab camera: {path}")
            product = self._rep.create.render_product(path, tuple(self.specs[label]["resolution_wh"]))
            annotator = self._rep.AnnotatorRegistry.get_annotator(kind)
            annotator.attach(product)
            self.products.append(product)
            self.annotators.append(annotator)

    def _ensure_render_products(self):
        from omni.replicator.core.scripts.annotators import AnnotatorRegistryError

        try:
            for annotator in self.annotators:
                annotator.get_node()
        except AnnotatorRegistryError:
            # File > Save As can clear SyntheticData's graph while leaving the
            # Python camera object alive. Recreate handles on the current stage.
            import omni.usd

            for product in self.products:
                product.destroy()
            self.stage = omni.usd.get_context().get_stage()
            self.products, self.annotators = [], []
            self._attach_render_products()
            print("LAB_CAMERA_REATTACHED after stage save/reload", flush=True)

    def world_from_cam(self):
        from pxr import UsdGeom
        from lab_camera import CV_FROM_USD

        cache = UsdGeom.XformCache()
        return np.asarray(cache.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.root + "/RGB180"))).T @ CV_FROM_USD

    def preview_rgb(self, *, render=False):
        """Read the latest rendered RGB without a task capture or physics step."""
        import cv2

        self._ensure_render_products()
        if render:
            self._rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=False)
        rgba = np.asarray(self.annotators[0].get_data())
        if rgba.shape != (self.height, self.width, 4):
            return None
        return cv2.remap(rgba[..., :3], *self._maps, cv2.INTER_LINEAR)

    def capture_request(self, q_init, task_instruction, *, pause_timeline=True):
        import cv2
        from lab_camera import align_depth180_to_rgb180

        self._ensure_render_products()
        self._rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=pause_timeline)
        rgb = np.asarray(self.annotators[0].get_data())[..., :3].copy()
        depth = np.asarray(self.annotators[1].get_data()).squeeze().copy()
        if rgb.shape != (self.height, self.width, 3) or depth.shape != (480,848):
            raise RuntimeError("Lab RGB/depth render is not ready")
        depth[~np.isfinite(depth)] = 0
        aligned = align_depth180_to_rgb180(depth, self.factory)
        rgb = cv2.remap(rgb, *self._maps, cv2.INTER_LINEAR)
        aligned = cv2.remap(aligned, *self._maps, cv2.INTER_NEAREST)
        return dict(rgb=rgb, depth=aligned.astype(np.float32), intrinsics=self.intrinsics.copy(),
                    world_from_cam=self.world_from_cam(), q_init=np.asarray(q_init, float).copy(),
                    task=task_instruction, camera_model=self.serial, timestamp=time.time(),
                    depth_source="isaac_geometric", world_frame="base_link")

    close = IsaacStereoCamera.close
