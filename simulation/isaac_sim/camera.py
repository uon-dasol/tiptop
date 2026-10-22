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
