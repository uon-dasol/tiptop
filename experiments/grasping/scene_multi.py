"""Three-class free-rigid-body scene, built from existing source USD references."""

from pathlib import Path

import grasp_scene
import numpy as np


def decorate(stage, record):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    target = record["target_object"]
    assert target["asset_scale"] == [0.1, 0.1, 0.1]
    grasp_scene.OBJECT = Path(target["asset"])
    bounds = grasp_scene.decorate(stage, record)
    for row in record["distractors"]:
        root = UsdGeom.Xform.Define(stage, row["root_path"])
        root.AddTransformOp().Set(Gf.Matrix4d(np.asarray(row["world_from_object"]).T.tolist()))
        asset = UsdGeom.Xform.Define(stage, row["root_path"] + "/Asset")
        asset.AddScaleOp().Set(Gf.Vec3f(*row["asset_scale"]))
        asset.GetPrim().GetReferences().AddReference(row["asset"])
        prim = stage.GetPrimAtPath(row["prim_path"])
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Distractor must reuse a real rigid USD")
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    # A wide camera showing all three classes on the table.
    camera = stage.GetPrimAtPath("/World/Camera")
    UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.35, -1.45, 1.25), Gf.Vec3d(0.60, 0.0, 0.48), Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    return bounds
