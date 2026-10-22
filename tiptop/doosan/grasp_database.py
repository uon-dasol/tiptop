"""Collected UON grasp candidates transformed by a sensor-derived CAD pose.

This supplies candidates, not a robot trajectory or a physical success claim.
All transforms use column vectors and metres. No Panda/M2T2 TCP offset applies.
"""

import json
from pathlib import Path

import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform


def load_database_grasps(
    path, *, class_id, world_from_cad, pose_source, world_from_planner_object, grasp_from_tcp, cad_from_db=None
):
    if pose_source != "rgbd_pose_estimator":
        raise ValueError("The control path requires a sensor-estimated CAD pose")
    world_cad = _rigid_transform(world_from_cad, "world_from_cad")
    world_planner = _rigid_transform(world_from_planner_object, "world_from_planner_object")
    cad_db = _rigid_transform(np.eye(4) if cad_from_db is None else cad_from_db, "cad_from_db")
    grasp_tcp = _rigid_transform(grasp_from_tcp, "grasp_from_tcp")
    document = json.loads(Path(path).read_text())
    metadata = document["metadata"]
    if (
        metadata["object"] != class_id
        or metadata["coordinate_frame"] != "object_zero_pose"
        or metadata["gripper_model"] != "UON_3finger_gripper"
    ):
        raise ValueError("DB object, coordinates or gripper do not match")
    rows = document["data"]
    if not rows:
        raise ValueError("No collected grasp candidates")
    db_grasps = np.stack([_rigid_transform(row["grasp_mat"], "DB grasp") for row in rows])
    scores = np.asarray([row["score"] for row in rows], np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Invalid DB ranking score")
    order = np.argsort(-scores, kind="stable")
    world_tcp = world_cad @ cad_db @ db_grasps[order] @ grasp_tcp
    return dict(
        world_from_tcp=world_tcp,
        grasps_obj=np.linalg.inv(world_planner) @ world_tcp,
        scores=scores[order],
        candidate_indices=order,
        source_scene_ids=[rows[index]["scene_id"] for index in order],
        source_pregrasp_indices=[rows[index]["source_pregrasp_index"] for index in order],
        pose_source=pose_source,
        real_tcp_calibrated=False,
        collision_filtered=False,
        physical_grasp_verified=False,
    )
