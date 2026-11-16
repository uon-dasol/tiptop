"""Collected UON grasp candidates transformed by a sensor-derived CAD pose.

This supplies candidates, not a robot trajectory or a physical success claim.
All transforms use column vectors and metres. No Panda/M2T2 TCP offset applies.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform


def _file_key(path):
    stat = Path(path).stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _document(path, mtime_ns, size):
    return json.loads(Path(path).read_text())


def database_grasp_matrices(document):
    """Return object-frame grasps at the recorded target origin.

    The single-object format already stores this origin in grasp_mat. V2/UON
    stores a base pose instead: target_points lies 237 mm along its local +Z.
    Use that recorded point after validating the offset; do not translate in
    world Z or change the source document, rotations, scores or TCP calibration.
    """
    rows = document["data"]
    grasps = np.stack([_rigid_transform(row["grasp_mat"], "DB grasp") for row in rows])
    metadata = document.get("metadata", {})
    if metadata.get("schema") != "dataset_v2_object_grasp_compact":
        return grasps
    if (metadata.get("schema_version") != 1 or metadata.get("gripper_model") != "UON_3finger_gripper"
            or metadata.get("coordinate_frame") != "object_zero_pose"):
        raise ValueError("Unsupported V2 grasp schema version, gripper or coordinate frame")
    try:
        targets = np.asarray([row["target_points"] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V2 grasp correction requires finite target_points") from error
    if targets.shape != (len(rows), 3) or not np.isfinite(targets).all():
        raise ValueError("V2 grasp correction requires finite [N, 3] target_points")
    local_offsets = np.einsum("nji,nj->ni", grasps[:, :3, :3], targets - grasps[:, :3, 3])
    # JSON coordinates are rounded; observed errors are below 2 micrometres.
    if not np.allclose(local_offsets, [0.0, 0.0, 0.237], atol=1e-5, rtol=0):
        raise ValueError("V2 target_points must be 237 mm along the grasp base's local +Z")
    grasps[:, :3, 3] = targets
    return grasps


@lru_cache(maxsize=4)
def _ranked_candidates(path, mtime_ns, size):
    document = _document(path, mtime_ns, size)
    rows = document["data"]
    db_grasps = database_grasp_matrices(document)
    scores = np.asarray([row["score"] for row in rows], np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Invalid DB ranking score")
    order = np.argsort(-scores, kind="stable")
    return (
        db_grasps[order],
        scores[order],
        order,
        [rows[index]["scene_id"] for index in order],
        [rows[index].get("source_pregrasp_index") for index in order],
        [rows[index].get("source_output_index") for index in order],
    )


def ensure_database_cache_capacity(objects):
    """Grow both caches before warming a sequence; do not resize during planning."""
    global _document, _ranked_candidates
    if not isinstance(objects, int) or isinstance(objects, bool) or objects <= 0:
        raise ValueError("DB cache capacity must be a positive integer")
    if objects > _document.cache_parameters()["maxsize"]:
        _document = lru_cache(maxsize=objects)(_document.__wrapped__)
        _ranked_candidates = lru_cache(maxsize=objects)(_ranked_candidates.__wrapped__)
    return _document.cache_parameters()["maxsize"]


def _validate_document(document, class_id):
    metadata = document["metadata"]
    if (metadata["object"] != class_id or metadata["coordinate_frame"] != "object_zero_pose"
            or metadata["gripper_model"] != "UON_3finger_gripper"):
        raise ValueError("DB object, coordinates or gripper do not match")
    if not document["data"]:
        raise ValueError("No collected grasp candidates")


def warm_database_grasps(path, *, class_id=None):
    """Read and rank the DB ahead of its first use; later loads of the same file reuse the result."""
    key = _file_key(path)
    if class_id is not None:
        _validate_document(_document(*key), class_id)
    _ranked_candidates(*key)


def load_database_grasps(
    path, *, class_id, world_from_cad, pose_source, world_from_planner_object, grasp_from_tcp, cad_from_db=None
):
    if pose_source != "rgbd_pose_estimator":
        raise ValueError("The control path requires a sensor-estimated CAD pose")
    world_cad = _rigid_transform(world_from_cad, "world_from_cad")
    world_planner = _rigid_transform(world_from_planner_object, "world_from_planner_object")
    cad_db = _rigid_transform(np.eye(4) if cad_from_db is None else cad_from_db, "cad_from_db")
    grasp_tcp = _rigid_transform(grasp_from_tcp, "grasp_from_tcp")
    key = _file_key(path)
    document = _document(*key)
    _validate_document(document, class_id)
    db_grasps, scores, order, scene_ids, pregrasp_indices, output_indices = _ranked_candidates(*key)
    world_tcp = world_cad @ cad_db @ db_grasps @ grasp_tcp
    return dict(
        world_from_tcp=world_tcp,
        grasps_obj=np.linalg.inv(world_planner) @ world_tcp,
        scores=scores.copy(),
        candidate_indices=order.copy(),
        source_scene_ids=list(scene_ids),
        source_pregrasp_indices=list(pregrasp_indices),
        source_output_indices=list(output_indices),
        pose_source=pose_source,
        real_tcp_calibrated=False,
        collision_filtered=False,
        physical_grasp_verified=False,
    )
