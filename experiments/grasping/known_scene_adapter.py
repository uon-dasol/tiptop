"""Isolated prototype for known CAD/DB -> existing cuTAMP grasp dictionaries.

No robot/HTTP/Isaac imports and no file writes. This is a review candidate only.
Transforms use metres and column vectors, T_A_B maps B into A.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform


def rigid(value, name="transform"):
    return _rigid_transform(value, name)


@dataclass(frozen=True)
class Instance:
    instance_id: str
    asset_key: str
    world_from_cad: np.ndarray
    pose_source: str
    frame_id: str = "world"
    units: str = "m"

    def __post_init__(self):
        if not self.instance_id or not self.asset_key or not self.pose_source:
            raise ValueError("Instance identity, asset identity and pose provenance are required")
        if self.frame_id != "world" or self.units != "m":
            raise ValueError("Normalize pose to world metres before creating an Instance")
        t = rigid(self.world_from_cad, "world_from_cad")
        t.flags.writeable = False
        object.__setattr__(self, "world_from_cad", t)


def resolve_instance(instances, *, instance_id=None, asset_key=None):
    """Asset identity is insufficient when several instances share that asset."""
    ids = [x.instance_id for x in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scene instance IDs")
    if bool(instance_id) == bool(asset_key):
        raise ValueError("Supply exactly one grounding key")
    matches = [x for x in instances if (x.instance_id == instance_id if instance_id else x.asset_key == asset_key)]
    if len(matches) != 1:
        raise ValueError(f"Grounding must select exactly one instance; matched {len(matches)}")
    return matches[0]


def db_to_planner(rows, instance, world_from_planner_object, *, cad_from_db=None, grasp_from_tcp=None):
    """Return CPU arrays named as expected by tiptop.planning.run_planning.

    The caller filters against the full scene, then explicitly converts arrays
    via tensor_args.to_device. Scores remain ranking values, not probabilities.
    """
    planner = rigid(world_from_planner_object, "world_from_planner_object")
    cad_db = rigid(np.eye(4) if cad_from_db is None else cad_from_db, "cad_from_db")
    grasp_tcp = rigid(np.eye(4) if grasp_from_tcp is None else grasp_from_tcp, "grasp_from_tcp")
    if not rows:
        raise ValueError("No candidates supplied")
    world_tcp = []
    metadata = []
    scores = []
    seen = set()
    for row in rows:
        cid = int(row["candidate_index"])
        if cid in seen:
            raise ValueError("Duplicate candidate IDs")
        seen.add(cid)
        score = float(row["score"])
        if not np.isfinite(score):
            raise ValueError("Candidate score must be finite")
        db_grasp = rigid(row["grasp_mat"], "db_from_grasp")
        world_tcp.append(instance.world_from_cad @ cad_db @ db_grasp @ grasp_tcp)
        scores.append(score)
        metadata.append(
            dict(
                candidate_index=cid,
                scene_id=str(row["scene_id"]),
                source_pregrasp_index=int(row["source_pregrasp_index"]),
                instance_id=instance.instance_id,
                asset_key=instance.asset_key,
            )
        )
    order = np.argsort(-np.asarray(scores), kind="stable")
    world_tcp = np.stack(world_tcp)[order]
    return dict(
        grasps_obj=np.linalg.inv(planner) @ world_tcp,
        confidences_pt=np.asarray(scores, dtype=np.float32)[order],
        world_from_tcp=world_tcp,
        metadata=[metadata[i] for i in order],
        frame_calibrated=False,
        physical_grasp_verified=False,
    )


def scene_digest(scene_revision, instances, model_hashes, *, table_pose, table_dimensions):
    """Bind a plan to the scene and source assets used for collision checking."""
    ids = [x.instance_id for x in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate instance IDs")
    dims = np.asarray(table_dimensions, dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("Invalid table dimensions")
    payload = dict(
        revision=scene_revision,
        models=model_hashes,
        table_pose=rigid(table_pose).tolist(),
        table_dimensions=dims.tolist(),
        instances=[
            dict(
                instance_id=x.instance_id,
                asset_key=x.asset_key,
                world_from_cad=x.world_from_cad.tolist(),
                pose_source=x.pose_source,
            )
            for x in sorted(instances, key=lambda x: x.instance_id)
        ],
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def assert_scene_matches(planned_digest, observed_digest):
    if planned_digest != observed_digest:
        raise ValueError("Scene changed since planning; observe and replan")
