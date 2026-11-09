"""Shared virtual wrist/fixed camera geometry; no device or simulator imports.

The observation carries this configuration so rendering and collision planning
use the same mount and fixed structures. These are virtual fixture settings,
not measured hardware calibrations.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from tiptop.camera_geometry import _rigid_transform, fixed_camera_envelopes


DEFAULT_PREPARATION = Path(__file__).resolve().parents[2] / "simulation/isaac_sim/config/wrist_camera.json"


def _positive_vector(value, name, size=3):
    vector = np.asarray(value, float)
    if vector.shape != (size,) or not np.isfinite(vector).all() or np.any(vector <= 0):
        raise ValueError(f"{name} requires {size} positive finite values")
    return vector


def load_camera_preparation(source):
    document = deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    if document.get("success", True) is not True:
        raise ValueError("Camera preparation reports failure")
    if "db_grasp_from_tcp" in document:
        document["db_grasp_from_tcp"] = _rigid_transform(document["db_grasp_from_tcp"], "db_grasp_from_tcp").tolist()
    profile = document["profile"]
    mount = profile.setdefault("mount", "wrist")
    if mount not in {"wrist", "fixed"}:
        raise ValueError("Virtual camera mount must be wrist or fixed")
    if mount == "wrist" and document.get("success") is not True:
        raise ValueError("A successful wrist camera preparation is required")
    parent = "link_6" if mount == "wrist" else "base_link"
    if profile["parent_link"] != parent:
        raise ValueError(f"{mount} camera requires parent_link={parent}")
    _rigid_transform(profile["parent_from_rig"], "parent_from_rig")
    for key in ["body_dims_m", "bracket_dims_m"]:
        _positive_vector(profile[key], key)
    bracket = np.asarray(profile["rig_from_bracket_xyz_m"], float)
    if bracket.shape != (3,) or not np.isfinite(bracket).all():
        raise ValueError("Invalid camera bracket offset")
    resolution = _positive_vector(profile["resolution_wh"], "resolution_wh", size=2)
    if not np.equal(resolution, np.floor(resolution)).all():
        raise ValueError("Camera resolution must be integer pixels")
    baseline, fov, margin = (float(profile[key]) for key in ["baseline_m", "horizontal_fov_deg", "collision_margin_m"])
    if not np.isfinite([baseline, fov, margin]).all() or baseline <= 0 or not 0 < fov < 180 or margin < 0:
        raise ValueError("Invalid stereo baseline, field of view or collision margin")
    q = np.asarray(document["q_capture"], float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Virtual capture posture requires six finite joint angles")
    names = {"fixed_camera_body", "fixed_camera_bracket", "configured_full_table", "observed_table"}
    for row in document.get("fixed_obstacles", []):
        if (
            not isinstance(row["name"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", row["name"])
            or row["name"] in names
        ):
            raise ValueError("Fixed structures require unique, non-reserved names")
        names.add(row["name"])
        _positive_vector(row["dims_m"], "fixed obstacle dimensions")
        _rigid_transform(row["base_from_obstacle"], "base_from_obstacle")
    return document


def camera_preparation_id(preparation):
    canonical = json.dumps(preparation, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def resolve_camera_preparation(observation, requested=None):
    """Use the current backend's geometry, rejecting a conflicting CLI file.

    Older wrist backends have no embedded configuration and retain their
    original default file. A new fixed backend always declares its geometry.
    """
    if "camera_preparation" not in observation:
        return load_camera_preparation(requested or DEFAULT_PREPARATION)
    actual = load_camera_preparation(json.loads(str(observation["camera_preparation"])))
    if requested is not None:
        selected = load_camera_preparation(requested)
        if camera_preparation_id(selected) != camera_preparation_id(actual):
            raise ValueError("Camera preparation differs from the live observation backend")
    return actual


def virtual_camera_obstacles(preparation):
    """Return fixed camera/structure boxes in base_link, without robot links."""
    result = deepcopy(preparation.get("fixed_obstacles", []))
    profile = preparation["profile"]
    if profile["mount"] != "fixed" or profile.get("collision_geometry_in_fixed_obstacles", False):
        return result
    result.extend(fixed_camera_envelopes(profile))
    return result


def camera_planning_options(preparation):
    from curobo.geom.types import Cuboid
    from scipy.spatial.transform import Rotation

    boxes = []
    for row in virtual_camera_obstacles(preparation):
        transform = np.asarray(row["base_from_obstacle"])
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        boxes.append(Cuboid(name=row["name"], dims=row["dims_m"], pose=[*transform[:3, 3], *quat]))
    return dict(
        camera_profile=preparation["profile"] if preparation["profile"]["mount"] == "wrist" else None,
        fixed_obstacles=boxes,
        fixed_base_world_contact=bool(preparation["profile"].get("fixed_base_world_contact", False)),
    )


def validate_virtual_camera_plan(plan, preparation):
    identity = plan.get("camera_preparation_sha256")
    if identity is None and preparation["profile"]["mount"] == "wrist":
        return  # Keep old wrist plans compatible with their original backend.
    if identity != camera_preparation_id(preparation):
        raise ValueError("Plan camera geometry differs from this runtime; observe and plan again")
