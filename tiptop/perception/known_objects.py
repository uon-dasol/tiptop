"""Registered CAD/DB provider following the common image perception front end."""

import json
from pathlib import Path

from tiptop.doosan.grasp_database import load_database_grasps
from tiptop.perception.foundation_pose import estimate_pose

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "tiptop/config/known_objects.json"


def normalize_label(value):
    return value.strip().lower().replace(" ", "_")


def registered_target(bboxes, grounded_atoms, *, catalog_path=DEFAULT_CATALOG):
    """Resolve one task target by explicit catalog aliases, never by label similarity."""
    entries = json.loads(Path(catalog_path).read_text())["objects"]
    target_labels = {atom["args"][0] for atom in grounded_atoms if atom.get("args")}
    matches = []
    for bbox in bboxes:
        label = bbox["label"]
        if label not in target_labels:
            continue
        for entry in entries:
            aliases = {normalize_label(v) for v in [entry["class_id"], *entry["aliases"]]}
            if normalize_label(label) in aliases:
                matches.append((label, entry["class_id"]))
    if len(matches) != 1:
        raise ValueError("Known-DB mode requires one uniquely identified catalog target")
    label, class_id = matches[0]
    return label, registered_object(class_id, catalog_path=catalog_path)


def registered_object(class_id, *, catalog_path=DEFAULT_CATALOG, asset_root=ROOT):
    entries = json.loads(Path(catalog_path).read_text())["objects"]
    ids = [entry["class_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registered class IDs")
    matches = [entry for entry in entries if entry["class_id"] == class_id]
    if len(matches) != 1:
        raise ValueError(f"Unregistered class: {class_id}")
    entry = dict(matches[0])
    for key in ["cad", "db"]:
        entry[key] = (Path(asset_root) / entry[key]).resolve()
        if not entry[key].is_file():
            raise FileNotFoundError(entry[key])
    return entry


async def estimate_registered_grasps(
    session, frame, visual, world_from_cam, *, entry, label, server_url, world_from_planner_object, grasp_from_tcp
):
    # Do not silently map a generic bottle label to a particular registered CAD.
    aliases = {normalize_label(value) for value in [entry["class_id"], *entry["aliases"]]}
    if normalize_label(label) not in aliases:
        raise ValueError(f"{label} is not a registered alias of {entry['class_id']}")
    indices = [i for i, bbox in enumerate(visual["bboxes"]) if bbox["label"] == label]
    if len(indices) != 1:
        raise ValueError("A registered grasp needs one uniquely grounded current instance")
    pose = await estimate_pose(
        session,
        server_url,
        frame,
        visual["depth_map"],
        visual["masks"][indices[0], 0],
        mesh_path=entry["cad"],
        mesh_scale=entry["cad_scale_to_m"],
        class_id=entry["class_id"],
        world_from_cam=world_from_cam,
    )
    grasps = load_database_grasps(
        entry["db"],
        class_id=entry["class_id"],
        world_from_cad=pose["world_from_cad"],
        pose_source=pose["pose_source"],
        world_from_planner_object=world_from_planner_object,
        grasp_from_tcp=grasp_from_tcp,
    )
    return dict(pose=pose, grasps=grasps)
