"""Registered CAD/DB provider following the common image perception front end."""

import json
import logging

import numpy as np

from tiptop.doosan.grasp_database import load_database_grasps, warm_database_grasps
from tiptop.object_catalog import DEFAULT_CATALOG, load_catalog
from tiptop.object_catalog import registered_object as registered_object
from tiptop.perception.cad_cache import load_cad_mesh
from tiptop.perception.foundation_pose import estimate_pose

_log = logging.getLogger(__name__)


def prefetch_registered_assets(class_id):
    """Read a class's CAD and grasp DB ahead of use; a failure resurfaces at the real load."""
    try:
        entry = registered_object(class_id)
        load_cad_mesh(entry["cad"], entry["cad_scale_to_m"])
        warm_database_grasps(entry["db"])
    except Exception as error:
        _log.warning("Asset prefetch failed: %s", error)


def known_object_detection_context(*, catalog_path=DEFAULT_CATALOG):
    """Catalog appearance/labels only; never current positions or simulator state."""
    entries = load_catalog(catalog_path)["objects"]
    rows = [
        dict(
            class_id=entry["class_id"], aliases=entry["aliases"], visual_description=entry.get("visual_description", "")
        )
        for entry in entries
    ]
    return (
        "Known-object catalog (use current image evidence, not the task name alone):\n"
        + json.dumps(rows, ensure_ascii=False)
        + "\n"
        "Use the matching canonical class_id as the label for a recognized catalog object. "
        "The task may use any of its aliases. Do not assign that label to a visually different object. "
        "If the requested object is not identifiable in the image, return no target predicate."
    )


def normalize_label(value):
    return value.strip().lower().replace(" ", "_")


def cad_support_point(vertices_m, world_from_cad):
    """Estimate support from the full posed CAD, including its occluded bottom."""
    vertices = np.asarray(vertices_m, float)
    pose = np.asarray(world_from_cad, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("CAD support needs finite 3D vertices")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("CAD support needs a finite world-from-CAD pose")
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    bottom = float(world[:, 2].min())
    support = world[world[:, 2] <= bottom + 0.001].mean(axis=0)
    support[2] = bottom
    return support


def validate_cad_mask_alignment(vertices_m, camera_from_cad, intrinsics, mask, *, minimum_iou=0.75):
    """Reject a gross CAD/mask silhouette mismatch before supplying DB grasps.

    Compare convex image silhouettes, so concavities/holes need no renderer.
    This is a full-view geometric check, not proof of semantic identity or pose
    accuracy. Occlusion or ambiguous matching must be resolved with a new view.
    """
    import cv2

    if not np.isfinite(minimum_iou) or not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be finite and between 0 and 1")
    vertices = np.asarray(vertices_m, float)
    transform = np.asarray(camera_from_cad, float)
    mask = np.asarray(mask, np.uint8)
    points = vertices @ transform[:3, :3].T + transform[:3, 3]
    if len(points) < 4 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("CAD silhouette needs finite geometry entirely in front of the camera")
    pixels = points @ np.asarray(intrinsics).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    detected = cv2.findNonZero(mask)
    if detected is None or len(detected) < 4:
        raise ValueError("CAD silhouette needs a nonempty object mask")
    projected_hull, mask_hull = np.zeros_like(mask), np.zeros_like(mask)
    cv2.fillConvexPoly(projected_hull, np.rint(cv2.convexHull(pixels.astype(np.float32))).astype(np.int32), 1)
    cv2.fillConvexPoly(mask_hull, cv2.convexHull(detected), 1)
    intersection = int(np.count_nonzero(projected_hull & mask_hull))
    union = int(np.count_nonzero(projected_hull | mask_hull))
    iou = intersection / union if union else 0.0
    if iou < minimum_iou:
        raise ValueError(
            f"CAD/mask silhouette mismatch (convex IoU={iou:.3f} < {minimum_iou:.3f}); "
            "check target identity, segmentation and pose before planning"
        )
    return dict(convex_silhouette_iou=iou, minimum_iou=minimum_iou, semantic_identity_verified=False)


def require_return_in_place(grounded_atoms):
    """The in-place runner executes only the return_in_place(target) preset."""
    if len(grounded_atoms) != 1 or grounded_atoms[0]["predicate"] != "return_in_place" or len(grounded_atoms[0]["args"]) != 1:
        raise ValueError(
            f"The in-place runner supports only return_in_place(target); Gemini returned {grounded_atoms}. "
            "Check experimental.return_in_place and the instruction."
        )


def registered_target(bboxes, grounded_atoms, *, catalog_path=DEFAULT_CATALOG):
    """Resolve one task target by explicit catalog aliases, never by label similarity."""
    entries = load_catalog(catalog_path)["objects"]
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


async def estimate_registered_grasps(
    session, frame, visual, world_from_cam, *, entry, label, server_url, world_from_planner_object, grasp_from_tcp,
    cad_mask_min_iou=0.75,
):
    # Do not silently map a generic bottle label to a particular registered CAD.
    aliases = {normalize_label(value) for value in [entry["class_id"], *entry["aliases"]]}
    if normalize_label(label) not in aliases:
        raise ValueError(f"{label} is not a registered alias of {entry['class_id']}")
    indices = [i for i, bbox in enumerate(visual["bboxes"]) if bbox["label"] == label]
    if len(indices) != 1:
        raise ValueError("A registered grasp needs one uniquely grounded current instance")
    box = visual["bboxes"][indices[0]].get("box_2d")
    if box is not None and (min(box[:2]) <= 0 or max(box[2:]) >= 1000):
        raise ValueError("Target box touches the image boundary; acquire a complete view before CAD pose estimation")
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
    vertices = np.asarray(load_cad_mesh(entry["cad"], entry["cad_scale_to_m"]).vertices)
    pose["cad_mask_alignment"] = validate_cad_mask_alignment(
        vertices, pose["camera_from_cad"], frame.intrinsics, visual["masks"][indices[0], 0],
        minimum_iou=cad_mask_min_iou,
    )
    grasps = load_database_grasps(
        entry["db"],
        class_id=entry["class_id"],
        world_from_cad=pose["world_from_cad"],
        pose_source=pose["pose_source"],
        world_from_planner_object=world_from_planner_object,
        grasp_from_tcp=grasp_from_tcp,
    )
    return dict(pose=pose, grasps=grasps, support_point_world=cad_support_point(vertices, pose["world_from_cad"]))
