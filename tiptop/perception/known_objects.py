"""Registered CAD/DB provider following the common image perception front end."""

import json
import logging

import numpy as np

from tiptop.console import log_tag
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


def _without_cad_outside_shadow(mask, silhouette, rgb):
    """Drop mask pixels outside the CAD silhouette that are no brighter than the table around the object.

    SAM3 masks often include the object's cast shadow; against the CAD silhouette those pixels only add a mismatch.
    Pixels inside the silhouette are never touched, so dark parts of the object itself still count.
    """
    import cv2

    def dilate(region, radius):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(region.astype(np.uint8), kernel).astype(bool)

    ring = dilate(mask, 9) & ~dilate(mask, 4)
    if not ring.any():
        return mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = gray < np.median(gray[ring]) * 1.05
    return mask & ~(~dilate(silhouette, 1) & dark)


def validate_cad_mask_alignment(mesh, camera_from_cad, intrinsics, mask, *, rgb, minimum_iou=0.75):
    """Reject a gross CAD/mask silhouette mismatch before supplying DB grasps.

    IoU of the CAD silhouette projected at the estimated pose and the detected mask, ignoring mask pixels that lie
    outside the silhouette and are as dark as the surrounding table (shadow). The IoU without that step is
    returned as `plain_silhouette_iou`. This is a full-view geometric check, not proof of semantic identity or pose
    accuracy. Occlusion or ambiguous matching must be resolved with a new view.
    """
    import cv2

    if not np.isfinite(minimum_iou) or not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be finite and between 0 and 1")
    transform = np.asarray(camera_from_cad, float)
    mask = np.asarray(mask, bool)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
        raise ValueError("The shadow check needs the uint8 RGB image on the same pixel grid as the mask")
    points = np.asarray(mesh.vertices, float) @ transform[:3, :3].T + transform[:3, 3]
    if len(points) < 4 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("CAD silhouette needs finite geometry entirely in front of the camera")
    if not mask.any():
        raise ValueError("CAD silhouette needs a nonempty object mask")
    pixels = points @ np.asarray(intrinsics).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    silhouette = np.zeros(mask.shape, np.uint8)
    for triangle in np.rint(pixels[np.asarray(mesh.faces)]).astype(np.int32):
        cv2.fillConvexPoly(silhouette, triangle, 1)
    silhouette = silhouette.astype(bool)

    def overlap(region):
        union = int(np.count_nonzero(silhouette | region))
        return np.count_nonzero(silhouette & region) / union if union else 0.0

    iou = overlap(_without_cad_outside_shadow(mask, silhouette, rgb))
    plain_iou = overlap(mask)
    if iou < minimum_iou:
        raise SilhouetteMismatchError(float(iou), minimum_iou, plain_iou=float(plain_iou))
    return dict(
        silhouette_iou=float(iou), plain_silhouette_iou=float(plain_iou), minimum_iou=minimum_iou,
        semantic_identity_verified=False,
    )


class SilhouetteMismatchError(ValueError):
    def __init__(self, iou, minimum_iou, tried=1, plain_iou=None):
        best = f"; best of {tried} SAM3 candidates" if tried > 1 else ""
        super().__init__(
            f"CAD/mask silhouette mismatch (IoU={iou:.3f} < {minimum_iou:.3f}){best}; "
            "check target identity, segmentation and pose before planning"
        )
        self.iou, self.minimum_iou, self.plain_iou, self.trials = iou, minimum_iou, plain_iou, []


def mask_cad_size_ratio(mask, depth_map, intrinsics, mesh):
    """Long side of the mask's image box in metres over the CAD's longest extent; None without valid depth."""
    ys, xs = np.nonzero(mask)
    depth = np.asarray(depth_map)[ys, xs]
    depth = depth[np.isfinite(depth) & (depth > 0)]
    if not depth.size:
        return None
    z = np.percentile(depth, 20)
    k = np.asarray(intrinsics)
    long_side = max((xs.max() - xs.min() + 1) * z / k[0, 0], (ys.max() - ys.min() + 1) * z / k[1, 1])
    return float(long_side / mesh.extents.max())


def _log_trial(trial, minimum_iou, label):
    """One terminal line for a verified candidate: SAM3 score, silhouette IoU, size against the CAD and pose score."""
    ok = trial["accepted"]
    name = label if trial["score"] is None else f"{trial['rank']}번째 후보"
    parts = [f"{'✓' if ok else '✗'} {name} {'채택' if ok else '거부'}"]
    if trial["score"] is not None:
        parts.append(f"SAM3 점수 {trial['score']:.4f}" + ("" if trial["appearance"] is None else f" · 외형 {trial['appearance']:.3f}"))
    plain = trial.get("plain_iou")
    parts.append(
        f"실루엣 IoU {trial['silhouette_iou']:.3f} {'≥' if ok else '<'} {minimum_iou:.3f}"
        + ("" if plain is None else f" (그림자 제외 · 원본 {plain:.3f})")
    )
    ratio = trial["size_ratio"]
    if ratio is not None:
        parts.append(f"크기 CAD의 {ratio:.2f}배" + (" ⚠ 크기 이상" if ratio > 1.7 or ratio < 0.3 else ""))
    if ok and trial.get("pose_score") is not None:
        parts.append(f"FoundationPose 점수 {trial['pose_score']:.1f}")
    log_tag("Perception", " · ".join(parts), "ok" if ok else "fail")
    if ok and trial["rank"] > 1:
        log_tag("Perception", f"앞선 {trial['rank'] - 1}개 후보는 실루엣 검사에서 거부되어 {trial['rank']}번째 후보로 진행합니다", "warn")


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
        class_label = bbox.get("class_id", label)
        for entry in entries:
            aliases = {normalize_label(v) for v in [entry["class_id"], *entry["aliases"]]}
            if normalize_label(class_label) in aliases:
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
    indices = [i for i, bbox in enumerate(visual["bboxes"]) if bbox["label"] == label]
    if len(indices) != 1:
        raise ValueError("A registered grasp needs one uniquely grounded current instance")
    bbox = visual["bboxes"][indices[0]]
    class_label = bbox.get("class_id", label)
    aliases = {normalize_label(value) for value in [entry["class_id"], *entry["aliases"]]}
    if normalize_label(class_label) not in aliases:
        raise ValueError(f"{class_label} is not a registered alias of {entry['class_id']}")
    box = bbox.get("box_2d")
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
    mesh = load_cad_mesh(entry["cad"], entry["cad_scale_to_m"])
    vertices = np.asarray(mesh.vertices)
    pose["cad_mask_alignment"] = validate_cad_mask_alignment(
        mesh, pose["camera_from_cad"], frame.intrinsics, visual["masks"][indices[0], 0],
        rgb=frame.rgb, minimum_iou=cad_mask_min_iou,
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


async def estimate_first_matching_candidate(session, frame, results, world_from_cam, *, switch_candidate, **options):
    """Verify the SAM3 candidates in score order; the first whose CAD silhouette matches its mask is the target.

    A detection without candidates (Gemini) is verified as the single candidate.
    Returns (results of the accepted candidate, label, entry, registered grasps, trials).
    """
    candidates = results.get("reference_candidates") or [None]
    minimum_iou = options.get("cad_mask_min_iou", 0.75)
    trials = []
    for rank, candidate in enumerate(candidates, 1):
        attempt = results if rank == 1 else switch_candidate(results, candidate)
        label, entry = registered_target(attempt["bboxes"], attempt["grounded_atoms"])
        mask = attempt["masks"][[b["label"] for b in attempt["bboxes"]].index(label), 0]
        trial = dict(
            rank=rank,
            candidate_index=None if candidate is None else candidate["index"],
            score=None if candidate is None else candidate["score"],
            appearance=None if candidate is None else candidate["appearance"],
            size_ratio=mask_cad_size_ratio(
                mask, attempt["depth_map"], frame.intrinsics, load_cad_mesh(entry["cad"], entry["cad_scale_to_m"])
            ),
        )
        trials.append(trial)
        try:
            registered = await estimate_registered_grasps(
                session, frame, attempt, world_from_cam, entry=entry, label=label, **options
            )
        except SilhouetteMismatchError as error:
            trial.update(silhouette_iou=error.iou, plain_iou=error.plain_iou, accepted=False)
            _log_trial(trial, minimum_iou, label)
            continue
        pose = registered["pose"]
        alignment = pose["cad_mask_alignment"]
        trial.update(
            silhouette_iou=alignment["silhouette_iou"], plain_iou=alignment.get("plain_silhouette_iou"), accepted=True,
            pose_score=pose.get("ranking_score"),
        )
        _log_trial(trial, minimum_iou, label)
        return attempt, label, entry, registered, trials
    best = max(trial["silhouette_iou"] for trial in trials)
    error = SilhouetteMismatchError(best, minimum_iou, tried=len(trials))
    error.trials = trials
    log_tag(
        "Perception",
        f"✗ 시험한 후보 {len(trials)}개 모두 실루엣 검사 실패 (최고 IoU {best:.3f} < {minimum_iou:.3f}) → "
        "물체 오인식 또는 포즈 불일치입니다. 마스크/자세/참조를 확인하세요",
        "fail",
    )
    raise error
