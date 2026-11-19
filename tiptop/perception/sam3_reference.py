"""Gemini-free reference detection client; no SAM3/Torch imports in this module."""

import asyncio
import io
import json
import logging
import re
import time
from functools import lru_cache, partial
from pathlib import Path

import aiohttp
import numpy as np

from tiptop.console import log_tag
from tiptop.perception.mask_cleanup import drop_stray_pieces
from tiptop.object_catalog import registered_object

_log = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
OBJECT_REFERENCE = "object"
OBJECT_CONFIDENCE_THRESHOLD = 0.3
DUPLICATE_OVERLAP = 0.5
MAX_REFERENCE_TRIALS = 8


class ReferenceDetectionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.perception_diagnostics = diagnostics


def parse_reference_task(instruction):
    """Resolve the catalog class from `<class_id> 집어`, the form task presets hand to perception."""
    match = re.fullmatch(r"\s*`?([A-Za-z][A-Za-z0-9_]*)`?\s*(?:을|를)?\s*집어\s*[.!]?\s*", instruction)
    if match is None:
        raise ValueError("Reference mode requires: <canonical_class_id> 집어")
    registered_object(match[1], check_files=False)
    return match[1]


def _hsv_histogram(rgb, box):
    """Hue/saturation distribution of the lit pixels inside a box; None if too few remain."""
    from PIL import Image

    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = np.ascontiguousarray(rgb[max(y0, 0) : y1, max(x0, 0) : x1])
    hsv = np.asarray(Image.fromarray(crop).convert("HSV")).reshape(-1, 3).astype(int)
    hsv = hsv[hsv[:, 2] > 60]  # the dark table
    if len(hsv) < 20:
        return None
    hist = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=[18, 8], range=[[0, 256], [0, 256]])[0].ravel()
    return hist / hist.sum()


@lru_cache(maxsize=32)
def _reference_histogram(path, modified_ns):
    from PIL import Image, ImageOps

    description = json.loads(Path(path).with_suffix(".json").read_text())
    image = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    histograms = [h for h in (_hsv_histogram(image, b) for b in (description.get("bboxes") or [description["bbox"]])) if h is not None]
    if not histograms:
        raise ValueError(f"Reference boxes of {path} have no usable pixels")
    mean = np.mean(histograms, axis=0)
    return mean / mean.sum()


def appearance_scores(rgb, boxes, reference_image):
    """Colour similarity (0-1) of each box to the reference exemplar boxes."""
    path = Path(reference_image)
    reference = _reference_histogram(str(path), path.stat().st_mtime_ns)
    histograms = [_hsv_histogram(rgb, box) for box in boxes]
    return np.array([0.0 if h is None else float(np.sqrt(h * reference).sum()) for h in histograms])


def unpack_reference_packet(packet, rgb_shape, class_id):
    """Validate a SAM3 server reply and return its masks, boxes, scores and metadata."""
    masks = np.asarray(packet["masks"])
    boxes = np.asarray(packet["boxes_xyxy"], float)
    scores = np.asarray(packet["scores"], float)
    metadata = json.loads(str(packet["metadata"]))
    n = len(scores)
    h, w = rgb_shape[:2]
    if (
        masks.dtype != np.bool_
        or masks.shape != (n, 1, h, w)
        or boxes.shape != (n, 4)
        or scores.shape != (n,)
        or not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or metadata.get("class_id") != class_id
    ):
        raise ValueError("SAM3 returned invalid or mismatched detection data")
    masks = np.stack([drop_stray_pieces(mask[0])[None] for mask in masks]) if n else masks
    return masks, boxes, scores, metadata


def usable_instance(box, mask, h, w):
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and mask.any()


def scene_box(class_id, index, box, h, w):
    return dict(
        label=f"{class_id}__instance_{index}",
        class_id=class_id,
        instance_index=index,
        box_2d=[box[1] / h * 1000, box[0] / w * 1000, box[3] / h * 1000, box[2] / w * 1000],
    )


def decode_reference_result(packet, rgb_shape, class_id, *, appearance_fn=None, appearance_weight=0.0):
    """Reject malformed results and restore the shared front-end contract.

    Candidates are ranked by SAM3 score, plus `appearance_weight` times `appearance_fn(metadata, boxes)` when given.
    Keep every usable returned instance in the scene. Ranking and MAX_REFERENCE_TRIALS apply only to
    selecting/verifying the grasp target, never to the scene masks used for collision geometry.
    """
    masks, boxes, scores, metadata = unpack_reference_packet(packet, rgb_shape, class_id)
    n = len(scores)
    h, w = rgb_shape[:2]
    metadata.update(candidate_scores=scores.tolist(), candidate_boxes_xyxy=boxes.tolist(), gemini_api_calls=0)
    if n == 0:
        raise ReferenceDetectionError(f"SAM3 did not detect {class_id} in the current image", metadata)

    def usable(i):
        return usable_instance(boxes[i], masks[i], h, w)

    ranking = scores
    if appearance_fn is not None:
        appearance = np.asarray(appearance_fn(metadata, boxes), float)
        ranking = scores + appearance_weight * appearance
        metadata.update(appearance_scores=appearance.tolist(), appearance_weight=appearance_weight)
    metadata["ranking_scores"] = ranking.tolist()
    order = np.argsort(-ranking, kind="stable")  # Equal rankings keep the server's order.
    index = int(order[0])
    if not usable(index):
        raise ReferenceDetectionError("SAM3 selected an empty mask or invalid target box", metadata)
    near = [int(i) for i in order if usable(i)]
    scene_indices = sorted(near)  # Keep masks and boxes in the server's instance order.

    def scene_box_of(i):
        return scene_box(class_id, i, boxes[i], h, w)

    candidates = [
        dict(
            index=i,
            score=float(scores[i]),
            ranking=float(ranking[i]),
            appearance=None if appearance_fn is None else float(appearance[i]),
            bboxes=[scene_box_of(i)],
        )
        for i in near[:MAX_REFERENCE_TRIALS]
    ]
    metadata.update(selected_index=index, scene_indices=scene_indices, trial_indices=[c["index"] for c in candidates])
    return dict(
        bboxes=[scene_box_of(i) for i in scene_indices],
        masks=masks[scene_indices],
        grounded_atoms=[dict(predicate="on", args=[scene_box_of(index)["label"], "table"])],
        semantic_source="sam3_reference",
        reference_diagnostics=metadata,
        reference_candidates=candidates,
    )


def decode_scene_objects(packet, rgb_shape):
    """Usable instances of the generic `object` reference, labelled like the target-class instances."""
    masks, boxes, scores, metadata = unpack_reference_packet(packet, rgb_shape, OBJECT_REFERENCE)
    h, w = rgb_shape[:2]
    indices = [i for i in range(len(scores)) if usable_instance(boxes[i], masks[i], h, w)]
    metadata.update(candidate_scores=scores.tolist(), candidate_boxes_xyxy=boxes.tolist(),
                    usable_scores=[float(scores[i]) for i in indices])
    return dict(
        bboxes=[scene_box(OBJECT_REFERENCE, i, boxes[i], h, w) for i in indices], masks=masks[indices], diagnostics=metadata
    )


def same_object(mask, other):
    """Whether two masks cover the same object: one lies mostly inside the other."""
    overlap = np.count_nonzero(mask & other)
    return overlap >= DUPLICATE_OVERLAP * min(np.count_nonzero(mask), np.count_nonzero(other))


def merge_scene_objects(result, scene, *, record_scores=False):
    """Add generic instances to the scene, except those already represented by a target-class instance.

    `record_scores` also keeps the SAM3 score of every usable generic instance, duplicates included.
    """
    known = result["masks"][:, 0]
    added = [i for i, mask in enumerate(scene["masks"][:, 0]) if not any(same_object(mask, other) for other in known)]
    result["bboxes"] = [*result["bboxes"], *[scene["bboxes"][i] for i in added]]
    result["masks"] = np.concatenate([result["masks"], scene["masks"][added]])
    result["reference_diagnostics"]["scene_objects"] = dict(
        confidence_threshold=scene["diagnostics"].get("confidence_threshold"),
        detected=len(scene["bboxes"]),
        added=len(added),
        labels=[scene["bboxes"][i]["label"] for i in added],
    )
    if record_scores:
        scores = scene["diagnostics"]["usable_scores"]
        result["reference_diagnostics"]["scene_objects"]["scores"] = {
            box["label"]: round(score, 4) for box, score in zip(scene["bboxes"], scores)
        }
    return result


def log_scene_object_scores(scene_objects):
    """One terminal line per `object` detection, highest score first: `물체 N. score S`, marked when it duplicates a target-class instance."""
    added = set(scene_objects["labels"])
    for number, (label, score) in enumerate(sorted(scene_objects["scores"].items(), key=lambda item: -item[1]), 1):
        log_tag("SAM3", f"물체 {number}. score {score:.2f}" + ("" if label in added else " · 대상 종류와 중복"), "dim")


def log_reference_detection(class_id, metadata, requested_threshold=None):
    """Terminal summary of the SAM3 candidates and which of them will be verified."""
    scores, boxes = metadata["candidate_scores"], metadata["candidate_boxes_xyxy"]
    ranking, appearance = metadata["ranking_scores"], metadata.get("appearance_scores")
    trial = metadata["trial_indices"]
    threshold = metadata.get("confidence_threshold")
    if requested_threshold is not None and threshold is not None and abs(threshold - requested_threshold) > 1e-9:
        log_tag(
            "SAM3",
            f"⚠ 요청한 임계값 {requested_threshold:.2f}과 서버가 적용한 값 {threshold:.2f}이 다릅니다 · SAM3 서버를 재시작하세요",
            "warn",
        )
    log_tag(
        "SAM3",
        f"{class_id}: 후보 {len(scores)}개 검출{'' if threshold is None else f' (점수 > {threshold:.2f})'} · "
        f"{'SAM3 점수 + 외형 유사도 × ' + format(metadata['appearance_weight'], 'g') if appearance else 'SAM3 점수'} 순으로 "
        f"상위 {len(trial)}개를 파지 대상으로 검증 (최대 {MAX_REFERENCE_TRIALS}개) · "
        f"장면 마스크 {len(metadata['scene_indices'])}개 유지",
        "info",
    )
    if appearance:
        weight = metadata["appearance_weight"]
        by_score = min(range(len(scores)), key=lambda i: (-scores[i], i))
        by_total = min(range(len(scores)), key=lambda i: (-ranking[i], i))
        log_tag(
            "SAM3",
            f"외형 유사도 보정 켜짐 · 가중치 {weight:g} · 참조 박스 {metadata.get('reference_box_count', '?')}개의 색 분포와 비교",
            "info",
        )
        if by_score != by_total:
            log_tag(
                "SAM3",
                f"순위 변경: SAM3 점수 1위 idx{by_score} (SAM3 {scores[by_score]:.4f}, 외형 {appearance[by_score]:.3f}) → "
                f"합산 1위 idx{by_total} (SAM3 {scores[by_total]:.4f}, 외형 {appearance[by_total]:.3f})",
                "warn",
            )
        else:
            log_tag("SAM3", f"외형 유사도를 반영해도 1위는 그대로입니다 (idx{by_total})", "dim")
    for rank, i in enumerate(sorted(range(len(scores)), key=lambda i: (-ranking[i], i)), 1):
        x1, y1, x2, y2 = boxes[i]
        value = (
            f"합산 {ranking[i]:.4f} (SAM3 {scores[i]:.4f} + 외형 {appearance[i]:.3f})" if appearance else f"점수 {scores[i]:.4f}"
        )
        if i in trial:
            note, kind = "검증 대상 · 장면 유지", "ok"
        elif i in metadata["scene_indices"]:
            note, kind = "장면 유지 · 파지 검증 생략", "dim"
        else:
            note, kind = "빈 마스크 또는 유효하지 않은 박스", "dim"
        log_tag(
            "SAM3",
            f"  {rank}위  {value}  박스 ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) {x2 - x1:.0f}×{y2 - y1:.0f}px  {note}",
            kind,
        )


def log_reference_failure(class_id, error):
    metadata = error.perception_diagnostics
    if metadata["candidate_scores"]:
        reason = "1위 후보의 마스크가 비었거나 박스가 이미지 밖입니다"
    else:
        threshold = metadata.get("confidence_threshold")
        reason = (
            f"{'' if threshold is None else f'점수 > {threshold:.2f} '}후보가 없습니다 "
            f"(참조 장면 {metadata.get('reference_scene')}에서 {class_id} 미검출 또는 모든 후보가 임계값 이하)"
        )
    log_tag("SAM3", f"실패: {reason}", "fail")


async def query_sam3(session, rgb, class_id, *, server_url, reference_scene, confidence_threshold, decode):
    stream = io.BytesIO()
    from tiptop.serialization import savez_fast

    savez_fast(
        stream, rgb=rgb, class_id=class_id, reference_scene=reference_scene, confidence_threshold=confidence_threshold
    )
    async with session.post(
        server_url.rstrip("/") + "/segment-reference",
        data=stream.getvalue(),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"SAM3 request failed ({response.status}): {(await response.text())[:1000]}")
        with np.load(io.BytesIO(await response.read()), allow_pickle=False) as packet:
            return decode(packet)


async def detect_reference(
    session, rgb, instruction, *, server_url, reference_scene, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    appearance_weight=None, scene_objects=False,
):
    """Find the grasp target with its own reference.

    With `scene_objects`, the `object` reference query starts right after the target query and is returned
    unfinished as `scene_objects_job`, so the caller can overlap it with other work and merge its result later.
    """
    started = time.perf_counter()
    class_id = parse_reference_task(instruction)
    query = partial(query_sam3, session, rgb, server_url=server_url, reference_scene=reference_scene)

    def decode_target(packet):
        try:
            return decode_reference_result(
                packet,
                rgb.shape,
                class_id,
                appearance_fn=None
                if appearance_weight is None
                else lambda metadata, boxes: appearance_scores(rgb, boxes, metadata["reference_image"]),
                appearance_weight=appearance_weight or 0.0,
            )
        except ReferenceDetectionError as error:
            log_reference_failure(class_id, error)
            raise

    result = await query(class_id, confidence_threshold=confidence_threshold, decode=decode_target)
    log_reference_detection(class_id, result["reference_diagnostics"], confidence_threshold)
    if scene_objects:
        result["scene_objects_job"] = asyncio.create_task(
            query(
                OBJECT_REFERENCE,
                confidence_threshold=OBJECT_CONFIDENCE_THRESHOLD,
                decode=lambda packet: decode_scene_objects(packet, rgb.shape),
            )
        )
    _log.info("Reference detection client total took %.2fs", time.perf_counter() - started)
    return result


async def check_reference_health(session, server_url):
    async with session.get(server_url.rstrip("/") + "/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
        response.raise_for_status()
        health = await response.json()
    if health.get("status") != "ready" or health.get("backend") != "sam3_reference":
        raise RuntimeError("The selected endpoint is not a ready SAM3 reference server")
    return health


def save_candidate_crops(rgb, diagnostics, trials, directory):
    """Write an overview and one crop per SAM3 candidate, labelled with its verification outcome.

    Crops cover every detected candidate, including those never verified, in ranking order (SAM3 score,
    or its sum with the appearance similarity); `idx` is the position in the SAM3 server's reply.
    """
    import cv2

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores, boxes = diagnostics["candidate_scores"], diagnostics["candidate_boxes_xyxy"]
    ranking, appearance = diagnostics["ranking_scores"], diagnostics.get("appearance_scores")
    queued = diagnostics["trial_indices"]
    tried = {trial["candidate_index"]: trial for trial in trials}
    colors = dict(accepted=(40, 200, 40), rejected=(230, 60, 60), skipped=(230, 200, 40))
    overview = np.ascontiguousarray(rgb).copy()
    for rank, i in enumerate(sorted(range(len(scores)), key=lambda i: (-ranking[i], i)), 1):
        trial = tried.get(i)
        if trial is not None:
            kind = "accepted" if trial["accepted"] else "rejected"
            note = f"IoU {trial['silhouette_iou']:.3f} {kind.upper()}"
        elif i in queued:
            kind, note = "skipped", "not tried: an earlier candidate matched"
        else:
            kind, note = "skipped", f"skipped: over the {MAX_REFERENCE_TRIALS}-candidate limit"
        color = colors[kind]
        x0, y0, x1, y1 = [int(round(v)) for v in boxes[i]]
        cv2.rectangle(overview, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overview, f"#{rank} {scores[i]:.3f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        pad = 50
        crop = np.ascontiguousarray(rgb[max(y0 - pad, 0) : y1 + pad, max(x0 - pad, 0) : x1 + pad]).copy()
        cv2.rectangle(crop, (x0 - max(x0 - pad, 0), y0 - max(y0 - pad, 0)), (x1 - max(x0 - pad, 0), y1 - max(y0 - pad, 0)), color, 1)
        scale = max(1, 360 // max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"#{rank} SAM3 {scores[i]:.4f} (idx{i})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if appearance:
            cv2.putText(
                crop, f"appearance {appearance[i]:.3f}  total {ranking[i]:.4f}", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
        cv2.putText(crop, note, (4, crop.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        name = f"rank{rank:02d}_{kind}_score{scores[i]:.3f}_idx{i}.png"
        cv2.imwrite(str(directory / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(directory / "overview.png"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))


def save_reference_image(diagnostics, directory):
    """Copy the reference image SAM3 was prompted with, plus a copy with its exemplar boxes drawn."""
    import shutil

    import cv2

    path = Path(diagnostics["reference_image"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, directory / f"reference{path.suffix}")
    description = json.loads(path.with_suffix(".json").read_text())
    boxes = description.get("bboxes") or [description["bbox"]]
    image = cv2.imread(str(path))
    for number, (x0, y0, x1, y1) in enumerate(boxes, 1):
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)
        cv2.putText(image, f"#{number}", (int(x0), max(int(y0) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite(str(directory / "reference_boxes.png"), image)
