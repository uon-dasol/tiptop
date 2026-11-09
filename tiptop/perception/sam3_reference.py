"""Gemini-free reference detection client; no SAM3/Torch imports in this module."""

import io
import json
import logging
import re
import time

import aiohttp
import numpy as np

from tiptop.object_catalog import registered_object

_log = logging.getLogger(__name__)


class ReferenceDetectionError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.perception_diagnostics = diagnostics


def parse_reference_task(instruction):
    """Resolve an explicit canonical-class pick-and-return command."""
    match = re.fullmatch(
        r"\s*`?([A-Za-z][A-Za-z0-9_]*)`?\s*(?:을|를)?\s*집(?:고|어서)\s*"
        r"(?:들어\s*올렸다가\s*)?제\s*자리(?:에|로)\s*놓(?:아\s*줘|아주세요|아|기)\s*[.!]?\s*",
        instruction,
    )
    if match is None:
        raise ValueError("Reference mode requires: <canonical_class_id>를 집고 제자리에 놓아")
    class_id = match.group(1)
    registered_object(class_id, check_files=False)
    return class_id


def decode_reference_result(packet, rgb_shape, class_id):
    """Reject malformed results and restore the shared front-end contract."""
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
    metadata.update(candidate_scores=scores.tolist(), candidate_boxes_xyxy=boxes.tolist(), gemini_api_calls=0)
    if n == 0:
        raise ReferenceDetectionError(f"SAM3 did not detect {class_id} in the current image", metadata)
    index = int(np.argmax(scores))  # Same selection policy as the data_gen real example.
    x1, y1, x2, y2 = boxes[index]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h) or not masks[index].any():
        raise ReferenceDetectionError("SAM3 selected an empty mask or invalid target box", metadata)
    metadata["selected_index"] = index
    return dict(
        bboxes=[dict(label=class_id, box_2d=[y1 / h * 1000, x1 / w * 1000, y2 / h * 1000, x2 / w * 1000])],
        masks=masks[index : index + 1],
        grounded_atoms=[dict(predicate="on", args=[class_id, "table"])],
        semantic_source="sam3_reference",
        reference_diagnostics=metadata,
    )


async def detect_reference(session, rgb, instruction, *, server_url, reference_scene):
    started = time.perf_counter()
    class_id = parse_reference_task(instruction)
    stream = io.BytesIO()
    from tiptop.serialization import savez_fast

    savez_fast(stream, rgb=rgb, class_id=class_id, reference_scene=reference_scene)
    async with session.post(
        server_url.rstrip("/") + "/segment-reference",
        data=stream.getvalue(),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"SAM3 request failed ({response.status}): {(await response.text())[:1000]}")
        with np.load(io.BytesIO(await response.read()), allow_pickle=False) as packet:
            result = decode_reference_result(packet, rgb.shape, class_id)
    _log.info("Reference detection client total took %.2fs", time.perf_counter() - started)
    return result


async def check_reference_health(session, server_url):
    async with session.get(server_url.rstrip("/") + "/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
        response.raise_for_status()
        health = await response.json()
    if health.get("status") != "ready" or health.get("backend") != "sam3_reference":
        raise RuntimeError("The selected endpoint is not a ready SAM3 reference server")
    return health
