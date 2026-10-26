"""Perception boundaries: mask/label alignment and invalid depth, no model calls."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tiptop.config import tiptop_cfg
from tiptop.perception import sam2
from tiptop.perception.cameras.frame import Frame
from tiptop.perception_wrapper import run_visual_perception


def test_unavailable_remote_sam2_fails_before_task_inference(monkeypatch):
    monkeypatch.setattr(tiptop_cfg().perception.sam, "mode", "remote")
    monkeypatch.setattr(tiptop_cfg().perception.sam, "url", "http://unavailable", raising=False)

    def unavailable(*args, **kwargs):
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(sam2.requests, "get", unavailable)
    with pytest.raises(RuntimeError, match="Failed to connect to SAM2"):
        sam2.sam2_client()


def test_visual_frontend_keeps_labels_masks_and_valid_depth_together(monkeypatch):
    monkeypatch.setattr(tiptop_cfg().perception, "depth_trunc_m", 2.0)
    monkeypatch.setattr(tiptop_cfg().perception.sam, "mode", "local")
    masks = np.array([[[1, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]], bool)
    monkeypatch.setattr(sam2, "download_sam2_checkpoint", lambda: "unused")
    received = []

    def segment(image, boxes, checkpoint):
        received.append(boxes)
        return masks, np.ones(2)

    monkeypatch.setattr(sam2, "_segment_local", segment)
    payload = dict(
        bboxes=[
            dict(label="pepper shaker", box_2d=[0, 0, 500, 1000]),
            dict(label="bottle", box_2d=[500, 0, 1000, 1000]),
        ],
        predicates=[dict(name="holding", args=["pepper shaker"])],
    )
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(payload))

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    frame = Frame(
        "fixture",
        1.0,
        np.zeros((2, 3, 3), np.uint8),
        np.diag([2.0, 2.0, 1.0]),
        np.array([[0.5, np.nan, 0], [np.inf, 1.0, 3.0]], np.float32),
    )
    transform = np.eye(4)
    transform[:3, 3] = [0.1, 0.2, 0.3]
    from tiptop.perception.known_objects import known_object_detection_context

    result = asyncio.run(
        run_visual_perception(
            None, frame, transform, "후추통 집어", client=client, object_context=known_object_detection_context()
        )
    )
    assert len(calls) == 1
    assert "black cylindrical screw cap" in calls[0]["contents"][1]
    assert "car_leather_conditioner_bottle" in calls[0]["contents"][1]
    assert result["masks"].shape == (2, 1, 2, 3)
    assert list(result["objects"]) == ["pepper_shaker", "bottle"]
    assert result["grounded_atoms"] == [dict(predicate="holding", args=["pepper_shaker"])]
    np.testing.assert_allclose(result["objects"]["pepper_shaker"]["world_xyz"], [[0.1, 0.2, 0.8]])
    np.testing.assert_allclose(result["objects"]["bottle"]["world_xyz"], [[0.6, 0.7, 1.3]])
    assert np.isnan(result["xyz_map"][0, 1]).all()
    assert np.isfinite(result["xyz_downsampled"]).all()
    np.testing.assert_allclose(received[0], [[0, 0, 3, 1], [0, 1, 3, 2]])


@pytest.mark.parametrize("box", [[0, 0, 0, 10], [0, 0, 1001, 10], [0, 1, 2]])
def test_malformed_box_cannot_silently_shift_label_alignment(monkeypatch, box):
    monkeypatch.setattr(sam2, "_segment_local", lambda *args: pytest.fail("No model call expected"))
    with pytest.raises(ValueError):
        sam2.sam2_segment_objects(Image.new("RGB", (8, 6)), [dict(label="bad", box_2d=box)])


def test_active_task_uses_new_image_and_mask_without_another_vlm_call(monkeypatch):
    from tiptop.perception import gemini

    async def forbidden(*args, **kwargs):
        pytest.fail("Continuing an active task must not spend another Gemini request")

    monkeypatch.setattr(gemini, "detect_and_translate_async", forbidden)
    monkeypatch.setattr(sam2, "download_sam2_checkpoint", lambda: "unused")
    monkeypatch.setattr(tiptop_cfg().perception.sam, "mode", "local")
    seen = []

    def segment(image, boxes, checkpoint):
        seen.append(np.asarray(image).copy())
        return np.ones((1, 1, 4, 4), bool), np.ones(1)

    monkeypatch.setattr(sam2, "_segment_local", segment)
    frame = Frame(
        "current", 2.0, np.full((4, 4, 3), 99, np.uint8), np.diag([100.0, 100.0, 1.0]), np.ones((4, 4), np.float32)
    )
    task = dict(
        instruction="후추통 집어",
        bboxes=[dict(label="pepper", box_2d=[0, 0, 1000, 1000])],
        grounded_atoms=[dict(predicate="holding", args=["pepper"])],
        masks=np.zeros((1, 1, 4, 4), bool),
    )
    result = asyncio.run(run_visual_perception(None, frame, np.eye(4), task["instruction"], active_task=task))
    np.testing.assert_array_equal(seen[0], frame.rgb)
    assert result["masks"].all()  # The old mask in task context is never consumed.
    assert result["objects"]["pepper"]["valid_pixels"] == 16
    assert result["semantic_source"] == "active_task_roi"
