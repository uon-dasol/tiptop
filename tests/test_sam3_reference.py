"""Mode isolation, current-frame transport and mask/label coordinates without models."""

import asyncio
import io
import json
import sys

import numpy as np
import pytest

from tiptop.perception import sam2, sam3_reference as reference
from tiptop.perception.cameras.frame import Frame
from tiptop.perception_wrapper import run_visual_perception


@pytest.mark.parametrize(
    "instruction",
    [
        "black_pepper_shaker를 집고 빈곳에 놓아",
        "`black_pepper_shaker` 집고 빈 곳에 놓아.",
        "black_pepper_shaker를 집어서 빈 곳에 놓아줘",
    ],
)
def test_explicit_reference_instruction(instruction):
    assert reference.parse_reference_task(instruction) == "black_pepper_shaker"


@pytest.mark.parametrize(
    "instruction",
    [
        "not_registered를 집고 빈곳에 놓아",
        "후추통을 집고 빈곳에 놓아",
        "black_pepper_shaker를 집어",
        "black_pepper_shaker를 집고 병 옆에 놓아",
    ],
)
def test_unsupported_instruction_fails_without_guessing(instruction):
    with pytest.raises(ValueError):
        reference.parse_reference_task(instruction)


def packet(h=4, w=6):
    masks = np.zeros((2, 1, h, w), bool)
    masks[0, 0, :2, :3] = True
    masks[1, 0, 2:, 3:] = True
    return dict(
        masks=masks,
        scores=np.array([0.6, 0.9]),
        boxes_xyxy=np.array([[0, 0, 3, 2], [3, 2, 6, 4]]),
        metadata=np.array(json.dumps(dict(class_id="black_pepper_shaker"))),
    )


def test_highest_score_keeps_canonical_class_and_original_coordinates():
    result = reference.decode_reference_result(packet(), (4, 6, 3), "black_pepper_shaker")
    assert result["bboxes"] == [dict(label="black_pepper_shaker", box_2d=[500, 500, 1000, 1000])]
    assert result["masks"].shape == (1, 1, 4, 6)
    assert result["masks"][0, 0, 2:, 3:].all()
    assert result["grounded_atoms"] == [dict(predicate="on", args=["black_pepper_shaker", "table"])]


@pytest.mark.parametrize("change", ["missing", "resolution", "class", "score", "mask", "box"])
def test_invalid_or_absent_detection_is_a_failure(change):
    value = packet()
    if change == "missing":
        value.update(masks=np.zeros((0, 1, 4, 6), bool), scores=np.array([]), boxes_xyxy=np.zeros((0, 4)))
    elif change == "resolution":
        value["masks"] = np.ones((2, 1, 6, 4), bool)
    elif change == "class":
        value["metadata"] = np.array(json.dumps(dict(class_id="wrong")))
    elif change == "score":
        value["scores"][1] = np.nan
    elif change == "mask":
        value["masks"][1] = False
    else:
        value["boxes_xyxy"][1, 2] = 99
    with pytest.raises(ValueError):
        reference.decode_reference_result(value, (4, 6, 3), "black_pepper_shaker")


def test_reference_frontend_uses_each_current_image_without_gemini_or_sam2(monkeypatch):
    from tiptop.perception import gemini

    async def forbidden(*args, **kwargs):
        pytest.fail("Reference mode must never call Gemini")

    monkeypatch.setattr(gemini, "detect_and_translate_async", forbidden)
    monkeypatch.setattr(sam2, "sam2_segment_objects", lambda *a: pytest.fail("SAM2 must be bypassed"))
    sent = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            stream = io.BytesIO()
            np.savez_compressed(stream, **packet())
            return stream.getvalue()

    class Session:
        def post(self, url, *, data, timeout):
            with np.load(io.BytesIO(data), allow_pickle=False) as request:
                sent.append(request["rgb"].copy())
                assert str(request["class_id"]) == "black_pepper_shaker"
            return Response()

    for color in (31, 97):
        frame = Frame("test", color, np.full((4, 6, 3), color, np.uint8), np.eye(3), np.ones((4, 6), np.float32))
        result = asyncio.run(
            run_visual_perception(
                Session(),
                frame,
                np.eye(4),
                "black_pepper_shaker를 집고 빈곳에 놓아",
                perception_mode="sam3_reference",
            )
        )
        assert result["semantic_source"] == "sam3_reference"
        assert result["objects"]["black_pepper_shaker"]["valid_pixels"] == 6
    assert len(sent) == 2
    assert np.all(sent[0] == 31) and np.all(sent[1] == 97)


def test_reference_output_is_accepted_by_the_existing_foundationpose_client():
    from tiptop.perception.foundation_pose import estimate_pose
    from tiptop.perception.known_objects import registered_target

    rgb = np.full((8, 12, 3), 42, np.uint8)
    value = dict(
        masks=np.ones((1, 1, 8, 12), bool),
        boxes_xyxy=np.array([[2, 2, 8, 7]]),
        scores=np.array([0.8]),
        metadata=packet()["metadata"],
    )
    detection = reference.decode_reference_result(value, rgb.shape, "black_pepper_shaker")
    label, entry = registered_target(detection["bboxes"], detection["grounded_atoms"])
    pose = np.eye(4)
    pose[2, 3] = 1

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return dict(camera_from_cad=pose.tolist(), class_id=label)

    class Session:
        def post(self, url, *, data, timeout):
            with np.load(io.BytesIO(data), allow_pickle=False) as request:
                np.testing.assert_array_equal(request["rgb"], rgb)
                np.testing.assert_array_equal(request["mask"], detection["masks"][0, 0])
                assert str(request["class_id"]) == "black_pepper_shaker"
                assert request["depth"].shape == rgb.shape[:2]
            return Response()

    frame = Frame("test", 1, rgb, np.eye(3))
    result = asyncio.run(
        estimate_pose(
            Session(),
            "http://test",
            frame,
            np.ones((8, 12)),
            detection["masks"][0, 0],
            mesh_path=entry["cad"],
            mesh_scale=entry["cad_scale_to_m"],
            class_id=label,
            world_from_cam=np.eye(4),
        )
    )
    np.testing.assert_array_equal(result["world_from_cad"], pose)


def test_no_detection_provenance_survives_task_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from tiptop.scripts import run_known_pick_place as runner

    async def fail(args):
        args._stage = "perception"
        raise reference.ReferenceDetectionError("No target", dict(reference_scene="Home/floor", gemini_api_calls=0))

    monkeypatch.setattr(runner, "run", fail)
    args = SimpleNamespace(output=tmp_path, task="test", perception_mode="sam3_reference")
    result = asyncio.run(runner.run_task(args))
    assert result["status"] == "failed" and result["gemini_api_calls"] == 0
    saved = json.loads((tmp_path / "task-result.json").read_text())
    assert saved["perception_diagnostics"]["reference_scene"] == "Home/floor"


@pytest.mark.parametrize(
    "extra, valid",
    [
        (["--perception-mode", "sam3_reference", "--interactive"], True),
        (["--perception-mode", "gemini_er", "--interactive", "--allow-gemini-once"], True),
        (["--perception-mode", "gemini_er", "--interactive"], False),
        (["--perception-mode", "sam3_reference", "--allow-gemini-once"], False),
        (["--perception-mode", "sam3_reference", "--active-task", "old.json"], False),
    ],
)
def test_cli_permissions_follow_selected_mode(monkeypatch, tmp_path, extra, valid):
    from tiptop.scripts import run_known_pick_place as runner

    called = []

    async def run(args):
        called.append(args.perception_mode)

    monkeypatch.setattr(runner, "run_interactive", run)
    monkeypatch.setattr(runner, "run_task", run)
    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path), *extra])
    if valid:
        runner.main()
        assert called
    else:
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 2 and not called
