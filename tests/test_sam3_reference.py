"""Mode isolation, current-frame transport and mask/label coordinates without models."""

import asyncio
import io
import json
import sys

import numpy as np
import pytest

from tiptop.perception import sam2, sam3_reference as reference
from tiptop.perception.mask_cleanup import drop_stray_pieces
from tiptop.perception.cameras.frame import Frame
from tiptop.perception_wrapper import run_visual_perception


@pytest.mark.parametrize(
    "instruction",
    ["black_pepper_shaker 집어", "`black_pepper_shaker` 집어.", "black_pepper_shaker를 집어", "black_pepper_shaker을 집어!"],
)
def test_explicit_reference_instruction(instruction):
    assert reference.parse_reference_task(instruction) == "black_pepper_shaker"


@pytest.mark.parametrize("class_id", ["spam_classic_20G_can", "grip_strengthener_30KG"])
def test_class_id_with_uppercase_is_parsed_verbatim(class_id):
    assert reference.parse_reference_task(f"{class_id} 집어") == class_id


@pytest.mark.parametrize(
    "instruction",
    [
        "not_registered를 집어",
        "SPAM_classic_20G_can을 집어",
        "후추통을 집어",
        "black_pepper_shaker를 집고 제자리에 놓아",
        "black_pepper_shaker를 집고 병 옆에 놓아",
        "black_pepper_shaker를 집고 상자에 놓아",
        "black_pepper_shaker를 집고 빈곳에 놓아",
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


def test_all_instances_are_retained_with_unique_labels_and_original_coordinates():
    result = reference.decode_reference_result(packet(), (4, 6, 3), "black_pepper_shaker")
    assert [box["label"] for box in result["bboxes"]] == ["black_pepper_shaker__instance_0", "black_pepper_shaker__instance_1"]
    assert [box["instance_index"] for box in result["bboxes"]] == [0, 1]
    assert result["bboxes"][1]["box_2d"] == [500, 500, 1000, 1000]
    assert all(box["class_id"] == "black_pepper_shaker" for box in result["bboxes"])
    np.testing.assert_array_equal(result["masks"], packet()["masks"])
    assert result["grounded_atoms"] == [dict(predicate="on", args=["black_pepper_shaker__instance_1", "table"])]


def test_stray_mask_pieces_are_dropped_but_large_secondary_pieces_stay():
    mask = np.zeros((40, 60), bool)
    mask[5:25, 5:25] = True          # main piece, 400 px
    mask[35:37, 50:52] = True        # 4 px far away: stray
    mask[5:13, 40:50] = True         # 80 px, 20% of the main piece: kept
    cleaned = drop_stray_pieces(mask)
    assert not cleaned[35:37, 50:52].any() and cleaned[5:13, 40:50].all() and cleaned[5:25, 5:25].all()
    single = np.zeros((10, 10), bool)
    single[2:5, 2:5] = True
    assert drop_stray_pieces(single) is single  # one piece, nothing to do


def test_reference_packet_masks_come_out_without_stray_pixels():
    data = packet(40, 60)
    data["masks"][0, 0, :20, :30] = True   # a 600-pixel instance
    data["masks"][0, 0, 39, 59] = True     # a lone pixel at the far corner
    masks = reference.unpack_reference_packet(data, (40, 60, 3), "black_pepper_shaker")[0]
    assert masks[0, 0, :20, :30].all() and not masks[0, 0, 39, 59]
    np.testing.assert_array_equal(masks[1], data["masks"][1])


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
    sent, thresholds = [], [0.5, 0.3]

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
                assert float(request["confidence_threshold"]) == thresholds[len(sent) - 1]
            return Response()

    for color, threshold in zip((31, 97), thresholds):
        frame = Frame("test", color, np.full((4, 6, 3), color, np.uint8), np.eye(3), np.ones((4, 6), np.float32))
        result = asyncio.run(
            run_visual_perception(
                Session(),
                frame,
                np.eye(4),
                "black_pepper_shaker 집어",
                perception_mode="sam3_reference",
                sam3_confidence_threshold=threshold,
            )
        )
        assert result["semantic_source"] == "sam3_reference"
        assert result["objects"]["black_pepper_shaker__instance_1"]["valid_pixels"] == 6
        assert len(result["objects"]) == 2
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
            return dict(camera_from_cad=pose.tolist(), class_id=entry["class_id"])

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
            class_id=entry["class_id"],
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


def several(scores, h=4, w=6):
    n = len(scores)
    masks = np.zeros((n, 1, h, w), bool)
    boxes = np.zeros((n, 4))
    for i in range(n):
        masks[i, 0, :2, : 1 + i % 3] = True
        boxes[i] = [0, 0, 1 + i % 3, 2]
    return dict(
        masks=masks, scores=np.array(scores), boxes_xyxy=boxes,
        metadata=np.array(json.dumps(dict(class_id="black_pepper_shaker", confidence_threshold=0.5, reference_scene="Lab"))),
    )


def test_candidates_follow_the_score_order_up_to_the_trial_limit():
    scores = [0.6, 0.9, 0.85, 0.5, 0.7, 0.65, 0.55, 0.95, 0.52, 0.8, 0.75]
    expected = sorted(range(len(scores)), key=lambda i: -scores[i])[: reference.MAX_REFERENCE_TRIALS]
    result = reference.decode_reference_result(several(scores), (4, 6, 3), "black_pepper_shaker")
    assert len(scores) > reference.MAX_REFERENCE_TRIALS
    assert [c["index"] for c in result["reference_candidates"]] == expected
    assert result["reference_diagnostics"]["trial_indices"] == expected
    assert result["masks"].shape[0] == len(scores)
    assert result["reference_diagnostics"]["scene_indices"] == list(range(len(scores)))
    assert result["bboxes"][7] == result["reference_candidates"][0]["bboxes"][0]


def test_equal_scores_keep_the_server_order_and_trials_are_capped():
    result = reference.decode_reference_result(several([0.8] * 10), (4, 6, 3), "black_pepper_shaker")
    assert [c["index"] for c in result["reference_candidates"]] == list(range(reference.MAX_REFERENCE_TRIALS))
    assert len(result["bboxes"]) == 10


def test_unusable_secondary_candidates_are_skipped_not_fatal():
    value = several([0.9, 0.85, 0.8, 0.7])
    value["masks"][1] = False
    value["boxes_xyxy"][2, 2] = 99
    result = reference.decode_reference_result(value, (4, 6, 3), "black_pepper_shaker")
    assert [c["index"] for c in result["reference_candidates"]] == [0, 3]
    assert [b["instance_index"] for b in result["bboxes"]] == [0, 3]
    np.testing.assert_array_equal(result["masks"], value["masks"][[0, 3]])


def test_switching_candidate_rebuilds_the_object_points_and_selection():
    from tiptop.perception_wrapper import select_reference_candidate

    detection = reference.decode_reference_result(several([0.9, 0.88]), (4, 6, 3), "black_pepper_shaker")
    depth = dict(
        valid_depth_mask=np.ones((4, 6), bool), xyz_map=np.zeros((4, 6, 3)), rgb_map=np.zeros((4, 6, 3)),
    )
    results = {**depth, **detection}
    switched = select_reference_candidate(results, detection["reference_candidates"][1])
    assert switched["reference_diagnostics"]["selected_index"] == 1
    assert switched["objects"]["black_pepper_shaker__instance_1"]["mask_pixels"] == int(several([0.9, 0.88])["masks"][1].sum())
    assert switched["masks"] is results["masks"]
    assert len(switched["objects"]) == 2
    assert switched["objects"]["black_pepper_shaker__instance_0"]["mask_pixels"] == 2
    assert switched["bboxes"] is results["bboxes"]
    assert switched["grounded_atoms"] == [dict(predicate="on", args=["black_pepper_shaker__instance_1", "table"])]
    assert results["grounded_atoms"] == [dict(predicate="on", args=["black_pepper_shaker__instance_0", "table"])]
    assert results["reference_diagnostics"]["selected_index"] == 0


def test_terminal_log_distinguishes_target_trials_from_retained_scene_masks(capsys):
    result = reference.decode_reference_result(several([0.99 - 0.01 * i for i in range(10)]), (4, 6, 3), "black_pepper_shaker")
    reference.log_reference_detection("black_pepper_shaker", result["reference_diagnostics"])
    text = capsys.readouterr().out
    assert "[SAM3]" in text and "검증 대상" in text and "장면 유지 · 파지 검증 생략" in text
    assert "장면 마스크 10개 유지" in text


def test_no_candidate_failure_is_logged_with_the_threshold(capsys):
    empty = several([])
    empty.update(masks=np.zeros((0, 1, 4, 6), bool), boxes_xyxy=np.zeros((0, 4)))
    with pytest.raises(reference.ReferenceDetectionError) as error:
        reference.decode_reference_result(empty, (4, 6, 3), "black_pepper_shaker")
    reference.log_reference_failure("black_pepper_shaker", error.value)
    text = capsys.readouterr().out
    assert "[SAM3]" in text and "점수 > 0.50" in text and "후보가 없습니다" in text



def test_server_value_that_differs_from_the_request_is_flagged(capsys):
    result = reference.decode_reference_result(several([0.9]), (4, 6, 3), "black_pepper_shaker")
    reference.log_reference_detection("black_pepper_shaker", result["reference_diagnostics"], 0.3)
    assert "서버를 재시작" in capsys.readouterr().out
    reference.log_reference_detection("black_pepper_shaker", result["reference_diagnostics"], 0.5)
    assert "서버를 재시작" not in capsys.readouterr().out


@pytest.mark.parametrize("flag, value", [("--sam3-confidence-threshold", "1.5"), ("--sam3-confidence-threshold", "nan")])
def test_sam3_confidence_threshold_option_is_validated(monkeypatch, tmp_path, flag, value):
    from tiptop.scripts import run_known_pick_place as runner

    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path)])
    assert runner.parse_args().sam3_confidence_threshold == reference.DEFAULT_CONFIDENCE_THRESHOLD
    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path), flag, value])
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_candidate_crops_cover_every_detected_candidate_with_its_outcome(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    diagnostics = dict(
        candidate_scores=[0.62, 0.59, 0.55, 0.30, 0.54],
        ranking_scores=[0.62, 0.59, 0.55, 0.30, 0.54],
        candidate_boxes_xyxy=[[10, 10, 40, 40], [50, 10, 80, 40], [90, 10, 120, 40], [10, 60, 40, 90], [50, 60, 80, 90]],
        trial_indices=[0, 1, 2],
    )
    trials = [
        dict(rank=1, candidate_index=0, silhouette_iou=0.66, accepted=False),
        dict(rank=2, candidate_index=1, silhouette_iou=0.81, accepted=True),
    ]
    reference.save_candidate_crops(rgb, diagnostics, trials, tmp_path / "candidates")
    names = sorted(path.name for path in (tmp_path / "candidates").iterdir())
    assert names == [
        "overview.png", "rank01_rejected_score0.620_idx0.png", "rank02_accepted_score0.590_idx1.png",
        "rank03_skipped_score0.550_idx2.png", "rank04_skipped_score0.540_idx4.png", "rank05_skipped_score0.300_idx3.png",
    ]


def test_reference_image_is_copied_with_its_exemplar_boxes_drawn(tmp_path):
    import cv2

    source = tmp_path / "ref" / "obj" / "0000.png"
    source.parent.mkdir(parents=True)
    cv2.imwrite(str(source), np.zeros((60, 80, 3), np.uint8))
    source.with_suffix(".json").write_text(json.dumps(dict(bboxes=[[5, 5, 20, 20], [40, 30, 70, 55]])))
    out = tmp_path / "candidates"
    reference.save_reference_image(dict(reference_image=str(source)), out)
    assert (out / "reference.png").read_bytes() == source.read_bytes()
    drawn = cv2.imread(str(out / "reference_boxes.png"))
    assert drawn.shape == (60, 80, 3) and drawn.any() and not cv2.imread(str(source)).any()


def test_appearance_similarity_favours_the_colour_of_the_reference_exemplars(tmp_path):
    import cv2

    source = tmp_path / "ref" / "0000.png"
    source.parent.mkdir()
    image = np.full((100, 160, 3), 20, np.uint8)
    image[10:40, 10:40] = (30, 200, 40)
    image[50:80, 60:90] = (30, 200, 40)
    cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    source.with_suffix(".json").write_text(json.dumps(dict(bboxes=[[10, 10, 40, 40], [60, 50, 90, 80]])))
    scene = np.full((100, 160, 3), 20, np.uint8)
    scene[10:40, 10:40] = (40, 190, 50)   # same colour family
    scene[10:40, 100:130] = (220, 40, 40)  # a red object
    similarity = reference.appearance_scores(scene, [[10, 10, 40, 40], [100, 10, 130, 40], [0, 60, 10, 70]], source)
    assert similarity[0] > 0.9 and similarity[1] < 0.2 and similarity[2] == 0.0   # the last box is only table


def test_appearance_ranking_reorders_candidates_and_is_recorded():
    value = several([0.60, 0.58, 0.40])
    similarity = np.array([0.1, 0.9, 0.5])
    off = reference.decode_reference_result(value, (4, 6, 3), "black_pepper_shaker")
    on = reference.decode_reference_result(
        value, (4, 6, 3), "black_pepper_shaker", appearance_fn=lambda metadata, boxes: similarity, appearance_weight=1.0
    )
    assert [c["index"] for c in off["reference_candidates"]] == [0, 1, 2]
    assert [c["index"] for c in on["reference_candidates"]] == [1, 2, 0]
    assert on["reference_diagnostics"]["ranking_scores"] == pytest.approx([0.7, 1.48, 0.9])
    assert on["reference_diagnostics"]["appearance_weight"] == 1.0 and on["reference_candidates"][0]["appearance"] == 0.9
    half = reference.decode_reference_result(
        value, (4, 6, 3), "black_pepper_shaker", appearance_fn=lambda metadata, boxes: similarity, appearance_weight=0.0
    )
    assert [c["index"] for c in half["reference_candidates"]] == [0, 1, 2]


def test_appearance_options_are_validated(monkeypatch, tmp_path):
    from tiptop.scripts import run_known_pick_place as runner

    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path)])
    args = runner.parse_args()
    assert args.appearance_ranking is False and args.appearance_weight == 1.0
    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path), "--appearance-ranking", "--appearance-weight", "-1"])
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_terminal_log_reports_the_appearance_ranking_and_whether_it_changed_the_winner(capsys):
    value = several([0.60, 0.58, 0.40])
    metadata = json.loads(str(value["metadata"]))
    metadata.update(reference_box_count=10)
    value["metadata"] = np.array(json.dumps(metadata))
    changed = reference.decode_reference_result(
        value, (4, 6, 3), "black_pepper_shaker", appearance_fn=lambda m, b: np.array([0.1, 0.9, 0.5]), appearance_weight=1.0
    )
    reference.log_reference_detection("black_pepper_shaker", changed["reference_diagnostics"])
    text = capsys.readouterr().out
    assert "외형 유사도 보정 켜짐" in text and "가중치 1" in text and "참조 박스 10개" in text
    assert "순위 변경: SAM3 점수 1위 idx0" in text and "합산 1위 idx1" in text and "합산 1.4800 (SAM3 0.5800 + 외형 0.900)" in text
    same = reference.decode_reference_result(
        value, (4, 6, 3), "black_pepper_shaker", appearance_fn=lambda m, b: np.array([0.9, 0.5, 0.1]), appearance_weight=1.0
    )
    reference.log_reference_detection("black_pepper_shaker", same["reference_diagnostics"])
    assert "1위는 그대로입니다" in capsys.readouterr().out
    off = reference.decode_reference_result(value, (4, 6, 3), "black_pepper_shaker")
    reference.log_reference_detection("black_pepper_shaker", off["reference_diagnostics"])
    assert "외형" not in capsys.readouterr().out


def object_packet(h=4, w=6):
    masks = np.zeros((4, 1, h, w), bool)
    masks[0, 0, :2, :3] = True  # the same pixels as the first target-class instance
    masks[1, 0, 2, 3:5] = True  # a fragment inside the second target-class instance
    masks[2, 0, :2, 4:] = True  # a separate object
    return dict(
        masks=masks,
        scores=np.array([0.7, 0.4, 0.5, 0.9]),
        boxes_xyxy=np.array([[0, 0, 3, 2], [3, 2, 5, 3], [4, 0, 6, 2], [0, 0, 1, 1]]),  # the last mask is empty
        metadata=np.array(json.dumps(dict(class_id="object", confidence_threshold=0.3))),
    )


def test_scene_objects_keep_only_usable_instances_with_generic_labels():
    scene = reference.decode_scene_objects(object_packet(), (4, 6, 3))
    assert [b["label"] for b in scene["bboxes"]] == [f"object__instance_{i}" for i in range(3)]
    assert {b["class_id"] for b in scene["bboxes"]} == {"object"} and len(scene["masks"]) == 3


def test_scene_objects_exclude_instances_already_found_by_the_target_class():
    result = reference.decode_reference_result(packet(), (4, 6, 3), "black_pepper_shaker")
    reference.merge_scene_objects(result, reference.decode_scene_objects(object_packet(), (4, 6, 3)))
    assert [b["label"] for b in result["bboxes"]] == [
        "black_pepper_shaker__instance_0", "black_pepper_shaker__instance_1", "object__instance_2"]
    assert len(result["masks"]) == 3
    assert result["reference_diagnostics"]["scene_objects"] == dict(
        confidence_threshold=0.3, detected=3, added=1, labels=["object__instance_2"])
    assert result["grounded_atoms"] == [dict(predicate="on", args=["black_pepper_shaker__instance_1", "table"])]


@pytest.mark.parametrize("scene_objects, classes, thresholds", [
    (False, ["black_pepper_shaker"], [0.5]),
    (True, ["black_pepper_shaker", "object"], [0.5, reference.OBJECT_CONFIDENCE_THRESHOLD]),
])
def test_scene_objects_are_requested_only_when_asked(scene_objects, classes, thresholds):
    sent = []

    class Response:
        status = 200

        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            stream = io.BytesIO()
            np.savez_compressed(stream, **self.value)
            return stream.getvalue()

    class Session:
        def post(self, url, *, data, timeout):
            with np.load(io.BytesIO(data), allow_pickle=False) as request:
                sent.append((str(request["class_id"]), float(request["confidence_threshold"])))
            return Response(object_packet() if sent[-1][0] == "object" else packet())

    async def detect():
        result = await reference.detect_reference(
            Session(), np.zeros((4, 6, 3), np.uint8), "black_pepper_shaker 집어", server_url="http://sam3",
            reference_scene="Lab/table", scene_objects=scene_objects)
        job = result.pop("scene_objects_job", None)
        assert (job is not None) == scene_objects
        assert len(result["bboxes"]) == 2  # the object query is still running, nothing merged yet
        if job is not None:
            reference.merge_scene_objects(result, await job)
        return result

    result = asyncio.run(detect())
    assert sent == list(zip(classes, thresholds))
    assert len(result["bboxes"]) == (3 if scene_objects else 2)


def test_attached_scene_objects_extend_masks_and_object_points():
    from tiptop.perception_wrapper import attach_scene_objects

    result = reference.decode_reference_result(packet(), (4, 6, 3), "black_pepper_shaker")
    scene = reference.decode_scene_objects(object_packet(), (4, 6, 3))
    depth = dict(valid_depth_mask=np.ones((4, 6), bool), xyz_map=np.zeros((4, 6, 3)), rgb_map=np.zeros((4, 6, 3)))
    result.update(depth, objects={})

    async def run():
        job = asyncio.create_task(asyncio.sleep(0, scene))
        return await attach_scene_objects(result, job)

    merged = asyncio.run(run())
    labels = [box["label"] for box in merged["bboxes"]]
    assert len(labels) == len(merged["masks"]) == 3 and set(merged["objects"]) == set(labels[2:])


def test_scene_object_scores_are_recorded_only_when_asked():
    scene = reference.decode_scene_objects(object_packet(), (4, 6, 3))

    def merged(**options):
        target = reference.decode_reference_result(packet(), (4, 6, 3), "black_pepper_shaker")
        return reference.merge_scene_objects(target, scene, **options)["reference_diagnostics"]["scene_objects"]

    assert "scores" not in merged()
    scores = merged(record_scores=True)["scores"]
    assert list(scores) == [f"object__instance_{i}" for i in range(3)]  # target-class duplicates included
    assert scores["object__instance_2"] == 0.5


def test_scene_object_scores_are_logged_one_per_line(capsys):
    reference.log_scene_object_scores(dict(labels=["object__instance_2"], scores={
        "object__instance_0": 0.5, "object__instance_2": 0.7}))
    lines = [line.split("\x1b[0m ", 1)[1].replace("\x1b[90m", "").replace("\x1b[0m", "") for line in capsys.readouterr().out.splitlines()]
    assert lines == ["물체 1. score 0.70", "물체 2. score 0.50 · 대상 종류와 중복"]
