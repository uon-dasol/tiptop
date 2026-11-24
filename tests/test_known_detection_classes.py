"""Gemini names objects freely and gives each one a separate catalog class, so duplicates stay distinguishable."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tiptop.config import tiptop_cfg
from tiptop.perception import sam2
from tiptop.perception.known_objects import (
    catalog_class_id, known_object_detection_context, normalize_known_detections, registered_target,
)
from tiptop.perception_wrapper import detect_and_segment

SPAM = "spam_classic_20G_can"


def entries():
    from tiptop.object_catalog import load_catalog

    return load_catalog()["objects"]


def box(label, y, **extra):
    return dict(label=label, box_2d=[y, 100, y + 100, 200], **extra)


def test_instances_with_their_own_names_resolve_through_the_separate_class_field():
    boxes = [box("closer", 600, class_id=SPAM), box("further", 300, class_id=SPAM), box("mug", 50, class_id=None)]
    label, entry = registered_target(boxes, [dict(predicate="holding", args=["closer"])])
    assert (label, entry["class_id"]) == ("closer", SPAM)
    label, _ = registered_target(boxes, [dict(predicate="holding", args=["further"])])
    assert label == "further"


def test_the_goal_must_name_exactly_one_catalog_instance():
    boxes = [box("closer", 600, class_id=SPAM), box("further", 300, class_id=SPAM)]
    both = [dict(predicate="holding", args=["closer"]), dict(predicate="holding", args=["further"])]
    for atoms in (both, [dict(predicate="holding", args=["mug"])], []):
        with pytest.raises(ValueError, match="uniquely identified catalog target"):
            registered_target(boxes, atoms)
    with pytest.raises(ValueError, match="uniquely identified catalog target"):
        registered_target([box("closer", 600, class_id=None)], [dict(predicate="holding", args=["closer"])])


def test_a_label_that_only_contains_the_class_name_is_still_not_a_class():
    # What Gemini did before the separate field existed; labels are never matched by similarity.
    assert catalog_class_id(box("closer_" + SPAM, 600), entries()) is None
    assert catalog_class_id(box(SPAM, 600), entries()) == SPAM


def test_detections_get_canonical_classes_and_unknown_ones_become_none(caplog):
    normalized = normalize_known_detections([
        box("a", 100, class_id="Spam Classic 20G Can"),   # catalog alias, other spelling
        box("b", 200, class_id="no_such_object"),
        box("c", 300, class_id=None),
        box(SPAM, 400),                                   # legacy answer: the label is the class
        box("d", 500),                                    # legacy answer, not a catalog name
    ])
    assert [b.get("class_id", "absent") for b in normalized] == [SPAM, None, None, SPAM, "absent"]
    assert "no_such_object" in caplog.text


def test_the_prompt_forces_unique_labels_and_a_class_id_from_the_allowed_list():
    context = known_object_detection_context()
    assert '"class_id": "an allowed class_id or null"' in context
    assert "MANDATORY" in context and "MUST NEVER" in context and "BEFORE YOU ANSWER" in context
    assert "'closer' and 'further'" in context and "NEVER in class_id" in context
    assert "as the label" not in context
    allowed = context[context.index("ALLOWED VALUES: "):context.index("\n", context.index("ALLOWED VALUES: "))]
    for entry in entries():  # every catalog id is offered verbatim as an allowed value
        assert json.dumps(entry["class_id"]) in allowed
    assert context.count("ALLOWED VALUES") == 3  # named in rule 1, listed once and checked before answering


def test_the_destination_rule_is_opt_in_and_asks_for_the_usable_part_only():
    assert "DESTINATION RULE" not in known_object_detection_context()  # fixed destinations stay unchanged
    context = known_object_detection_context(perceived_destination=True)
    rule = context[context.index("6. DESTINATION RULE"):context.index("BEFORE YOU ANSWER")]
    assert "not a box or a flat surface" in rule and "actual shape" in rule
    assert "inside of a bowl" in rule and "never its handle" in rule
    assert "'<destination>_inside'" in rule and "class_id null" in rule
    assert "MUST NOT also detect the whole destination" in rule
    assert "box or a flat surface, detect it normally" in rule
    # Everything before the rule is identical, so the class_id rules apply in both cases.
    assert context.replace(rule, "") == known_object_detection_context()


def test_the_base_template_stays_free_of_destination_handling():
    from tiptop.perception.gemini import load_prompt

    prompt = load_prompt("detect_and_translate").format(task_instruction="put the cap in the bowl")
    assert "only the part" not in prompt.lower() and "_inside" not in prompt


@pytest.mark.parametrize("pick_only, expects_rule", [(True, False), (False, True)])
def test_only_tasks_with_a_perceived_destination_get_the_destination_rule(monkeypatch, tmp_path, pick_only, expects_rule):
    from tiptop import tiptop_run

    seen = {}

    class Captured(Exception):
        pass

    async def capture(*args, **kwargs):
        seen.update(kwargs)
        raise Captured

    monkeypatch.setattr(tiptop_run, "run_visual_perception", capture)
    observation = SimpleNamespace(frame=SimpleNamespace(rgb=np.zeros((4, 4, 3), np.uint8)), world_from_cam=np.eye(4))
    with pytest.raises(Captured):
        asyncio.run(tiptop_run.run_perception(
            None, observation, "x", tmp_path, include_workspace=False, log_to_rerun=False,
            grasp_provider="known_db", pick_only=pick_only,
        ))
    assert ("DESTINATION RULE" in seen["object_context"]) is expects_rule
    assert "ALLOWED VALUES" in seen["object_context"]


def test_gemini_instances_reach_segmentation_with_their_classes(monkeypatch):
    monkeypatch.setattr(tiptop_cfg().perception.sam, "mode", "local")
    monkeypatch.setattr(sam2, "download_sam2_checkpoint", lambda: "unused")
    monkeypatch.setattr(sam2, "_segment_local", lambda image, boxes, checkpoint: (np.ones((2, 6, 8), bool), np.ones(2)))
    payload = dict(
        bboxes=[box("closer to robot", 600, class_id=SPAM), box("further from robot", 300, class_id=SPAM)],
        predicates=[dict(name="holding", args=["closer to robot"])],
    )

    async def generate(**kwargs):
        return SimpleNamespace(text=json.dumps(payload))

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    result = asyncio.run(detect_and_segment(
        np.zeros((6, 8, 3), np.uint8), "로봇과 가까이에 있는 스팸 집어", client=client,
        object_context=known_object_detection_context(),
    ))
    assert [b["label"] for b in result["bboxes"]] == ["closer_to_robot", "further_from_robot"]
    assert [b["class_id"] for b in result["bboxes"]] == [SPAM, SPAM]
    assert result["grounded_atoms"] == [dict(predicate="holding", args=["closer_to_robot"])]
    label, entry = registered_target(result["bboxes"], result["grounded_atoms"])
    assert (label, entry["class_id"]) == ("closer_to_robot", SPAM)


def test_the_prompt_asks_for_a_korean_display_name_and_lists_the_catalog_names():
    context = known_object_detection_context()
    assert '"name_ko": "Korean display name"' in context
    assert "5. NAME RULE" in context and "Hangul" in context and "copy its name_ko from the catalog" in context
    assert '"name_ko": "스팸"' in context  # catalog rows carry the Korean names to copy
    assert "NAME RULE" in known_object_detection_context(perceived_destination=True)


def test_display_names_are_kept_when_usable_and_dropped_otherwise():
    normalized = normalize_known_detections([
        box("a", 100, class_id=SPAM, name_ko="  스팸  "),
        box("b", 200, class_id=None, name_ko="바가지 안쪽"),
        box("c", 300, class_id=None, name_ko=""),
        box("d", 400, class_id=None, name_ko=7),
        box("e", 500, class_id=None),
    ])
    assert [b.get("name_ko", "absent") for b in normalized] == ["스팸", "바가지 안쪽", "absent", "absent", "absent"]
