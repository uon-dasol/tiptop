"""Gemini translation of the in-place preset: prompt selection, rendering and goal check."""

from types import SimpleNamespace

import pytest

from tiptop.perception import gemini
from tiptop.perception.known_objects import registered_target, require_return_in_place


def experimental(monkeypatch, **flags):
    values = dict(pick_place_next_to=False, return_in_place=False) | flags
    monkeypatch.setattr(gemini, "tiptop_cfg", lambda: SimpleNamespace(experimental=SimpleNamespace(**values)))


@pytest.mark.parametrize(
    "flags, name",
    [
        ({}, "detect_and_translate"),
        ({"pick_place_next_to": True}, "detect_and_translate_pick_place_next_to"),
        ({"return_in_place": True}, "detect_and_translate_return_in_place"),
    ],
)
def test_prompt_template_follows_experimental_flags(monkeypatch, flags, name):
    experimental(monkeypatch, **flags)
    assert gemini._detect_and_translate_prompt_name() == name


def test_both_flags_are_rejected(monkeypatch):
    experimental(monkeypatch, pick_place_next_to=True, return_in_place=True)
    with pytest.raises(ValueError, match="different prompts"):
        gemini._detect_and_translate_prompt_name()


def test_preset_template_renders_and_keeps_the_original_predicates():
    original = gemini.load_prompt("detect_and_translate")
    preset = gemini.load_prompt("detect_and_translate_return_in_place")
    assert "return_in_place" not in original
    rendered = preset.format(task_instruction="컵 집고 제자리에 놓아")
    assert 'task instruction: "컵 집고 제자리에 놓아".' in rendered
    assert '"name": "return_in_place", "args": ["white_cup"]' in rendered
    for line in ("- on(movable, surface)", "- holding(movable)"):
        assert line in rendered


def test_translated_preset_atom_passes_through_the_parser():
    bboxes, atoms = gemini._parse_response(
        '{"bboxes": [{"box_2d": [1, 2, 3, 4], "label": "white_cup"}],'
        ' "predicates": [{"name": "return_in_place", "args": ["white_cup"]}]}'
    )
    assert atoms == [dict(predicate="return_in_place", args=["white_cup"])]
    require_return_in_place(atoms)


@pytest.mark.parametrize(
    "atoms",
    [
        [],
        [dict(predicate="on", args=["cup", "plate"])],
        [dict(predicate="holding", args=["cup"])],
        [dict(predicate="return_in_place", args=["cup", "plate"])],
        [dict(predicate="return_in_place", args=["cup"]), dict(predicate="return_in_place", args=["mug"])],
        [dict(predicate="return_in_place", args=["cup"]), dict(predicate="on", args=["cup", "plate"])],
    ],
)
def test_in_place_runner_rejects_every_other_goal(atoms):
    with pytest.raises(ValueError, match="supports only return_in_place"):
        require_return_in_place(atoms)


def test_preset_atom_resolves_the_catalog_target_by_its_first_argument():
    atoms = [dict(predicate="return_in_place", args=["pepper_shaker"])]
    label, entry = registered_target([dict(label="pepper_shaker"), dict(label="orange_bottle")], atoms)
    assert label == "pepper_shaker"
    assert entry["class_id"] == "black_pepper_shaker"
