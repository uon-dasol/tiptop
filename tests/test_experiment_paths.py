"""Moving evidence must preserve its contents and keep new outputs separate."""

import json

import pytest

from experiments.grasping import paths


def test_relocated_inputs_preserve_evidence_and_external_asset_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "archive")
    paths.DATA_ROOT.mkdir()
    old = "/old-checkout/simulation/isaac_sim/experiments/nextstage-20261020"
    external = "/media/uon/data1/pepper.obj"
    document = {
        "candidate_source_path": old + "/plans/example/candidate-ik.json",
        "nested": [{old + "/runs/a/pick.json": external}],
        "unrelated": old + "-unrelated/file.json",
    }
    evidence = paths.DATA_ROOT / "pick.json"
    evidence.write_text(json.dumps(document))
    original = evidence.read_bytes()

    restored = paths.read_json(old + "/pick.json")

    assert restored["candidate_source_path"] == str(paths.DATA_ROOT / "plans/example/candidate-ik.json")
    assert restored["nested"] == [{str(paths.DATA_ROOT / "runs/a/pick.json"): external}]
    assert restored["unrelated"] == document["unrelated"]
    assert evidence.read_bytes() == original


@pytest.mark.parametrize("suffix", ["", "new-analysis"])
def test_work_directory_cannot_overwrite_archive(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "archive")
    monkeypatch.setattr(paths, "WORK_ROOT", paths.DATA_ROOT / suffix)
    with pytest.raises(ValueError, match="outside the input archive"):
        paths.ensure_workdirs()
    assert not paths.DATA_ROOT.exists()
