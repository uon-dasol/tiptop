"""Sequence readiness, strict preparation failures and retention beyond old LRU limits."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from tiptop.object_catalog import sequence_objects
from tiptop.perception import sequence


def objects(tmp_path, count=2):
    rows = []
    for index in range(count):
        name = f"object_{index}"
        cad, db = tmp_path / (name+".obj"), tmp_path / (name+".json")
        trimesh.creation.box().export(cad)
        db.write_text(json.dumps(dict(metadata=dict(object=name, coordinate_frame="object_zero_pose",
            gripper_model="UON_3finger_gripper"), data=[dict(grasp_mat=np.eye(4).tolist(), score=.9,
                scene_id="scene", source_pregrasp_index=0)])))
        rows.append(dict(class_id=name, cad=cad, db=db, cad_scale_to_m=.001, aliases=[]))
    return rows


def test_local_preparation_retains_more_than_four_databases_and_eight_cads(tmp_path, monkeypatch, capsys):
    from tiptop.doosan import grasp_database
    from tiptop.perception import cad_cache

    selected = objects(tmp_path, 9)
    reads = []
    original_mesh, original_json = cad_cache.trimesh.load, grasp_database.json.loads
    monkeypatch.setattr(cad_cache.trimesh, "load", lambda *a, **kw: reads.append("cad") or original_mesh(*a, **kw))
    monkeypatch.setattr(grasp_database.json, "loads", lambda *a, **kw: reads.append("db") or original_json(*a, **kw))
    report = sequence.prepare_local_assets(selected)
    assert report["cache_capacity"]["cad"] >= 9 and report["cache_capacity"]["db"] >= 9
    assert reads.count("cad") == reads.count("db") == 9
    output = capsys.readouterr().out
    assert '\033[33mLOCAL_PREPARE 1/9 object_0 준비 완료' in output
    assert '\033[33mLOCAL_PREPARE 9/9 object_8 준비 완료' in output
    before = list(reads)
    sequence.prepare_local_assets(list(reversed(selected)))
    assert reads == before
    wrong = json.loads(selected[0]["db"].read_text())
    wrong["metadata"]["object"] = "different"
    selected[0]["db"].write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="DB object"):
        sequence.prepare_local_assets(selected)
    assert '\033[31mLOCAL_PREPARE_FAILED 1/9 object_0' in capsys.readouterr().out


def test_list_resolution_is_ordered_deduplicated_and_all_is_explicit(tmp_path):
    selected = objects(tmp_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(objects=[{**r, "cad": str(r["cad"]), "db": str(r["db"])} for r in selected])))
    assert [r["class_id"] for r in sequence_objects(["object_1", "object_0", "object_1"], catalog_path=catalog)] == ["object_1", "object_0"]
    assert len(sequence_objects(["all"], catalog_path=catalog)) == 2
    for invalid in ([], ["all", "object_0"], ["unknown"]):
        with pytest.raises(ValueError):
            sequence_objects(invalid, catalog_path=catalog)


class Response:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError("HTTP preparation failure")

    async def json(self):
        return self.data


class Session:
    def __init__(self, failure=None):
        self.calls, self.failure = [], failure

    def get(self, url, **kwargs):
        self.calls.append(("GET", url))
        if url.endswith("/health"):
            return Response(dict(preparation_supported=self.failure != "old_server"))
        assert url == "http://sam3/reference"
        p = kwargs["params"]
        return Response(dict(success=True, class_id=p["class_id"], reference_scene=p["scene"],
                             inference_performed=self.failure == "inference"))

    def post(self, url, *, json, **kwargs):
        self.calls.append(("POST", url))
        assert url == "http://pose/prepare"
        rows = [dict(class_id=r["class_id"], success=True, cache_hit=True) for r in json["objects"]]
        if self.failure == "missing_class":
            rows.pop()
        return Response(dict(success=self.failure != "load", inference_performed=False, objects=rows),
                        400 if self.failure == "load" else 200)


@pytest.mark.parametrize("failure", [None, "old_server", "missing_class", "load", "inference"])
def test_sequence_only_prepares_assets_and_references_and_never_ignores_failure(tmp_path, monkeypatch, failure):
    selected, local = objects(tmp_path), []
    monkeypatch.setattr(sequence, "sequence_objects", lambda names: selected)
    monkeypatch.setattr(sequence, "prepare_local_assets", lambda values: local.append(values) or dict(success=True))
    session = Session(failure)
    request = sequence.prepare_sequence(session, ["object_0", "object_1"], pose_url="http://pose",
                                        sam3_url="http://sam3", reference_scene="Lab/table")
    if failure:
        with pytest.raises(sequence.SequencePreparationError) as captured:
            asyncio.run(request)
        report = captured.value.report
        assert report["success"] is False and report["stage"] != "ready"
    else:
        report = asyncio.run(request)
        assert report["success"] and report["stage"] == "ready" and len(report["references"]) == 2
    assert report["inference_performed"] is False and "elapsed_s" in report
    assert all(url.endswith(("/health", "/prepare", "/reference")) for method, url in session.calls)
    if failure == "old_server":
        assert not local and len(session.calls) == 1


def test_prepare_failure_prevents_interactive_input_and_task_execution(tmp_path, monkeypatch):
    from tiptop.scripts import run_known_pick_place as runner

    async def fail(args, output):
        raise sequence.SequencePreparationError("failed CAD", dict(success=False))

    def no_instruction(prompt):
        pytest.fail("Task input must not start until preparation succeeds")

    monkeypatch.setattr(runner, "prepare_before_tasks", fail)
    args = SimpleNamespace(output=tmp_path, perception_mode="sam3_reference")
    result = asyncio.run(runner.run_interactive(args, read_instruction=no_instruction))
    assert result["success"] is False and result["executed"] is False


def test_preparation_failure_stops_before_run_and_is_recorded(tmp_path, monkeypatch):
    from tiptop.scripts import run_known_pick_place as runner

    async def fail(args, output):
        raise sequence.SequencePreparationError("failed CAD", dict(success=False))

    async def forbidden(args):
        pytest.fail("Perception/robot path must not start on preparation failure")

    monkeypatch.setattr(runner, "prepare_before_tasks", fail)
    monkeypatch.setattr(runner, "run", forbidden)
    args = SimpleNamespace(output=tmp_path, perception_mode="sam3_reference", preload_objects=["object_0"], task="test")
    result = asyncio.run(runner.run_task(args))
    assert result["failed_stage"] == "sequence_preparation"
    assert json.loads((tmp_path / "task-result.json").read_text())["success"] is False


def test_prepare_only_cannot_be_combined_with_robot_execution(monkeypatch):
    import sys
    from tiptop.scripts import run_known_pick_place as runner

    monkeypatch.setattr(sys, "argv", ["run", "--prepare-only", "--preload-objects", "all", "--execute"])
    with pytest.raises(SystemExit) as captured:
        runner.main()
    assert captured.value.code == 2


def test_interactive_sequence_prepares_once_and_keeps_caches_for_later_tasks(tmp_path, monkeypatch):
    from tiptop.scripts import run_known_pick_place as runner

    prepares, runs = [], []

    async def prepare(session, class_ids, **kwargs):
        prepares.append(list(class_ids))
        return dict(success=True, class_ids=class_ids, elapsed_s=.1, inference_performed=False)

    async def task(args):
        assert args._sequence_prepared
        runs.append(args.task)
        return dict(status="planned", success=None, executed=False)

    monkeypatch.setattr(sequence, "prepare_sequence", prepare)
    monkeypatch.setattr(runner, "run", task)
    args = SimpleNamespace(output=tmp_path, perception_mode="sam3_reference", preload_objects=["one", "two"],
                           pose_url="http://pose", sam3_url="http://sam3", reference_scene="Lab/table")
    instructions = iter(["one 집고 제자리에 놓아", "two 집고 제자리에 놓아", "/quit"])
    asyncio.run(runner.run_interactive(args, read_instruction=lambda _: next(instructions)))
    assert prepares == [["one", "two"]] and len(runs) == 2
    assert json.loads((tmp_path / "sequence-preparation.json").read_text())["success"]


def test_prepare_only_never_enters_observation_or_execution_path(tmp_path, monkeypatch):
    from tiptop.scripts import run_known_pick_place as runner

    async def prepare(session, class_ids, **kwargs):
        return dict(success=True, class_ids=class_ids, elapsed_s=.1, inference_performed=False)

    async def forbidden(args):
        pytest.fail("Preparation only must not call the task pipeline")

    monkeypatch.setattr(sequence, "prepare_sequence", prepare)
    monkeypatch.setattr(runner, "run", forbidden)
    args = SimpleNamespace(output=tmp_path, perception_mode="sam3_reference", preload_objects=["one"],
                           pose_url="http://pose", sam3_url="http://sam3", reference_scene="Lab/table")
    result = asyncio.run(runner.run_preparation_only(args))
    assert result["success"] and not result["executed"]
    assert (tmp_path / "sequence-preparation.json").exists()
    assert not (tmp_path / "observation.npz").exists()
