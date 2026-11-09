"""CAD and grasp-DB loads are read once per file version and never shared mutably."""

import json
import os

import numpy as np
import trimesh

from tiptop.doosan import grasp_database
from tiptop.doosan.grasp_database import load_database_grasps, warm_database_grasps
from tiptop.perception import cad_cache
from tiptop.perception.cad_cache import load_cad_mesh


def write_db(path, scores):
    rows = [
        dict(grasp_mat=np.eye(4).tolist(), score=score, scene_id=f"scene{index}", source_pregrasp_index=index)
        for index, score in enumerate(scores)
    ]
    metadata = dict(object="pepper", coordinate_frame="object_zero_pose", gripper_model="UON_3finger_gripper")
    path.write_text(json.dumps(dict(metadata=metadata, data=rows)))


def load(path):
    return load_database_grasps(
        path,
        class_id="pepper",
        world_from_cad=np.eye(4),
        pose_source="rgbd_pose_estimator",
        world_from_planner_object=np.eye(4),
        grasp_from_tcp=np.eye(4),
    )


def test_cad_is_read_once_scaled_and_copied_per_caller(tmp_path, monkeypatch):
    path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(10, 20, 30)).export(path)
    reads = []
    original = trimesh.load
    monkeypatch.setattr(cad_cache.trimesh, "load", lambda *args, **kwargs: reads.append(args) or original(*args, **kwargs))
    cad_cache._scaled_cad.cache_clear()

    first, second = load_cad_mesh(path, 0.001), load_cad_mesh(path, 0.001)
    assert len(reads) == 1
    np.testing.assert_allclose(first.bounds, [[-0.005, -0.01, -0.015], [0.005, 0.01, 0.015]])
    first.vertices *= 2
    np.testing.assert_allclose(second.bounds, [[-0.005, -0.01, -0.015], [0.005, 0.01, 0.015]])
    assert load_cad_mesh(path, 0.001).bounds[1, 0] == second.bounds[1, 0]
    load_cad_mesh(path, 1.0)
    assert len(reads) == 2


def test_cad_reloads_after_the_file_changes(tmp_path):
    path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(path)
    cad_cache._scaled_cad.cache_clear()
    assert np.isclose(load_cad_mesh(path, 1.0).extents[0], 1.0)
    trimesh.creation.box(extents=(2, 1, 1)).export(path)
    os.utime(path, ns=(1, 1))
    assert np.isclose(load_cad_mesh(path, 1.0).extents[0], 2.0)


def test_database_is_parsed_once_and_results_are_independent(tmp_path, monkeypatch):
    path = tmp_path / "db.json"
    write_db(path, [0.7, 0.9, 0.8])
    grasp_database._document.cache_clear()
    grasp_database._ranked_candidates.cache_clear()
    parsed = []
    original = json.loads
    monkeypatch.setattr(grasp_database.json, "loads", lambda text: parsed.append(1) or original(text))

    warm_database_grasps(path)
    first, second = load(path), load(path)
    assert len(parsed) == 1
    assert first["candidate_indices"].tolist() == [1, 2, 0]
    assert first["source_scene_ids"] == ["scene1", "scene2", "scene0"]
    first["scores"][:] = 0
    first["candidate_indices"][:] = 0
    first["source_scene_ids"].clear()
    assert second["scores"].tolist() == np.float32([0.9, 0.8, 0.7]).tolist()
    assert load(path)["candidate_indices"].tolist() == [1, 2, 0]
    assert load(path)["source_scene_ids"] == ["scene1", "scene2", "scene0"]


def test_database_reloads_after_the_file_changes(tmp_path):
    path = tmp_path / "db.json"
    write_db(path, [0.1, 0.2])
    assert load(path)["candidate_indices"].tolist() == [1, 0]
    write_db(path, [0.5, 0.2, 0.9])
    os.utime(path, ns=(1, 1))
    assert load(path)["candidate_indices"].tolist() == [2, 0, 1]
