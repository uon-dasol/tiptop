"""DB-only discovery across datasets does not require perception metadata or CAD."""

import json

import numpy as np
import pytest

from tiptop.doosan.grasp_database import load_database_grasps, warm_database_grasps
from tiptop.object_catalog import database_entries, registered_database, registered_object


def write_database(root, name, *, v2=False):
    suffix = "_UON_3finger_gripper" if v2 else ""
    path = root / name / "UON_3finger_gripper" / f"{name}{suffix}_merged_grasp_zero_pose.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    index_key = "source_output_index" if v2 else "source_pregrasp_index"
    rows = [dict(grasp_mat=np.eye(4).tolist(), score=score, scene_id=f"Home/table/{index:04}",
                 **{index_key: index}) for index, score in enumerate([0.6, 0.9])]
    metadata = dict(object=name, coordinate_frame="object_zero_pose", gripper_model="UON_3finger_gripper")
    path.write_text(json.dumps(dict(metadata=metadata, data=rows)))
    return path


def write_catalog(tmp_path, root, extra_roots, objects):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(dict(db_root=str(root), additional_db_roots=[str(p) for p in extra_roots],
                                    objects=objects)))
    return path


def test_both_datasets_are_available_without_cad_or_csv(tmp_path):
    primary, extra = tmp_path / "original", tmp_path / "new"
    old_db = write_database(primary, "pepper")
    new_db = write_database(extra, "chair_leg_caps", v2=True)
    catalog = write_catalog(tmp_path, primary, [extra], [dict(class_id="pepper", db=str(old_db))])
    entries = database_entries(catalog_path=catalog)
    assert entries == [dict(class_id="pepper", db=old_db), dict(class_id="chair_leg_caps", db=new_db)]
    entry = registered_database("chair_leg_caps", catalog_path=catalog)
    warm_database_grasps(entry["db"], class_id=entry["class_id"])
    result = load_database_grasps(
        entry["db"], class_id=entry["class_id"], world_from_cad=np.eye(4),
        pose_source="rgbd_pose_estimator", world_from_planner_object=np.eye(4), grasp_from_tcp=np.eye(4),
    )
    assert result["candidate_indices"].tolist() == [1, 0]
    assert result["source_scene_ids"] == ["Home/table/0001", "Home/table/0000"]
    assert result["source_output_indices"] == [1, 0]
    assert result["source_pregrasp_indices"] == [None, None]
    np.testing.assert_allclose(result["world_from_tcp"], np.repeat(np.eye(4)[None], 2, axis=0))
    # A discovered DB does not silently invent a perception/CAD registration.
    with pytest.raises(ValueError, match="Unregistered class"):
        registered_object("chair_leg_caps", catalog_path=catalog)


def test_explicit_db_and_first_extra_root_win_without_loading_review_files(tmp_path):
    primary, extra, later = (tmp_path / name for name in ("original", "new", "later"))
    old_db = write_database(primary, "pepper")
    write_database(extra, "pepper", v2=True)
    new_db = write_database(extra, "cup")
    write_database(later, "cup", v2=True)
    (new_db.parent / "cup_grasp_review.json").write_text("not a database")
    review_only = extra / "review_only" / "UON_3finger_gripper"
    review_only.mkdir(parents=True)
    (review_only / "review_only_grasp_review.json").write_text("not a database")
    catalog = write_catalog(tmp_path, primary, [extra, later], [dict(class_id="pepper", db=str(old_db))])
    assert database_entries(catalog_path=catalog) == [dict(class_id="pepper", db=old_db),
                                                     dict(class_id="cup", db=new_db)]


def test_optional_root_can_be_absent_but_missing_explicit_db_still_fails(tmp_path):
    missing = tmp_path / "missing.json"
    catalog = write_catalog(tmp_path, tmp_path, [tmp_path / "unmounted"],
                            [dict(class_id="pepper", db=str(missing))])
    assert registered_database("pepper", catalog_path=catalog, check_files=False)["db"] == missing
    with pytest.raises(FileNotFoundError):
        registered_database("pepper", catalog_path=catalog)
    with pytest.raises(ValueError, match="Unregistered grasp database"):
        registered_database("unknown", catalog_path=catalog)


def v2_document(offset=0.237, **metadata):
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_euler("xyz", [[10, 20, 30], [90, 0, 45]], degrees=True).as_matrix()
    bases = np.repeat(np.eye(4)[None], 2, axis=0)
    bases[:, :3, :3] = rotations
    bases[:, :3, 3] = [[0.1, -0.2, 0.76], [0.0, 0.3, 0.5]]
    targets = bases[:, :3, 3] + rotations[:, :, 2] * offset
    rows = [dict(grasp_mat=base.tolist(), target_points=target.tolist(), score=0.5, scene_id=str(index),
                 source_output_index=index) for index, (base, target) in enumerate(zip(bases, targets))]
    meta = dict(object="obj", coordinate_frame="object_zero_pose", gripper_model="UON_3finger_gripper",
                schema="dataset_v2_object_grasp_compact", schema_version=1) | metadata
    return dict(metadata=meta, data=rows), bases, targets


def test_v2_grasp_origin_moves_to_target_along_recorded_local_z():
    from tiptop.doosan.grasp_database import database_grasp_matrices

    document, bases, targets = v2_document()
    grasps = database_grasp_matrices(document)
    np.testing.assert_allclose(grasps[:, :3, 3], targets, atol=1e-12)
    np.testing.assert_array_equal(grasps[:, :3, :3], bases[:, :3, :3])
    assert document["data"][0]["grasp_mat"] == bases[0].tolist()


def test_single_object_grasps_are_unchanged():
    from tiptop.doosan.grasp_database import database_grasp_matrices

    document, bases, _ = v2_document()
    document["metadata"] = dict(object="obj", coordinate_frame="object_zero_pose",
                                gripper_model="UON_3finger_gripper", schema="single_object_grasp_compact")
    np.testing.assert_array_equal(database_grasp_matrices(document), bases)


@pytest.mark.parametrize("document", [v2_document(offset=0.2)[0], v2_document(schema_version=2)[0],
                                      v2_document(gripper_model="other")[0]])
def test_v2_rejects_unexpected_offset_or_schema(document):
    from tiptop.doosan.grasp_database import database_grasp_matrices

    with pytest.raises(ValueError):
        database_grasp_matrices(document)


def test_v2_loader_places_tcp_at_target(tmp_path):
    document, _, targets = v2_document()
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(document))
    result = load_database_grasps(path, class_id="obj", world_from_cad=np.eye(4), pose_source="rgbd_pose_estimator",
                                  world_from_planner_object=np.eye(4), grasp_from_tcp=np.eye(4))
    np.testing.assert_allclose(result["world_from_tcp"][:, :3, 3], targets[result["candidate_indices"]], atol=1e-9)
    assert result["source_output_indices"] == [int(i) for i in result["candidate_indices"]]
