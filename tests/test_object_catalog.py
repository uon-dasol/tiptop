"""Data-root routing must agree between the client and the pose server."""

import json

import pytest

from tiptop.object_catalog import load_catalog, registered_object, sequence_objects
from tiptop.scripts import foundation_pose_server


def write_catalog(tmp_path, **settings):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(settings))
    return path


def test_separate_roots_agree_with_foundationpose_boundary(tmp_path, monkeypatch):
    cad_root, db_root = tmp_path / "models", tmp_path / "grasps"
    mesh = cad_root / "cup/edited/cup.obj"
    db = db_root / "cup/UON_3finger_gripper/cup_merged_grasp_zero_pose.json"
    for path in (mesh, db):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    mesh.write_text("usemtl paint\n")
    mesh.with_suffix(".mtl").write_text(
        "newmtl paint\nmap_Kd /old/export/cup.bmp\nnewmtl unused\nmap_Kd deleted.webp\n"
    )
    (mesh.parent.parent / "cup.bmp").write_bytes(b"texture")
    catalog = write_catalog(
        tmp_path,
        cad_root=str(cad_root),
        db_root=str(db_root),
        objects=[
            dict(class_id="cup", cad="cup/edited/cup.obj", db="cup/UON_3finger_gripper/cup_merged_grasp_zero_pose.json")
        ],
    )
    entry = registered_object("cup", catalog_path=catalog)
    assert entry["cad"] == mesh and entry["db"] == db
    monkeypatch.setattr(foundation_pose_server, "CATALOG", catalog)
    assert foundation_pose_server.default_asset_root() == cad_root
    assert all(level == "ok" for level, _ in foundation_pose_server.cad_findings(cad_root))
    assert any("outside the asset root" in message for _, message in foundation_pose_server.cad_findings(db_root))


@pytest.mark.parametrize("absolute", [False, True])
def test_existing_catalog_paths_remain_compatible(tmp_path, absolute):
    mesh, db = tmp_path / "mesh.obj", tmp_path / "db.json"
    mesh.touch()
    db.touch()
    catalog = write_catalog(
        tmp_path,
        objects=[dict(class_id="cup", cad=str(mesh) if absolute else mesh.name, db=str(db) if absolute else db.name)],
    )
    entry = registered_object("cup", catalog_path=catalog, asset_root=tmp_path)
    assert entry["cad"] == mesh and entry["db"] == db


def test_missing_external_file_never_uses_a_project_copy(tmp_path):
    (tmp_path / "mesh.obj").touch()
    (tmp_path / "db.json").touch()
    missing_root = tmp_path / "unmounted"
    catalog = write_catalog(
        tmp_path,
        cad_root=str(missing_root),
        db_root=str(missing_root),
        objects=[dict(class_id="cup", cad="mesh.obj", db="db.json")],
    )
    with pytest.raises(FileNotFoundError) as error:
        registered_object("cup", catalog_path=catalog, asset_root=tmp_path)
    assert error.value.args[0] == missing_root / "mesh.obj"
    # CLI defaults can be constructed before --cad/--data overrides are parsed.
    assert registered_object("cup", catalog_path=catalog, check_files=False)["cad"] == missing_root / "mesh.obj"


def test_duplicate_class_cannot_select_a_different_database(tmp_path):
    catalog = write_catalog(
        tmp_path,
        objects=[
            dict(class_id="cup", cad="one.obj", db="one.json"),
            dict(class_id="cup", cad="two.obj", db="two.json"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        registered_object("cup", catalog_path=catalog, check_files=False)


def bundled_assets(root, name):
    mesh = root / name / "edited" / f"{name}.obj"
    db = root / name / "UON_3finger_gripper" / f"{name}_UON_3finger_gripper_merged_grasp_zero_pose.json"
    for path in (mesh, db):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    mesh.with_suffix(".mtl").touch()
    return mesh, db


def test_bundled_models_use_their_dataset_and_preserve_explicit_objects(tmp_path, monkeypatch):
    original, extra = tmp_path / "original", tmp_path / "new"
    old_mesh, old_db = bundled_assets(original, "cup")
    bundled_assets(extra, "cup")  # A same-named discovered object cannot replace the explicit one.
    mesh, db = bundled_assets(extra, "chair_leg_caps")
    catalog = write_catalog(tmp_path, cad_root=str(original), db_root=str(original),
        additional_datasets=[dict(root=str(extra), cad_scale_to_m=.001)],
        objects=[dict(class_id="cup", cad=str(old_mesh), db=str(old_db), cad_scale_to_m=.001)])

    old = registered_object("cup", catalog_path=catalog)
    assert old["cad"] == old_mesh and old["db"] == old_db
    new = registered_object("chair_leg_caps", catalog_path=catalog)
    assert new["cad"] == mesh and new["db"] == db and new["cad_scale_to_m"] == .001
    assert new["aliases"] == ["chair leg caps"]
    assert len(sequence_objects(["all"], catalog_path=catalog)) == 2
    monkeypatch.setattr(foundation_pose_server, "CATALOG", catalog)
    assert foundation_pose_server.default_asset_root() == original
    assert foundation_pose_server.additional_asset_roots() == [extra]
    assert all(level == "ok" for level, _ in foundation_pose_server.cad_findings(original))
    preloaded = foundation_pose_server.resolve_preload_objects(["chair_leg_caps"])
    assert preloaded == [dict(class_id="chair_leg_caps", mesh_path=str(mesh), mesh_scale=.001)]

    mesh.unlink()
    with pytest.raises(FileNotFoundError) as error:
        registered_object("chair_leg_caps", catalog_path=catalog)
    assert error.value.args[0] == mesh


def test_bundled_relative_root_and_missing_optional_dataset(tmp_path):
    mesh, db = bundled_assets(tmp_path / "new", "cup")
    catalog = write_catalog(tmp_path, additional_datasets=[dict(root="new", cad_scale_to_m=.001),
        dict(root="unmounted", cad_scale_to_m=.001)], objects=[])
    entry = registered_object("cup", catalog_path=catalog, asset_root=tmp_path)
    assert entry["cad"] == mesh and entry["db"] == db


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_bundled_model_scale_must_be_explicit_and_positive(tmp_path, scale):
    catalog = write_catalog(tmp_path, additional_datasets=[dict(root="new", cad_scale_to_m=scale)], objects=[])
    with pytest.raises(ValueError, match="CAD scale"):
        load_catalog(catalog)
