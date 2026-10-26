"""Data-root routing must agree between the client and the pose server."""

import json

import pytest

from tiptop.object_catalog import registered_object
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
