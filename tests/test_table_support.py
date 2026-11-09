"""Table support modes: RANSAC contact choice, the configured table and the CAD-bottom gap check."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from curobo.geom.types import Cuboid, Mesh

from tiptop.perception.placement import check_support_gap, table_support_z


def test_support_height_restores_the_lowered_table_box():
    assert table_support_z(-0.014) == pytest.approx(0.006)


def test_cad_bottom_gap_is_signed_and_limited():
    assert check_support_gap("bottle", 0.012, 0.006) == pytest.approx(0.006)
    with pytest.raises(ValueError, match=r"36\.0 mm below"):
        check_support_gap("bottle", -0.030, 0.006)
    with pytest.raises(ValueError, match="above"):
        check_support_gap("bottle", 0.080, 0.006)
    with pytest.raises(ValueError, match="not finite"):
        check_support_gap("bottle", float("nan"), 0.006)


def test_declared_table_is_lowered_like_the_ransac_box():
    from tiptop.tiptop_run import declared_table_cuboid

    declared = Cuboid(name="configured_full_table", dims=[0.78, 2.5, 0.03], pose=[0, 0.5, -0.015, 1, 0, 0, 0])
    table = declared_table_cuboid(declared)
    assert table.name == "table"
    np.testing.assert_allclose(table.dims, declared.dims)
    np.testing.assert_allclose(table.pose[:2], [0, 0.5])
    # A box whose top is 20 mm under the declared surface restores that surface as the support height.
    assert table_support_z(table.pose[2] + table.dims[2] / 2) == pytest.approx(0.0)
    rotated = Cuboid(name="configured_full_table", dims=[1, 1, 0.03], pose=[0, 0, 0, 0.7071, 0, 0, 0.7071])
    with pytest.raises(ValueError, match="axis-aligned"):
        declared_table_cuboid(rotated)


@pytest.fixture
def scene(monkeypatch):
    """process_scene_geometry with a RANSAC table whose top is at z=0.02, so its support plane is at z=0.04."""
    from tiptop import tiptop_run as main

    table_calls = []

    def segment_table(*args, **kwargs):
        table_calls.append(kwargs)
        return trimesh.creation.box(extents=[1, 1, 0.04])

    hull = trimesh.creation.box(extents=[0.04, 0.04, 0.08])
    monkeypatch.setattr(main, "segment_table_with_ransac", segment_table)
    monkeypatch.setattr(main, "log_curobo_mesh_to_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(main, "segment_pointcloud_by_masks", lambda *a, **kw: ({"pepper": hull}, {"pepper": object()}))
    monkeypatch.setattr(
        main,
        "convert_trimesh_to_curobo_mesh",
        lambda mesh, label: Mesh(name=label, pose=[0.5, 0.1, 0.4, 1, 0, 0, 0], vertices=mesh.vertices, faces=mesh.faces),
    )
    monkeypatch.setattr(main, "tensor_args", SimpleNamespace(to_device=lambda array: array))
    tcp = np.eye(4)[None]
    tcp[0, :3, 3] = [0.6, 0.2, 0.5]

    def run(mode, cad_bottom_z, declared_table=None):
        registered = {
            "pepper": dict(
                grasps=dict(world_from_tcp=tcp, scores=np.array([0.8])),
                support_point_world=np.array([0.5, 0.1, cad_bottom_z]),
            )
        }
        return main.process_scene_geometry(
            np.zeros((2, 2, 3)),
            np.zeros((2, 2, 3)),
            np.ones((1, 1, 2, 2), bool),
            [dict(label="pepper")],
            {},
            registered_grasps=registered,
            table_support=mode,
            declared_table=declared_table,
        )

    return SimpleNamespace(run=run, table_calls=table_calls)


def test_ransac_cad_scores_planes_with_cad_bottoms_and_only_records_the_gap(scene):
    result = scene.run("ransac-cad", 0.4)
    np.testing.assert_allclose(scene.table_calls[0]["support_points_world"], [[0.5, 0.1, 0.4]])
    assert result.table_support["mode"] == "ransac-cad"
    assert result.table_support["support_z_m"] == pytest.approx(0.04)
    assert result.table_support["cad_bottom_gap_m"]["pepper"] == pytest.approx(0.36)


def test_ransac_mask_uses_mask_bottoms_and_rejects_far_cad_bottoms(scene):
    result = scene.run("ransac-mask", 0.045)
    assert "support_points_world" not in scene.table_calls[0]
    assert result.table_support["cad_bottom_gap_m"]["pepper"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="above the table support plane"):
        scene.run("ransac-mask", 0.4)


def test_declared_table_skips_ransac_and_checks_the_cad_bottom(scene):
    declared = Cuboid(name="configured_full_table", dims=[0.78, 2.5, 0.03], pose=[0, 0.5, -0.015, 1, 0, 0, 0])
    result = scene.run("declared", 0.01, declared)
    assert scene.table_calls == []
    assert result.table_cuboid.name == "table"
    assert result.table_support["support_z_m"] == pytest.approx(0.0)
    assert result.table_support["cad_bottom_gap_m"]["pepper"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match=r"36\.0 mm below"):
        scene.run("declared", -0.036, declared)
    with pytest.raises(ValueError, match="needs a configured full table"):
        scene.run("declared", 0.01)


def test_unknown_table_support_is_rejected(scene):
    with pytest.raises(ValueError, match="Unknown table support"):
        scene.run("fixed", 0.01)


def test_cli_defaults_to_the_current_behaviour_and_accepts_every_mode():
    from tiptop.scripts.run_known_pick_place import configured_full_table, parse_args

    assert parse_args(["--task", "pepper 집고 제자리에 놓아"]).table_support == "ransac-cad"
    for mode in ("ransac-cad", "ransac-mask", "declared"):
        assert parse_args(["--task", "pepper 집고 제자리에 놓아", "--table-support", mode]).table_support == mode
    table = Cuboid(name="configured_full_table", dims=[1, 1, 0.03], pose=[0, 0, -0.015, 1, 0, 0, 0])
    other = Cuboid(name="configured_frame", dims=[1, 1, 0.03], pose=[0, 0, 1, 1, 0, 0, 0])
    assert configured_full_table([other, table]) is table
    with pytest.raises(ValueError, match="exactly one"):
        configured_full_table([other])
