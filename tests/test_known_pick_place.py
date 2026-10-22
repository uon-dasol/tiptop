"""Small CPU checks for the new provider boundary and observed placement goal."""

import numpy as np
import pytest

from tiptop.perception.placement import select_observed_free_region
from tiptop.perception.known_objects import registered_target


def test_alias_requires_unique_known_target():
    label, entry = registered_target(
        [dict(label="pepper_shaker"), dict(label="orange_bottle")],
        [dict(predicate="on", args=["pepper_shaker", "table"])],
    )
    assert label == "pepper_shaker"
    assert entry["class_id"] == "black_pepper_shaker"
    with pytest.raises(ValueError, match="uniquely"):
        registered_target([dict(label="orange_bottle")], [dict(predicate="on", args=["orange_bottle", "table"])])


def test_empty_region_requires_observed_support_and_avoids_occupancy():
    x, y = np.meshgrid(np.arange(0, 1, 0.005), np.arange(-0.4, 0.4, 0.005))
    table = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.4)]
    target = np.array([0.45, -0.15, 0.45])
    obstacle = np.array([[0.60, -0.15, 0.45]])
    # A large invisible patch must not be chosen as free space.
    table = table[~((table[:, 0] > 0.55) & (table[:, 1] > 0.05))]
    result = select_observed_free_region(
        np.vstack([table, obstacle]),
        table_bounds=[[0, -0.4, 0.37], [1, 0.4, 0.38]],
        target_center=target,
        footprint_radius=0.05,
    )
    center = np.asarray(result["center_xy_m"])
    assert np.linalg.norm(center - target[:2]) >= 0.14
    assert np.linalg.norm(center - obstacle[0, :2]) > result["checked_radius_m"]
    assert result["observed_support_fraction"] >= 0.98
    assert not result["unknown_space_is_free"]
    with pytest.raises(ValueError, match="No sufficiently observed"):
        select_observed_free_region(
            obstacle, table_bounds=[[0, -0.4, 0.37], [1, 0.4, 0.38]], target_center=target, footprint_radius=0.05
        )


def test_registered_tcp_is_rebased_without_panda_correction(monkeypatch):
    import trimesh
    from types import SimpleNamespace
    from tiptop import tiptop_run as main
    from curobo.geom.types import Mesh

    hull = trimesh.creation.box(extents=[0.04, 0.04, 0.08])
    table = trimesh.creation.box(extents=[1, 1, 0.04])
    monkeypatch.setattr(main, "segment_table_with_ransac", lambda *a: table)
    monkeypatch.setattr(main, "log_curobo_mesh_to_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(main, "segment_pointcloud_by_masks", lambda *a, **kw: ({"pepper": hull}, {"pepper": object()}))
    monkeypatch.setattr(
        main,
        "convert_trimesh_to_curobo_mesh",
        lambda mesh, label: Mesh(
            name=label, pose=[0.5, 0.1, 0.4, 1, 0, 0, 0], vertices=mesh.vertices, faces=mesh.faces
        ),
    )
    monkeypatch.setattr(main, "tensor_args", SimpleNamespace(to_device=lambda array: array))

    def forbidden():
        raise AssertionError("DB poses must not use the M2T2/Panda correction")

    monkeypatch.setattr(main, "m2t2_to_tiptop_transform", forbidden)
    tcp = np.eye(4)[None]
    tcp[0, :3, 3] = [0.6, 0.2, 0.5]
    registered = {"pepper": dict(grasps=dict(world_from_tcp=tcp, scores=np.array([0.8])))}
    result = main.process_scene_geometry(
        np.zeros((2, 2, 3)),
        np.zeros((2, 2, 3)),
        np.ones((1, 1, 2, 2), bool),
        [dict(label="pepper")],
        {},
        registered_grasps=registered,
    )
    np.testing.assert_allclose(result.grasps["pepper"]["grasps_obj"][0, :3, 3], [0.1, 0.1, 0.1])
