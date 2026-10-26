"""Small CPU checks for the new provider boundary and observed placement goal."""

import numpy as np
import pytest

from tiptop.perception.known_objects import registered_target
from tiptop.perception.placement import select_observed_free_region


def test_alias_requires_unique_known_target():
    label, entry = registered_target(
        [dict(label="pepper_shaker"), dict(label="orange_bottle")],
        [dict(predicate="on", args=["pepper_shaker", "table"])],
    )
    assert label == "pepper_shaker"
    assert entry["class_id"] == "black_pepper_shaker"
    label, entry = registered_target(
        [dict(label="leather_care_bottle"), dict(label="orange_bottle")],
        [dict(predicate="on", args=["leather_care_bottle", "table"])],
    )
    assert label == "leather_care_bottle"
    assert entry["class_id"] == "car_leather_conditioner_bottle"
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


def test_known_rectangular_support_does_not_count_the_footprint_twice():
    x, y = np.meshgrid(np.arange(0.6, 0.8001, 0.005), np.arange(-0.4, 0.4001, 0.005))
    table = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.4)]
    args = dict(
        table_bounds=[[0.6, -0.4, 0.37], [0.8, 0.4, 0.38]], target_center=[0.65, -0.3, 0.45], footprint_radius=0.13
    )
    # A long object fits this observed strip, although its enclosing disk does not.
    with pytest.raises(ValueError, match="No observed empty"):
        select_observed_free_region(table, **args)
    region = select_observed_free_region(table, **args, footprint_size_xy=[0.08, 0.24])
    np.testing.assert_allclose(region["size_xy_m"], [0.11, 0.27])
    assert region["occupancy_shape"] == "rectangle" and region["observed_support_fraction"] >= 0.98
    assert not region["unknown_space_is_free"]
    with pytest.raises(ValueError, match="No observed empty"):
        select_observed_free_region(table[table[:, 1] < -0.25], **args, footprint_size_xy=[0.08, 0.24])


def test_registered_tcp_is_rebased_without_panda_correction(monkeypatch):
    from types import SimpleNamespace

    import trimesh
    from curobo.geom.types import Mesh

    from tiptop import tiptop_run as main

    hull = trimesh.creation.box(extents=[0.04, 0.04, 0.08])
    table = trimesh.creation.box(extents=[1, 1, 0.04])
    table_calls = []

    def segment_table(*args, **kwargs):
        table_calls.append(kwargs)
        return table

    monkeypatch.setattr(main, "segment_table_with_ransac", segment_table)
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
    registered = {
        "pepper": dict(
            grasps=dict(world_from_tcp=tcp, scores=np.array([0.8])), support_point_world=np.array([0.5, 0.1, 0.4])
        )
    }
    result = main.process_scene_geometry(
        np.zeros((2, 2, 3)),
        np.zeros((2, 2, 3)),
        np.ones((1, 1, 2, 2), bool),
        [dict(label="pepper")],
        {},
        registered_grasps=registered,
    )
    np.testing.assert_allclose(result.grasps["pepper"]["grasps_obj"][0, :3, 3], [0.1, 0.1, 0.1])
    np.testing.assert_allclose(table_calls[0]["support_points_world"], [[0.5, 0.1, 0.4]])


def test_release_height_accounts_for_sampled_envelope_and_planner_buffer_once():
    from tiptop.perception.placement import placement_surface_height

    # A cropped hull starts 12 mm above the CAD bottom, and its 3 mm
    # spheres extend below the hull. Include cuTAMP's separate 2 mm buffer.
    cad_bottom, sphere_bottom = -0.04, -0.04 + 0.012 - 0.003
    top = placement_surface_height(support_z=0.4, cad_bottom=cad_bottom, sphere_bottom=sphere_bottom)
    placed_origin = top + 0.002 - sphere_bottom
    assert placed_origin + cad_bottom == pytest.approx(0.403)
    with pytest.raises(ValueError, match="finite"):
        placement_surface_height(support_z=np.nan, cad_bottom=0, sphere_bottom=0)


@pytest.mark.parametrize(
    "options",
    [dict(), dict(placement_support_tolerance_m=0.002, motion_position_tolerance_m=0.005)],
)
def test_release_height_retains_nominal_clearance_with_relaxed_tolerances(options):
    from tiptop.doosan.known_task import KnownTaskPlanningSettings
    from tiptop.perception.placement import placement_surface_height

    settings = KnownTaskPlanningSettings(**options)
    cad_bottom, sphere_bottom = -0.04, -0.031
    top = placement_surface_height(
        support_z=0.4,
        cad_bottom=cad_bottom,
        sphere_bottom=sphere_bottom,
        release_clearance=settings.release_clearance_m,
    )
    # Apply both allowed downward translation errors to the nominal placement.
    lowest_cad_bottom = (
        top
        + 0.002
        - sphere_bottom
        + cad_bottom
        - settings.placement_support_tolerance_m
        - settings.motion_position_tolerance_m
    )
    assert lowest_cad_bottom == pytest.approx(0.405)


@pytest.mark.parametrize(
    "options",
    [
        dict(motion_timeout_s=0),
        dict(motion_timeout_s=float("nan")),
        dict(motion_position_tolerance_m=-0.01),
        dict(motion_rotation_tolerance=float("inf")),
        dict(placement_support_tolerance_m=0),
        dict(placement_xy_tolerance_m=-0.01),
        dict(release_clearance_m=0.003),
        dict(release_clearance_m=0.02),
        dict(release_clearance_m=float("nan")),
    ],
)
def test_planning_settings_reject_invalid_or_inconsistent_limits(options):
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    with pytest.raises(ValueError):
        KnownTaskPlanningSettings(**options)


def test_approach_bound_preserves_database_score_order(tmp_path):
    import json

    from scipy.spatial.transform import Rotation

    from tiptop.doosan.grasp_database import load_database_grasps
    from tiptop.doosan.known_task import approach_tilt_mask

    # A higher-scoring 10-degree grasp precedes a 2-degree grasp when both
    # satisfy the requested bound. A still higher score cannot bypass it.
    matrices = np.repeat(np.eye(4)[None], 3, axis=0)
    matrices[:, :3, :3] = Rotation.from_euler("y", [[2], [10], [20]], degrees=True).as_matrix() @ np.diag([1, -1, -1])
    path = tmp_path / "grasps.json"
    path.write_text(
        json.dumps(
            dict(
                metadata=dict(
                    object="pepper", coordinate_frame="object_zero_pose", gripper_model="UON_3finger_gripper"
                ),
                data=[
                    dict(grasp_mat=matrix.tolist(), score=score, scene_id="scene0001", source_pregrasp_index=index)
                    for index, (matrix, score) in enumerate(zip(matrices, [0.8, 0.95, 0.99]))
                ],
            )
        )
    )
    result = load_database_grasps(
        path,
        class_id="pepper",
        world_from_cad=np.eye(4),
        pose_source="rgbd_pose_estimator",
        world_from_planner_object=np.eye(4),
        grasp_from_tcp=np.eye(4),
    )
    poses = result["world_from_tcp"]
    assert result["candidate_indices"][approach_tilt_mask(poses, 15)].tolist() == [1, 0]
    assert result["candidate_indices"][approach_tilt_mask(poses)].tolist() == [0]
    np.testing.assert_array_equal(poses, matrices[result["candidate_indices"]])
