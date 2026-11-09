"""Small CPU checks for the new provider boundary and observed placement goal."""

import numpy as np
import pytest

from tiptop.perception.known_objects import registered_target


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


@pytest.mark.parametrize(
    "options",
    [
        dict(motion_timeout_s=0),
        dict(motion_timeout_s=float("nan")),
        dict(motion_position_tolerance_m=-0.01),
        dict(motion_rotation_tolerance=float("inf")),
        dict(approach_distance_m=0),
        dict(approach_distance_m=.04),
        dict(lift_distance_m=0),
        dict(lift_distance_m=.005),
        dict(lift_hold_s=0),
        dict(lift_hold_s=float("nan")),
        dict(max_wrist_travel_deg=0),
        dict(max_wrist_travel_deg=float("inf")),
        dict(min_table_clearance_m=0),
        dict(min_table_clearance_m=float("nan")),
        dict(min_engagement_depth_m=-0.001),
        dict(min_engagement_depth_m=float("nan")),
    ],
)
def test_planning_settings_reject_invalid_or_inconsistent_limits(options):
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    with pytest.raises(ValueError):
        KnownTaskPlanningSettings(**options)


@pytest.mark.parametrize("name", [
    "approach_speed_factor", "descend_far_speed_factor", "descend_near_speed_factor",
    "lift_speed_factor", "home_speed_factor",
])
@pytest.mark.parametrize("value", [0, 1.01, float("nan"), float("inf")])
def test_stage_speed_factors_require_positive_finite_ratios_at_most_one(name, value):
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    with pytest.raises(ValueError, match=name):
        KnownTaskPlanningSettings(**{name: value})


def test_engagement_depth_is_measured_from_the_posed_cad_top_and_zero_disables_the_bound():
    from tiptop.doosan.known_task import KnownTaskPlanningSettings, engagement_depth_m

    assert KnownTaskPlanningSettings().min_engagement_depth_m == pytest.approx(0.02)
    assert KnownTaskPlanningSettings(min_engagement_depth_m=0).min_engagement_depth_m == 0
    # A 5 cm tall CAD whose origin is on its bottom, posed 1 cm above the table and turned about world Z.
    vertices = np.array([[0, 0, 0], [.03, .02, .05], [-.03, .01, .02]])
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[2, 3] = .01
    tcp = np.tile(np.eye(4), (3, 1, 1))
    tcp[:, 2, 3] = [.06, .04, .02]
    np.testing.assert_allclose(engagement_depth_m(tcp, pose, vertices), [0, .02, .04], atol=1e-12)


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
