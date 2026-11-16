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


def test_instance_target_resolves_its_class_without_parsing_the_instance_name(monkeypatch):
    from tiptop.perception import known_objects

    entry = dict(class_id="black_pepper_shaker", aliases=["pepper_shaker"])
    monkeypatch.setattr(known_objects, "load_catalog", lambda *a: dict(objects=[entry]))
    monkeypatch.setattr(known_objects, "registered_object", lambda class_id, **k: dict(entry))
    boxes = [dict(label=f"instance_{i}", class_id=entry["class_id"], instance_index=i) for i in range(2)]
    label, target = registered_target(boxes, [dict(predicate="holding", args=["instance_1"])])
    assert label == "instance_1" and target["class_id"] == entry["class_id"]


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


@pytest.mark.parametrize("surrounding_obstacles", [True, False])
@pytest.mark.parametrize("target_index", [1, 2])
def test_all_reference_masks_become_hulls_with_only_the_selected_target_movable(
    monkeypatch, surrounding_obstacles, target_index,
):
    import json
    from types import SimpleNamespace

    from curobo.geom.types import Cuboid
    from cutamp.utils.common import get_world_cfg

    from tiptop import tiptop_run as main
    from tiptop.perception.sam3_reference import decode_reference_result
    from tiptop.perception_wrapper import select_reference_candidate

    # Four separated, non-planar visible surfaces. The fourth is outside the target trial limit.
    h, w = 12, 48
    y, x = np.mgrid[:h, :w]
    xyz = np.stack([x * .01, y * .01, .10 + .02 * np.sin(x) * np.cos(y)], axis=-1)
    rgb = np.full_like(xyz, .5)
    masks = np.zeros((4, 1, h, w), bool)
    boxes = []
    for i in range(4):
        start = i * 12 + 2
        masks[i, 0, 2:10, start:start+8] = True
        boxes.append([start, 2, start+8, 10])
    detection = decode_reference_result(
        dict(masks=masks, boxes_xyxy=boxes, scores=[.9, .85, .8, .75],
             metadata=np.array(json.dumps(dict(class_id="black_pepper_shaker", confidence_threshold=.5)))),
        (h, w, 3), "black_pepper_shaker",
    )
    # Rejecting the first target must preserve its hull and the untried fourth instance.
    results = dict(detection, xyz_map=xyz, rgb_map=rgb, valid_depth_mask=np.ones((h, w), bool))
    selected = select_reference_candidate(results, detection["reference_candidates"][target_index])
    monkeypatch.setattr(main, "tensor_args", SimpleNamespace(to_device=lambda array: array))
    monkeypatch.setattr(main, "log_curobo_mesh_to_rerun", lambda *a, **k: None)
    table = Cuboid(name="configured_full_table", dims=[1, 1, .03], pose=[0, 0, -.015, 1, 0, 0, 0])
    scene = main.process_scene_geometry(
        xyz, rgb, selected["masks"], selected["bboxes"], {}, table_support="declared", declared_table=table,
        registered_grasps={f"black_pepper_shaker__instance_{target_index}": dict(
            support_point_world=np.array([.2, .05, .01]),
            grasps=dict(world_from_tcp=np.eye(4)[None], scores=np.array([.8])),
        )},
    )
    assert len(scene.object_meshes) == 4
    target = f"black_pepper_shaker__instance_{target_index}"
    assert set(scene.grasps) == {target}
    env, surfaces = main.create_tamp_environment(
        scene.object_meshes, scene.table_cuboid, [dict(predicate="holding", args=[target])],
        include_workspace=False, movable_labels=set(scene.grasps),
        collision_labels=None if surrounding_obstacles else {target},
    )
    assert [obj.name for obj in env.movables] == [target]
    assert [obj.name for obj in surfaces] == ["table"]
    obstacle_names = {f"black_pepper_shaker__instance_{i}" for i in range(4) if i != target_index} if surrounding_obstacles else set()
    assert {obj.name for obj in env.statics} == {"table", *obstacle_names}
    static_world = get_world_cfg(env, include_movables=False)
    assert {mesh.name for mesh in static_world.mesh} == obstacle_names
    assert {mesh.name for mesh in get_world_cfg(env, include_movables=True).mesh} == {target, *obstacle_names}
    assert len(scene.object_meshes) == len(selected["masks"]) == len(selected["bboxes"]) == 4
    with pytest.raises(ValueError, match="must remain in the collision scene"):
        main.create_tamp_environment(scene.object_meshes, scene.table_cuboid,
                                     [dict(predicate="holding", args=[target])],
                                     include_workspace=False, collision_labels=set())


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
        dict(num_particles=0),
        dict(num_particles=-4),
        dict(num_particles=16.5),
        dict(robot_world_tolerance_m=0),
        dict(robot_world_tolerance_m=float("nan")),
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


def test_num_particles_defaults_to_32_and_reaches_the_planning_settings_from_the_command_line():
    from tiptop.doosan.known_task import KnownTaskPlanningSettings
    from tiptop.scripts.run_known_pick_place import parse_args

    assert KnownTaskPlanningSettings().num_particles == 32
    base = ["--task", "black_pepper_shaker 집고 제자리에 놓아", "--camera-url", "http://127.0.0.1:13348"]
    assert parse_args(base).planning_settings.num_particles == 32
    assert parse_args([*base, "--num-particles", "16"]).planning_settings.num_particles == 16


def test_candidate_stage_line_reports_counts_drop_time_and_colour(capsys):
    from tiptop.doosan.known_task import log_candidate_stage

    log_candidate_stage("Grasp", "① 접근각", 4892, 223, 0.031, " · 메모", enough=32)
    log_candidate_stage("Grasp", "② 깊이", 223, 0, enough=32)
    log_candidate_stage("IK", "③ 전달", 100, 20, enough=32)
    green, red, yellow = capsys.readouterr().out.splitlines()
    assert "4,892 → 223 (−4,669) · 메모 · 0.03s" in green and "\033[1;32m[Grasp]" in green
    assert "223 → 0 (−223)" in red and "\033[1;31m" in red
    assert "100 → 20 (−80)" in yellow and "\033[1;33m[IK]" in yellow


def test_robot_world_tolerance_defaults_to_5_mm_and_reaches_the_planning_settings():
    from tiptop.doosan.known_task import KnownTaskPlanningSettings
    from tiptop.scripts.run_known_pick_place import parse_args

    assert KnownTaskPlanningSettings().robot_world_tolerance_m == pytest.approx(0.005)
    base = ["--task", "black_pepper_shaker 집고 제자리에 놓아", "--camera-url", "http://127.0.0.1:13348"]
    assert parse_args(base).planning_settings.robot_world_tolerance_m == pytest.approx(0.005)
    assert parse_args([*base, "--robot-world-tolerance-m", "0.001"]).planning_settings.robot_world_tolerance_m == 0.001


def test_planning_applies_the_tolerance_to_robot_to_world_only_and_never_mutates_the_upstream_table(monkeypatch):
    import tiptop.planning as planning
    from cutamp.scripts.utils import default_constraint_to_tol
    from cutamp.task_planning.constraints import Collision

    seen = {}
    monkeypatch.setattr(planning, "run_cutamp", lambda *args, **kwargs: seen.update(checker=args[3]) or (None, 0, "no plan"))
    upstream = default_constraint_to_tol[Collision.type].copy()
    planning.run_planning(None, None, None, None, None, None, [], robot_world_tolerance=0.005)
    assert seen["checker"]._get_tol(Collision.type, "robot_to_world") == 0.005
    assert seen["checker"]._get_tol(Collision.type, "movable_to_world") == 1e-3
    assert default_constraint_to_tol[Collision.type] == upstream
    planning.run_planning(None, None, None, None, None, None, [])
    assert seen["checker"]._get_tol(Collision.type, "robot_to_world") == 1e-3
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="Robot-to-world"):
            planning.run_planning(None, None, None, None, None, None, [], robot_world_tolerance=bad)


def test_noisy_loggers_are_quieted_but_cutamp_warnings_and_tiptop_logs_stay():
    import logging

    from tiptop.scripts.run_known_pick_place import quiet_noisy_loggers

    names = ("curobo", "cutamp")
    before = {name: logging.getLogger(name).level for name in names}
    try:
        logging.getLogger().setLevel(logging.INFO)
        quiet_noisy_loggers()
        curobo, cutamp, tiptop = (logging.getLogger(name) for name in ("curobo", "cutamp.algorithm", "tiptop.planning"))
        assert not curobo.isEnabledFor(logging.WARNING) and curobo.isEnabledFor(logging.ERROR)
        assert not cutamp.isEnabledFor(logging.INFO) and cutamp.isEnabledFor(logging.WARNING)
        assert tiptop.isEnabledFor(logging.INFO)
    finally:
        for name, level in before.items():
            logging.getLogger(name).setLevel(level)


def test_tiptop_progress_info_is_hidden_but_timing_lines_warnings_and_other_loggers_stay():
    import logging

    from tiptop.scripts.run_known_pick_place import _TiptopProgressFilter

    def record(name, level, message):
        return logging.LogRecord(name, level, __file__, 1, message, None, None)

    keep = _TiptopProgressFilter().filter
    hidden = [
        "Created mesh for homestar__instance_3: 3248 pts, centroid=[0.04 0.79 0.05]",
        "Movables: ['homestar__instance_9']",
        "Created TAMP environment with 1 movables, 1 surfaces",
        "Executing step 1/7: Pick.approach. Action type: trajectory (70 waypoints)",
        "Completed step 3/7: {'success': True}",
        "Found plan with 7 steps",
        "Saved planning input to tiptop_outputs/m1013/tasks/task-9/planning-input.pkl",
    ]
    assert not any(keep(record("tiptop.tiptop_run", logging.INFO, text)) for text in hidden)
    for text in ("Pose estimation client total took 0.86s", "cuTAMP planning took: 4.04s", "Home planning took 0.34s (speed factor 0.9)"):
        assert keep(record("tiptop.planning", logging.INFO, text))
    assert keep(record("tiptop.execute_plan", logging.WARNING, "Executing step failed"))
    assert keep(record("cutamp.algorithm", logging.INFO, "Found plan with 7 steps"))
