"""Projection, image identity and the read-only viewer; no devices or inference."""

import asyncio
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen, Request

import numpy as np
import pytest
from PIL import Image

from tiptop.perception_preview import (TaskPreview, project_grasps, project_grasp_boxes,
                                       gripper_mesh_parts, final_grasp_joints, preview_saved_task)
from tiptop.scripts.serve_perception_viewer import make_server


def test_projection_uses_camera_frame_and_rejects_points_behind_camera():
    camera = np.eye(4)
    camera[:3, 3] = [2, 3, 4]
    tcp = camera.copy()
    tcp[2, 3] += 1
    behind = camera.copy()
    behind[2, 3] -= 1
    k = np.array([[100, 0, 40], [0, 100, 30], [0, 0, 1]])
    result = project_grasps([tcp, behind], camera, k, [[0, 0, -.1, .01]], [7, 8], [.8, .7])
    assert len(result) == 1 and result[0]["id"] == 7
    np.testing.assert_allclose(result[0]["axes"][:3], [[40, 30], [43.5, 30], [40, 33.5]])
    assert result[0]["spheres"][0][2] == pytest.approx(100*.01/.9)


def test_recorded_sweep_boxes_use_object_zero_frame_and_preserve_four_corners():
    boxes = np.array([[[0, 0, 0], [.01, 0, 0], [.01, .02, 0], [0, .02, 0]]]*3)
    original = boxes.copy()
    world = np.array([[0, -1, 0, .3], [1, 0, 0, .4], [0, 0, 1, 1.2], [0, 0, 0, 1.]])
    camera = np.eye(4)
    camera[:3, 3] = [.1, .2, .2]
    k = np.array([[100, 0, 40], [0, 100, 30], [0, 0, 1]])
    projected = project_grasp_boxes(boxes, world, camera, k)
    assert [box['finger'] for box in projected] == [1, 2, 3]
    np.testing.assert_allclose(projected[0]['points'], [[60, 50], [60, 51], [58, 51], [58, 50]])
    np.testing.assert_allclose(projected[0]['world_points'][0], [.3, .4, 1.2])
    np.testing.assert_array_equal(boxes, original)


@pytest.mark.parametrize('ambiguous', [False, True])
@pytest.mark.parametrize('source_key', ['source_pregrasp_index', 'source_output_index'])
def test_final_sweep_boxes_reconcile_pose_not_score_and_skip_ambiguous_sources(tmp_path, monkeypatch, ambiguous, source_key):
    from yourdfpy import URDF
    import tiptop.object_catalog as catalog

    preview = snapshot(tmp_path)
    robot = URDF.load(str(preview.urdf_path), load_meshes=False)
    tcp = robot.get_transform('gripper_tcp', 'base_link')
    other = tcp.copy()
    if not ambiguous:
        other[0, 3] += .1
    corners = np.array([[[0, 0, 0], [.01, 0, 0], [.01, .02, 0], [0, .02, 0]]]*3) + tcp[:3, 3]
    rows = [dict(grasp_mat=pose.tolist(), score=score, scene_id=scene, **{source_key: index},
                 grasp_boxes=corners.tolist())
            for pose, score, scene, index in [(tcp, .2, '0033', 321), (other, .9, '0099', 999)]]
    db = tmp_path / 'source-db.json'
    db.write_text(json.dumps(dict(metadata=dict(object='test', coordinate_frame='object_zero_pose'), data=rows)))
    (tmp_path / 'registered-pose.json').write_text(json.dumps(dict(class_id='test', pose=dict(world_from_cad=np.eye(4).tolist()))))
    monkeypatch.setattr(catalog, 'registered_object', lambda _: dict(class_id='test', db=db))
    preview.grasps(dict(world_from_tcp=np.array([other, tcp]), candidate_indices=[1, 0], scores=[.9, .2]),
                   np.eye(4), np.diag([100, 100, 1]), max_tilt_deg=180)
    preview.update('planning', dict(supplied_candidate_ids=[0, 1]))
    preview.publish_plan(completed_plan(), np.eye(4), np.diag([100, 100, 1]))
    final = preview.state['final_grasp']
    assert len(final['outlines']) == 3  # only the finger tips are published
    assert preview.state['db_source']['path'] == str(db)
    assert len(preview.state['grasps'][0]['grasp_boxes']) == 3
    if ambiguous:
        assert 'grasp_boxes' not in final
        assert '불명확' in final['db_match_note']
    else:
        assert final['db_reference']['index'] == 0
        assert final['db_reference']['scene_id'] == '0033'
        assert final['db_reference'][source_key] == 321
        assert final['db_reference']['position_error_mm'] == pytest.approx(0)
        assert len(final['grasp_boxes']) == 3
        # Side geometry may arrive after both the archive and the final plan.
        preview.side_scene(np.array([[0, 0, 0], [.1, .1, .1]]), 0, np.eye(4))
        for box in final['grasp_boxes']:
            world = np.asarray(box['world_points'])
            np.testing.assert_allclose(np.asarray(box['side_points'])[:, 0], .05-world[:, 0])
            np.testing.assert_allclose(np.asarray(box['side_points'])[:, 1], world[:, 2])


def test_sweep_template_follows_corrected_candidate_and_final_fk_in_both_views(tmp_path):
    from yourdfpy import URDF

    preview = snapshot(tmp_path)
    final_tcp = URDF.load(str(preview.urdf_path), load_meshes=False).get_transform('gripper_tcp', 'base_link')
    target_tcp = final_tcp.copy()
    target_tcp[0, 3] += .004  # IK endpoint need not exactly equal the DB target.
    world_from_object = np.array([[0, -1, 0, .2], [1, 0, 0, .3], [0, 0, 1, .1], [0, 0, 0, 1.]])
    correction = np.diag([-1., -1., 1., 1.])
    recorded_grasp = np.linalg.inv(world_from_object) @ target_tcp @ correction
    local = np.array([[[.02, 0, -.008], [.04, 0, -.008], [.04, .01, -.008], [.02, .01, -.008]]]*3)
    recorded_boxes = local @ recorded_grasp[:3, :3].T + recorded_grasp[:3, 3]
    row = dict(grasp_mat=recorded_grasp.tolist(), grasp_boxes=recorded_boxes.tolist(),
               scene_id='test', source_pregrasp_index=123)
    preview._database_context = ([row], world_from_object)
    preview.side_scene(np.array([[0, 0, 0], [.1, .1, .1]]), 0, np.eye(4))
    camera = np.eye(4)
    camera[:3, 3] = [-.1, .1, -.2]
    k = np.array([[100, 0, 40], [0, 100, 30], [0, 0, 1]])
    preview.grasps(dict(world_from_tcp=[target_tcp], candidate_indices=[0], scores=[.8]),
                   camera, k, max_tilt_deg=180)
    plan = completed_plan()
    plan['supplied_candidate_ids'] = [0]
    preview.publish_plan(plan, camera, k)
    candidate, final = preview.state['grasps'][0], preview.state['final_grasp']
    assert final['db_reference']['index'] == 0
    assert final['db_reference']['position_error_mm'] == pytest.approx(4)
    original_world = recorded_boxes @ world_from_object[:3, :3].T + world_from_object[:3, 3]
    for grasp, tcp in [(candidate, target_tcp), (final, final_tcp)]:
        assert grasp['grasp_boxes_frame'] == 'displayed_tcp'
        # The DB-to-TCP correction is undone: the template is the DB grasp frame at the displayed TCP.
        database_frame = tcp @ correction
        expected = local @ database_frame[:3, :3].T + database_frame[:3, 3]
        for i, box in enumerate(grasp['grasp_boxes']):
            np.testing.assert_allclose(box['world_points'], expected[i], atol=1e-12)
            camera_points = expected[i] - camera[:3, 3]
            pixels = camera_points @ k.T
            np.testing.assert_allclose(box['points'], pixels[:, :2]/pixels[:, 2:])
            np.testing.assert_allclose(box['side_points'], preview._side_points(expected[i]))
            np.testing.assert_allclose(grasp['db_grasp_boxes_original'][i]['world_points'], original_world[i])
    # At the exact DB target the template equals the recorded geometry; the FK endpoint residual moves it.
    np.testing.assert_allclose(candidate['grasp_boxes'][0]['world_points'], original_world[0], atol=1e-12)
    assert not np.allclose(final['grasp_boxes'][0]['world_points'], original_world[0])
    # The source JSON geometry is neither overwritten nor rotated in place.
    np.testing.assert_array_equal(row['grasp_boxes'], recorded_boxes)
    np.testing.assert_allclose(np.asarray(candidate['grasp_boxes'][0]['world_points'])
                               - np.asarray(final['grasp_boxes'][0]['world_points']), [[.004, 0, 0]]*4, atol=1e-12)


def test_planned_target_disambiguates_nearby_candidates_even_when_another_is_closer_to_fk(tmp_path):
    from yourdfpy import URDF

    preview = snapshot(tmp_path)
    tcp = URDF.load(str(preview.urdf_path), load_meshes=False).get_transform('gripper_tcp', 'base_link')
    chosen, neighbor = tcp.copy(), tcp.copy()
    chosen[0, 3] += .004
    neighbor[0, 3] += .001  # Both are in tolerance; nearest FK is not the chosen goal.
    local = np.array([[[0, 0, 0], [.01, 0, 0], [.01, .02, 0], [0, .02, 0]]]*3)
    rows = [dict(grasp_mat=pose.tolist(), grasp_boxes=(local @ pose[:3, :3].T + pose[:3, 3]).tolist(),
                 scene_id=str(index), source_pregrasp_index=index)
            for index, pose in enumerate((chosen, neighbor))]
    preview._database_context = (rows, np.eye(4))
    camera, k = np.eye(4), np.diag([100, 100, 1])
    preview.grasps(dict(world_from_tcp=[chosen, neighbor], candidate_indices=[0, 1], scores=[.2, .9]),
                   camera, k, max_tilt_deg=180)
    preview.update('planning', dict(supplied_candidate_ids=[0, 1]))
    plan = completed_plan()
    plan['selected_grasp'] = dict(world_from_tcp=chosen.tolist(), source='completed_motion_plan')
    preview.publish_plan(plan, camera, k)
    final = preview.state['final_grasp']
    assert final['db_reference']['index'] == 0
    assert final['db_reference']['match_method'] == 'planned_target'
    assert final['db_reference']['position_error_mm'] == pytest.approx(4)
    assert len(final['grasp_boxes']) == 3
    np.testing.assert_allclose(final['grasp_boxes'][0]['world_points'], local[0] @ tcp[:3, :3].T + tcp[:3, 3])
    np.testing.assert_array_equal(final['joint_positions_rad'], plan['steps'][1]['positions'][-1])
    # An unrecognized target must not fall back to an unrelated nearby DB row.
    plan['selected_grasp']['world_from_tcp'][0][3] += .1
    preview.publish_plan(plan, camera, k)
    assert 'grasp_boxes' not in preview.state['final_grasp']


def test_cad_projection_contains_three_fingers_and_respects_the_model_tcp(tmp_path):
    from pathlib import Path
    import xml.etree.ElementTree as ET

    source = Path(__file__).resolve().parents[1] / 'tiptop/doosan/assets/m1013_uon/m1013_uon.urdf'
    original = dict(gripper_mesh_parts(str(source)))
    assert {f'gripper_finger{i}' for i in (1, 2, 3)} <= original.keys()
    robot = ET.parse(source)
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', str((source.parent / mesh.get('filename')).resolve()))
    origin = robot.find("joint[@name='gripper_tcp_joint']/origin")
    xyz = np.fromstring(origin.get('xyz'), sep=' ')
    xyz[2] += .01
    origin.set('xyz', ' '.join(map(str, xyz)))
    modified = tmp_path / 'gripper.urdf'
    robot.write(modified)
    for name, vertices in gripper_mesh_parts(str(modified)):
        np.testing.assert_allclose(vertices, original[name]-[0, 0, .01], atol=1e-12)
    tcp = np.eye(4)
    tcp[2, 3] = 1
    result = project_grasps([tcp], np.eye(4), np.diag([100, 100, 1]), [[0, 0, -.1, .01]],
                            [7], [.8], tool_parts=list(original.items()))
    assert len(result[0]['outlines']) == 3 and not any('points' in part for part in result[0]['outlines'])
    tips = {part['part']: part['tip'] for part in result[0]['outlines'] if 'finger' in part['part']}
    assert set(tips) == {f'gripper_finger{i}' for i in (1, 2, 3)}
    for name, tip in tips.items():
        vertices = original[name]
        expected = vertices[vertices[:, 2] >= vertices[:, 2].max() - .015].mean(axis=0)
        np.testing.assert_allclose(tip, 100*expected[:2]/(expected[2] + 1), rtol=1e-6)


def snapshot(root, identity="first"):
    preview = TaskPreview(root, observation_id=identity, task="bottle 집고 제자리에 놓아", depth_source="sensor")
    rgb = np.full((6, 8, 3), 128, np.uint8)
    preview.update("observation", dict(rgb=rgb, depth=np.zeros((6, 8))))
    return preview


def completed_plan():
    return dict(steps=[
        dict(type='trajectory', label='Pick.approach', positions=[[0.]*6, [.1]*6]),
        dict(type='trajectory', label='Pick.descend', positions=[[.1]*6, [0.]*6]),
        dict(type='gripper', label='Pick.close', action='close'),
        dict(type='trajectory', label='Place.lift', positions=[[0.]*6, [.2]*6]),
    ], supplied_candidate_ids=[9999])


def test_final_grasp_comes_from_commanded_endpoint_and_preserves_archived_candidates(tmp_path):
    preview = snapshot(tmp_path)
    candidate = dict(id=9999, score=1., axes=[[99, 99]]*4, outlines=[])
    preview.update('grasps', dict(grasps=[candidate], db_total=1, tilt_candidates=1, max_tilt_deg=10))
    assert preview.state['final_grasp'] is None
    plan = completed_plan()
    np.testing.assert_array_equal(final_grasp_joints(plan), [0.]*6)
    preview.publish_plan(plan, np.eye(4), np.diag([100, 100, 1]))
    final = preview.state['final_grasp']
    assert final is not None
    assert final['source_step'] == 'Pick.descend' and final['before_action'] == 'Pick.close'
    np.testing.assert_array_equal(final['joint_positions_rad'], plan['steps'][1]['positions'][-1])
    assert not np.allclose(final['axes'], candidate['axes'])
    assert 'id' not in final and 'score' not in final
    assert preview.state['grasps'] == [candidate]


def test_saved_final_grasp_does_not_require_a_database_or_pose_estimation_file(tmp_path):
    np.savez_compressed(tmp_path/'observation.npz', observation_id='saved', rgb=np.zeros((6, 8, 3), np.uint8),
                        intrinsics=np.diag([100, 100, 1]), world_from_cam=np.eye(4), depth_source='sensor',
                        depth=np.ones((6, 8)))
    (tmp_path/'task-result.json').write_text(json.dumps(dict(task='saved task')))
    (tmp_path/'plan.json').write_text(json.dumps(completed_plan()))
    preview = preview_saved_task(tmp_path)
    assert preview.state['final_grasp'] is not None
    assert preview.state['grasps'] == []
    assert not (tmp_path/'registered-pose.json').exists()


@pytest.mark.parametrize('source', ['plan', 'virtual_observation'])
def test_saved_candidates_use_recorded_db_alignment(tmp_path, monkeypatch, source):
    correction = np.diag([-1., -1., 1., 1.]).tolist()
    np.savez(tmp_path/'observation.npz', observation_id='saved', rgb=np.zeros((6,8,3),np.uint8),
             intrinsics=np.eye(3), world_from_cam=np.eye(4), depth=np.ones((6,8)),
             camera_preparation=json.dumps({'db_grasp_from_tcp': correction} if source == 'virtual_observation' else {}))
    (tmp_path/'task-result.json').write_text(json.dumps(dict(task='saved task')))
    (tmp_path/'registered-pose.json').write_text(json.dumps(dict(class_id='fixture', pose={'world_from_cad': np.eye(4).tolist()})))
    (tmp_path/'db.json').write_text(json.dumps({'data': [{'score':1., 'grasp_mat':np.eye(4).tolist()}]}))
    if source == 'plan':
        (tmp_path/'plan.json').write_text(json.dumps({'db_grasp_from_tcp': correction}))
    monkeypatch.setattr('tiptop.object_catalog.registered_object', lambda _: {'db':tmp_path/'db.json'})
    received = []
    monkeypatch.setattr(TaskPreview, 'grasps', lambda self, candidates, *a, **kw: received.append(candidates))
    monkeypatch.setattr(TaskPreview, 'publish_plan', lambda *a: None)
    preview_saved_task(tmp_path)
    np.testing.assert_allclose(received[0]['world_from_tcp'], [correction])


def test_side_view_keeps_metric_heights_and_uses_the_final_commanded_gripper(tmp_path):
    preview = snapshot(tmp_path)
    vertices = np.array([[.1, .2, .03], [.2, .2, .03], [.1, .3, .13], [.2, .3, .13]])
    camera = np.eye(4)
    # Camera roll must not turn height into a camera-depth measurement.
    camera[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    preview.side_scene(vertices, .03, camera)
    scene = preview.state['side_scene']
    assert scene['viewpoint'] == 'right_table_end'
    assert scene['view_label'] == 'Right table 끝 → 로봇'
    np.testing.assert_array_equal(scene['horizontal_world'], [-1, 0, 0])
    np.testing.assert_allclose(np.asarray(scene['points'])[:, 0], [.05, -.05, .05, -.05])
    np.testing.assert_allclose(np.asarray(scene['points'])[:, 1], [0, 0, .1, .1])
    assert scene['object_top_m'] == pytest.approx(.1)
    # Moving/rotating the RGB camera must not change the operator's side view.
    preview.side_scene(vertices, .03, np.eye(4))
    np.testing.assert_allclose(preview.state['side_scene']['points'], scene['points'])
    preview.publish_plan(completed_plan(), np.eye(4), np.diag([100, 100, 1]))
    final = preview.state['final_grasp']
    side = final['side']
    assert side['axes'][0][1] == pytest.approx(final['world_from_tcp'][2][3]-.03)
    assert side['axes'][0][0] == pytest.approx(.15-final['world_from_tcp'][0][3])
    assert len(side['outlines']) == 3
    tcp = np.asarray(final['world_from_tcp'])
    for name, vertices in gripper_mesh_parts(str(preview.urdf_path)):
        if 'finger' in name:
            from tiptop.perception_preview import finger_tip

            world_tip = finger_tip(vertices) @ tcp[:3, :3].T + tcp[:3, 3]
            shown = next(part['tip'] for part in side['outlines'] if part['part'] == name)
            np.testing.assert_allclose(shown, [.15-world_tip[0], world_tip[2]-.03])
    lowest = min((vertices @ tcp[:3, :3].T + tcp[:3, 3])[:, 2].min()
                 for name, vertices in gripper_mesh_parts(str(preview.urdf_path)) if 'finger' in name)
    assert side['finger_lowest_m'] == pytest.approx(lowest-.03)


def test_layers_preserve_rgb_and_missing_depth_is_black(tmp_path):
    preview = snapshot(tmp_path)
    mask = np.zeros((1, 1, 6, 8), bool)
    mask[0, 0, 2:4, 3:5] = True
    preview.update("detection", dict(masks=mask, bboxes=[dict(label="bottle", box_2d=[0, 0, 500, 500])]))
    state = json.loads((preview.directory / "state.json").read_text())
    assert state["observation_id"] == "first"
    np.testing.assert_array_equal(Image.open(preview.directory / state["images"]["rgb"]), 128)
    assert not np.asarray(Image.open(preview.directory / state["images"]["depth"])).any()
    overlay = np.asarray(Image.open(preview.directory / state["images"]["mask"]))
    assert overlay[2, 3, 3] > 0 and overlay[0, 0, 3] == 0
    assert state["depth_valid_percent"] == 0


def test_sam3_target_moves_to_the_current_candidate_without_changing_scene_layers(tmp_path):
    preview = snapshot(tmp_path)
    boxes = [dict(label=f"bottle__instance_{i}", class_id="bottle", instance_index=i,
                  box_2d=[0, i*250, 500, (i+1)*250]) for i in range(3)]
    masks = np.zeros((3, 1, 6, 8), bool)
    for i in range(3):
        masks[i, 0, 1:4, i*2:i*2+2] = True
    preview.update("detection", dict(bboxes=boxes, masks=masks, semantic_source="sam3_reference",
                                     grounded_atoms=[dict(predicate="on", args=[boxes[0]["label"], "table"])]))
    assert preview.state["target_label"] == boxes[0]["label"]
    mask_file = preview.state["images"]["mask"]
    preview.update("target", dict(label=boxes[1]["label"]))
    state = json.loads((preview.directory / "state.json").read_text())
    assert state["target_label"] == boxes[1]["label"]
    assert state["boxes"] == boxes and state["images"]["mask"] == mask_file
    assert state["target_final"] is False  # candidates are still being verified
    preview.update("target_final", {})
    assert json.loads((preview.directory / "state.json").read_text())["target_final"] is True


def test_old_worker_snapshots_use_the_saved_selected_instance_not_the_first_box(tmp_path):
    from tiptop.scripts.serve_perception_viewer import snapshot_target_label

    boxes = [dict(label=f"bottle__instance_{i}") for i in range(4)]
    data = dict(detector="sam3_reference", boxes=boxes)
    semantic = tmp_path / "semantic-task.json"
    for index in (0, 1, 3):
        semantic.write_text(json.dumps(dict(grounded_atoms=[dict(predicate="on", args=[boxes[index]["label"], "table"])])))
        assert snapshot_target_label(data, tmp_path) == boxes[index]["label"]
    # During verification the live target is newer than the saved semantic task.
    data["target_label"] = boxes[2]["label"]
    assert snapshot_target_label(data, tmp_path) == boxes[2]["label"]
    del data["target_label"]
    semantic.write_text("{")
    (tmp_path / "registered-pose.json").write_text(json.dumps(dict(label=boxes[1]["label"])))
    assert snapshot_target_label(data, tmp_path) == boxes[1]["label"]


def test_obstacle_preview_uses_actual_world_mesh_and_samples_only_observed_points(tmp_path):
    from types import SimpleNamespace
    import trimesh
    from curobo.geom.types import Mesh
    from tiptop.perception_preview import build_obstacle_scene

    hull = trimesh.creation.box(extents=[.03, .05, .09])
    mesh = Mesh(name="target", vertices=hull.vertices.tolist(), faces=hull.faces.tolist(),
                pose=[.2, .6, .08, 2**-.5, 0, 0, 2**-.5])
    before = np.asarray(mesh.vertices).copy()
    points = np.arange(90, dtype=float).reshape(-1, 3)/1000
    boxes = [dict(label="target", class_id="bottle"), dict(label="other", class_id="bottle")]
    objects = dict(target=dict(world_xyz=points), other=dict(world_xyz=[[np.nan, 0, 0], [.4, .7, .1]]))
    scene = build_obstacle_scene(dict(target=mesh), objects, boxes, .03, "target", point_limit=8)
    target, other = scene["objects"]
    np.testing.assert_allclose(target["vertices"], hull.vertices @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) + [.2, .6, .08], atol=1e-5)
    assert len(target["points"]) == 8 and target["observed_points"] == 30
    assert target["collision_mesh"] and len(target["faces"]) == 12
    assert not other["collision_mesh"] and other["points"] == [[.4, .7, .1]]
    assert scene["target_label"] == "target" and scene["support_z_m"] == .03
    np.testing.assert_array_equal(mesh.vertices, before)
    np.testing.assert_array_equal(objects["target"]["world_xyz"], points)

    preview = snapshot(tmp_path)
    preview.obstacles(SimpleNamespace(table_cuboid=SimpleNamespace(pose=[0, 0, -.005], dims=[1, 1, .03]),
                                     grasps={"target": {}}, object_meshes={"target": mesh}),
                      dict(objects=objects, bboxes=boxes))
    saved = json.loads((preview.directory / "obstacles.json").read_text())
    assert saved == preview.state["obstacle_scene"]
    assert saved["support_z_m"] == pytest.approx(.03)
    assert len(saved["objects"]) == 2
    preview.obstacles(SimpleNamespace(table_cuboid=SimpleNamespace(pose=[0, 0, -.005], dims=[1, 1, .03]),
                                     grasps={"target": {}}, object_meshes={"target": mesh}),
                      dict(objects=objects, bboxes=boxes, surrounding_obstacles=False),
                      collision_labels={"target"})
    saved = json.loads((preview.directory / "obstacles.json").read_text())
    assert saved == preview.state["obstacle_scene"]
    assert saved["surrounding_obstacles"] is False
    assert [obj["label"] for obj in saved["objects"]] == ["target"]
    assert len(boxes) == len(objects) == 2


def test_obstacle_camera_matches_rgb_optical_projection():
    from tiptop.perception_preview import rgb_camera_view

    world_from_cam = np.array([[0, -1, 0, .2], [1, 0, 0, .6], [0, 0, 1, .4], [0, 0, 0, 1.]])
    intrinsics = np.array([[500, 0, 320], [0, 510, 240], [0, 0, 1.]])
    camera = rgb_camera_view(world_from_cam, intrinsics, 640, 480)
    point_cam = np.array([.1, -.2, 1., 1.])
    world = world_from_cam @ point_cam
    restored = np.asarray(camera["camera_from_world"]) @ world
    pixel = np.asarray(camera["intrinsics"]) @ restored[:3]
    np.testing.assert_allclose(pixel[:2]/pixel[2], [370, 138])
    assert camera["width"] == 640 and camera["height"] == 480


def test_viewer_recovers_rgb_camera_for_each_new_task_from_an_older_worker(tmp_path):
    from tiptop.perception_preview import rgb_camera_view

    server = make_server(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for index in (1, 2):
            output = tmp_path / f"task-{index}"
            preview = snapshot(output, str(index))
            # The observation packet already includes the 180-degree RGB rotation.
            world_from_cam = np.diag([-1., -1., 1., 1.])
            world_from_cam[:3, 3] = [index*.1, .7, 1.]
            intrinsics = np.array([[500., 0, 317.8], [0, 510., 250.3], [0, 0, 1.]])
            np.savez(output / "observation.npz", world_from_cam=world_from_cam, intrinsics=intrinsics)
            preview.update("obstacles", dict(objects=[]))
            with urlopen(f"http://127.0.0.1:{server.server_port}/state") as response:
                data = json.load(response)
            assert data["observation_id"] == str(index)
            assert data["obstacle_scene"]["camera"] == rgb_camera_view(
                world_from_cam, intrinsics, preview.state["width"], preview.state["height"])
            # Recovering display metadata does not alter the saved worker state.
            assert "camera" not in json.loads((preview.directory / "state.json").read_text())["obstacle_scene"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("rotation_deg", [0, 180])
def test_obstacle_camera_respects_the_already_rotated_rgb_packet(rotation_deg):
    from tiptop.perception_preview import rgb_camera_view

    width, height = 640, 480
    intrinsics = np.array([[500., 0, 301.2], [0, 510., 228.7], [0, 0, 1.]])
    native_from_rotated = np.eye(4)
    native_pixel = np.array([351.2, 126.7])
    if rotation_deg == 180:
        native_from_rotated[:2, :2] *= -1
        intrinsics[0, 2] = width - 1 - intrinsics[0, 2]
        intrinsics[1, 2] = height - 1 - intrinsics[1, 2]
    camera = rgb_camera_view(native_from_rotated, intrinsics, width, height)
    point = np.asarray(camera["camera_from_world"]) @ [.1, -.2, 1., 1.]
    pixel = np.asarray(camera["intrinsics"]) @ point[:3]
    expected = [width-1, height-1] - native_pixel if rotation_deg else native_pixel
    np.testing.assert_allclose(pixel[:2]/pixel[2], expected)


@pytest.mark.parametrize("scene_first", [True, False])
def test_archived_candidate_side_matches_its_own_pose_in_either_update_order(tmp_path, scene_first):
    preview = snapshot(tmp_path)
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, .1]])
    poses = np.repeat(np.diag([1., -1., -1., 1.])[None], 2, axis=0)
    poses[:, :3, 3] = [[0, 0, 1], [.2, 0, 1.2]]
    if scene_first:
        preview.side_scene(vertices, 0, np.eye(4))
    preview.grasps(dict(world_from_tcp=poses, candidate_indices=[7, 8], scores=[.9, .8]),
                   np.eye(4), np.diag([100, 100, 1]))
    if not scene_first:
        preview.side_scene(vertices, 0, np.eye(4))
    first, second = preview.state["grasps"]
    np.testing.assert_allclose(np.asarray(second["side"]["axes"])[0]-np.asarray(first["side"]["axes"])[0], [-.2, .2])
    assert preview.state["final_grasp"] is None


def test_viewer_follows_new_observation_and_has_no_control_endpoint(tmp_path):
    first = snapshot(tmp_path / "task-001", "first")
    server = make_server(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/state") as response:
            state = json.load(response)
        assert state["observation_id"] == "first"
        with urlopen(url + state["images"]["rgb"]) as response:
            assert response.headers["Content-Type"] == "image/png"
        second = snapshot(tmp_path / "task-002", "second")
        with urlopen(url + "/state") as response:
            assert json.load(response)["observation_id"] == "second"
        # Previous images remain immutable; overlays cannot borrow the new task's RGB.
        assert (first.directory / first.state["images"]["rgb"]).is_file()
        assert (second.directory / "state.json").is_file()
        for route in ("/execute", "/image?path=../../etc/passwd"):
            with pytest.raises(HTTPError) as error:
                urlopen(url + route)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + "/execute", data=b"{}"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_visual_frontend_publishes_depth_and_detection_independently(monkeypatch):
    from tiptop.perception.cameras.frame import Frame
    from tiptop import perception_wrapper as frontend
    from tiptop.perception import sam3_reference

    events = []
    depth_done = threading.Event()

    async def depth(*args):
        # Detection must have reached the UI before this branch finishes.
        for _ in range(100):
            if depth_done.is_set():
                break
            await asyncio.sleep(.001)
        assert depth_done.is_set()
        return dict(valid_depth_mask=np.ones((2, 3), bool), xyz_map=np.zeros((2, 3, 3)),
                    rgb_map=np.zeros((2, 3, 3)), depth_map=np.ones((2, 3)))

    async def detect(*args, **kwargs):
        return dict(bboxes=[dict(label="bottle")], masks=np.ones((1, 1, 2, 3), bool), grounded_atoms=[])

    def update(event, data):
        events.append(event)
        if event == "detection":
            depth_done.set()

    monkeypatch.setattr(frontend, "predict_scene_depth", depth)
    monkeypatch.setattr(sam3_reference, "detect_reference", detect)
    frame = Frame("test", 0, np.zeros((2, 3, 3), np.uint8), np.eye(3))
    result = asyncio.run(frontend.run_visual_perception(
        None, frame, np.eye(4), "unused", perception_mode="sam3_reference", on_update=update))
    assert events == ["detection", "depth"]
    assert result["objects"]["bottle"]["valid_pixels"] == 6
