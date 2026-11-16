"""CPU boundaries for sensor pose -> UON DB, without loading models or devices."""

import asyncio
import io
import json

import numpy as np
import pytest

from simulation.isaac_sim.observation_server import require_current_result
from tiptop.doosan.grasp_database import load_database_grasps
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.foundation_pose import estimate_pose
from tiptop.perception.known_objects import SilhouetteMismatchError, estimate_registered_grasps


def make_db(tmp_path):
    grasp = np.eye(4)
    grasp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    grasp[:3, 3] = [0, 0.02, 0]
    row = dict(grasp_mat=grasp.tolist(), score=0.8, scene_id="scene0001", source_pregrasp_index=7)
    path = tmp_path / "db.json"
    path.write_text(
        json.dumps(
            dict(
                metadata=dict(
                    object="pepper", coordinate_frame="object_zero_pose", gripper_model="UON_3finger_gripper"
                ),
                data=[dict(row, score=0.7), row],
            )
        )
    )
    return path


def test_noncommuting_cad_db_grasp_tcp_and_planner_transforms(tmp_path):
    world_cad = np.eye(4)
    world_cad[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world_cad[:3, 3] = [0.5, 0.1, 0.4]
    cad_db, grasp_tcp, planner = np.eye(4), np.eye(4), np.eye(4)
    cad_db[0, 3], grasp_tcp[2, 3] = 0.1, 0.03
    planner[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    planner[:3, 3] = [0.2, -0.3, 0.4]
    result = load_database_grasps(
        make_db(tmp_path),
        class_id="pepper",
        world_from_cad=world_cad,
        pose_source="rgbd_pose_estimator",
        world_from_planner_object=planner,
        grasp_from_tcp=grasp_tcp,
        cad_from_db=cad_db,
    )
    np.testing.assert_allclose(result["world_from_tcp"][0, :3, 3], [0.51, 0.2, 0.4])
    np.testing.assert_allclose(result["world_from_tcp"][0, :3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(planner @ result["grasps_obj"], result["world_from_tcp"])
    assert result["candidate_indices"].tolist() == [1, 0]
    assert not result["collision_filtered"]
    with pytest.raises(ValueError, match="sensor-estimated"):
        load_database_grasps(
            tmp_path / "not-read.json",
            class_id="pepper",
            world_from_cad=world_cad,
            pose_source="simulator_ground_truth",
            world_from_planner_object=planner,
            grasp_from_tcp=np.eye(4),
        )


class PoseSession:
    def __init__(self, class_id="pepper"):
        self.packets = []
        self.class_id = class_id

    def post(self, url, *, data, **kwargs):
        packet = np.load(io.BytesIO(data), allow_pickle=False)
        self.packets.append({key: packet[key] for key in packet.files})
        pose = np.eye(4)
        pose[:3, 3] = [0.02, -0.03, 0.6]
        result = dict(class_id=self.class_id, camera_from_cad=pose.tolist(), ranking_score=2.0)

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def json(self):
                return result

        return Response()


@pytest.mark.parametrize("label", ["pepper", "pepper__instance_1"])
def test_current_rgb_mask_depth_pose_and_db_join_without_ground_truth(tmp_path, label):
    import trimesh

    frame = Frame("current-camera", 12.0, np.full((4, 4, 3), 17, np.uint8), np.diag([100.0, 100.0, 1.0]))
    depth = np.full((4, 4), 0.6, np.float32)
    depth[0, 0] = np.nan
    bbox = dict(label=label)
    if label != "pepper":
        bbox.update(class_id="pepper", instance_index=1)
    visual = dict(bboxes=[bbox], masks=np.ones((1, 1, 4, 4), bool), depth_map=depth)
    world_cam = np.eye(4)
    world_cam[:3, 3] = [0.3, 0.1, 0.2]
    entry = dict(
        class_id="pepper", aliases=["pepper"], cad=tmp_path / "mesh.obj", cad_scale_to_m=0.001, db=make_db(tmp_path)
    )
    mesh = trimesh.creation.box(extents=[18, 18, 1])
    mesh.apply_translation([-11, 39, 0])
    mesh.export(entry["cad"])
    session = PoseSession()
    result = asyncio.run(
        estimate_registered_grasps(
            session,
            frame,
            visual,
            world_cam,
            entry=entry,
            label=label,
            server_url="http://local-model",
            world_from_planner_object=np.eye(4),
            grasp_from_tcp=np.eye(4),
        )
    )
    assert len(session.packets) == 1
    np.testing.assert_array_equal(session.packets[0]["rgb"], frame.rgb)
    assert str(session.packets[0]["class_id"]) == "pepper"
    assert session.packets[0]["depth"][0, 0] == 0
    assert not any("world" in key or "ground_truth" in key for key in session.packets[0])
    np.testing.assert_allclose(result["pose"]["world_from_cad"][:3, 3], [0.32, 0.07, 0.8])
    assert len(result["grasps"]["candidate_indices"]) == 2
    assert not result["grasps"]["physical_grasp_verified"]
    visual["bboxes"][0]["box_2d"] = [838, 281, 1000, 492]
    with pytest.raises(ValueError, match="image boundary"):
        asyncio.run(
            estimate_registered_grasps(
                session,
                frame,
                visual,
                world_cam,
                entry=entry,
                label=label,
                server_url="http://local-model",
                world_from_planner_object=np.eye(4),
                grasp_from_tcp=np.eye(4),
            )
        )
    assert len(session.packets) == 1


def _bright_object_on_dark_table(mask, table=40, object_value=200):
    rgb = np.full((*mask.shape, 3), table, np.uint8)
    rgb[mask] = object_value
    return rgb


def _shadow_scene():
    """A bright 21x11 box with a dark shadow strip glued to its right side, inside the SAM3 mask."""
    import trimesh

    cad = trimesh.creation.box(extents=[0.1, 0.2, 0.05])
    transform = np.eye(4)
    transform[2, 3] = 1.0
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    box = np.zeros((100, 100), bool)
    box[40:61, 45:56] = True
    shadow = np.zeros_like(box)
    shadow[40:61, 56:61] = True
    mask = box | shadow
    rgb = _bright_object_on_dark_table(box)
    rgb[shadow] = 25                                    # darker than the table
    return cad, transform, k, mask, rgb


def test_dark_shadow_outside_the_cad_is_ignored_and_the_plain_iou_is_reported():
    from tiptop.perception.known_objects import validate_cad_mask_alignment

    cad, transform, k, mask, rgb = _shadow_scene()
    result = validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb, minimum_iou=0.9)
    # The 1 px band next to the CAD stays as tolerance; the rest of the shadow strip is ignored.
    assert result["silhouette_iou"] > 0.9
    assert result["plain_silhouette_iou"] < 0.75
    with pytest.raises(SilhouetteMismatchError) as error:
        validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb, minimum_iou=0.95)
    assert error.value.iou == pytest.approx(result["silhouette_iou"])
    assert error.value.plain_iou == pytest.approx(result["plain_silhouette_iou"])


def test_bright_mask_pixels_outside_the_cad_still_count_as_a_mismatch():
    from tiptop.perception.known_objects import validate_cad_mask_alignment

    cad, transform, k, mask, rgb = _shadow_scene()
    rgb[:, 56:61] = 200                                                  # the "shadow" strip is really bright object
    with pytest.raises(SilhouetteMismatchError):
        validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb, minimum_iou=0.9)


def test_dark_pixels_inside_the_cad_silhouette_are_never_removed():
    from tiptop.perception.known_objects import validate_cad_mask_alignment

    cad, transform, k, mask, rgb = _shadow_scene()
    baseline = validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb)["silhouette_iou"]
    rgb[44:57, 47:54] = 20                                               # a dark part of the object itself, inside the CAD
    assert validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb)["silhouette_iou"] == pytest.approx(baseline)


def test_the_rgb_image_must_match_the_mask_grid():
    from tiptop.perception.known_objects import validate_cad_mask_alignment

    cad, transform, k, mask, rgb = _shadow_scene()
    with pytest.raises(ValueError, match="RGB image"):
        validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb[:50])
    with pytest.raises(ValueError, match="RGB image"):
        validate_cad_mask_alignment(cad, transform, k, mask, rgb=rgb.astype(float))


def test_pose_mask_mismatch_is_rejected_even_when_the_server_returned_a_pose():
    import trimesh

    from tiptop.perception.known_objects import validate_cad_mask_alignment

    def box(scale):
        return trimesh.creation.box(extents=np.array([0.1, 0.2, 0.05]) * scale)

    transform = np.eye(4)
    transform[2, 3] = 1.0
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    mask = np.zeros((100, 100), bool)
    mask[40:61, 45:56] = True
    rgb = _bright_object_on_dark_table(mask)
    result = validate_cad_mask_alignment(box(1), transform, k, mask, rgb=rgb)
    assert result["silhouette_iou"] > 0.9 and not result["semantic_identity_verified"]
    # A pose-fitting server can return a transform for the wrong, larger CAD.
    with pytest.raises(ValueError, match="target identity"):
        validate_cad_mask_alignment(box(1.4), transform, k, mask, rgb=rgb)
    relaxed = validate_cad_mask_alignment(box(1.1), transform, k, mask, rgb=rgb, minimum_iou=0.75)
    assert 0.75 <= relaxed["silhouette_iou"] < 0.8
    with pytest.raises(ValueError, match="target identity"):
        validate_cad_mask_alignment(box(1.1), transform, k, mask, rgb=rgb, minimum_iou=0.8)
    with pytest.raises(ValueError, match="minimum_iou"):
        validate_cad_mask_alignment(box(1), transform, k, mask, rgb=rgb, minimum_iou=float("nan"))


def test_concave_cad_is_not_matched_by_a_mask_filling_its_convex_hull():
    import trimesh

    from tiptop.perception.known_objects import validate_cad_mask_alignment

    ends = [trimesh.creation.box(extents=[0.03, 0.06, 0.002], transform=trimesh.transformations.translation_matrix([x, 0, 0]))
            for x in (-0.06, 0.06)]
    dumbbell = trimesh.util.concatenate([*ends, trimesh.creation.box(extents=[0.09, 0.01, 0.002])])
    transform = np.eye(4)
    transform[2, 3] = 1.0
    k = np.array([[1000.0, 0, 200], [0, 1000.0, 200], [0, 0, 1]])
    shape = np.zeros((400, 400), bool)
    shape[170:230, 125:155] = shape[170:230, 245:275] = shape[195:205, 155:245] = True
    assert validate_cad_mask_alignment(dumbbell, transform, k, shape, rgb=_bright_object_on_dark_table(shape))["silhouette_iou"] > 0.9
    blob = np.zeros((400, 400), bool)
    blob[170:230, 125:275] = True
    with pytest.raises(ValueError, match="target identity"):
        validate_cad_mask_alignment(dumbbell, transform, k, blob, rgb=_bright_object_on_dark_table(blob))


def test_pose_rejects_wrong_cad_identity():
    frame = Frame("camera", 1.0, np.zeros((4, 4, 3), np.uint8), np.diag([100.0, 100.0, 1.0]))
    with pytest.raises(ValueError, match="different CAD"):
        asyncio.run(
            estimate_pose(
                PoseSession(class_id="another_object"),
                "http://local-model",
                frame,
                np.ones((4, 4)),
                np.ones((4, 4), bool),
                mesh_path="mesh.obj",
                mesh_scale=0.001,
                class_id="pepper",
                world_from_cam=np.eye(4),
            )
        )


def test_pose_crop_preserves_native_pixels_and_optical_rays():
    k = np.array([[600., 0, 400.], [0, 610., 300.], [0, 0, 1.]])
    rgb = np.arange(600 * 800 * 3, dtype=np.uint8).reshape(600, 800, 3)
    frame = Frame("camera", 1., rgb, k)
    mask = np.zeros((600, 800), bool)
    mask[250:270, 350:370] = True
    session = PoseSession()
    result = asyncio.run(estimate_pose(
        session, "http://local-model", frame, np.ones((600, 800), np.float32), mask,
        mesh_path="mesh.obj", mesh_scale=.001, class_id="pepper", world_from_cam=np.eye(4),
    ))
    packet = session.packets[0]
    assert result["input_crop_xyxy"] == [250, 150, 470, 370]
    np.testing.assert_array_equal(packet["rgb"], rgb[150:370, 250:470])
    np.testing.assert_array_equal(packet["mask"], mask[150:370, 250:470])
    np.testing.assert_allclose(
        np.linalg.inv(packet["intrinsics"]) @ [360 - 250, 260 - 150, 1],
        np.linalg.inv(k) @ [360, 260, 1],
    )
    np.testing.assert_allclose(k[:2, 2], [400, 300])


@pytest.mark.parametrize("result_id,current_q", [("previous-frame", np.zeros(6)), ("current-frame", np.ones(6) * 0.02)])
def test_live_result_rejects_wrong_frame_or_changed_robot(result_id, current_q):
    with pytest.raises(ValueError):
        require_current_result(result_id, "current-frame", current_q, np.zeros(6))
    require_current_result("current-frame", "current-frame", np.zeros(6), np.zeros(6))


class _Candidates:
    """Fakes for the candidate loop: no models, no catalog files."""

    @staticmethod
    def results(scores):
        masks = np.ones((1, 1, 4, 6), bool)
        candidates = [
            dict(index=i, score=s, appearance=None, masks=masks, bboxes=[dict(label="pepper", box_2d=[0, 0, 1, 1])])
            for i, s in enumerate(scores)
        ]
        value = dict(
            bboxes=candidates[0]["bboxes"], masks=masks, grounded_atoms=[], depth_map=np.full((4, 6), 1.0),
            reference_candidates=candidates,
        )
        return value

    @staticmethod
    def patch(monkeypatch, outcomes):
        import trimesh

        from tiptop.perception import known_objects

        calls = []

        async def estimate(session, frame, visual, world_from_cam, *, entry, label, **options):
            outcome = outcomes[len(calls)]
            calls.append(visual["tag"] if "tag" in visual else 0)
            if isinstance(outcome, Exception):
                raise outcome
            return dict(pose=dict(cad_mask_alignment=dict(silhouette_iou=outcome), ranking_score=88.0))

        monkeypatch.setattr(known_objects, "estimate_registered_grasps", estimate)
        monkeypatch.setattr(known_objects, "registered_target", lambda *a, **k: ("pepper", dict(cad="x", cad_scale_to_m=1.0)))
        monkeypatch.setattr(known_objects, "load_cad_mesh", lambda *a: trimesh.creation.box(extents=[0.05] * 3))
        return calls


def _verify(results, switch=lambda r, c: dict(r, bboxes=c["bboxes"], masks=c["masks"], tag=c["index"])):
    from types import SimpleNamespace

    from tiptop.perception.known_objects import estimate_first_matching_candidate

    frame = SimpleNamespace(intrinsics=np.diag([100.0, 100.0, 1.0]))
    return asyncio.run(
        estimate_first_matching_candidate(None, frame, results, np.eye(4), switch_candidate=switch, cad_mask_min_iou=0.75)
    )


def test_a_rejected_first_candidate_hands_over_to_the_next_one(monkeypatch, capsys):
    calls = _Candidates.patch(monkeypatch, [SilhouetteMismatchError(0.66, 0.75), 0.83])
    attempt, label, entry, registered, trials = _verify(_Candidates.results([0.58, 0.57]))
    assert attempt["tag"] == 1 and label == "pepper" and calls == [0, 1]
    assert [(t["rank"], t["accepted"]) for t in trials] == [(1, False), (2, True)]
    assert trials[0]["silhouette_iou"] == pytest.approx(0.66) and trials[1]["silhouette_iou"] == pytest.approx(0.83)
    text = capsys.readouterr().out
    assert "✗ 1번째 후보 거부" in text and "✓ 2번째 후보 채택" in text and "SAM3 점수 0.5700" in text and "앞선 1개" in text


def test_all_candidates_rejected_reports_the_best_iou_and_the_trials(monkeypatch, capsys):
    _Candidates.patch(monkeypatch, [SilhouetteMismatchError(0.66, 0.75), SilhouetteMismatchError(0.71, 0.75)])
    with pytest.raises(SilhouetteMismatchError, match=r"IoU=0.710 < 0.750\); best of 2 SAM3 candidates") as error:
        _verify(_Candidates.results([0.58, 0.57]))
    assert [t["accepted"] for t in error.value.trials] == [False, False]
    assert "모두 실루엣 검사 실패" in capsys.readouterr().out


def test_an_error_other_than_a_silhouette_mismatch_is_not_retried(monkeypatch):
    calls = _Candidates.patch(monkeypatch, [ValueError("pose server down")])
    with pytest.raises(ValueError, match="pose server down"):
        _verify(_Candidates.results([0.58, 0.57]))
    assert len(calls) == 1


def test_a_detection_without_candidates_is_verified_as_the_single_candidate(monkeypatch, capsys):
    calls = _Candidates.patch(monkeypatch, [0.9])
    results = _Candidates.results([0.9])
    del results["reference_candidates"]
    attempt, label, entry, registered, trials = _verify(results)
    assert len(trials) == 1 and trials[0]["score"] is None and calls == [0]
    assert "✓ pepper 채택" in capsys.readouterr().out


def test_target_verification_keeps_all_masks_and_stable_instance_ids(monkeypatch):
    import trimesh

    from tiptop.perception import known_objects
    from tiptop.perception.sam3_reference import decode_reference_result
    from tiptop.perception_wrapper import select_reference_candidate

    masks = np.ones((4, 1, 4, 6), bool)
    masks[1, 0, :2] = False
    detection = decode_reference_result(
        dict(masks=masks, boxes_xyxy=np.tile([1, 1, 5, 3], (4, 1)), scores=[.9, .8, .7, .6],
             metadata=np.array(json.dumps(dict(class_id="pepper")))),
        (4, 6, 3), "pepper",
    )
    results = dict(detection, depth_map=np.ones((4, 6)), xyz_map=np.zeros((4, 6, 3)),
                   rgb_map=np.zeros((4, 6, 3)), valid_depth_mask=np.ones((4, 6), bool))
    entry = dict(cad="unused", cad_scale_to_m=1.0)
    monkeypatch.setattr(known_objects, "registered_target", lambda boxes, atoms: (atoms[0]["args"][0], entry))
    monkeypatch.setattr(known_objects, "load_cad_mesh", lambda *a: trimesh.creation.box(extents=[.05] * 3))
    seen = []

    async def estimate(session, frame, visual, world_from_cam, *, label, **options):
        seen.append(label)
        assert visual["masks"] is results["masks"]
        assert visual["bboxes"] is results["bboxes"]
        if len(seen) == 1:
            raise SilhouetteMismatchError(.4, .75)
        return dict(pose=dict(cad_mask_alignment=dict(silhouette_iou=.9)))

    monkeypatch.setattr(known_objects, "estimate_registered_grasps", estimate)
    accepted, label, _, _, trials = _verify(results, switch=select_reference_candidate)
    assert seen == ["pepper__instance_0", "pepper__instance_1"]
    assert label == "pepper__instance_1" and len(trials) == 2
    assert len(accepted["bboxes"]) == 4 and len(accepted["objects"]) == 4
    np.testing.assert_array_equal(accepted["masks"], masks)
    assert results["grounded_atoms"][0]["args"][0] == "pepper__instance_0"
