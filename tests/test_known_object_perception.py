"""CPU boundaries for sensor pose -> UON DB, without loading models or devices."""

import asyncio
import io
import json

import numpy as np
import pytest

from tiptop.doosan.grasp_database import load_database_grasps
from tiptop.perception.cameras.frame import Frame
from tiptop.perception.foundation_pose import estimate_pose
from tiptop.perception.known_objects import estimate_registered_grasps
from simulation.isaac_sim.observation_server import require_current_result


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


def test_current_rgb_mask_depth_pose_and_db_join_without_ground_truth(tmp_path):
    frame = Frame("current-camera", 12.0, np.full((4, 4, 3), 17, np.uint8), np.diag([100.0, 100.0, 1.0]))
    depth = np.full((4, 4), 0.6, np.float32)
    depth[0, 0] = np.nan
    visual = dict(bboxes=[dict(label="pepper")], masks=np.ones((1, 1, 4, 4), bool), depth_map=depth)
    world_cam = np.eye(4)
    world_cam[:3, 3] = [0.3, 0.1, 0.2]
    entry = dict(
        class_id="pepper", aliases=["pepper"], cad=tmp_path / "mesh.obj", cad_scale_to_m=0.001, db=make_db(tmp_path)
    )
    session = PoseSession()
    result = asyncio.run(
        estimate_registered_grasps(
            session,
            frame,
            visual,
            world_cam,
            entry=entry,
            label="pepper",
            server_url="http://local-model",
            world_from_planner_object=np.eye(4),
            grasp_from_tcp=np.eye(4),
        )
    )
    assert len(session.packets) == 1
    np.testing.assert_array_equal(session.packets[0]["rgb"], frame.rgb)
    assert session.packets[0]["depth"][0, 0] == 0
    assert not any("world" in key or "ground_truth" in key for key in session.packets[0])
    np.testing.assert_allclose(result["pose"]["world_from_cad"][:3, 3], [0.32, 0.07, 0.8])
    assert len(result["grasps"]["candidate_indices"]) == 2
    assert not result["grasps"]["physical_grasp_verified"]


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


@pytest.mark.parametrize("result_id,current_q", [("previous-frame", np.zeros(6)), ("current-frame", np.ones(6) * 0.02)])
def test_live_result_rejects_wrong_frame_or_changed_robot(result_id, current_q):
    with pytest.raises(ValueError):
        require_current_result(result_id, "current-frame", current_q, np.zeros(6))
    require_current_result("current-frame", "current-frame", np.zeros(6), np.zeros(6))
