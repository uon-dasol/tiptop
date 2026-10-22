"""Meaningful frame/identity/staleness checks for the isolated prototype."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.grasping.known_scene_adapter import (
    Instance,
    assert_scene_matches,
    db_to_planner,
    resolve_instance,
    rigid,
    scene_digest,
)


def T(xyz=(0, 0, 0), angles=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    t[:3, 3] = xyz
    return t


class Checks(unittest.TestCase):
    def setUp(self):
        self.instance = Instance(
            "pepper-01", "black_pepper_shaker", T((0.55, -0.12, 0.44), (0.7, -0.2, 0.3)), "simulation_ground_truth"
        )
        self.rows = [
            dict(
                candidate_index=i,
                score=s,
                scene_id="0029",
                source_pregrasp_index=i,
                grasp_mat=T((0.01, 0.02, 0.03), (0.1, 0.2, 0.3)),
            )
            for i, s in [(3, 0.6), (4, 0.8)]
        ]

    def test_rotated_and_recentered_planner_mesh(self):
        planner = T((0.6, -0.1, 0.5), (0.3, 0.4, -0.6))
        out = db_to_planner(self.rows, self.instance, planner)
        for pose, metadata in zip(out["grasps_obj"], out["metadata"]):
            row = next(r for r in self.rows if r["candidate_index"] == metadata["candidate_index"])
            np.testing.assert_allclose(planner @ pose, self.instance.world_from_cad @ row["grasp_mat"], atol=1e-12)

    def test_separate_cad_db_and_tcp_mount_frames(self):
        cad_db = T((0.001, -0.002, 0), (0, 0, 0.4))
        grasp_tcp = T((0, 0, 0.01), (0.2, 0, 0))
        out = db_to_planner(self.rows, self.instance, np.eye(4), cad_from_db=cad_db, grasp_from_tcp=grasp_tcp)
        np.testing.assert_allclose(
            out["world_from_tcp"][0], self.instance.world_from_cad @ cad_db @ self.rows[1]["grasp_mat"] @ grasp_tcp
        )

    def test_shared_asset_needs_instance_grounding(self):
        other = Instance("pepper-02", self.instance.asset_key, T((0.7, 0.1, 0.44)), "simulation_ground_truth")
        with self.assertRaisesRegex(ValueError, "matched 2"):
            resolve_instance([self.instance, other], asset_key=self.instance.asset_key)
        self.assertIs(resolve_instance([self.instance, other], instance_id="pepper-02"), other)

    def test_duplicate_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resolve_instance([self.instance, self.instance], instance_id="pepper-01")

    def test_units_camera_and_reflection_rejected(self):
        with self.assertRaises(ValueError):
            Instance("a", "b", np.eye(4), "camera_pose", frame_id="camera")
        with self.assertRaises(ValueError):
            Instance("a", "b", np.eye(4), "unknown", units="mm")
        reflection = np.eye(4)
        reflection[0, 0] = -1
        with self.assertRaises(ValueError):
            rigid(reflection)

    def test_stale_scene_with_same_revision_rejected(self):
        args = dict(
            scene_revision=1, model_hashes={"arm": "abc"}, table_pose=np.eye(4), table_dimensions=[0.8, 0.8, 0.04]
        )
        one = scene_digest(instances=[self.instance], **args)
        moved = Instance(
            "pepper-01", self.instance.asset_key, T((0.551, -0.12, 0.44), (0.7, -0.2, 0.3)), "simulation_ground_truth"
        )
        two = scene_digest(instances=[moved], **args)
        with self.assertRaisesRegex(ValueError, "Scene changed"):
            assert_scene_matches(one, two)

    def test_score_and_raw_provenance_stay_paired(self):
        out = db_to_planner(self.rows, self.instance, np.eye(4))
        self.assertEqual([m["candidate_index"] for m in out["metadata"]], [4, 3])
        self.assertAlmostEqual(float(out["confidences_pt"][0]), 0.8)
        self.assertFalse(out["physical_grasp_verified"])

    def test_input_pose_is_immutable(self):
        original = T((0.55, 0, 0.4))
        obj = Instance("a", "b", original, "test")
        original[0, 3] = 99
        self.assertEqual(obj.world_from_cad[0, 3], 0.55)
        with self.assertRaises(ValueError):
            obj.world_from_cad[0, 3] = 99


if __name__ == "__main__":
    unittest.main(verbosity=2)
