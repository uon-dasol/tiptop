"""Geometric equivalence and stopping behavior of the CAD candidate filter."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from tiptop.doosan.grasp_screening import screen_cad_grasps


def reference_indices(tcp, pose, tool, candidates, hull):
    """Original full point/triangle test, independent of the new plane shortcut."""
    accepted = []
    inverse = np.linalg.inv(pose)
    for index in candidates:
        transform = inverse @ tcp[index]
        local = tool[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        _, distance, _ = trimesh.proximity.closest_point_naive(hull, local)
        inside = (np.einsum("sfj,fj->sf", local[:, None] - hull.triangles_center[None], hull.face_normals)
                  <= 1e-9).all(axis=1)
        if (np.where(inside, -distance, distance) - tool[:, 3]).min() >= .0005:
            accepted.append(int(index))
    return accepted


def transforms(positions):
    matrices = np.repeat(np.eye(4)[None], len(positions), axis=0)
    matrices[:, :3, 3] = positions
    return matrices


def test_face_edge_corner_inside_and_clearance_boundary_match_original():
    hull = trimesh.creation.box(extents=[2, 2, 2]).convex_hull
    tcp = transforms([[1.1005 + offset, 0, 0] for offset in [-1e-10, 0, 1e-10]]
                     + [[1.08, 1.08, 0], [1.06, 1.06, 0], [0, 0, 0], [2, 0, 0]])
    tool = np.array([[0, 0, 0, .1]])
    result = screen_cad_grasps(tcp, np.eye(4), tool, np.arange(len(tcp)), hull)
    assert result.indices == reference_indices(tcp, np.eye(4), tool, np.arange(len(tcp)), hull)
    assert 3 in result.indices  # Clears the corner although no single plane certifies clearance.
    assert 4 not in result.indices and 5 not in result.indices
    assert result.distance_queries > 0 and result.complete


def test_all_gripper_spheres_must_clear_and_ordered_prefix_stops_before_the_tail():
    hull = trimesh.creation.box(extents=[2, 2, 2]).convex_hull
    # Sphere 2 makes the first candidate collide, although sphere 1 clears.
    tool = np.array([[0, 0, 0, .1], [-.3, 0, 0, .1]])
    tcp = transforms([[1.2, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    candidates = np.array([0, 2, 1, 3])  # Input order is the supplied score order.
    result = screen_cad_grasps(tcp, np.eye(4), tool, candidates, hull, limit=2)
    assert result.indices == reference_indices(tcp, np.eye(4), tool, candidates, hull)[:2] == [2, 1]
    assert result.checked == 3 and result.available == 4 and not result.complete
    assert result.distance_queries == 0  # Inside rejection and well-separated acceptances.


@pytest.mark.parametrize("seed", [0, 41, 183])
def test_rotated_object_and_tool_geometry_match_original_over_random_candidates(seed):
    rng = np.random.default_rng(seed)
    hull = trimesh.convex.convex_hull(rng.normal(size=(36, 3)) * [.11, .07, .2])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    pose[:3, 3] = [.3, .7, -.1]
    local_tcp = transforms(rng.uniform(-.4, .4, (128, 3)))
    local_tcp[:, :3, :3] = Rotation.random(128, random_state=rng).as_matrix()
    tcp = pose @ local_tcp
    tool = np.column_stack([rng.uniform(-.04, .04, (4, 3)), rng.uniform(.005, .05, 4)])
    candidates = rng.permutation(len(tcp))
    expected = reference_indices(tcp, pose, tool, candidates, hull)
    result = screen_cad_grasps(tcp, pose, tool, candidates, hull, limit=48)
    assert result.indices == expected[:48]
    assert result.checked <= len(candidates)


def test_empty_candidate_set_returns_no_grasps():
    result = screen_cad_grasps(np.empty((0, 4, 4)), np.eye(4), np.array([[0, 0, 0, .1]]), [],
                               trimesh.creation.box().convex_hull)
    assert result.indices == [] and result.checked == 0 and result.complete
