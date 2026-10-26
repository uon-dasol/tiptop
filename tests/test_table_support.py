"""A single visible bowl bottom must not move the supporting table upward."""
import numpy as np
import open3d as o3d
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from tiptop.perception.known_objects import cad_support_point
from tiptop.perception.segmentation import segment_table_with_ransac


def test_support_comes_from_the_posed_full_cad_including_hidden_vertices():
    vertices = trimesh.creation.box(extents=[0.1, 0.2, 0.05]).vertices
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('x', 25, degrees=True).as_matrix()
    pose[:3, 3] = [0.6, 0.1, 0.45]
    support = cad_support_point(vertices, pose)
    world = vertices @ pose[:3, :3].T + pose[:3, 3]
    assert support[2] == pytest.approx(world[:, 2].min())
    np.testing.assert_allclose(support[:2], world[np.isclose(world[:, 2], world[:, 2].min()), :2].mean(0))


def test_one_object_with_visible_bottom_above_table_uses_cad_support():
    x, y = np.meshgrid(np.linspace(-0.25, 0.25, 100), np.linspace(-0.25, 0.25, 100))
    # Small depth variation avoids an unrelated zero-volume trimesh box.
    z = 0.4 + 0.0003 * np.sin(32 * x) * np.sin(24 * y)
    xyz = np.stack([x, y, z], axis=-1)
    mask = (abs(x) < 0.04) & (abs(y) < 0.04)
    xyz[mask, 2] = 0.445  # The visible bowl interior is 45 mm above the real table.
    o3d.utility.random.seed(0)
    table = segment_table_with_ransac(
        xyz, np.zeros_like(xyz), mask[None, None], support_points_world=np.array([[0, 0, 0.4]])
    )
    assert table.bounds[1, 2] + 0.02 == pytest.approx(0.4, abs=0.001)
    assert np.all(table.extents[:2] > 0.4)


@pytest.mark.parametrize('points', [np.zeros((3,)), np.full((1, 3), np.nan)])
def test_invalid_support_cannot_select_an_arbitrary_table(points):
    with pytest.raises(ValueError, match='Table support points'):
        segment_table_with_ransac(
            np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.ones((1, 1, 4, 4), bool),
            support_points_world=points,
        )
