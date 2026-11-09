"""Robot-sphere point filter must match the exhaustive point-by-sphere test exactly."""

import numpy as np
import pytest

from tiptop.perception.robot_filter import points_inside_spheres


def exhaustive(points, spheres, margin_m):
    distance = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=-1)
    return np.any(distance <= spheres[None, :, 3] + margin_m, axis=1)


@pytest.mark.parametrize("seed", range(5))
def test_matches_exhaustive_test_including_nonfinite_points(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (20000, 3))
    points[rng.choice(len(points), 200)] = np.nan
    points[rng.choice(len(points), 100)] = np.inf
    spheres = np.column_stack([rng.uniform(-0.8, 0.8, (40, 3)), rng.uniform(0.02, 0.15, 40)])
    expected = exhaustive(points, spheres, 0.003)
    result = points_inside_spheres(points, spheres, 0.003)
    assert expected.any()
    np.testing.assert_array_equal(result, expected)


def test_point_on_the_margin_boundary_is_inside():
    spheres = np.array([[0.0, 0.0, 0.0, 0.1]])
    points = np.array([[0.103, 0.0, 0.0], [0.1031, 0.0, 0.0], [0.0, -0.103, 0.0]])
    np.testing.assert_array_equal(points_inside_spheres(points, spheres, 0.003), [True, False, True])


def test_no_spheres_or_points_removes_nothing():
    assert not points_inside_spheres(np.zeros((5, 3)), np.zeros((0, 4))).any()
    assert points_inside_spheres(np.zeros((0, 3)), np.array([[0, 0, 0, 1.0]])).shape == (0,)


def test_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="points"):
        points_inside_spheres(np.zeros((4, 2)), np.zeros((1, 4)))
    with pytest.raises(ValueError, match="spheres"):
        points_inside_spheres(np.zeros((4, 3)), np.zeros((1, 3)))
