import numpy as np
import pytest
from tiptop.doosan.planning import prepare_joint_path


def test_stationary_endpoint_preserves_sample_count_positions_and_timing():
    positions = np.array([[0] * 6, [.01] * 6])
    velocities = np.array([[0] * 6, [8.8e-5] * 6])
    q, v, durations = prepare_joint_path(positions, velocities, .02)
    np.testing.assert_array_equal(q, positions)
    np.testing.assert_array_equal(v[:-1], velocities[:-1])
    assert np.count_nonzero(v[-1]) == 0
    np.testing.assert_array_equal(velocities[-1], [8.8e-5] * 6)
    assert durations == [.02] * 2


def test_moving_endpoint_cannot_be_relabelled_as_stopped():
    with pytest.raises(ValueError, match='too large'):
        prepare_joint_path([[0] * 6], [[.1] * 6], .02)
