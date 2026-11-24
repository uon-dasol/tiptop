import numpy as np
import pytest

from tiptop.doosan.leg_blend import blend_legs

DT = 0.02


def smooth_leg(start, end, seconds):
    s = np.linspace(0, 1, round(seconds / DT) + 1)[:, None]
    profile = 10 * s**3 - 15 * s**4 + 6 * s**5
    position = start + profile * (end - start)
    rate = (30 * s**2 - 60 * s**3 + 30 * s**4) / seconds
    return position, rate * (end - start)


def legs():
    home, approach, grasp = np.zeros(6), np.array([.3, .2, -.1, 0, .1, 0]), np.array([.3, .2, -.1, 0, .1, .1])
    first, v1 = smooth_leg(home, approach, 1.5)
    second, v2 = smooth_leg(approach, grasp, 0.6)
    tcp = lambda q: q[:, :3] * 0.5 + q[:, 3:6] * 2.0  # a TCP that moves with the joints
    return first, v1, second, v2, tcp


def test_blend_is_continuous_ends_at_the_grasp_and_saves_time():
    first, v1, second, v2, tcp = legs()
    position, velocity, saved = blend_legs(first, v1, second, v2, DT, tcp(first), tcp(second), radius_m=0.02)
    assert saved > 0
    assert len(position) == len(first) + len(second) - 1 - saved
    np.testing.assert_allclose(position[0], first[0])
    np.testing.assert_allclose(position[-1], second[-1])
    assert np.abs(velocity[[0, -1]]).max() < 1e-6
    # The path never stops in the middle: a stop would show as a joint-speed minimum near zero.
    assert np.linalg.norm(velocity[5:-5], axis=1).min() > 1e-3
    assert np.abs(np.diff(position, axis=0)).max() < 0.02  # no jump at the connection
    limit = max(np.abs(v1).max(), np.abs(v2).max())
    assert np.abs(velocity).max() <= limit + 1e-9


def test_blend_keeps_the_final_descent_untouched():
    first, v1, second, v2, tcp = legs()
    position, _, _ = blend_legs(first, v1, second, v2, DT, tcp(first), tcp(second), radius_m=0.02)
    np.testing.assert_allclose(position[-5:], second[-5:])


def test_blend_refuses_legs_that_do_not_meet():
    first, v1, second, v2, tcp = legs()
    with pytest.raises(ValueError, match="start where"):
        blend_legs(first, v1, second + 0.01, v2, DT, tcp(first), tcp(second))


def test_blend_returns_none_when_the_legs_are_too_short_to_cut():
    first, v1, second, v2, tcp = legs()
    assert blend_legs(first, v1, second, v2, DT, tcp(first), tcp(second), radius_m=5.0) is None


def test_blend_can_cut_the_second_leg_farther_from_the_stop_than_the_first():
    first, v1, second, v2, tcp = legs()
    near = blend_legs(first, v1, second, v2, DT, tcp(first), tcp(second), radius_m=0.02)
    far = blend_legs(first, v1, second, v2, DT, tcp(first), tcp(second), radius_m=0.02, second_radius_m=0.06)
    assert near is not None and far is not None
    assert far[2] + 5 >= near[2]  # replacing more of the second leg changes the time saved by only a few samples
    np.testing.assert_allclose(far[0][-5:], second[-5:])

