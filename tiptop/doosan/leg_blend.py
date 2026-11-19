"""Join two consecutive legs that stop at a shared point into one path that does not stop in between."""

import numpy as np

DEFAULT_BLEND_RADIUS_M = 0.03


def _excess_limits(position, velocity, acceleration, jerk, limits):
    return (np.abs(velocity).max(0) > limits[0] + 1e-9).any() or (np.abs(acceleration).max(0) > limits[1] + 1e-9).any() \
        or (np.abs(jerk).max(0) > limits[2] + 1e-9).any()


def quintic_connection(start, end, steps, dt):
    """Samples strictly between two (position, velocity, acceleration) states, joined by a quintic in time."""
    duration = steps * dt
    powers = np.arange(6)
    rows = []
    for tau in (0.0, duration):
        rows += [tau ** powers,
                 powers * tau ** np.maximum(powers - 1, 0),
                 powers * np.maximum(powers - 1, 0) * tau ** np.maximum(powers - 2, 0)]
    # Rows are position, velocity and acceleration at the start, then at the end.
    matrix = np.array([rows[0], rows[1], rows[2], rows[3], rows[4], rows[5]], dtype=float)
    boundary = np.stack([start[0], start[1], start[2], end[0], end[1], end[2]])
    coefficients = np.linalg.solve(matrix, boundary)  # [6, dof]
    tau = np.arange(1, steps) * dt
    basis = tau[:, None] ** powers
    d_basis = powers * tau[:, None] ** np.maximum(powers - 1, 0)
    return basis @ coefficients, d_basis @ coefficients


def blend_legs(first, first_velocity, second, second_velocity, dt, first_tcp, second_tcp, *,
                    radius_m=DEFAULT_BLEND_RADIUS_M, max_steps=120):
    """Cut both legs `radius_m` from their shared stop and join the cut ends with one smooth connection.

    `first` and `second` are joint positions [N, dof] sampled every `dt` with their velocities; `second` starts
    where `first` stops.
    `first_tcp`/`second_tcp` are the matching TCP positions [N, 3]. The speed, acceleration and jerk of the joined
    path never exceed the largest the two original legs reach. Returns (position, velocity, steps_saved), or None
    when no connection fits those limits.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    v1, v2 = np.asarray(first_velocity, float), np.asarray(second_velocity, float)
    if not np.allclose(first[-1], second[0], atol=1e-6, rtol=0):
        raise ValueError("The descent must start where the approach stops")
    a1, a2 = np.gradient(v1, dt, axis=0), np.gradient(v2, dt, axis=0)
    jerk = [np.gradient(a, dt, axis=0) for a in (a1, a2)]
    limits = [np.maximum(np.abs(x).max(0), np.abs(y).max(0)) for x, y in ((v1, v2), (a1, a2), tuple(jerk))]
    stop = first_tcp[-1]
    cut_first = np.flatnonzero(np.linalg.norm(first_tcp - stop, axis=1) >= radius_m)
    cut_second = np.flatnonzero(np.linalg.norm(second_tcp - stop, axis=1) >= radius_m)
    if len(cut_first) == 0 or len(cut_second) == 0:
        return None
    i, j = int(cut_first[-1]), int(cut_second[0])
    removed = (len(first) - 1 - i) + j
    start, end = (first[i], v1[i], a1[i]), (second[j], v2[j], a2[j])
    for steps in range(2, min(max_steps, removed) + 1):
        position, velocity = quintic_connection(start, end, steps, dt)
        full_position = np.concatenate([first[: i + 1], position, second[j:]])
        full_velocity = np.concatenate([v1[: i + 1], velocity, v2[j:]])
        full_acceleration = np.gradient(full_velocity, dt, axis=0)
        full_jerk = np.gradient(full_acceleration, dt, axis=0)
        # Only the junction is new; the remaining samples came from the original plans.
        window = slice(max(i - 2, 0), i + steps + 3)
        if not _excess_limits(full_position[window], full_velocity[window], full_acceleration[window],
                              full_jerk[window], limits):
            return full_position, full_velocity, removed - steps
    return None
