"""Prepare cuRobo samples for blocking M1013 execution."""
import math

import numpy as np


def prepare_joint_path(positions, velocities, dt):
    """Preserve sample positions and timing; zero only tiny endpoint residues."""
    q, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) == 0 or v.shape != q.shape:
        raise ValueError('Expected matching nonempty [N, 6] position and velocity arrays')
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Expected finite samples and positive dt')
    if np.max(np.abs(v[-1])) > 5e-3:
        raise ValueError('Terminal velocity is too large for a stationary endpoint')
    v = v.copy()
    v[-1] = 0
    return q, v, [float(dt)] * len(q)
