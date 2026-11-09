"""Mask scene points that lie inside the robot's collision spheres."""

import numpy as np


def points_inside_spheres(points, spheres, margin_m=0.0):
    """Boolean mask of finite points within `radius + margin_m` of any sphere centre.

    Only points inside the spheres' joint bounding box, then inside each sphere's own box, get an exact
    distance test, so a mostly empty robot volume costs far less than testing every point against every sphere.
    """
    points = np.asarray(points, float)
    spheres = np.asarray(spheres, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N, 3] points, got {points.shape}")
    if spheres.ndim != 2 or spheres.shape[1] != 4:
        raise ValueError(f"Expected [S, 4] spheres (x, y, z, radius), got {spheres.shape}")
    inside = np.zeros(len(points), bool)
    if not len(spheres):
        return inside
    centers, radii = spheres[:, :3], spheres[:, 3] + margin_m
    low, high = (centers - radii[:, None]).min(0), (centers + radii[:, None]).max(0)
    candidates = np.flatnonzero(np.isfinite(points).all(1) & (points >= low).all(1) & (points <= high).all(1))
    near = points[candidates]
    for center, radius in zip(centers, radii):
        box = np.flatnonzero((np.abs(near - center) <= radius).all(1))
        if box.size:
            hit = np.linalg.norm(near[box] - center, axis=1) <= radius
            inside[candidates[box[hit]]] = True
    return inside
