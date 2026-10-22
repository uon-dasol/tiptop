"""Select a table region supported by this observation, without simulator poses."""

import numpy as np
from scipy.spatial import cKDTree


def select_observed_free_region(
    xyz, *, table_bounds, target_center, footprint_radius, region_size=0.10, min_translation=0.14, resolution=0.01
):
    """Return a small placement surface; occluded/unobserved cells are not free.

    The whole region plus a conservative object footprint must have measured
    table support, and must avoid every observed above-table point. The planner
    still checks the robot, carried object and approach/retreat trajectories.
    """
    points = np.asarray(xyz, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    bounds = np.asarray(table_bounds, float)
    target = np.asarray(target_center, float)
    if (
        bounds.shape != (2, 3)
        or target.shape != (3,)
        or not len(points)
        or not np.isfinite(bounds).all()
        or not np.isfinite(target).all()
        or min(footprint_radius, region_size, min_translation, resolution) <= 0
    ):
        raise ValueError("Placement needs finite observed geometry and positive dimensions")
    # Existing TiPToP table box is lowered 20 mm. Recover the measured support
    # height from nearby points, rather than using its collision-margin height.
    reference = bounds[1, 2] + 0.02
    flat = points[np.abs(points[:, 2] - reference) < 0.008]
    if len(flat) < 30:
        raise ValueError("No sufficiently observed table support")
    height = float(np.median(flat[:, 2]))
    obstacles = points[points[:, 2] > height + 0.012]
    support = cKDTree(flat[:, :2])
    occupancy = cKDTree(obstacles[:, :2]) if len(obstacles) else None
    radius = region_size / np.sqrt(2) + footprint_radius + 0.015
    low, high = bounds[0, :2] + radius, bounds[1, :2] - radius
    grid = [np.arange(low[i], high[i] + resolution * 0.1, resolution * 2) for i in range(2)]
    offsets = np.arange(-radius, radius + resolution * 0.1, resolution)
    dx, dy = np.meshgrid(offsets, offsets)
    disk = np.c_[dx.ravel(), dy.ravel()]
    disk = disk[np.linalg.norm(disk, axis=1) <= radius]
    candidates = []
    for x in grid[0]:
        for y in grid[1]:
            center = np.array([x, y])
            distance = float(np.linalg.norm(center - target[:2]))
            if distance < min_translation:
                continue
            clearance = float(occupancy.query(center)[0]) if occupancy else float("inf")
            if clearance <= radius:
                continue
            # Require nearly every support sample; small depth holes may be
            # interpolated over one grid cell, large unseen patches may not.
            coverage = float(np.mean(support.query(center + disk)[0] <= resolution * 1.5))
            if coverage < 0.98:
                continue
            candidates.append((distance, -coverage, center, coverage, clearance))
    if not candidates:
        raise ValueError("No observed empty region large enough for this object; acquire another view")
    _, _, center, coverage, clearance = min(candidates, key=lambda row: row[:2])
    return dict(
        center_xy_m=center.tolist(),
        size_m=region_size,
        support_z_m=height,
        observed_support_fraction=coverage,
        obstacle_clearance_m=clearance,
        checked_radius_m=float(radius),
        minimum_translation_m=min_translation,
        source="current_rgbd_observed_table",
        unknown_space_is_free=False,
    )
