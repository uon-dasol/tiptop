"""Select a table region supported by this observation, without simulator poses."""

import numpy as np
from scipy.spatial import cKDTree


def placement_surface_height(
    *, support_z, cad_bottom, sphere_bottom, release_clearance=0.003, planner_buffer=0.002, activation_distance=0.0
):
    """Match cuTAMP's sphere support to a camera-estimated physical CAD bottom.

    Both bottoms use the same object frame and world-up axis. cuTAMP places
    min(z-radius) at surface + activation + 2 mm. Account for that buffer and
    the ACTUAL sampled sphere envelope once, rather than adding another drop
    height to a partial-hull crop correction. No simulator state is used.
    """
    values = np.asarray(
        [support_z, cad_bottom, sphere_bottom, release_clearance, planner_buffer, activation_distance], float
    )
    if not np.isfinite(values).all() or min(release_clearance, planner_buffer, activation_distance) < 0:
        raise ValueError("Placement heights must be finite metres with non-negative clearances")
    return float(support_z + release_clearance + sphere_bottom - cad_bottom - planner_buffer - activation_distance)


def select_observed_free_region(
    xyz,
    *,
    table_bounds,
    target_center,
    footprint_radius,
    region_size=0.10,
    min_translation=0.14,
    resolution=0.01,
    footprint_size_xy=None,
):
    """Return a small placement surface; occluded/unobserved cells are not free.

    Known objects use a rectangular support surface containing their footprint
    plus margins. Legacy radius inputs use a disk containing the centre region
    and footprint. Both need measured support and avoid above-table points. The planner
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
    # A known object's rectangular region is the SUPPORT SURFACE, not a set of
    # possible centres. cuTAMP constrains the object onto this whole surface.
    # Include its measured CAD footprint once, rather than adding it again to
    # a small centre-region disk (which needlessly rejects long handled items).
    size_xy = None
    if footprint_size_xy is not None:
        extent = np.asarray(footprint_size_xy, float)
        if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError("Known placement needs two positive footprint dimensions in metres")
        size_xy = np.maximum(region_size, extent + 0.03)
        half = size_xy / 2 + 0.005
        radius = float(np.linalg.norm(half))
        boundary = half
    else:
        radius = region_size / np.sqrt(2) + footprint_radius + 0.015
        boundary = radius
    low, high = bounds[0, :2] + boundary, bounds[1, :2] - boundary
    grid = [np.arange(low[i], high[i] + resolution * 0.1, resolution * 2) for i in range(2)]
    limits = [radius, radius] if size_xy is None else half
    dx, dy = np.meshgrid(*[np.arange(-value, value + resolution * 0.1, resolution) for value in limits])
    disk = np.c_[dx.ravel(), dy.ravel()]
    if size_xy is None:
        disk = disk[np.linalg.norm(disk, axis=1) <= radius]
    candidates = []
    for x in grid[0]:
        for y in grid[1]:
            center = np.array([x, y])
            distance = float(np.linalg.norm(center - target[:2]))
            if distance < min_translation:
                continue
            clearance = float(occupancy.query(center)[0]) if occupancy else float("inf")
            if occupancy:
                if size_xy is None:
                    occupied = clearance <= radius
                else:
                    nearby = occupancy.query_ball_point(center, radius)
                    occupied = bool(nearby) and bool(
                        np.any(np.all(np.abs(obstacles[nearby, :2] - center) <= half, axis=1))
                    )
                if occupied:
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
    result = dict(
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
    if size_xy is not None:
        result.update(
            size_xy_m=size_xy.tolist(),
            footprint_size_xy_m=extent.tolist(),
            occupancy_shape="rectangle",
            footprint_margin_m=0.015,
            support_margin_m=0.005,
        )
    return result
