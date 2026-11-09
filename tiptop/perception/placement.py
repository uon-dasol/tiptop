"""Placement frame and height helpers, without simulator poses."""

import numpy as np

# segment_table_with_ransac places the top of its table box this far below the measured support plane.
TABLE_BOX_LOWERING_M = 0.02
TABLE_SUPPORT_MODES = ("ransac-cad", "ransac-mask", "declared")
MAX_SUPPORT_GAP_M = 0.03


def table_support_z(table_box_top_z):
    """Measured support-plane height of a table box that was lowered by TABLE_BOX_LOWERING_M."""
    return float(table_box_top_z + TABLE_BOX_LOWERING_M)


def check_support_gap(label, cad_bottom_z, support_z, limit_m=MAX_SUPPORT_GAP_M):
    """Signed height of the posed CAD bottom over the support plane; reject poses farther than the limit."""
    gap = float(cad_bottom_z - support_z)
    if not np.isfinite(gap):
        raise ValueError(f"{label}: CAD bottom or table support height is not finite")
    if abs(gap) > limit_m:
        direction = "above" if gap > 0 else "below"
        raise ValueError(
            f"{label}: estimated CAD bottom is {abs(gap) * 1000:.1f} mm {direction} the table support plane "
            f"(limit {limit_m * 1000:.0f} mm); the object pose is likely wrong"
        )
    return gap


def restrict_pick_place_area(table_bounds, target_vertices, bounds_xy):
    """Keep the whole picked CAD and placement surface inside the task area."""
    table = np.asarray(table_bounds, float).copy()
    area = np.asarray(bounds_xy, float)
    vertices = np.asarray(target_vertices, float)
    if area.shape != (2,2) or not np.isfinite(area).all() or np.any(area[1] <= area[0]):
        raise ValueError("Pick/place area requires ordered finite XY bounds")
    if np.any(vertices[:,:2].min(0) < area[0]) or np.any(vertices[:,:2].max(0) > area[1]):
        raise ValueError("Target object footprint is outside the configured pick/place area")
    table[0,:2] = np.maximum(table[0,:2], area[0])
    table[1,:2] = np.minimum(table[1,:2], area[1])
    if np.any(table[1,:2] <= table[0,:2]):
        raise ValueError("No observed table inside the configured pick/place area")
    return table
