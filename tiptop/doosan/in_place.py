"""Path and pick-region helpers shared by the planners."""

import numpy as np

from tiptop.robot_limits import load_robot_limits


def source_region(table_bounds, vertices, bounds_xy=None):
    from tiptop.perception.placement import restrict_pick_place_area, table_support_z

    table, vertices = np.asarray(table_bounds), np.asarray(vertices)
    if bounds_xy is not None:
        restrict_pick_place_area(table, vertices, bounds_xy)
    low, high = vertices.min(0), vertices.max(0)
    return dict(source="pick_source", support_z_m=table_support_z(table[1, 2]),
                initial_cad_bounds_xyz_m=[low.tolist(), high.tolist()],
                center_xy_m=((low+high)/2)[:2].tolist(),
                size_xy_m=(high-low+.04)[:2].tolist(), return_xy_tolerance_m=.02)


SPEED_FACTOR = 1.0  # every planned leg runs at the planner's full speed


def stationary_boundaries(path, *, terminal_velocity_tolerance_rad_s=None):
    """Remove only boundary finite-difference residues at an initial hold.

    cuRobo can report a small initial velocity despite two coincident initial
    positions.
    All positions, interior velocities and sample times are preserved.
    """
    if terminal_velocity_tolerance_rad_s is None:
        terminal_velocity_tolerance_rad_s = load_robot_limits()["terminal_velocity_tolerance_rad_s"]
    result = path.clone()
    if bool((result.velocity[-1].abs() > terminal_velocity_tolerance_rad_s).any()):
        raise ValueError("Nonstationary final trajectory boundary")
    if bool((result.velocity[0] != 0).any()):
        import torch

        # Compare positions at float precision, rather than inventing a minimum
        # robot speed. These paths are requested from stationary JointStates.
        scale = torch.maximum(result.position[0].abs(), result.position[1].abs()).clamp_min(1)
        tolerance = 4*torch.finfo(result.position.dtype).eps*scale
        if len(result.position) < 2 or bool(((result.position[1]-result.position[0]).abs() > tolerance).any()):
            raise ValueError("Nonstationary initial trajectory boundary")
    result.velocity[[0, -1]] = 0
    return result


def wrist_travel_deg(path):
    """Largest J4 or J5 excursion from the first sample of a path, in degrees."""
    wrist = path.position[:, 3:5]
    return float(np.degrees((wrist - wrist[0]).abs().max().item()))
