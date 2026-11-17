"""Pick, carry the object home with it attached, and open the gripper over the fixed box."""

import numpy as np


def drop_inside_box(home_tcp, grasp_tcp, world_from_cad, cad_vertices, region, *, tolerance_m):
    """Check that the held object's CAD centre at home is over the box floor and its bottom is not below the box top."""
    release_cad = np.asarray(home_tcp) @ np.linalg.inv(grasp_tcp) @ np.asarray(world_from_cad)
    vertices = np.asarray(cad_vertices) @ release_cad[:3, :3].T + release_cad[:3, 3]
    low, high = vertices.min(0), vertices.max(0)
    center, half = np.asarray(region["center_xy_m"]), np.asarray(region["size_xy_m"]) / 2
    object_center = (low[:2] + high[:2]) / 2
    if np.any(np.abs(object_center - center) > half):
        raise ValueError("Object centre at home is not over the box floor")
    if low[2] < region["support_z_m"] - tolerance_m:
        raise ValueError("Held object reaches below the box top at home")
    return dict(drop_cad_bottom_z_m=float(low[2]), drop_height_m=float(low[2] - region["support_z_m"]),
                drop_cad_bounds_xyz_m=[low.tolist(), high.tolist()], center_inside_box=True)


def run_home_drop(solver, region, world_from_cad, cad_vertices, tolerance_m, *args, **kwargs):
    """Call the home-drop solver and reject a candidate whose held object would miss the box."""
    from cutamp.motion_solver import MotionPlanningError

    steps = solver(*args, **kwargs)
    world = args[2]
    home_tcp = world.kin_model.get_state(steps[4]["plan"].position[-1:]).ee_pose.get_matrix()[0]
    try:
        drop_inside_box(home_tcp.cpu().numpy().astype(float), np.asarray(steps[1]["grasp_target_world_from_tcp"]),
                        world_from_cad, cad_vertices, region, tolerance_m=tolerance_m)
    except ValueError as error:
        raise MotionPlanningError(str(error)) from error
    return steps


def validate_home_drop_plan(plan, tcp_at, world_from_cad, cad_vertices, region, *, lift_distance_m,
                            position_tolerance_m):
    steps = plan["steps"]
    expected = ["Pick.approach", "Pick.descend", "Pick.close", "Place.lift", "Home", "Place.open"]
    if [step["label"] for step in steps] != expected:
        raise ValueError("Home-drop plan has an unexpected action sequence")
    if steps[2].get("action") != "close" or steps[5].get("action") != "open":
        raise ValueError("Home-drop plan must close before lift and open after reaching home")
    if not np.allclose(steps[4]["positions"][0], steps[3]["positions"][-1], atol=1e-7, rtol=0):
        raise ValueError("Home must start at the lifted joints")
    if not np.allclose(steps[4]["positions"][-1], plan["q_init"], atol=1e-7, rtol=0):
        raise ValueError("Home-drop plan must finish at its captured home")
    grasp = tcp_at(steps[1]["positions"][-1])
    lifted = tcp_at(steps[3]["positions"][-1])
    delta = lifted[:3, 3] - grasp[:3, 3]
    if delta[2] < lift_distance_m - position_tolerance_m - .002:
        raise ValueError("Planned lift does not raise the TCP sufficiently")
    drop = drop_inside_box(tcp_at(steps[4]["positions"][-1]), grasp, world_from_cad, cad_vertices, region,
                           tolerance_m=position_tolerance_m + .002)
    return dict(tcp_lift_xyz_m=delta.tolist(), drops_at_home=True, **drop)
