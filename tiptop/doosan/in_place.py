"""Pick, lift and release above the observed source; no placement-region search."""

import logging
import time

import numpy as np

_log = logging.getLogger(__name__)


def source_region(table_bounds, vertices, bounds_xy=None):
    from tiptop.perception.placement import restrict_pick_place_area, table_support_z

    table, vertices = np.asarray(table_bounds), np.asarray(vertices)
    if bounds_xy is not None:
        restrict_pick_place_area(table, vertices, bounds_xy)
    low, high = vertices.min(0), vertices.max(0)
    return dict(source="in_place_return", support_z_m=table_support_z(table[1, 2]),
                initial_cad_bounds_xyz_m=[low.tolist(), high.tolist()],
                center_xy_m=((low+high)/2)[:2].tolist(),
                size_xy_m=(high-low+.04)[:2].tolist(), return_xy_tolerance_m=.02)


def reverse_path(path):
    """Reverse positions AND velocity signs, preserving every sample and timing."""
    from curobo.types.state import JointState

    return JointState(position=path.position.flip(0).clone(), velocity=-path.velocity.flip(0).clone())


def stationary_boundaries(path, *, terminal_velocity_tolerance_rad_s=2.5e-3):
    """Remove only boundary finite-difference residues at an initial hold.

    cuRobo can report a small initial velocity despite two coincident initial
    positions. A reversed path needs that held endpoint to finish at rest too.
    All positions, interior velocities and sample times are preserved.
    """
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


def join_paths(first, second):
    """Join two rest-to-rest paths that meet at one waypoint."""
    import torch
    from curobo.types.state import JointState

    if not torch.allclose(first.position[-1], second.position[0], atol=1e-6, rtol=0):
        raise ValueError("Paths to join do not meet")
    return JointState(position=torch.cat([first.position, second.position[1:]]),
                      velocity=torch.cat([first.velocity, second.velocity[1:]]))


def wrist_travel_deg(path):
    """Largest J4 or J5 excursion from the first sample of a path, in degrees."""
    wrist = path.position[:, 3:5]
    return float(np.degrees((wrist - wrist[0]).abs().max().item()))


def make_in_place_solver(settings):
    """Use cuTAMP's grasp, lift back along the descend path, then release and return home.

    The approach stops `approach_distance_m` behind the grasp and the descend passes through a stop
    `lift_distance_m` behind it, so the last stretch is the same path the lift reverses. The lift is not
    collision-checked with the object attached. Home starts at the lifted joints. Nothing is sent to
    any execution backend here.
    """
    def solve(plan_info, best_particle, world, config, timer, visualizer,
              obj_to_initial_pose, timeline="curobo", motion_gen=None):
        import torch
        from curobo.rollout.cost.pose_cost import PoseCostMetric
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        from cutamp.motion_solver import MotionPlanningError
        from cutamp.utils.common import action_6dof_to_mat4x4

        picks = [op for op in plan_info["plan_skeleton"] if op.operator.name == "Pick"]
        if len(picks) != 1 or any(op.operator.name not in {"MoveFree", "Pick"}
                                  for op in plan_info["plan_skeleton"]):
            raise MotionPlanningError("In-place return requires one Pick and no Place or transport operator")
        obj, grasp, _ = picks[0].values
        mg = motion_gen
        mg.detach_object_from_robot("attached_object")
        for name, pose in obj_to_initial_pose.items():
            mg.world_coll_checker.enable_obstacle(enable=True, name=name)
            mg.world_collision.update_obstacle_pose(name, Pose.from_matrix(pose))
        plan_config = MotionGenPlanConfig(timeout=settings.motion_timeout_s,
            enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor)
        straight = plan_config.clone()
        straight.pose_cost_metric = PoseCostMetric(hold_partial_pose=True,
            hold_vec_weight=world.tensor_args.to_device([.1, .1, .1, .1, .1, 0]), project_to_goal_frame=True)
        # cuRobo reuses and updates the start-state buffer across planning legs.
        # Keep home separate from every state passed to that mutable buffer.
        home_position = best_particle["q0"][None].clone()
        q0 = JointState.from_position(home_position.clone())
        obj_from_grasp = best_particle[grasp].clone()
        if obj_from_grasp.shape != (4, 4):
            obj_from_grasp = action_6dof_to_mat4x4(obj_from_grasp)
        target = obj_to_initial_pose[obj] @ obj_from_grasp @ world.tool_from_ee

        def behind(distance):
            offset = torch.eye(4, device=world.device)
            offset[2, 3] = -distance
            return target @ offset

        def leg(start, goal, options, label, *, speed_factor,
                terminal_velocity_tolerance_rad_s=2.5e-3, joint_goal=False):
            options = options.clone()
            options.time_dilation_factor = speed_factor
            started = time.perf_counter()
            result = (mg.plan_single_js(start, goal, options) if joint_goal
                      else mg.plan_single(start, Pose.from_matrix(goal), options))
            if not bool(result.success):
                raise MotionPlanningError(f"{label}: {result.status}")
            try:
                path = stationary_boundaries(result.get_interpolated_plan(),
                    terminal_velocity_tolerance_rad_s=terminal_velocity_tolerance_rad_s)
            except ValueError as error:
                raise MotionPlanningError(f"{label}: {error}") from error
            travel = wrist_travel_deg(path)
            if travel > settings.max_wrist_travel_deg:
                raise MotionPlanningError(
                    f"{label}: J4/J5 travel {travel:.0f} deg exceeds {settings.max_wrist_travel_deg:g} deg"
                )
            _log.info("%s planning took %.2fs (speed factor %.3g, terminal tolerance %.4g rad/s)",
                      label, time.perf_counter() - started, speed_factor, terminal_velocity_tolerance_rad_s)
            return dict(type="trajectory", label=label, plan=path, dt=float(result.interpolation_dt))

        approach = leg(q0, behind(settings.approach_distance_m), plan_config, "Pick.approach",
                       speed_factor=settings.approach_speed_factor, terminal_velocity_tolerance_rad_s=4e-3)
        above = approach
        if settings.approach_distance_m > settings.lift_distance_m:
            above = leg(JointState.from_position(approach["plan"].position[-1:]), behind(settings.lift_distance_m),
                        straight, "Pick.descend far", speed_factor=settings.descend_far_speed_factor)
        final = leg(JointState.from_position(above["plan"].position[-1:]), target, straight, "Pick.descend",
                    speed_factor=settings.descend_near_speed_factor)
        descend = final
        if above is not approach:
            if above["dt"] != final["dt"]:
                raise MotionPlanningError("Pick.descend: legs have different sample times")
            descend = dict(final, plan=join_paths(above["plan"], final["plan"]))
            travel = wrist_travel_deg(descend["plan"])
            if travel > settings.max_wrist_travel_deg:
                raise MotionPlanningError(
                    f"Pick.descend: J4/J5 travel {travel:.0f} deg exceeds {settings.max_wrist_travel_deg:g} deg"
                )
        lift_plan = reverse_path(final["plan"])
        lift_ratio = settings.lift_speed_factor / settings.descend_near_speed_factor
        # Keep the checked joint positions and stationary endpoints. Changing
        # lift speed requires scaling its velocity and sample time together.
        if lift_ratio != 1:
            lift_plan.velocity = lift_plan.velocity * lift_ratio
        lift = dict(type="trajectory", label="Place.lift", plan=lift_plan, dt=final["dt"] / lift_ratio)

        lifted = lift["plan"].position[-1:].repeat(2, 1)
        hold = dict(type="trajectory", label="Place.hold",
                    plan=JointState(position=lifted, velocity=torch.zeros_like(lifted)),
                    dt=settings.lift_hold_s/2)
        home = leg(JointState.from_position(lift["plan"].position[-1:]),
                   JointState.from_position(home_position.clone()),
                   plan_config, "Home", speed_factor=settings.home_speed_factor,
                   terminal_velocity_tolerance_rad_s=4e-3, joint_goal=True)
        # Carry the successful particle's exact goal with its completed path.
        # The viewer must not infer its DB source from the FK residual alone.
        descend["grasp_target_world_from_tcp"] = target.detach().cpu().numpy().tolist()
        return [approach, descend, dict(type="gripper", label="Pick.close", action="close"), lift, hold,
                dict(type="gripper", label="Place.open", action="open"), home]

    return solve


def validate_in_place_plan(plan, tcp_at, *, lift_distance_m, position_tolerance_m):
    steps = plan["steps"]
    expected = ["Pick.approach", "Pick.descend", "Pick.close", "Place.lift", "Place.hold",
                "Place.open", "Home"]
    if [step["label"] for step in steps] != expected:
        raise ValueError("In-place plan has an unexpected action sequence")
    if steps[2].get("action") != "close" or steps[5].get("action") != "open":
        raise ValueError("In-place plan must close before lift and open at the lifted hold")
    grasp_q = np.asarray(steps[1]["positions"][-1])
    release_q = np.asarray(steps[4]["positions"][-1])
    if not np.allclose(steps[6]["positions"][0], release_q, atol=1e-7, rtol=0):
        raise ValueError("Home must start at the lifted release joints")
    if not np.allclose(steps[-1]["positions"][-1], plan["q_init"], atol=1e-7, rtol=0):
        raise ValueError("In-place plan must finish at its captured home")
    if (not np.allclose(steps[4]["positions"], steps[3]["positions"][-1], atol=1e-7, rtol=0)
            or np.any(np.abs(steps[4]["velocities"]) > 1e-7)):
        raise ValueError("The lifted hold must remain stationary")
    grasp = tcp_at(grasp_q)
    lifted = tcp_at(steps[3]["positions"][-1])
    delta = lifted[:3, 3]-grasp[:3, 3]
    if delta[2] < lift_distance_m-position_tolerance_m-.002:
        raise ValueError("Planned lift does not raise the TCP sufficiently")
    return dict(tcp_lift_xyz_m=delta.tolist(), release_at_lifted_pose=True,
                returns_to_home=True)
