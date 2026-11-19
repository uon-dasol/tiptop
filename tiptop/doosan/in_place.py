"""Pick, lift and carry home; no placement-region search."""

import logging
import time

import numpy as np

from tiptop.robot_limits import load_robot_limits

_log = logging.getLogger(__name__)


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
HOME_DROP_OFFSET_M = 0.05  # cuTAMP's fixed approach and retract offset, used by `home_drop`


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


def attach_object_at_grasp(mg, world, obj, world_from_obj, grasp_state):
    """Attach the object's sampled collision spheres to the gripper, as cuTAMP's own Pick does."""
    from curobo.geom.sphere_fit import SphereFitType
    from curobo.geom.types import Sphere
    from curobo.types.math import Pose

    obstacle = mg.world_model.get_obstacle(obj)
    original = obstacle.get_bounding_spheres

    def bounding_spheres(self, *args, **kwargs):
        spheres = world.get_collision_spheres(obj)
        pose = Pose.from_matrix(world_from_obj)
        if kwargs.get("pre_transform_pose") is not None:
            pose = kwargs["pre_transform_pose"].multiply(pose)
        points = pose.transform_points(self.tensor_args.to_device(spheres[:, :3].cpu().numpy())).cpu().view(-1, 3).numpy()
        radii = spheres[:, 3].cpu().numpy()
        return [Sphere(name=f"{self.name}_sph_{i}", pose=[*points[i], 1, 0, 0, 0], radius=radii[i])
                for i in range(len(points))]

    obstacle.get_bounding_spheres = bounding_spheres.__get__(obstacle)
    try:
        mg.attach_objects_to_robot(grasp_state, object_names=[obj], surface_sphere_radius=0.005,
                                   sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                                   voxelize_method="subdivide")
    finally:
        obstacle.get_bounding_spheres = original


def make_in_place_solver(settings):
    """Plan pick, lift and the carry home as cuTAMP's own Pick and Place do, then open at home.

    The approach stops `HOME_DROP_OFFSET_M` behind the grasp, one straight descend reaches it, and the object is
    attached at the grasp. The lift is a new straight plan `HOME_DROP_OFFSET_M` back, checked with the object; if the
    grasp pose collides as a start state the lift is planned once without the object, as cuTAMP's Place does.
    Home is then planned with the object and the gripper opens at the captured home. Nothing is sent to any
    execution backend here.
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

        class StartStateCollision(MotionPlanningError):
            pass

        picks = [op for op in plan_info["plan_skeleton"] if op.operator.name == "Pick"]
        if len(picks) != 1 or any(op.operator.name not in {"MoveFree", "Pick"}
                                  for op in plan_info["plan_skeleton"]):
            raise MotionPlanningError("Home drop requires one Pick and no Place or transport operator")
        obj, grasp, _ = picks[0].values
        mg = motion_gen
        mg.detach_object_from_robot("attached_object")
        for name, pose in obj_to_initial_pose.items():
            mg.world_coll_checker.enable_obstacle(enable=True, name=name)
            mg.world_collision.update_obstacle_pose(name, Pose.from_matrix(pose))
        plan_config = MotionGenPlanConfig(timeout=settings.motion_timeout_s,
            enable_finetune_trajopt=False, time_dilation_factor=SPEED_FACTOR)
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

        def leg(start, goal, options, label, *, joint_goal=False):
            started = time.perf_counter()
            result = (mg.plan_single_js(start, goal, options) if joint_goal
                      else mg.plan_single(start, Pose.from_matrix(goal), options))
            if not bool(result.success):
                collides = getattr(result.status, "name", None) == "INVALID_START_STATE_WORLD_COLLISION"
                raise (StartStateCollision if collides else MotionPlanningError)(f"{label}: {result.status}")
            try:
                path = stationary_boundaries(result.get_interpolated_plan())
            except ValueError as error:
                raise MotionPlanningError(f"{label}: {error}") from error
            travel = wrist_travel_deg(path)
            if travel > settings.max_wrist_travel_deg:
                raise MotionPlanningError(
                    f"{label}: J4/J5 travel {travel:.0f} deg exceeds {settings.max_wrist_travel_deg:g} deg"
                )
            _log.info("%s planning took %.2fs", label, time.perf_counter() - started)
            return dict(type="trajectory", label=label, plan=path, dt=float(result.interpolation_dt))

        approach = leg(q0, behind(HOME_DROP_OFFSET_M), plan_config, "Pick.approach")
        descend = leg(JointState.from_position(approach["plan"].position[-1:]), target, straight, "Pick.descend")
        grasp_position = descend["plan"].position[-1:]
        attach_object_at_grasp(mg, world, obj, obj_to_initial_pose[obj], JointState.from_position(grasp_position))
        try:
            lift = leg(JointState.from_position(grasp_position.clone()), behind(HOME_DROP_OFFSET_M),
                       straight, "Place.lift")
        except StartStateCollision:
            # The picked object rests on its support, so only this lift is planned without it.
            kinematics_config = mg.kinematics.kinematics_config
            object_spheres = kinematics_config.get_link_spheres("attached_object").clone()
            kinematics_config.detach_object("attached_object")
            try:
                lift = leg(JointState.from_position(grasp_position.clone()), behind(HOME_DROP_OFFSET_M),
                           plan_config, "Place.lift")
            finally:
                kinematics_config.attach_object(sphere_tensor=object_spheres, link_name="attached_object")

        home = leg(JointState.from_position(lift["plan"].position[-1:]),
                   JointState.from_position(home_position.clone()),
                   plan_config, "Home", joint_goal=True)
        # Carry the successful particle's exact goal with its completed path.
        # The viewer must not infer its DB source from the FK residual alone.
        descend["grasp_target_world_from_tcp"] = target.detach().cpu().numpy().tolist()
        close = dict(type="gripper", label="Pick.close", action="close")
        release = dict(type="gripper", label="Place.open", action="open")
        mg.detach_object_from_robot("attached_object")
        return [approach, descend, close, lift, home, release]

    return solve
