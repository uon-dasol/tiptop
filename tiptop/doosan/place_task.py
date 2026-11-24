"""Small adapters around cuTAMP's existing On/Place planner, without device access."""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tiptop.doosan.in_place import SPEED_FACTOR, stationary_boundaries, wrist_travel_deg
from tiptop.doosan.leg_blend import blend_legs

CUTAMP_APPROACH_M = 0.05  # cuTAMP ends the carry this far before the placement pose, then descends straight
TRANSPORT_APPROACH_M = 0.02  # this adapter ends the carry this far before it, drops the descent and releases there
CARRY_SECOND_CUT_M = 0.06  # the transport is cut this far from the lift end: nearer it still creeps away, and dips
RELEASE_LOW_TOLERANCE_M = 0.015  # the object may be released this much lower than that point; any higher is accepted


def move_along_approach(goal, distance_m):
    """The same pose moved `distance_m` along its own approach (z) axis."""
    from curobo.types.math import Pose

    matrix = goal.get_matrix()[0].clone()
    matrix[:3, 3] += matrix[:3, 2] * distance_m
    return Pose.from_matrix(matrix)


def turn_about_approach(goal, angle_rad):
    """The same pose turned about its own approach (z) axis; only J6 has to change to reach it."""
    from curobo.types.math import Pose

    matrix = goal.get_matrix()[0].clone()
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    turn = matrix.new_tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, :3] = matrix[:3, :3] @ turn
    return Pose.from_matrix(matrix)


def release_j6_turn(release_j6, grasp_j6, home_j6):
    """Turn in radians about the approach axis that brings J6 at the release between its grasp and home values.

    Dropping into the box does not care about the object's yaw, so the release may be turned freely. Any J6 between
    the grasp and home values costs the same total rotation; the nearest one is used, and no turn when already between.
    """
    low, high = sorted((float(grasp_j6), float(home_j6)))
    equivalent = release_j6 + 2 * math.pi * round(((low + high) / 2 - release_j6) / (2 * math.pi))
    return min(max(equivalent, low), high) - equivalent


def release_turn_candidates(release_j6, grasp_j6, home_j6):
    """Turns about the approach axis to try, best first: the nearest J6 between grasp and home, the other end, none.

    Every J6 between the grasp and home values costs the same total rotation. When J6 is already between them the
    first candidate is no turn.
    """
    low, high = sorted((float(grasp_j6), float(home_j6)))
    equivalent = release_j6 + 2 * math.pi * round(((low + high) / 2 - release_j6) / (2 * math.pi))
    ends = sorted((low - equivalent, high - equivalent), key=abs)
    result = []
    for turn in (release_j6_turn(release_j6, grasp_j6, home_j6), *ends, 0.0):
        if all(abs(turn - other) > 1e-9 for other in result):
            result.append(turn)
    return result


def footprint_overhang_m(vertices, region):
    """How far the vertices reach outside the destination's top in XY; negative when inside, by that much to spare."""
    low, high = vertices.min(0)[:2], vertices.max(0)[:2]
    center, half = np.asarray(region["center_xy_m"]), np.asarray(region["size_xy_m"]) / 2
    return float(max((center - half - low).max(), (high - (center + half)).max()))


def choose_release_turn(candidates, overhang_of, margin_m):
    """The first turn that keeps the footprint inside with `margin_m` to spare, else the one closest to fitting."""
    best = None
    for turn in candidates:
        overhang = overhang_of(turn)
        if overhang <= -margin_m:
            return turn
        if best is None or overhang < best[1]:
            best = (turn, overhang)
    return best[0]


def cad_bottom_correction(spheres, world_from_object, world_from_cad, cad_vertices):
    """Align the partial-hull sphere support height with the complete CAD bottom."""
    object_from_cad = np.linalg.inv(np.asarray(world_from_object)) @ np.asarray(world_from_cad)
    local = np.asarray(cad_vertices) @ object_from_cad[:3, :3].T + object_from_cad[:3, 3]
    sampled_bottom = (np.asarray(spheres)[:, 2] - np.asarray(spheres)[:, 3]).min()
    return float(sampled_bottom - local[:, 2].min())


@contextmanager
def placement_height_context(height_by_surface):
    """Offset placement sampling AND its support cost; leave physical obstacles in place.

    cuTAMP's ordinary collision activation distance and 2 mm support buffer remain.
    Keep substitutions scoped to the serialized planning call, including retries.
    """
    if not height_by_surface or not any(height_by_surface.values()):
        yield
        return
    from cutamp.cost_function import CostFunction
    import cutamp.particle_initialization as initialization

    initialize_cost = CostFunction.__init__
    sample = initialization.place_4dof_sampler

    def cost_init(self, *args, **kwargs):
        initialize_cost(self, *args, **kwargs)
        for surface, height in height_by_surface.items():
            if surface in self.surface_to_target_z:
                self.surface_to_target_z[surface] += height

    def sample_place(num_samples, obj, obj_spheres, surface, *args, **kwargs):
        placements = sample(num_samples, obj, obj_spheres, surface, *args, **kwargs)
        height = height_by_surface.get(surface.name, 0)
        if height:
            placements = placements.clone()
            placements[:, 2] += height
        return placements

    with (patch.object(CostFunction, "__init__", cost_init),
          patch.object(initialization, "place_4dof_sampler", sample_place)):
        yield


def stay_at(start):
    """A finished plan that does not move: one sample at `start`."""
    from curobo.types.state import JointState

    path = JointState.from_position(start.position.clone())
    return SimpleNamespace(success=True, interpolation_dt=0.02, optimized_dt=0.02, optimized_plan=path,
                           get_interpolated_plan=lambda: path)


def joined_path(motion_gen, first, second, *, second_radius_m=None):
    """The two consecutive legs' paths joined without a stop in between, or None if that is not possible."""
    from curobo.types.state import JointState

    first_path, second_path = first.get_interpolated_plan(), second.get_interpolated_plan()
    dt = float(first.interpolation_dt)
    if abs(dt - float(second.interpolation_dt)) > 1e-9:
        return None

    def tcp_positions(path):
        return motion_gen.kinematics.get_state(path.position).ee_pose.position.cpu().numpy()

    joined = blend_legs(first_path.position.cpu().numpy(), first_path.velocity.cpu().numpy(),
                        second_path.position.cpu().numpy(), second_path.velocity.cpu().numpy(), dt,
                        tcp_positions(first_path), tcp_positions(second_path), second_radius_m=second_radius_m)
    if joined is None:
        return None
    position, velocity, _ = joined
    path = JointState.from_position(first_path.position.new_tensor(position))
    path.velocity = first_path.velocity.new_tensor(velocity)
    if not bool(motion_gen.check_constraints(JointState.from_position(path.position.clone())).feasible.all()):
        return None
    return path


def adapt_place_steps(steps, settings):
    """Name and validate the existing upstream legs without altering positions or timing.

    The straight descent to the placement pose and the retract after the release are planned as motions that
    stay in place and are dropped here: the object is released where the carry ends and Home starts there.
    A leg joined into its neighbour stays in place too and is dropped the same way: the Pick descent into the
    approach, the Place lift into the transport.
    """
    result, counts = [], dict(Pick=0, Place=0, GoToInitial=0)
    labels = dict(Pick=["Pick.approach", "Pick.descend"],
                  Place=["Place.lift", "Place.transport", None],
                  GoToInitial=[None, "Home"])
    for step in steps:
        current = dict(step)
        operation = step["label"].split("(", 1)[0]
        if operation not in labels:
            raise ValueError(f"Unexpected standard Place step: {step['label']}")
        if step["type"] == "gripper":
            expected = "close" if operation == "Pick" else "open" if operation == "Place" else None
            if step.get("action") != expected:
                raise ValueError("Standard Place gripper order changed")
            current["label"] = f"{operation}.{expected}"
        else:
            index = counts[operation]
            if index >= len(labels[operation]):
                raise ValueError("Standard Place leg count changed")
            label = current["label"] = labels[operation][index]
            counts[operation] += 1
            if (operation, index) in (("Pick", 1), ("Place", 0)) and len(step["plan"].position) == 1:
                label = current["label"] = None
            if label is None:
                if len(step["plan"].position) != 1:
                    raise ValueError("A leg replaced by staying in place must not move")
                continue
            path = step["plan"]
            try:
                path = stationary_boundaries(path)
            except ValueError as error:
                terminal_velocity = float(path.velocity[-1].abs().max())
                raise ValueError(f"{label}: {error} (terminal velocity {terminal_velocity:.4g} rad/s)") from error
            travel = wrist_travel_deg(path)
            if travel > settings.max_wrist_travel_deg:
                raise ValueError(f"{label}: J4/J5 travel {travel:.0f} deg exceeds {settings.max_wrist_travel_deg:g} deg")
            current.update(plan=path)
        result.append(current)
    if counts != dict(Pick=2, Place=3, GoToInitial=2):
        raise ValueError("Expected one complete standard Pick/Place and return home")
    return result


def validate_place_plan(plan, tcp_at, world_from_cad, cad_vertices, region, *, overhang_tolerance_mm=0.0):
    """Check the actual release endpoint using full CAD, before accepting a candidate.

    The release is `TRANSPORT_APPROACH_M` before the placement pose along the approach axis, so the CAD bottom is
    expected that far above the requested bottom, measured along the axis' vertical component.
    """
    steps = plan["steps"]
    labels = [step["label"] for step in steps]
    picks = (["Pick.approach", "Pick.descend"], ["Pick.approach"])
    carries = (["Place.lift", "Place.transport"], ["Place.transport"])
    if not any(labels == pick + ["Pick.close"] + carry + ["Place.open", "Home"] for pick in picks for carry in carries):
        raise ValueError("Expected Pick, attached transport, release above the destination and home")
    close, release = labels.index("Pick.close"), labels.index("Place.open")
    if steps[close].get("action") != "close" or steps[release].get("action") != "open":
        raise ValueError("Expected Pick, attached transport, release above the destination and home")
    previous = np.asarray(plan["q_init"])
    for step in steps:
        if step["type"] != "trajectory":
            continue
        positions = np.asarray(step["positions"])
        if not np.allclose(positions[0], previous, atol=1e-6, rtol=0):
            raise ValueError(f"{step['label']}: discontinuous joint path")
        previous = positions[-1]
    if not np.allclose(previous, plan["q_init"], atol=1e-3, rtol=0):
        raise ValueError("Place plan must return to the captured home joints")
    grasp_tcp = tcp_at(steps[close - 1]["positions"][-1])
    release_tcp = tcp_at(steps[release - 1]["positions"][-1])
    release_cad = release_tcp @ np.linalg.inv(grasp_tcp) @ np.asarray(world_from_cad)
    vertices = np.asarray(cad_vertices) @ release_cad[:3, :3].T + release_cad[:3, 3]
    low, high = vertices.min(0), vertices.max(0)
    # No XY tolerance outside the destination: a tall object must not merely have
    # its TCP inside the box while its footprint hangs over the rim.
    overhang = footprint_overhang_m(vertices, region)
    if overhang * 1000 > overhang_tolerance_mm:
        raise ValueError(f"Full CAD footprint at release does not fit inside the destination "
                         f"(overhang {overhang * 1000:.0f} mm, allowed {overhang_tolerance_mm:g} mm)")
    bottom = float(low[2])
    target_z = region["release_bottom_z_m"] - TRANSPORT_APPROACH_M * float(release_tcp[2, 2])
    if target_z - bottom > RELEASE_LOW_TOLERANCE_M:
        raise ValueError(f"Release CAD bottom {bottom:.3f}m is below requested {target_z:.3f}m")
    if bottom < region["support_z_m"] - .002:
        raise ValueError("Release CAD penetrates the destination top")
    return dict(release_cad_bottom_z_m=bottom, release_height_m=bottom - region["support_z_m"],
                requested_release_height_m=region["release_height_m"],
                release_cad_bounds_xyz_m=[low.tolist(), high.tolist()],
                world_from_cad_at_release=release_cad.tolist(),
                footprint_overhang_m=overhang, full_cad_inside_destination=overhang <= 0, returns_to_home=True)


def run_standard_place(solver, settings, region, world_from_cad, cad_vertices, *args, **kwargs):
    """Call the original solver; reject unsuitable candidates so cuTAMP can try the next one."""
    from curobo.types.state import JointState
    from cutamp.motion_solver import MotionPlanningError
    from cutamp.utils.common import action_6dof_to_mat4x4
    from tiptop.planning import serialize_plan

    plan_info, best_particle, world = args[:3]
    skeleton = plan_info["plan_skeleton"]
    picks = [op for op in skeleton if op.operator.name == "Pick"]
    places = [op for op in skeleton if op.operator.name == "Place"]
    if len(picks) != 1 or len(places) != 1 or any(
            op.operator.name not in {"MoveFree", "MoveHolding", "Pick", "Place"} for op in skeleton):
        raise MotionPlanningError("This adapter expects one standard Pick and Place")
    motion_gen = kwargs["motion_gen"]

    release_q = best_particle.get(places[0].values[-1])
    home_q = best_particle["q0"]

    def tcp_at(q):
        state = world.kin_model.get_state(world.tensor_args.to_device(np.asarray(q)[None]))
        return state.ee_pose.get_matrix().detach().cpu().numpy()[0]

    class MotionAdapter:
        """Name each leg and check its J4/J5 travel as soon as it is planned."""
        attached = False
        released = False
        carry_legs = 0
        pick_legs = 0
        approach_result = None
        lift_result = None
        grasp_j6 = None
        grasp_tcp = None
        turn = 0.0
        turn_report = "방향 검사 없음"

        def release_turn(self, goal):
            """J6 turn that shortens the carry and the way home while the object still lands inside the destination.

            The turn also rotates the held object, so each candidate is checked against the destination's top first.
            No turn when cuTAMP's release joints are unknown.
            """
            if release_q is None or self.grasp_j6 is None:
                return 0.0
            candidates = release_turn_candidates(float(release_q.reshape(-1)[5]), self.grasp_j6,
                                                 float(home_q.reshape(-1)[5]))
            release_tcp = goal.get_matrix()[0].detach().cpu().numpy().astype(float)
            cad_from_grasp = np.linalg.inv(self.grasp_tcp) @ np.asarray(world_from_cad, float)

            def overhang(turn):
                tcp = release_tcp.copy()
                c, s = math.cos(turn), math.sin(turn)
                tcp[:3, :3] = tcp[:3, :3] @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                cad = tcp @ cad_from_grasp
                return footprint_overhang_m(np.asarray(cad_vertices) @ cad[:3, :3].T + cad[:3, 3], region)

            self.turn = choose_release_turn(candidates, overhang, settings.motion_position_tolerance_m)
            self.turn_report = (f"방향 후보 {len(candidates)}개 중 J6 {math.degrees(self.turn):+.0f}° 선택, "
                                f"예상 벗어남 {overhang(self.turn) * 1000:+.0f} mm")
            return self.turn

        def leg_label(self):
            if self.attached:
                return ("Place.lift", "Place.transport")[min(self.carry_legs, 1)]
            return ("Pick.approach", "Pick.descend")[min(self.pick_legs, 1)]

        def check_wrist_travel(self, label, value):
            travel = wrist_travel_deg(value.get_interpolated_plan())
            if travel > settings.max_wrist_travel_deg:
                raise MotionPlanningError(f"{label}: J4/J5 travel {travel:.0f} deg exceeds {settings.max_wrist_travel_deg:g} deg")

        def __getattr__(self, key):
            return getattr(motion_gen, key)

        def attach_objects_to_robot(self, *args, **kwargs):
            value = motion_gen.attach_objects_to_robot(*args, **kwargs)
            self.attached = True
            self.carry_legs = 0
            return value

        def detach_object_from_robot(self, *args, **kwargs):
            value = motion_gen.detach_object_from_robot(*args, **kwargs)
            if self.attached:
                self.released = True
            self.attached = False
            return value

        def plan_single(self, start, goal, options):
            if self.released or (self.attached and self.carry_legs == 2):
                # The descent to the placement pose and the retract after the release are left out: the box is open.
                return stay_at(start)
            options = options.clone()
            options.time_dilation_factor = SPEED_FACTOR
            if self.attached and self.carry_legs == 0:
                self.grasp_j6 = float(start.position.reshape(-1)[5])
                self.grasp_tcp = tcp_at(start.position.reshape(-1).detach().cpu().numpy())
            if self.attached and self.carry_legs == 1:  # Place.transport: the first carry leg is the lift
                goal = move_along_approach(goal, CUTAMP_APPROACH_M - TRANSPORT_APPROACH_M)
                turn = self.release_turn(goal)
                if turn:
                    goal = turn_about_approach(goal, turn)
            value = motion_gen.plan_single(start, goal, options)
            if bool(value.success):
                self.check_wrist_travel(self.leg_label(), value)
                if self.attached:
                    if self.carry_legs == 0:
                        self.lift_result = value
                    else:
                        path = joined_path(motion_gen, self.lift_result, value, second_radius_m=CARRY_SECOND_CUT_M)
                        if path is not None:
                            # The lift now lives inside the transport's path, which starts at the grasp pose.
                            lift = self.lift_result.get_interpolated_plan()
                            hold = stay_at(JointState.from_position(lift.position[:1].clone())).get_interpolated_plan()
                            self.lift_result.get_interpolated_plan = lambda: hold
                            value.get_interpolated_plan = lambda: path
                    self.carry_legs += 1
                elif not self.released:
                    if self.pick_legs == 0:
                        self.approach_result = value
                    else:
                        path = joined_path(motion_gen, self.approach_result, value)
                        if path is not None:
                            # The descent now lives inside the approach's path, which ends at the grasp pose.
                            self.approach_result.get_interpolated_plan = lambda: path
                            value = stay_at(JointState.from_position(path.position[-1:].clone()))
                    self.pick_legs += 1
            return value

        def plan_single_js(self, start, goal, options):
            options = options.clone()
            options.time_dilation_factor = SPEED_FACTOR
            value = motion_gen.plan_single_js(start, goal, options)
            if bool(value.success):
                self.check_wrist_travel("Home", value)
            return value

    adapter = MotionAdapter()
    try:
        # The upstream visualizer mutates the q0 view on some versions. Keep
        # this candidate's home and transforms private across all motion legs.
        particle = {key: value.clone() for key, value in best_particle.items()}
        steps = solver(plan_info, particle, *args[2:], **dict(kwargs, motion_gen=adapter))
        steps = adapt_place_steps(steps, settings)
        grasp = best_particle[picks[0].values[1]].clone()
        if grasp.shape != (4, 4):
            grasp = action_6dof_to_mat4x4(grasp)
        initial_poses = kwargs["obj_to_initial_pose"] if "obj_to_initial_pose" in kwargs else args[6]
        grasp_target = initial_poses[picks[0].values[0]] @ grasp @ world.tool_from_ee
        close = next(i for i, step in enumerate(steps) if step["label"] == "Pick.close")
        steps[close - 1]["grasp_target_world_from_tcp"] = grasp_target.detach().cpu().numpy().tolist()

        check = validate_place_plan(serialize_plan(steps, best_particle["q0"].detach().cpu().numpy()), tcp_at,
                                    world_from_cad, cad_vertices, region,
                                    overhang_tolerance_mm=settings.footprint_overhang_tolerance_mm)
        next(step for step in steps if step["label"] == "Place.transport")["placement_check"] = check
        return steps
    except ValueError as error:
        detail = f" · {adapter.turn_report}" if "footprint" in str(error) else ""
        raise MotionPlanningError(f"{error}{detail}") from error
    finally:
        # Upstream does not restore its temporary sphere method if attachment
        # raises. Avoid leaking that state into another candidate or task.
        for obj in motion_gen.world_model.mesh:
            if hasattr(obj, "old_get_bounding_spheres"):
                obj.get_bounding_spheres = obj.old_get_bounding_spheres
                del obj.old_get_bounding_spheres
        motion_gen.detach_object_from_robot("attached_object")
