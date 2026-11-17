"""Small adapters around cuTAMP's existing On/Place planner, without device access."""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from tiptop.doosan.in_place import SPEED_FACTOR, stationary_boundaries, wrist_travel_deg


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


def adapt_place_steps(steps, settings):
    """Name and validate the existing upstream legs without altering positions or timing."""
    result, counts = [], dict(Pick=0, Place=0, GoToInitial=0)
    labels = dict(Pick=["Pick.approach", "Pick.descend"],
                  Place=["Place.lift", "Place.transport", "Place.descend"],
                  GoToInitial=["Place.retract", "Home"])
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


def validate_place_plan(plan, tcp_at, world_from_cad, cad_vertices, region, *, position_tolerance_m):
    """Check the actual release endpoint using full CAD, before accepting a candidate."""
    steps = plan["steps"]
    expected = ["Pick.approach", "Pick.descend", "Pick.close", "Place.lift", "Place.transport",
                "Place.descend", "Place.open", "Place.retract", "Home"]
    if [step["label"] for step in steps] != expected or steps[2].get("action") != "close" or steps[6].get("action") != "open":
        raise ValueError("Expected Pick, attached transport, release at destination, retract and home")
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
    grasp_tcp = tcp_at(steps[1]["positions"][-1])
    release_tcp = tcp_at(steps[5]["positions"][-1])
    release_cad = release_tcp @ np.linalg.inv(grasp_tcp) @ np.asarray(world_from_cad)
    vertices = np.asarray(cad_vertices) @ release_cad[:3, :3].T + release_cad[:3, 3]
    low, high = vertices.min(0), vertices.max(0)
    center, half = np.asarray(region["center_xy_m"]), np.asarray(region["size_xy_m"]) / 2
    # No XY tolerance outside the destination: a tall object must not merely have
    # its TCP inside the box while its footprint hangs over the rim.
    if np.any(low[:2] < center - half) or np.any(high[:2] > center + half):
        raise ValueError("Full CAD footprint at release does not fit inside the destination")
    bottom = float(low[2])
    target_z = region["release_bottom_z_m"]
    if abs(bottom - target_z) > position_tolerance_m + .002:
        raise ValueError(f"Release CAD bottom {bottom:.3f}m differs from requested {target_z:.3f}m")
    if bottom < region["support_z_m"] - .002:
        raise ValueError("Release CAD penetrates the destination top")
    return dict(release_cad_bottom_z_m=bottom, release_height_m=bottom - region["support_z_m"],
                requested_release_height_m=region["release_height_m"],
                release_cad_bounds_xyz_m=[low.tolist(), high.tolist()],
                world_from_cad_at_release=release_cad.tolist(),
                full_cad_inside_destination=True, returns_to_home=True)


def run_standard_place(solver, settings, region, world_from_cad, cad_vertices, *args, **kwargs):
    """Call the original solver; reject unsuitable candidates so cuTAMP can try the next one."""
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

    class MotionAdapter:
        """Name each leg and check its J4/J5 travel as soon as it is planned."""
        attached = False
        released = False
        carry_legs = 0
        pick_legs = 0

        def leg_label(self):
            if self.released:
                return "Place.retract"
            if self.attached:
                return ("Place.lift", "Place.transport", "Place.descend")[min(self.carry_legs, 2)]
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
            options = options.clone()
            options.time_dilation_factor = SPEED_FACTOR
            value = motion_gen.plan_single(start, goal, options)
            if bool(value.success):
                self.check_wrist_travel(self.leg_label(), value)
                if self.attached:
                    self.carry_legs += 1
                elif not self.released:
                    self.pick_legs += 1
            return value

        def plan_single_js(self, start, goal, options):
            options = options.clone()
            options.time_dilation_factor = SPEED_FACTOR
            value = motion_gen.plan_single_js(start, goal, options)
            if bool(value.success):
                self.check_wrist_travel("Home", value)
            return value

    try:
        # The upstream visualizer mutates the q0 view on some versions. Keep
        # this candidate's home and transforms private across all motion legs.
        particle = {key: value.clone() for key, value in best_particle.items()}
        steps = solver(plan_info, particle, *args[2:], **dict(kwargs, motion_gen=MotionAdapter()))
        steps = adapt_place_steps(steps, settings)
        grasp = best_particle[picks[0].values[1]].clone()
        if grasp.shape != (4, 4):
            grasp = action_6dof_to_mat4x4(grasp)
        initial_poses = kwargs["obj_to_initial_pose"] if "obj_to_initial_pose" in kwargs else args[6]
        grasp_target = initial_poses[picks[0].values[0]] @ grasp @ world.tool_from_ee
        steps[1]["grasp_target_world_from_tcp"] = grasp_target.detach().cpu().numpy().tolist()

        def tcp_at(q):
            state = world.kin_model.get_state(world.tensor_args.to_device(np.asarray(q)[None]))
            return state.ee_pose.get_matrix().detach().cpu().numpy()[0]

        check = validate_place_plan(serialize_plan(steps, best_particle["q0"].detach().cpu().numpy()), tcp_at,
                                    world_from_cad, cad_vertices, region,
                                    position_tolerance_m=settings.motion_position_tolerance_m)
        steps[5]["placement_check"] = check
        return steps
    except ValueError as error:
        raise MotionPlanningError(str(error)) from error
    finally:
        # Upstream does not restore its temporary sphere method if attachment
        # raises. Avoid leaking that state into another candidate or task.
        for obj in motion_gen.world_model.mesh:
            if hasattr(obj, "old_get_bounding_spheres"):
                obj.get_bounding_spheres = obj.old_get_bounding_spheres
                del obj.old_get_bounding_spheres
        motion_gen.detach_object_from_robot("attached_object")
