"""Placement geometry, task isolation, and the original Place solver with an in-memory motion backend."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, Holding, On

from tiptop.doosan.known_task import KnownTaskPlanningSettings
from tiptop.doosan.place_task import cad_bottom_correction, placement_height_context, run_standard_place, validate_place_plan
from tiptop.launcher import build_command, load_presets
from tiptop.task_presets import FIXED_BOX_NAME, TaskSettings, prepare_task_environment


@pytest.fixture
def scene():
    obj = Cuboid(name="object", dims=[.04, .04, .06], pose=[0, .6, .03, 1, 0, 0, 0])
    table = Cuboid(name="table", dims=[.73, .755, .03], pose=[0, .8475, -.035, 1, 0, 0, 0])
    full = Cuboid(name="configured_full_table", dims=[.78, 2.5, .03], pose=[0, 0, -.015, 1, 0, 0, 0])
    env = TAMPEnvironment("test", [obj], [table], {"Movable": [obj], "Surface": [table]},
                          frozenset({Holding.ground("object")}))
    observed = SimpleNamespace(grasps={"object": {}}, table_cuboid=table, perception_results={})
    return env, observed, full


@pytest.fixture(scope="module")
def upstream():
    # cuRobo creates two CUDA Pose defaults while importing MotionGen. Only
    # those unused defaults need CPU allocation for this in-memory solver test.
    original = Pose.from_list.__func__

    def cpu_default(cls, *args, **kwargs):
        kwargs["tensor_args"] = TensorDeviceType(device="cpu")
        return original(cls, *args, **kwargs)

    with patch.object(Pose, "from_list", classmethod(cpu_default)):
        import cutamp.motion_solver as solver
        import cutamp.cost_function as costs
        import cutamp.particle_initialization as initialization
    return SimpleNamespace(solver=solver, costs=costs, initialization=initialization)


def test_home_drop_holds_the_object_and_models_the_box_as_four_5_mm_walls(scene):
    original, observed, full = scene
    settings = TaskSettings(preset="home_drop")
    env, region = prepare_task_environment(original, observed, settings, fixed_obstacles=[full])
    assert env.goal_state == frozenset({Holding.ground("object")})
    assert all(item.name != FIXED_BOX_NAME for item in env.statics)
    walls = {item.name: item for item in env.statics if item.name.startswith(FIXED_BOX_NAME)}
    assert sorted(walls) == [f"{FIXED_BOX_NAME}_wall_{side}" for side in ("back", "front", "left", "right")]
    # The walls enclose the original 0.35 x 0.20 x 0.15 m footprint, each 5 mm thick.
    low = np.min([np.asarray(w.pose[:3]) - np.asarray(w.dims) / 2 for w in walls.values()], axis=0)
    high = np.max([np.asarray(w.pose[:3]) + np.asarray(w.dims) / 2 for w in walls.values()], axis=0)
    np.testing.assert_allclose(low, [-.175, .25, 0], atol=1e-9)
    np.testing.assert_allclose(high, [.175, .45, .15], atol=1e-9)
    assert walls[f"{FIXED_BOX_NAME}_wall_front"].dims[1] == pytest.approx(.005)
    assert walls[f"{FIXED_BOX_NAME}_wall_left"].dims[0] == pytest.approx(.005)
    np.testing.assert_allclose(settings.drop_region()["size_xy_m"], [.34, .19])
    # Preparing an already prepared environment again keeps exactly the four walls.
    again, _ = prepare_task_environment(env, observed, settings, fixed_obstacles=[full])
    assert sum(item.name.startswith(FIXED_BOX_NAME) for item in again.statics) == 4
    assert len(original.statics) == 1


def test_home_drop_only_asks_perception_for_the_object():
    settings = TaskSettings(preset="home_drop")
    assert settings.perception_instruction("ice_pack 집고 박스에 놓아") == "ice_pack 집어"
    with pytest.raises(ValueError, match="박스에 놓기"):
        settings.perception_instruction("ice_pack")


@pytest.mark.parametrize("preset", ["home_drop", "perceived_surface"])
def test_installed_workspace_box_cannot_be_disabled_by_a_stale_task_payload(preset):
    assert TaskSettings(preset=preset, fixed_box_present=False).fixed_box_present


@pytest.mark.parametrize("options", [dict(box_center_x_m=.38), dict(box_center_y_m=1.24), dict(box_bottom_z_m=.05)])
def test_box_must_fit_on_the_real_full_table(scene, options):
    env, observed, full = scene
    with pytest.raises(ValueError, match="tabletop"):
        prepare_task_environment(env, observed, TaskSettings(preset="home_drop", **options), fixed_obstacles=[full])
    with pytest.raises(ValueError, match="configured_full_table"):
        prepare_task_environment(env, observed, TaskSettings(preset="home_drop"))


@pytest.mark.parametrize("options", [dict(release_height_m=-.01), dict(release_height_m=float("nan")),
                                    dict(box_size_x_m=0), dict(box_center_y_m=float("inf"))])
def test_invalid_destination_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        TaskSettings(preset="home_drop", **options)


def test_perceived_on_goal_reuses_its_surface_and_retains_installed_fixed_box(scene):
    env, observed, full = scene
    destination = Cuboid(name="detected_box", dims=[.2, .2, .1], pose=[0, .3, .05, 1, 0, 0, 0])
    env.statics.append(destination)
    env.type_to_objects["Surface"].append(destination)
    env.goal_state = frozenset({On.ground("object", destination.name), HandEmpty.ground()})
    result, region = prepare_task_environment(env, observed, TaskSettings(preset="perceived_surface"), fixed_obstacles=[full])
    assert result.goal_state == env.goal_state
    assert region["destination"] == destination.name and region["support_z_m"] == pytest.approx(.1)
    assert any(item.name == FIXED_BOX_NAME for item in result.statics)
    with pytest.raises(ValueError, match="Gemini"):
        TaskSettings(preset="perceived_surface").validate_perception_mode("sam3_reference")


def test_perceived_table_support_is_corrected_once_even_when_prepared_again(scene):
    env, observed, full = scene
    env.goal_state = frozenset({On.ground("object", "table"), HandEmpty.ground()})
    for _ in range(2):
        env, region = prepare_task_environment(env, observed, TaskSettings(preset="perceived_surface"), fixed_obstacles=[full])
        assert region["support_z_m"] == pytest.approx(0, abs=1e-9)
        assert region["source_support_z_m"] == pytest.approx(0, abs=1e-9)
    assert observed.table_cuboid.pose[2] == -.035


@pytest.mark.parametrize("mode", ["sam3_reference", "gemini_er"])
def test_launcher_and_worker_parser_keep_placement_separate_from_perception(mode):
    from tiptop.scripts.run_known_pick_place import parse_args

    values = dict(load_presets()["presets"][mode], task="후추통 집고 박스에 놓아", task_preset="home_drop")
    for height in (.05, 0):
        command = build_command(mode, dict(values, release_height_m=height), "plan")
        assert not {"--execute", "--check", "--gateway-url"} & set(command)
        args = parse_args(command[3:])
        assert args.task_settings.preset == "home_drop" and args.task_settings.release_height_m == 0
        assert not any(key.startswith("--box-") for key in command)
        # Explicit CLI overrides remain available for deliberate offline experiments.
        args = parse_args([*command[3:], "--release-height-m", str(height)])
        assert args.task_settings.release_height_m == height
        assert args.task_settings.perception_instruction(args.task) == "black_pepper_shaker 집어"
    args = parse_args(build_command(mode, dict(values, fixed_box_present=False), "plan")[3:])
    assert args.task_settings.fixed_box_present  # Installed geometry cannot be disabled by a stale payload.


def test_release_height_offsets_sampling_and_support_cost_and_restores_both(upstream, monkeypatch):
    cost_type = upstream.costs.CostFunction
    initialization = upstream.initialization

    def baseline_init(self):
        self.surface_to_target_z = {"box": .112, "table": .002}
        self.stable_placement_constraints = [SimpleNamespace(params=("object", "g", "p", "box"))]
        self.world = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(placement_check="aabb")
        self.surface_to_aabb = {"box": (torch.tensor([-.12,.235]), torch.tensor([.12,.385]))}

    def baseline_sample(count, obj, spheres, surface):
        return torch.tensor([[0., 0., .112, 0.]]).repeat(count, 1)

    monkeypatch.setattr(cost_type, "__init__", baseline_init)
    monkeypatch.setattr(initialization, "place_4dof_sampler", baseline_sample)
    for height in (.05, 0, .02):
        with placement_height_context({"box": height}):
            cost = cost_type()
            assert cost.surface_to_target_z == {"box": pytest.approx(.112+height), "table": .002}
            samples = initialization.place_4dof_sampler(3, None, None, SimpleNamespace(name="box"))
            assert samples[:, 2].tolist() == pytest.approx([.112+height]*3)
            assert initialization.place_4dof_sampler(1, None, None, SimpleNamespace(name="table"))[0, 2] == pytest.approx(.112)
        assert cost_type.__init__ is baseline_init and initialization.place_4dof_sampler is baseline_sample
    with pytest.raises(RuntimeError):
        with placement_height_context({"box": .05}):
            raise RuntimeError("cancelled")
    assert cost_type.__init__ is baseline_init and initialization.place_4dof_sampler is baseline_sample
    # The actual upstream cost must accept the elevated object instead of
    # driving its bottom back down onto the physical box during optimization.
    rollout = dict(action_to_pose_ts={"p": 0}, num_particles=1)
    spheres = {"object": torch.tensor([[[[0.,.31,.165,.003]]]])}
    with placement_height_context({"box": .05}):
        cost = cost_type().stable_placement_costs(rollout, spheres)["values"]["box_support"]
        assert float(cost.item()) == pytest.approx(0, abs=1e-7)
    cost = cost_type().stable_placement_costs(rollout, spheres)["values"]["box_support"]
    assert float(cost.item()) == pytest.approx(.05, abs=1e-7)


def completed_plan():
    q = np.zeros(6)
    steps = []
    for label, xyz in [("Pick.approach", [0,.6,.2]), ("Pick.descend", [0,.6,.04]),
                       ("Pick.close", None), ("Place.lift", [0,.6,.09]),
                       ("Place.transport", [0,.31,.25]), ("Place.descend", [0,.31,.20]),
                       ("Place.open", None), ("Place.retract", [0,.31,.25]), ("Home", [0,0,0])]:
        if xyz is None:
            steps.append(dict(type="gripper", label=label, action=label.split(".")[1]))
        else:
            end = np.array([*xyz, 0, 0, 0.])
            steps.append(dict(type="trajectory", label=label, positions=np.array([q, end]),
                              velocities=np.zeros((2,6)), dt=.02))
            q = end
    return dict(q_init=np.zeros(6), steps=steps)


def tcp_at(q):
    result = np.diag([1., -1., -1., 1.])
    result[:3, 3] = np.asarray(q)[:3]
    return result


def placement_inputs():
    initial = np.eye(4)
    initial[1, 3] = .6
    vertices = np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (0,.06)])
    region = dict(center_xy_m=[0,.31], size_xy_m=[.24,.15], support_z_m=.11,
                  release_bottom_z_m=.16, release_height_m=.05)
    return initial, vertices, region


def test_full_cad_not_tcp_defines_the_release_height_and_footprint():
    initial, vertices, region = placement_inputs()
    check = validate_place_plan(completed_plan(), tcp_at, initial, vertices, region, position_tolerance_m=.01)
    assert check["release_cad_bottom_z_m"] == pytest.approx(.16)
    assert check["release_height_m"] == pytest.approx(.05)
    bad = completed_plan()
    bad["steps"][5]["positions"][-1, 0] = .15
    bad["steps"][7]["positions"][0, 0] = .15
    with pytest.raises(ValueError, match="footprint"):
        validate_place_plan(bad, tcp_at, initial, vertices, region, position_tolerance_m=.01)
    bad = completed_plan()
    bad["steps"][5]["positions"][-1, 2] += .04
    bad["steps"][7]["positions"][0, 2] += .04
    with pytest.raises(ValueError, match="bottom"):
        validate_place_plan(bad, tcp_at, initial, vertices, region, position_tolerance_m=.01)


def test_partial_top_observation_does_not_change_the_requested_full_cad_release_height():
    world_from_object = np.eye(4)
    world_from_object[:3, 3] = [.2, .6, .02]
    world_from_cad = np.eye(4)
    world_from_cad[:3, 3] = [.2, .6, 0]
    vertices = np.array([[0,0,0], [.02,.02,.06]])
    # The partial observation's lowest sphere bottom is 8 mm above the real CAD.
    spheres = np.array([[0,0,-.009,.003], [0,0,.02,.003]])
    correction = cad_bottom_correction(spheres, world_from_object, world_from_cad, vertices)
    assert correction == pytest.approx(.008)
    # Pose sampling originally aligns sphere_bottom with top+release. Adding
    # this correction makes the complete CAD bottom meet the same target.
    target = .11 + .05
    object_z = target + correction - (spheres[:,2]-spheres[:,3]).min()
    assert object_z - .02 == pytest.approx(target)


@pytest.mark.parametrize("failure", [None, "attached_collision", "oversize", "planned_speed_boundary"])
def test_existing_solver_generates_pick_attached_transport_release_and_home(upstream, failure):
    """Real cuTAMP solve_curobo, fake in-memory MotionGen: no GPU planning or network."""
    tensor = TensorDeviceType(device="cpu")
    calls, attachments, sphere_states = [], [], []
    original, vertices, region = placement_inputs()
    initial = torch.tensor(original, dtype=torch.float32)
    grasp = torch.diag(torch.tensor([1., -1., -1., 1.]))
    grasp[2, 3] = .04

    def fk(q):
        q = q if q.ndim == 2 else q[None]
        matrices = torch.diag(torch.tensor([1., -1., -1., 1.])).repeat(len(q), 1, 1)
        matrices[:, :3, 3] = q[:, :3]
        return SimpleNamespace(ee_pose=SimpleNamespace(get_matrix=lambda: matrices))

    class Motion:
        attached = False
        spheres_attached = False
        collided = False  # Only the grasp pose, as a start state, touches the support.

        def detach_object_from_robot(self, *args):
            self.attached = False
            self.spheres_attached = False

        def attach_objects_to_robot(self, *args, **kwargs):
            self.attached = True
            self.spheres_attached = True
            attachments.append(True)

        def result(self, start, end, options):
            q = torch.stack([start.position[0], start.position[0], end, end])
            path = JointState(position=q, velocity=torch.zeros_like(q), tensor_args=tensor)
            if failure == "planned_speed_boundary":
                path.velocity[-1, 0] = .003 * options.time_dilation_factor / .9
            return SimpleNamespace(success=True, interpolation_dt=.02*.9/options.time_dilation_factor, optimized_dt=.02,
                                   optimized_plan=path, get_interpolated_plan=lambda: path)

        def plan_single(self, start, goal, options):
            target = goal.get_matrix()[0, :3, 3]
            calls.append((self.attached, target.clone(), options.time_dilation_factor))
            sphere_states.append(self.spheres_attached)
            if failure == "attached_collision" and self.spheres_attached and not self.collided:
                self.collided = True
                return SimpleNamespace(success=False, status=SimpleNamespace(name="INVALID_START_STATE_WORLD_COLLISION"))
            return self.result(start, torch.cat([target, torch.zeros(3)]), options)

        def plan_single_js(self, start, goal, options):
            return self.result(start, goal.position[0], options)

    noop = lambda *args, **kwargs: None
    obstacle = SimpleNamespace(get_bounding_spheres=noop)
    mg = Motion()
    mg.kinematics = SimpleNamespace(kinematics_config=SimpleNamespace(
        get_link_spheres=lambda name: torch.ones(2, 4),
        detach_object=lambda name: setattr(mg, "spheres_attached", False),
        attach_object=lambda **kwargs: setattr(mg, "spheres_attached", True)))
    mg.world_model = SimpleNamespace(mesh=[obstacle], get_obstacle=lambda _: obstacle)
    mg.world_coll_checker = SimpleNamespace(enable_obstacle=noop)
    mg.world_collision = SimpleNamespace(update_obstacle_pose=noop)
    world = SimpleNamespace(device=torch.device("cpu"), tensor_args=tensor, tool_from_ee=torch.eye(4),
                            kin_model=SimpleNamespace(get_state=fk))
    op = lambda name, values: SimpleNamespace(operator=SimpleNamespace(name=name), values=values,
                                              name=f"{name}({','.join(values)})")
    skeleton = [op("MoveFree", ["q0","t0","q1"]), op("Pick", ["object","g","q1"]),
                op("MoveHolding", ["object","g","q1","t1","q2"]),
                op("Place", ["object","g","p","box","q2"])]
    visualizer = SimpleNamespace(**{key: (lambda *a, **k: 0.) for key in (
        "set_time_seconds", "set_joint_positions", "log_mat4x4", "log_joint_trajectory",
        "log_joint_trajectory_with_mat4x4")})
    timer = SimpleNamespace(time=lambda *a, **k: nullcontext(), get_summary=lambda *a: {})
    settings = KnownTaskPlanningSettings()
    particle = dict(q0=torch.tensor([0.,.9,.6,0,0,0]), g=grasp, p=torch.tensor([0.,.31,.16,0.]))
    if failure == "oversize":
        vertices[:, 0] *= 10

    def run():
        return run_standard_place(upstream.solver.solve_curobo, settings, region, original, vertices,
                                  dict(plan_skeleton=skeleton), particle, world,
                                  SimpleNamespace(warmup_motion_gen=False, time_dilation_factor=.9,
                                                  robot="m1013_uon", grasp_dof=6),
                                  timer, visualizer, obj_to_initial_pose={"object": initial}, motion_gen=mg)

    if failure == "oversize":
        with pytest.raises(upstream.solver.MotionPlanningError, match="footprint"):
            run()
        assert not mg.attached and not hasattr(obstacle, "old_get_bounding_spheres")
        return
    result = run()
    assert [step["label"] for step in result] == [step["label"] for step in completed_plan()["steps"]]
    assert attachments == [True] and not mg.attached
    if failure == "attached_collision":
        # cuTAMP retries the first lift once with the picked object detached, then carries it again.
        assert [attached for attached, _, _ in calls] == [False,False,True,True,True,True,False]
        assert sphere_states == [False,False,True,False,True,True,False]
        assert [speed for _, _, speed in calls] == [1.0] * 7
        return
    assert [attached for attached, _, _ in calls] == [False,False,True,True,True,False]
    assert [speed for _, _, speed in calls] == [1.0] * 6
    assert result[5]["placement_check"]["release_height_m"] == pytest.approx(.05, abs=1e-6)
    assert result[1]["dt"] == pytest.approx(.02*.9)
    assert result[4]["dt"] == pytest.approx(.02*.9)
    assert not hasattr(obstacle, "old_get_bounding_spheres")
    assert torch.equal(particle["q0"], torch.tensor([0.,.9,.6,0,0,0]))


def test_placement_preview_projects_physical_box_and_release_height_separately(tmp_path):
    import json
    from tiptop.perception_preview import TaskPreview

    region = TaskSettings(preset="home_drop", release_height_m=.05).box_region()
    preview = TaskPreview(tmp_path, observation_id="offline", task="test", depth_source="saved")
    preview.update("observation", dict(rgb=np.zeros((720,1280,3), np.uint8)))
    camera = np.diag([-1., 1., -1., 1.])
    camera[:3, 3] = [0., .76, 1.025]
    k = np.array([[610.,0,638.], [0,610.,361.], [0,0,1.]])
    preview.placement(region, camera, k)
    stored = json.loads((preview.directory / "state.json").read_text())
    assert stored["placement"]["region"]["release_bottom_z_m"] == pytest.approx(.20)
    box = stored["obstacle_scene"]["objects"][0]
    vertices = np.asarray(box["vertices"])
    assert vertices[:,2].max() == .15 and vertices[:,2].min() == 0
    assert box["collision_mesh"] and box["source"] == "configured"
    assert stored["placement"]["top"] != stored["placement"]["release"]
    assert np.asarray(stored["placement"]["top"]).min() >= 0
