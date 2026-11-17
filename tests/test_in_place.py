"""Pick-and-carry path mathematics, without hardware or inference."""
import numpy as np
import pytest
import torch
from curobo.types.state import JointState

from tiptop.doosan.in_place import source_region, stationary_boundaries, wrist_travel_deg


def path():
    q=torch.zeros((4,6))
    q[2:,0]=torch.tensor([.01,.02])
    v=torch.zeros_like(q)
    v[0,0]=.003
    v[2,0]=.01
    return JointState(position=q,velocity=v)


def test_boundary_residue_is_removed_only_at_a_stationary_initial_hold():
    raw=path()
    clean=stationary_boundaries(raw)
    assert clean.velocity[0,0]==0 and raw.velocity[0,0]!=0
    assert torch.equal(clean.position,raw.position)
    assert torch.equal(clean.velocity[1:-1],raw.velocity[1:-1])
    raw.position[1,0]=.01
    with pytest.raises(ValueError,match='initial'):
        stationary_boundaries(raw)
    raw=path()
    raw.velocity[-1,0]=.1
    with pytest.raises(ValueError,match='final'):
        stationary_boundaries(raw)


def test_terminal_velocity_up_to_the_default_tolerance_is_zeroed_and_more_is_rejected():
    raw = path()
    raw.velocity[-1, 0] = .0047
    clean = stationary_boundaries(raw)
    assert torch.count_nonzero(clean.velocity[[0, -1]]) == 0
    assert torch.equal(clean.position, raw.position)
    raw.velocity[-1, 0] = .0051
    with pytest.raises(ValueError, match='final'):
        stationary_boundaries(raw)


def test_wrist_travel_counts_only_j4_and_j5_excursion_from_the_start():
    q=torch.zeros((5,6))
    q[:,5]=torch.linspace(0,np.deg2rad(190),5)  # J6 may turn freely
    assert wrist_travel_deg(JointState(position=q)) == 0
    q[2,3]=np.deg2rad(40)   # an excursion that returns still counts
    q[4,4]=-np.deg2rad(165)
    assert wrist_travel_deg(JointState(position=q)) == pytest.approx(165)


def test_source_region_needs_neither_box_nor_free_space_observations():
    region=source_region([[-.39,-1.25,-.05],[.39,1.25,-.02]],[[.1,.6,0],[.15,.65,.1]],
                         [[-.365,.47],[.365,1.225]])
    assert region['source']=='pick_source'
    assert region['support_z_m']==pytest.approx(0)


def run_home_drop(monkeypatch, order):
    """Plan one home_drop particle against fakes; `order` names each plan_single call in sequence."""
    from types import SimpleNamespace
    from tiptop.doosan import in_place
    from tiptop.doosan.in_place import make_in_place_solver
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    events, starts, constrained = [], {}, {}

    def result(label, start, end=None):
        q = start.position[-1:].repeat(4, 1)
        end = q[0] + torch.tensor([.02, 0, 0, 0, 0, 0]) if end is None else end
        q[2], q[3] = (q[0] + end) / 2, end
        v = torch.zeros_like(q)
        v[2, 0] = .02
        events.append(label)
        starts[label] = start.position[-1:].clone()
        return SimpleNamespace(success=True, interpolation_dt=.02, status=None,
                               get_interpolated_plan=lambda: JointState(position=q, velocity=v))

    def plan_single(start, goal, options):
        label = next(calls)
        constrained[label] = options.pose_cost_metric is not None
        if label == "lift-failed":
            events.append(label)
            return SimpleNamespace(success=False, status=SimpleNamespace(name="INVALID_START_STATE_WORLD_COLLISION"))
        return result(label, start)

    noop = lambda *a, **k: None
    monkeypatch.setattr(in_place, "attach_object_at_grasp", lambda *a, **k: events.append("attach"))
    calls = iter(order)
    kinematics_config = SimpleNamespace(
        get_link_spheres=lambda name: torch.ones(2, 4),
        detach_object=lambda name: events.append("detach"),
        attach_object=lambda **k: events.append("reattach"),
    )
    mg = SimpleNamespace(
        detach_object_from_robot=noop,
        world_coll_checker=SimpleNamespace(enable_obstacle=noop),
        world_collision=SimpleNamespace(update_obstacle_pose=noop),
        kinematics=SimpleNamespace(kinematics_config=kinematics_config),
        plan_single=plan_single,
        plan_single_js=lambda start, goal, options: result("home", start, goal.position[0]),
    )
    world = SimpleNamespace(device=torch.device("cpu"), tool_from_ee=torch.eye(4),
                            tensor_args=SimpleNamespace(to_device=lambda x: torch.tensor(x)))
    pick = SimpleNamespace(operator=SimpleNamespace(name="Pick"), values=["object", "grasp", None])
    steps = make_in_place_solver(KnownTaskPlanningSettings())(
        dict(plan_skeleton=[pick]), dict(q0=torch.zeros(6), grasp=torch.eye(4)), world,
        SimpleNamespace(time_dilation_factor=.7), None, None, dict(object=torch.eye(4)), motion_gen=mg)
    return steps, events, starts, constrained


LABELS = ["Pick.approach", "Pick.descend", "Pick.close", "Place.lift", "Home", "Place.open"]


def test_home_drop_descends_in_one_leg_and_lifts_with_the_object_attached(monkeypatch):
    steps, events, starts, constrained = run_home_drop(monkeypatch, ["approach", "descend", "lift"])
    # One descend, the object joins the robot at the grasp, the lift is a new straight plan, then Home.
    assert events == ["approach", "descend", "attach", "lift", "home"]
    assert [s["label"] for s in steps] == LABELS
    assert constrained["lift"]
    assert torch.equal(starts["lift"], steps[1]["plan"].position[-1:])
    assert torch.equal(starts["home"], steps[3]["plan"].position[-1:])
    assert torch.equal(steps[4]["plan"].position[-1], torch.zeros(6))
    # The completed descend carries the actual grasp target; no other leg does.
    np.testing.assert_allclose(steps[1]["grasp_target_world_from_tcp"], np.eye(4))
    assert "grasp_target_world_from_tcp" not in steps[0]


def test_home_drop_replans_the_lift_without_the_object_when_the_grasp_pose_collides(monkeypatch):
    steps, events, starts, constrained = run_home_drop(
        monkeypatch, ["approach", "descend", "lift-failed", "lift"])
    # Same fallback as cuTAMP's Place: detach, plan the lift unconstrained, reattach, then carry on with the object.
    assert events == ["approach", "descend", "attach", "lift-failed", "detach", "lift", "reattach", "home"]
    assert [s["label"] for s in steps] == LABELS
    assert constrained["lift-failed"] and not constrained["lift"]
    assert torch.equal(starts["lift"], steps[1]["plan"].position[-1:])
