"""Return-path mathematics and success criteria, without hardware or inference."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from curobo.types.state import JointState

from tiptop.doosan.in_place import (
    join_paths, source_region, reverse_path, stationary_boundaries, validate_in_place_plan, wrist_travel_deg)


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


def test_reverse_changes_velocity_sign_and_preserves_the_exact_joint_path():
    forward=stationary_boundaries(path())
    backward=reverse_path(forward)
    assert torch.equal(backward.position,forward.position.flip(0))
    assert torch.equal(backward.velocity,-forward.velocity.flip(0))
    assert torch.equal(reverse_path(backward).position,forward.position)


def test_joined_descend_keeps_both_rests_and_drops_only_the_repeated_waypoint():
    far=stationary_boundaries(path())
    near=JointState(position=far.position[-1:]+torch.tensor([[0,0,0,0,0,0.],[.01,0,0,0,0,0],[.02,0,0,0,0,0]]),
                    velocity=torch.zeros((3,6)))
    joined=join_paths(far,near)
    assert len(joined.position)==len(far.position)+len(near.position)-1
    assert torch.equal(joined.position[:4],far.position) and torch.equal(joined.position[4:],near.position[1:])
    assert torch.equal(joined.velocity[3],torch.zeros(6)) and torch.equal(joined.velocity[[0,-1]],torch.zeros((2,6)))
    assert torch.equal(reverse_path(near).position,near.position.flip(0))
    with pytest.raises(ValueError,match='meet'):
        join_paths(far,JointState(position=near.position+1,velocity=near.velocity))


def plan():
    def leg(label,q):
        return dict(type='trajectory',label=label,positions=np.asarray(q).tolist(),velocities=np.zeros_like(q).tolist(),dt=.5)
    home=[0.]*6
    grasp=[.2,.7,.2,0,0,0]
    upper=[.2,.7,.25,0,0,0]
    a=leg('Pick.approach',[home,upper])
    b=leg('Pick.descend',[upper,grasp])
    c=leg('Place.lift',[grasp,upper])
    hold=leg('Place.hold',[upper,upper])
    return dict(q_init=home,steps=[a,b,dict(type='gripper',label='Pick.close',action='close'),c,hold,
        dict(type='gripper',label='Place.open',action='open'),leg('Home',[upper,home])])


def tcp_at(q):
    pose=np.eye(4)
    pose[:3,3]=q[:3]
    return pose


def test_in_place_sequence_opens_at_lifted_pose_and_then_returns_home():
    from tiptop.doosan.tamp import verify_gateway_contract
    value=plan()
    result=validate_in_place_plan(value,tcp_at,lift_distance_m=.05,position_tolerance_m=.01)
    assert result['release_at_lifted_pose'] is True
    assert len(verify_gateway_contract(value))==7


@pytest.mark.parametrize('bad',['return','home','open','hold','short_lift'])
def test_changed_return_or_hold_cannot_pass_the_plan_guard(bad):
    value=deepcopy(plan())
    if bad=='return': value['steps'][6]['positions'][0][0]+=.02
    elif bad=='home': value['steps'][6]['positions'][-1][0]+=.02
    elif bad=='open': value['steps'][5]['action']='close'
    elif bad=='hold': value['steps'][4]['positions'][0][0]+=.01
    else: value['steps'][3]['positions'][-1][2]=.205
    with pytest.raises(ValueError):
        validate_in_place_plan(value,tcp_at,lift_distance_m=.05,position_tolerance_m=.01)


def test_source_region_needs_neither_box_nor_free_space_observations():
    region=source_region([[-.39,-1.25,-.05],[.39,1.25,-.02]],[[.1,.6,0],[.15,.65,.1]],
                         [[-.365,.47],[.365,1.225]])
    assert region['source']=='in_place_return'
    assert region['support_z_m']==pytest.approx(0)


def test_completed_solver_carries_its_actual_target_with_the_descend_path():
    from types import SimpleNamespace
    from tiptop.doosan.in_place import make_in_place_solver
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    initial = torch.eye(4)
    initial[:3, 3] = torch.tensor([.2, .3, .4])
    grasp = torch.eye(4)
    grasp[0, 3] = .02
    tool = torch.eye(4)
    tool[2, 3] = .01
    paths = []

    def leg(*args):
        q = torch.zeros((2, 6))
        value = JointState(position=q, velocity=torch.zeros_like(q))
        paths.append(value)
        return SimpleNamespace(success=True, interpolation_dt=.02, get_interpolated_plan=lambda: value)

    noop = lambda *args, **kwargs: None
    mg = SimpleNamespace(detach_object_from_robot=noop,
                         world_coll_checker=SimpleNamespace(enable_obstacle=noop),
                         world_collision=SimpleNamespace(update_obstacle_pose=noop),
                         plan_single=leg, plan_single_js=leg)
    world = SimpleNamespace(device=torch.device('cpu'), tool_from_ee=tool,
                            tensor_args=SimpleNamespace(to_device=lambda x: torch.tensor(x)))
    pick = SimpleNamespace(operator=SimpleNamespace(name='Pick'), values=['object', 'grasp', None])
    result = make_in_place_solver(KnownTaskPlanningSettings())(
        dict(plan_skeleton=[pick]), dict(q0=torch.zeros(6), grasp=grasp), world,
        SimpleNamespace(time_dilation_factor=.4), None, None, dict(object=initial), motion_gen=mg)
    assert len(result) == 7
    np.testing.assert_allclose(result[1]['grasp_target_world_from_tcp'], (initial @ grasp @ tool).numpy())
    assert result[1]['label'] == 'Pick.descend' and 'grasp_target_world_from_tcp' not in result[0]
    # Metadata leaves every generated joint sample untouched.
    for value in paths:
        assert torch.equal(value.position, torch.zeros_like(value.position))


@pytest.mark.parametrize("overrides", [{}, dict(approach_distance_m=.05)])
def test_every_leg_runs_at_the_fixed_speed_and_lift_preserves_time_velocity_consistency(overrides, monkeypatch):
    from types import SimpleNamespace
    from tiptop.doosan import in_place
    from tiptop.doosan.in_place import make_in_place_solver
    from tiptop.doosan.known_task import KnownTaskPlanningSettings

    settings = KnownTaskPlanningSettings(**overrides)
    calls, generated = [], []

    def result(start, options, end=None):
        q = start.position[-1:].repeat(4, 1)
        end = q[0] + torch.tensor([.02, 0, 0, 0, 0, 0]) if end is None else end
        q[2], q[3] = (q[0] + end) / 2, end
        v = torch.zeros_like(q)
        v[0, 0] = .003  # The duplicate initial positions permit the existing residue cleanup.
        v[2, 0] = .02
        value = JointState(position=q, velocity=v)
        calls.append(options.time_dilation_factor)
        generated.append(value.clone())
        return SimpleNamespace(success=True, interpolation_dt=.02, get_interpolated_plan=lambda: value)

    noop = lambda *a, **k: None
    mg = SimpleNamespace(
        detach_object_from_robot=noop,
        world_coll_checker=SimpleNamespace(enable_obstacle=noop),
        world_collision=SimpleNamespace(update_obstacle_pose=noop),
        plan_single=lambda start, goal, options: result(start, options),
        plan_single_js=lambda start, goal, options: result(start, options, goal.position[0]),
    )
    world = SimpleNamespace(device=torch.device('cpu'), tool_from_ee=torch.eye(4),
                            tensor_args=SimpleNamespace(to_device=lambda x: torch.tensor(x)))
    pick = SimpleNamespace(operator=SimpleNamespace(name='Pick'), values=['object', 'grasp', None])
    steps = make_in_place_solver(settings)(
        dict(plan_skeleton=[pick]), dict(q0=torch.zeros(6), grasp=torch.eye(4)), world,
        SimpleNamespace(time_dilation_factor=.7), None, None, dict(object=torch.eye(4)), motion_gen=mg)
    has_far = settings.approach_distance_m > settings.lift_distance_m
    assert calls == [in_place.SPEED_FACTOR] * (4 if has_far else 3)
    assert [s['label'] for s in steps] == [
        'Pick.approach', 'Pick.descend', 'Pick.close', 'Place.lift', 'Place.hold', 'Place.open', 'Home']
    near = generated[2 if has_far else 1]
    lift = steps[3]
    assert torch.equal(lift['plan'].position, near.position.flip(0))
    assert torch.equal(steps[1]['plan'].position[-len(near.position):], near.position)
    assert lift['dt'] == pytest.approx(.02)
    # Same joint displacement per sample after reversing and retiming; the
    # original descent positions, timing, and raw velocities stay untouched.
    assert float(lift['plan'].velocity[1, 0]) * lift['dt'] == pytest.approx(-.02 * .02)
    assert torch.count_nonzero(lift['plan'].velocity[[0, -1]]) == 0
    assert near.velocity[0, 0] == pytest.approx(.003)
    assert steps[4]['dt'] * len(steps[4]['plan'].position) == settings.lift_hold_s
    assert torch.equal(steps[-1]['plan'].position[-1], torch.zeros(6))


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
    steps = make_in_place_solver(KnownTaskPlanningSettings(), drop_at_home=True)(
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


def test_home_drop_replans_the_lift_without_the_object_when_the_grasp_pose_collides(monkeypatch):
    steps, events, starts, constrained = run_home_drop(
        monkeypatch, ["approach", "descend", "lift-failed", "lift"])
    # Same fallback as cuTAMP's Place: detach, plan the lift unconstrained, reattach, then carry on with the object.
    assert events == ["approach", "descend", "attach", "lift-failed", "detach", "lift", "reattach", "home"]
    assert [s["label"] for s in steps] == LABELS
    assert constrained["lift-failed"] and not constrained["lift"]
    assert torch.equal(starts["lift"], steps[1]["plan"].position[-1:])


def test_simulator_accepts_source_return_without_transport_but_requires_table_contact():
    from simulation.isaac_sim.task_runtime import evaluate_pick_place
    args=dict(baseline=[.1,.6,.04],final=[.1,.6,.04],trace=[dict(position=[.1,.6,.09])],
              forces=[0,0,0],support=.2,region=dict(source='in_place_return',center_xy_m=[.1,.6],size_xy_m=[.1,.1]),
              max_other_displacement=0)
    assert evaluate_pick_place(**args)['success'] is True
    args['support']=0
    assert evaluate_pick_place(**args)['success'] is False
