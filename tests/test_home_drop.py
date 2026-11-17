import numpy as np
import pytest

from tiptop.doosan.home_drop import drop_inside_box, validate_home_drop_plan

REGION = dict(center_xy_m=[0, .37], size_xy_m=[.34, .19], support_z_m=.15)
CAD = np.array([[-.02, -.02, -.03], [.02, .02, .03]])  # a 4 x 4 x 6 cm block around its own origin
# The block's centre sits 3 cm below the TCP while it is grasped at (0, 0.8, 0.03).
CAD_AT_GRASP = np.eye(4)
CAD_AT_GRASP[:3, 3] = [0, .8, 0]


def tcp(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_object_carried_home_is_checked_in_its_hand_frame():
    grasp, home = tcp(0, .8, .03), tcp(-.03, .35, .28)  # the held block keeps its pose relative to the TCP
    check = drop_inside_box(home, grasp, CAD_AT_GRASP, CAD, REGION, tolerance_m=.0045)
    assert check["drop_cad_bottom_z_m"] == pytest.approx(.28 - .03 - .03)
    assert check["drop_height_m"] == pytest.approx(.28 - .03 - .03 - .15)


@pytest.mark.parametrize("home, message", [
    (tcp(.2, .35, .28), "not over the box floor"),   # beyond the side wall
    (tcp(0, .8, .28), "not over the box floor"),     # not over the box at all
    (tcp(0, .35, .15), "below the box top"),              # would sit inside the wall height
])
def test_drop_is_rejected_when_the_held_object_misses_or_dips_into_the_box(home, message):
    with pytest.raises(ValueError, match=message):
        drop_inside_box(home, tcp(0, .8, .03), CAD_AT_GRASP, CAD, REGION, tolerance_m=.0045)


def test_object_may_extend_past_the_wall_when_its_centre_is_over_the_box():
    home = tcp(0, .37 + .09, .28)  # the 4 cm block's far side is 1 cm past the 9.5 cm half depth
    check = drop_inside_box(home, tcp(0, .8, .03), CAD_AT_GRASP, CAD, REGION, tolerance_m=.0045)
    assert check["center_inside_box"] and check["drop_cad_bounds_xyz_m"][1][1] > .37 + .095


def plan(labels=("Pick.approach", "Pick.descend", "Pick.close", "Place.lift", "Home", "Place.open")):
    q_grasp, q_lift, q_home = np.full(6, .1), np.full(6, .2), np.zeros(6)
    positions = {"Pick.approach": [q_home, q_grasp], "Pick.descend": [q_grasp, q_grasp],
                 "Place.lift": [q_grasp, q_lift], "Home": [q_lift, q_home]}
    steps = []
    for label in labels:
        step = dict(label=label, type="trajectory", positions=positions.get(label, []))
        if label == "Pick.close":
            step.update(type="gripper", action="close")
        if label == "Place.open":
            step.update(type="gripper", action="open")
        steps.append(step)
    return dict(steps=steps, q_init=q_home)


def fake_tcp_at(q):
    if np.allclose(q, 0):
        return tcp(-.03, .35, .28)
    return tcp(0, .8, .03 if np.allclose(q, .1) else .08)


def test_valid_home_drop_plan_passes_and_reports_the_drop():
    result = validate_home_drop_plan(plan(), fake_tcp_at, CAD_AT_GRASP, CAD, REGION, lift_distance_m=.05,
                                     position_tolerance_m=.0025)
    assert result["drops_at_home"] and result["center_inside_box"]


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["steps"].reverse(), "unexpected action sequence"),
    (lambda p: p["steps"][4].update(positions=[np.ones(6), np.zeros(6)]), "start at the lifted joints"),
    (lambda p: p.update(q_init=np.ones(6)), "finish at its captured home"),
])
def test_home_drop_plan_with_a_wrong_sequence_or_endpoints_is_rejected(mutate, message):
    value = plan()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_home_drop_plan(value, fake_tcp_at, CAD_AT_GRASP, CAD, REGION, lift_distance_m=.05,
                                position_tolerance_m=.0025)
