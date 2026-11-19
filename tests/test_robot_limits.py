import json

import pytest
import torch
from curobo.types.state import JointState

from tiptop.doosan.in_place import stationary_boundaries
from tiptop.doosan.model import m1013_uon_curobo_cfg
from tiptop.robot_limits import DEFAULTS, load_robot_limits, save_robot_limits, validate_limits


def test_defaults_are_used_until_something_is_saved(tmp_path):
    assert load_robot_limits() == DEFAULTS
    assert not (tmp_path / "robot_limits.json").exists()


def test_save_roundtrips_and_load_reads_it(tmp_path):
    values = dict(DEFAULTS, max_acceleration=[6, 6, 6, 6, 6, 9], terminal_velocity_tolerance_rad_s=0.02)
    save_robot_limits(values)
    assert load_robot_limits()["max_acceleration"] == [6.0] * 5 + [9.0]
    assert json.loads((tmp_path / "robot_limits.json").read_text())["terminal_velocity_tolerance_rad_s"] == 0.02


@pytest.mark.parametrize("bad", [
    dict(max_acceleration=[5.0] * 5),
    dict(max_acceleration=[5.0] * 5 + [float("nan")]),
    dict(max_acceleration=[5.0] * 5 + [0.0]),
    dict(max_jerk=[160.0] * 5 + [5000.0]),
    dict(terminal_velocity_tolerance_rad_s=-1),
    dict(terminal_velocity_tolerance_rad_s=True),
    dict(unknown=1),
])
def test_invalid_limits_are_rejected_and_not_saved(tmp_path, bad):
    with pytest.raises(ValueError):
        save_robot_limits(dict(DEFAULTS, **bad))
    assert not (tmp_path / "robot_limits.json").exists()


def test_missing_fields_fall_back_to_the_defaults():
    assert validate_limits({}) == DEFAULTS


def test_planner_config_uses_the_saved_limits_for_the_six_joints_and_keeps_the_gripper_entry():
    original = m1013_uon_curobo_cfg()["robot_cfg"]["kinematics"]["cspace"]
    assert original["max_acceleration"][:6] == DEFAULTS["max_acceleration"] and len(original["max_acceleration"]) == 7
    save_robot_limits(dict(DEFAULTS, max_acceleration=[4, 4, 4, 4, 4, 4], max_jerk=[100] * 6))
    cspace = m1013_uon_curobo_cfg()["robot_cfg"]["kinematics"]["cspace"]
    assert cspace["max_acceleration"][:6] == [4.0] * 6 and cspace["max_jerk"][:6] == [100.0] * 6
    assert cspace["max_acceleration"][6] == original["max_acceleration"][6]


def test_stationary_tolerance_default_follows_the_saved_value():
    q = torch.zeros((4, 6))
    q[2:, 0] = torch.tensor([.01, .02])
    velocity = torch.zeros_like(q)
    velocity[-1, 0] = .02
    path = JointState(position=q, velocity=velocity)
    with pytest.raises(ValueError, match="final"):
        stationary_boundaries(path)
    save_robot_limits(dict(DEFAULTS, terminal_velocity_tolerance_rad_s=.03))
    assert float(stationary_boundaries(path).velocity[-1].abs().max()) == 0
