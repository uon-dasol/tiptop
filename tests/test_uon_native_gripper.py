"""Native UON protocol completion/stale feedback/cancellation, with no ROS or motor."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "ros2_ws/src/execution/tiptop_execution/tiptop_execution/uon_gripper.py"
spec = importlib.util.spec_from_file_location("uon_native_contract", path)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def profile():
    return dict(
        calibrated=True,
        position_unit="dynamixel_counts",
        current_unit="dynamixel_current_counts",
        stroke_min_counts=0,
        stroke_max_counts=1000,
        open_position_counts=1000,
        closed_position_counts=0,
        motor_current_limit_counts=500,
        open_current_limit_counts=50,
        close_current_limit_counts=100,
        contact_current_threshold_counts=60,
        position_tolerance_counts=2,
        settle_s=0.03,
        contact_settle_s=0.05,
        command_timeout_s=0.4,
        feedback_max_age_s=0.1,
    )


class Device:
    def __init__(self, mode="endpoint"):
        self.now, self.position, self.target = 0.0, 1000, None
        self.mode, self.commands = mode, []

    def publish(self, target, limit):
        self.target = target
        self.commands.append((target, limit))

    def sleep(self, seconds):
        self.now += seconds
        if self.target is not None:
            desired = self.target if self.mode == "endpoint" else max(400, self.target)
            self.position += max(-100, min(100, desired - self.position))

    def feedback(self):
        current = 65536 - 80 if self.mode == "contact" and self.position == 400 else 5
        stamp = self.now if self.mode not in {"stale", "frozen"} else (-1 if self.mode == "stale" else 0.0)
        return dict(received_at=stamp, position_counts=self.position, current_counts=current)

    def controller(self, settings=None):
        return native.NativeUONGripper(
            settings or profile(), self.publish, self.feedback, clock=lambda: self.now, sleep=self.sleep
        )


def test_endpoint_completion_uses_native_counts_and_post_command_feedback():
    device = Device()
    result = device.controller().run("close")
    assert result["endpoint_reached"] and result["position_counts"] == 0
    assert device.commands == [(0, 100)]
    assert not result["physical_grasp_verified"] and not result["force_parameter_applied"]
    assert device.controller().run("open")["position_counts"] == 1000
    assert device.commands[-1] == (1000, 50)


@pytest.mark.parametrize("position,expected", [(1000, "open"), (998, "open"), (0, "closed"), (500, "intermediate")])
def test_reported_position_state_uses_feedback_without_actuation(position, expected):
    device = Device()
    device.position = position
    state = device.controller().state()
    assert state["position_state"] == expected
    assert state["position_counts"] == position and state["open_position_counts"] == 1000
    assert not device.commands


def test_opening_from_closed_waits_for_measured_open_endpoint():
    device = Device()
    device.position = 0
    result = device.controller().run("open")
    assert result["success"] and result["endpoint_reached"]
    assert result["position_counts"] == 1000 and device.now >= .1


def test_stable_closing_contact_completes_before_unreachable_endpoint():
    device = Device("contact")
    result = device.controller().run("close")
    assert result["motor_contact_inferred"] and not result["endpoint_reached"]
    assert result["position_counts"] == 400 and result["current_counts"] == -80
    assert not result["physical_grasp_verified"]
    assert device.commands == [(0, 100)]


def test_low_current_stall_times_out_and_holds_the_observed_position():
    device = Device("no-contact")
    with pytest.raises(TimeoutError):
        device.controller().run("close")
    assert device.commands == [(0, 100), (400, 100)]


@pytest.mark.parametrize("mode", ["stale", "frozen"])
def test_missing_new_feedback_cannot_report_completion(mode):
    device = Device(mode)
    with pytest.raises(RuntimeError, match="stale"):
        device.controller().run("open")
    if mode == "stale":
        assert not device.commands


def test_cancel_does_not_continue_closing_and_sends_measured_hold():
    device = Device()
    with pytest.raises(RuntimeError, match="cancelled"):
        device.controller().run("close", cancelled=lambda: device.now >= 0.05)
    assert device.commands[0] == (0, 100)
    assert device.commands[-1] == (device.position, 100) and device.position > 0


def test_unmeasured_profile_or_mismatched_units_cannot_publish():
    device = Device()
    for changes in [dict(calibrated=False), dict(position_unit="rad"), dict(close_current_limit_counts=501)]:
        with pytest.raises(ValueError):
            device.controller(dict(profile(), **changes))
    assert not device.commands
    assert native.signed_current_counts(65535) == -1
    assert native.signed_current_counts(-80) == -80
    with pytest.raises(ValueError):
        native.signed_current_counts(float("nan"))


def test_unconfigured_example_does_not_enable_commands():
    import json

    settings = json.loads((path.parent.parent / "config/uon_native.example.json").read_text())
    device = Device()
    with pytest.raises(ValueError, match="calibration"):
        device.controller(settings)
    assert not device.commands


def test_operator_profile_opens_at_3400_and_readiness_uses_the_same_target():
    import json
    from tiptop.doosan.readiness import check_gateway_state

    settings = json.loads((path.parent.parent / "config/uon_native.measured.json").read_text())
    device = Device()
    controller = device.controller(settings)
    # Existing observations and the opposite closed end must be readable before
    # the initial opening command; a feedback ceiling of 3600 would reject them.
    for position in (1800, 3127, 3706, 6321):
        device.position = position
        assert controller.state()["position_state"] == "intermediate"
    result = controller.run("open")
    assert device.commands == [(3400, 50)]
    assert result["endpoint_reached"] and controller.state()["position_state"] == "open"
    state = dict(success=True, robot_system=0, motion_status=0, joint_positions=[0.]*6,
                 gripper=dict(backend="uon_native", feedback_fresh=True, **controller.state()),
                 connections=dict(trajectory_action_ready=True, stop_service_ready=True,
                                  gripper_command_subscribers=1, gripper_command_publishers=1,
                                  gripper_state_publishers=1))
    assert check_gateway_state(state, execute=True, require_open=True)["gripper_open"]
    state["gripper"]["position_counts"] = 1800
    with pytest.raises(ValueError, match="did not reach its open position"):
        check_gateway_state(state, execute=True, require_open=True)
