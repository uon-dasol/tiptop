"""No physical commands: check guards and cancellation with a recording client."""

import threading

import pytest

from tiptop.doosan.execution import execute_measured_plan
from tiptop.execute_plan import ExecutionFailure


def plan():
    return dict(
        version="1.0.0",
        robot="m1013_uon",
        robot_system=0,
        observation_id="current",
        q_init=[0.0] * 6,
        steps=[
            dict(
                type="trajectory", label="move", positions=[[0.0] * 6, [0.01] * 6], velocities=[[0.0] * 6] * 2, dt=1.0
            ),
            dict(type="gripper", label="close", action="close"),
        ],
    )


class Client:
    def __init__(self, *, moved=False, fail=False, cancelled=None):
        self.calls, self.moved, self.fail, self.cancelled = [], moved, fail, cancelled

    def get_state(self):
        return dict(
            success=True,
            robot_system=0,
            motion_status=0,
            joint_positions=[0.1 if self.moved else 0.0] * 6,
            gripper=dict(backend="uon_native", feedback_fresh=True),
            connections=dict(
                trajectory_action_ready=True,
                stop_service_ready=True,
                gripper_command_subscribers=1,
                gripper_command_publishers=1,
                gripper_state_publishers=1,
            ),
        )

    def execute_joint_impedance_path(self, **kwargs):
        self.calls.append("move")
        if self.cancelled is not None:
            self.cancelled.set()
        return dict(success=not self.fail, error="trajectory failed")

    def close_gripper(self, **kwargs):
        self.calls.append("close")
        return dict(success=True)

    def stop(self):
        self.calls.append("stop")


@pytest.fixture(autouse=True)
def measured_backend(monkeypatch):
    # Actual backend/fingerprint rejection is covered in test_doosan_execution.
    monkeypatch.setattr("tiptop.doosan.execution.validate_plan_backend", lambda *a: None)


def test_controller_completion_does_not_claim_object_success():
    client = Client()
    result = execute_measured_plan(plan(), client)
    assert client.calls == ["move", "close"]
    assert result["execution_success"] and result["success"] is None
    assert not result["physical_grasp_verified"] and not result["physical_placement_verified"]


def test_moved_robot_and_virtual_plan_send_no_commands():
    client = Client(moved=True)
    with pytest.raises(ValueError, match="changed since observation"):
        execute_measured_plan(plan(), client)
    assert client.calls == []
    wrong = dict(plan(), robot_system=1)
    with pytest.raises(ValueError, match="physical"):
        execute_measured_plan(wrong, Client())


def test_motion_failure_or_cancel_prevents_the_next_gripper_command():
    for fail in [True, False]:
        cancel = threading.Event()
        client = Client(fail=fail, cancelled=None if fail else cancel)
        with pytest.raises(ExecutionFailure):
            execute_measured_plan(plan(), client, cancelled=cancel)
        assert client.calls == ["move", "stop"]


def test_missing_driver_connection_blocks_arm_before_any_command():
    client = Client()
    state = client.get_state()
    state["connections"]["gripper_command_subscribers"] = 0
    client.get_state = lambda: state
    with pytest.raises(ValueError, match="driver"):
        execute_measured_plan(plan(), client)
    assert client.calls == []


@pytest.mark.parametrize("status", ["stale", "failed", "cancelled"])
def test_stale_gripper_or_unrecovered_failure_blocks_commands(status):
    client = Client()
    state = client.get_state()
    if status == "stale":
        state["gripper"]["feedback_fresh"] = False
    else:
        state["execution"] = dict(status=status)
    client.get_state = lambda: state
    with pytest.raises(ValueError, match="feedback|recover"):
        execute_measured_plan(plan(), client)
    assert client.calls == []
