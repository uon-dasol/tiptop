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
    def __init__(self, *, moved=False, fail=False, cancelled=None, error="trajectory failed"):
        self.calls, self.moved, self.fail, self.cancelled, self.error = [], moved, fail, cancelled, error
        self.position = 213

    def get_state(self):
        return dict(
            success=True,
            robot_system=0,
            motion_status=0,
            joint_positions=[0.1 if self.moved else 0.0] * 6,
            gripper=dict(backend="uon_native", feedback_fresh=True, position_counts=self.position,
                         open_position_counts=1800, position_tolerance_counts=10),
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
        return dict(success=not self.fail, error=self.error)

    def close_gripper(self, **kwargs):
        self.calls.append("close")
        return dict(success=True)

    def open_gripper(self, **kwargs):
        self.calls.append("open")
        self.position = 1800
        return dict(success=True, endpoint_reached=True)

    def stop(self):
        self.calls.append("stop")


@pytest.fixture(autouse=True)
def measured_backend(monkeypatch):
    # Actual backend/fingerprint rejection is covered in test_doosan_execution.
    monkeypatch.setattr("tiptop.doosan.execution.validate_plan_backend", lambda *a: None)


def test_controller_completion_does_not_claim_object_success():
    client = Client()
    result = execute_measured_plan(plan(), client)
    assert client.calls == ["open", "move", "close"]
    assert result["gripper_preparation"]["verified_state"]["position_counts"] == 1800
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
        assert client.calls == ["open", "move", "stop"]


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


@pytest.mark.parametrize("mode", ["command_failed", "not_open", "stale", "arm_moved", "cancelled"])
def test_opening_failure_never_sends_arm_or_close_command(mode):
    cancel = threading.Event()
    client = Client()

    def open_gripper():
        client.calls.append("open")
        if mode == "cancelled":
            cancel.set()
        if mode == "arm_moved":
            client.moved = True
        client.position = 213 if mode == "not_open" else 1800
        if mode == "stale":
            get_state = client.get_state

            def stale_state():
                state = get_state()
                state["gripper"]["feedback_fresh"] = False
                return state

            client.get_state = stale_state
        return dict(success=mode != "command_failed", endpoint_reached=True)

    client.open_gripper = open_gripper
    with pytest.raises((ExecutionFailure, ValueError)):
        execute_measured_plan(plan(), client, cancelled=cancel)
    assert client.calls == ["open", "stop"]


def test_old_gateway_without_opening_configuration_sends_no_commands():
    client = Client()
    state = client.get_state()
    del state["gripper"]["open_position_counts"]
    client.get_state = lambda: state
    with pytest.raises(ValueError, match="updated gateway"):
        execute_measured_plan(plan(), client)
    assert client.calls == []


def test_failed_motion_report_preserves_stage_and_stop_result(tmp_path):
    import json

    path = tmp_path / "execution.json"
    client = Client(fail=True)
    with pytest.raises(ExecutionFailure):
        execute_measured_plan(plan(), client, report_path=path)
    report = json.loads(path.read_text())
    assert report["finished"] and not report["execution_success"]
    assert report["events"][0]["step"] == 0
    failed = report["events"][-1]
    assert failed["step"] == 1 and failed["type"] == "trajectory" and failed["status"] == "failed"
    assert failed["result"]["error"] == "trajectory failed"
    assert "stop_result" in report and client.calls == ["open", "move", "stop"]


@pytest.mark.parametrize("success,acknowledged", [(True, True), (True, False), (False, True)])
def test_gateway_stop_acknowledgement_controls_fallback_stop(tmp_path, success, acknowledged):
    import json

    client = Client()
    stopped = dict(success=success, arm_stop_acknowledged=acknowledged, stop_requested=True)

    def failed_motion(**kwargs):
        client.calls.append("move")
        return dict(success=False, error="trajectory failed", stop_result=stopped)

    client.execute_joint_impedance_path = failed_motion
    path = tmp_path / "execution.json"
    with pytest.raises(ExecutionFailure):
        execute_measured_plan(plan(), client, report_path=path)
    expected = ["open", "move"] if success and acknowledged else ["open", "move", "stop"]
    assert client.calls == expected
    if success and acknowledged:
        assert json.loads(path.read_text())["stop_result"] == stopped


def test_initial_opening_report_is_distinct_from_planned_gripper_actions(tmp_path):
    import json

    path = tmp_path / "execution.json"
    execute_measured_plan(plan(), Client(), report_path=path)
    report = json.loads(path.read_text())
    assert report["execution_success"] and not report["physical_grasp_verified"]
    assert [(s["step"], s["method"]) for s in report["events"]] == [
        (0, "open_gripper"), (1, "execute_joint_impedance_path"), (2, "close_gripper")]


def test_execution_timings_separate_initial_opening_from_planned_motion(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from tiptop.doosan import execution

    clock = [0.0]
    unix_time = execution.time.time
    monkeypatch.setattr(execution, "time", SimpleNamespace(perf_counter=lambda: clock[0], time=unix_time))

    class TimedClient(Client):
        def get_state(self):
            clock[0] += 0.25
            return super().get_state()

        def open_gripper(self, **kwargs):
            clock[0] += 2
            return super().open_gripper(**kwargs)

        def execute_joint_impedance_path(self, **kwargs):
            clock[0] += 5
            return super().execute_joint_impedance_path(**kwargs)

        def close_gripper(self, **kwargs):
            clock[0] += 1
            return super().close_gripper(**kwargs)

    client = TimedClient()
    path = tmp_path / "execution.json"
    result = execute_measured_plan(plan(), client, report_path=path)
    assert client.calls == ["open", "move", "close"]
    assert result["execution_duration_s"] == 6  # Existing definition stays planned steps only.
    assert result["execution_timings"] == dict(
        preflight_s=0.25, initial_gripper_open_s=2.25, planned_steps_s=6, postflight_s=0.25,
    )
    report = json.loads(path.read_text())
    assert report["execution_timings"] == result["execution_timings"]
    assert [event["duration_s"] for event in report["events"]] == [2, 5, 1]


@pytest.mark.parametrize("status,code,color,words", [(5, 0, "\033[33m", "취소됨"), (6, -5, "\033[31m", "중단됨")])
def test_trajectory_failure_is_explained_in_korean_with_matching_color(capsys, status, code, color, words):
    client = Client(fail=True, error=f"Trajectory failed: status={status}, code={code}, ")
    with pytest.raises(ExecutionFailure):
        execute_measured_plan(plan(), client)
    out = capsys.readouterr().out
    assert color in out and words in out and f"status={status}" in out
