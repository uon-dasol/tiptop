"""Execution contract tests; no ROS daemon, device, camera, or GPU required."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tiptop.doosan.client import DoosanUONClient, validate_plan_backend
from tiptop.execute_plan import ExecutionFailure, execute_cutamp_plan

_path = (
    Path(__file__).resolve().parents[1] / "ros2_ws/src/execution/tiptop_execution" / "tiptop_execution/trajectory.py"
)
_spec = importlib.util.spec_from_file_location("trajectory_contract", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_trajectory = _module.validate_trajectory


def test_saved_plan_cannot_cross_virtual_real_or_state_only_backends():
    virtual = dict(success=True, robot_system=1)
    physical = dict(success=True, robot_system=0, gripper=dict(backend="uon_native"))
    validate_plan_backend(dict(robot_system=1), virtual)
    # Legacy virtual plans remain usable, but are never accepted on a real arm.
    validate_plan_backend({}, virtual)
    with pytest.raises(ValueError, match="differ"):
        validate_plan_backend(dict(robot_system=1), physical)
    with pytest.raises(ValueError, match="measured"):
        validate_plan_backend({}, physical)
    with pytest.raises(ValueError, match="read-only"):
        validate_plan_backend(dict(robot_system=0), dict(physical, read_only=True))
    with pytest.raises(ValueError, match="actual robot_system"):
        validate_plan_backend({}, dict(success=True))


def test_saved_physical_plan_uses_same_measured_executor_as_live_cli(monkeypatch, tmp_path, capsys):
    import json
    import sys

    from tiptop.doosan import __main__ as cli

    path = tmp_path / "physical.json"
    path.write_text(json.dumps(dict(version="1.0.0", robot_system=0)))
    calls = []

    def execute(plan, client, **kwargs):
        calls.append(plan)
        return dict(success=None, execution_success=True, status="execution_completed_unverified")

    monkeypatch.setattr("tiptop.doosan.execution.execute_measured_plan", execute)
    monkeypatch.setattr(cli, "execute_cutamp_plan", lambda *a: pytest.fail("Physical plan bypassed measured checks"))
    monkeypatch.setattr(sys, "argv", ["tiptop.doosan", "run", str(path)])
    cli.main()
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["success"] is None


def test_joint_units_velocities_and_variable_durations(monkeypatch):
    """A nonuniform timed path crosses the client boundary without losing qdot."""
    client = DoosanUONClient()
    q = np.array([[0] * 6, [0.01] * 6, [0.03] * 6])
    v = np.array([[0] * 6, [0.02] * 6, [0] * 6])
    durations = [0.05, 0.5, 1.0]
    captured = {}

    def request(path, payload, timeout):
        captured.update(payload)
        assert path == "/trajectory"
        return {"success": True}

    monkeypatch.setattr(client, "_request", request)
    assert client.execute_joint_impedance_path(q, v, durations)["success"]
    positions, velocities, times = validate_trajectory(captured)
    np.testing.assert_array_equal(positions, q)
    np.testing.assert_array_equal(velocities, v)
    np.testing.assert_allclose(times, [0.05, 0.55, 1.55])


@pytest.mark.parametrize(
    "update",
    [
        {"positions": [[0] * 7], "velocities": [[0] * 7]},  # Franka's seven joints
        {"positions": [[10] * 6]},  # Degrees accidentally passed as radians
        {"durations": [0]},
        {"velocities": [[float("nan")] * 6]},
        {"velocities": [[0.1] * 6]},  # A moving endpoint cannot be a blocking completion
    ],
)
def test_invalid_paths_are_rejected(update):
    payload = {"positions": [[0] * 6], "velocities": [[0] * 6], "durations": [1]}
    payload.update(update)
    with pytest.raises(ValueError):
        validate_trajectory(payload)


def test_execution_stops_before_gripper_after_arm_failure():
    calls = []

    class Client:
        def execute_joint_impedance_path(self, **kwargs):
            calls.append("arm")
            return {"success": False, "error": "goal rejected"}

        def close_gripper(self, **kwargs):
            calls.append("close")
            return {"success": True}

    steps = [
        {"type": "trajectory", "label": "move", "positions": [[0] * 6], "velocities": [[0] * 6], "dt": 1},
        {"type": "gripper", "label": "close", "action": "close"},
    ]
    with pytest.raises(ExecutionFailure, match="goal rejected"):
        execute_cutamp_plan(steps, Client())
    assert calls == ["arm"]
