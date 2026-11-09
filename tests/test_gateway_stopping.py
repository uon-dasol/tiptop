"""Stop coordination with recording callbacks; no ROS or devices."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "ros2_ws/src/execution/tiptop_execution/tiptop_execution/stopping.py"
spec = importlib.util.spec_from_file_location("gateway_stopping", path)
stopping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stopping)


@pytest.mark.parametrize("failed", [None, "gripper", "cancel", "arm", "all"])
def test_each_stop_operation_is_attempted_despite_other_failures(failed):
    calls = []

    def action(name):
        def run():
            calls.append(name)
            if failed in {name, "all"}:
                raise RuntimeError(name)
        return run

    result = stopping.stop_components(
        hold_gripper=action("gripper"), cancel_trajectory=action("cancel"), stop_arm=action("arm"))
    assert calls == ["gripper", "cancel", "arm"]
    assert result["success"] == (failed is None)
    assert result["arm_stop_acknowledged"] == (failed not in {"arm", "all"})
    assert result["stop_requested"]
