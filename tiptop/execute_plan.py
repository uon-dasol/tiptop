import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tiptop.utils import RobotClient

_log = logging.getLogger(__name__)


class ExecutionFailure(Exception):
    """Failure in executing plan on robot."""


def execute_cutamp_plan(cutamp_plan: list[dict], client: "RobotClient | None" = None) -> None:
    """Execute cuTAMP actions or serialized TiPToP steps in order."""
    if client is None:
        from tiptop.utils import get_robot_client

        client = get_robot_client()

    start_time = time.perf_counter()
    for step, action_dict in enumerate(cutamp_plan):
        action_start_time = time.perf_counter()
        action_type = action_dict["type"]
        action_label = action_dict["label"]

        # Form log message
        msg = f"Executing step {step + 1}/{len(cutamp_plan)}: {action_label}. Action type: {action_dict['type']}"
        if action_type == "gripper":
            msg += f" ({action_dict['action']})"
        elif action_type == "trajectory":
            positions = action_dict["plan"].position if "plan" in action_dict else action_dict["positions"]
            msg += f" ({len(positions)} waypoints)"
        else:
            raise ValueError(f"Unknown action type in cuTAMP plan: {action_dict['type']}")
        _log.info(msg)

        # Now execute the actions
        if action_type == "gripper":
            action = action_dict["action"]
            if action == "open":
                result = client.open_gripper(speed=1.0)
            elif action == "close":
                result = client.close_gripper(speed=1.0)
            else:
                raise ValueError(f"Unknown gripper action: {action}")

        elif action_type == "trajectory":
            # Extract joint position and velocity waypoints for the trajectory
            if "plan" in action_dict:
                waypoints = action_dict["plan"].position.cpu().numpy()
                velocities = action_dict["plan"].velocity.cpu().numpy()
            else:
                waypoints = np.asarray(action_dict["positions"])
                velocities = np.asarray(action_dict["velocities"])
            timings = [action_dict["dt"]] * len(waypoints)
            result = client.execute_joint_impedance_path(
                joint_confs=waypoints, joint_vels=velocities, durations=timings
            )

        else:
            raise ValueError(f"Unexpected action type in cuTAMP plan: {action_dict['type']}")

        # Raise error if execution failed
        if result is None:
            raise RuntimeError("Fatal error: result should not be None")
        if not result["success"]:
            raise ExecutionFailure(result["error"])
        _log.info("Completed step %s/%s: %s", step + 1, len(cutamp_plan), result)

        action_duration = time.perf_counter() - action_start_time
        _log.debug(f"Executing {action_type} action took {action_duration:.2f}s")

    # Now we're done executing plan open-loop without any failures on the controller side
    duration = time.perf_counter() - start_time
    _log.info(f"Plan execution took {duration:.2f}s")
