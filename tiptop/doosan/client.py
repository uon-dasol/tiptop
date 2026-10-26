"""Blocking RobotClient API over the ROS2 execution gateway's JSON/HTTP API."""

import json
from functools import cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def validate_plan_backend(plan, state):
    """Check a saved plan's actual/virtual backend before any command is sent."""
    if not state.get("success"):
        raise RuntimeError(state.get("error", "Robot state unavailable"))
    if state.get("read_only"):
        raise ValueError("The gateway is read-only; no plan can execute")
    system = state.get("robot_system")
    if system not in (0, 1):
        raise ValueError("The gateway must report its actual robot_system")
    expected = plan.get("robot_system")
    if expected is not None and expected != system:
        raise ValueError("Saved plan and actual robot_system differ")
    if system == 0:
        from tiptop.doosan.hardware import hardware_profile_fingerprint, load_hardware_profile

        if expected != 0 or plan.get("robot") != "m1013_uon" or not plan.get("hardware_profile"):
            raise ValueError("A physical run requires a plan made with measured --hardware-profile inputs")
        if state.get("gripper", {}).get("backend") != "uon_native":
            raise ValueError("A physical run requires the configured native UON gripper backend")
        profile = load_hardware_profile(plan["hardware_profile"])
        if plan.get("hardware_profile_sha256") != hardware_profile_fingerprint(profile):
            raise ValueError("Hardware geometry or camera calibration changed after planning; acquire and plan again")


class DoosanUONClient:
    """M1013 joint paths (rad, rad/s, seconds) and the UON model gripper.

    The historical execute_joint_impedance_path name implements position trajectory
    execution here, not Franka torque impedance. Commands are never retried.
    """

    def __init__(self, url: str = "http://127.0.0.1:8090"):
        self.url = url.rstrip("/")

    def _request(self, path, payload=None, timeout=10.0):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            return json.load(exc)
        except (URLError, TimeoutError, OSError) as exc:
            return {
                "success": False,
                "error": f"Gateway communication failed: {exc}. Query state before another command.",
            }

    def get_state(self):
        return self._request("/state")

    def get_joint_positions(self):
        state = self.get_state()
        if not state["success"]:
            raise RuntimeError(state["error"])
        return state["joint_positions"]

    def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
        try:
            durations = [float(dt) for dt in durations]
            payload = {
                "positions": [[float(q) for q in row] for row in joint_confs],
                "velocities": [[float(v) for v in row] for row in joint_vels],
                "durations": durations,
            }
            return self._request("/trajectory", payload, timeout=max(10.0, sum(durations) + 20.0))
        except (ValueError, TypeError) as exc:
            return {"success": False, "error": str(exc)}

    def open_gripper(self, speed=1.0, force=0.1):
        return self._request("/gripper", {"action": "open", "speed": speed, "force": force}, timeout=180.0)

    def close_gripper(self, speed=1.0, force=0.1):
        return self._request("/gripper", {"action": "close", "speed": speed, "force": force}, timeout=180.0)

    def stop(self):
        return self._request("/stop", {})

    def close(self):
        """HTTP calls have no persistent device connection to close."""


@cache
def get_doosan_client():
    from tiptop.config import tiptop_cfg

    cfg = tiptop_cfg()
    if cfg.robot.dof != 6:
        raise ValueError("M1013 requires robot.dof=6")
    return DoosanUONClient(f"http://{cfg.robot.host}:{cfg.robot.port}")
