"""Blocking RobotClient API over the ROS2 execution gateway's JSON/HTTP API."""

import json
import re
from functools import cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_TRAJECTORY_ERROR = re.compile(r"Trajectory failed: status=(-?\d+), code=(-?\d+)")
# action_msgs/GoalStatus
_GOAL_STATUS = {
    0: ("UNKNOWN", "알 수 없음"),
    1: ("ACCEPTED", "수락됨"),
    2: ("EXECUTING", "실행 중"),
    3: ("CANCELING", "취소 중"),
    4: ("SUCCEEDED", "성공"),
    5: ("CANCELED", "취소 요청으로 종료"),
    6: ("ABORTED", "컨트롤러가 스스로 종료"),
}
# control_msgs/FollowJointTrajectory error_code
_ERROR_CODE = {
    0: ("SUCCESSFUL", "오류 없음"),
    -1: ("INVALID_GOAL", "잘못된 목표"),
    -2: ("INVALID_JOINTS", "관절 불일치"),
    -3: ("OLD_HEADER_TIMESTAMP", "오래된 타임스탬프"),
    -4: ("PATH_TOLERANCE_VIOLATED", "이동 중 경로 오차 초과"),
    -5: ("GOAL_TOLERANCE_VIOLATED", "허용 시간 안에 목표 도달 실패"),
}
_HEADLINE = {3: "궤적 실행 취소 중", 5: "궤적 실행 취소됨", 6: "궤적 실행 중단됨"}
_YELLOW, _RED = "\033[33m", "\033[31m"


def describe_trajectory_error(message) -> tuple[str, str] | None:
    """Korean explanation and ANSI color for a gateway 'Trajectory failed: status=, code=' error, else None."""
    match = _TRAJECTORY_ERROR.search(str(message))
    if match is None:
        return None
    status, code = (int(value) for value in match.groups())
    status_name, status_text = _GOAL_STATUS.get(status, ("?", "알 수 없는 상태"))
    code_name, code_text = _ERROR_CODE.get(code, ("?", "알 수 없는 코드"))
    headline = _HEADLINE.get(status, "궤적 실행 실패")
    text = f"{headline} | status={status} {status_name} ({status_text}) · code={code} {code_name} ({code_text})"
    return text, _YELLOW if status in (3, 5) else _RED


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
