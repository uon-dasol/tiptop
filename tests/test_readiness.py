"""Connection checks use GET only, never inference or actuator commands."""

import asyncio
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np
import pytest

from tiptop.doosan.readiness import check_gateway_state, check_readiness


@pytest.fixture
def services():
    packet = dict(
        rgb=np.zeros((4, 6, 3), np.uint8),
        intrinsics=np.diag([100, 100, 1]),
        q_init=np.zeros(6),
        world_from_cam=np.eye(4),
        observation_id="fresh-frame",
        camera_model="isaac",
        stereo_right_rgb=np.zeros((4, 6, 3), np.uint8),
        stereo_baseline_m=0.095,
    )
    requests, failures = [], set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(("POST", self.path))
            self.send_error(405)

        def do_GET(self):
            requests.append(("GET", self.path))
            path = urlsplit(self.path).path
            if path in failures:
                self.send_error(503)
                return
            if path == "/camera/observe":
                stream = io.BytesIO()
                np.savez_compressed(stream, **packet)
                data = stream.getvalue()
            else:
                responses = {
                    "/camera/health": dict(success=True),
                    "/pose/health": dict(status="healthy", model_loaded=True),
                    "/depth/health": dict(status="healthy", model_loaded=True),
                    "/sam3/health": dict(status="ready", backend="sam3_reference", inference_calls=0),
                    "/sam3/reference": dict(success=True, inference_performed=False),
                    "/gateway/state": gateway_state(),
                }
                data = json.dumps(responses[path]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Type", "application/octet-stream" if path.startswith("/camera/") else "application/json"
            )
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", packet, requests, failures
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def options(url):
    return SimpleNamespace(
        perception_mode="sam3_reference",
        task="black_pepper_shaker를 집고 빈곳에 놓아",
        camera_url=url + "/camera",
        depth_url=url + "/depth",
        pose_url=url + "/pose",
        sam3_url=url + "/sam3",
        reference_scene="Home/Uon_1F_Home/floor",
        hardware_profile=None,
        gateway_url=None,
        execute=True,
        active_task=None,
    )


def test_ready_services_capture_once_and_never_post(services, monkeypatch):
    url, packet, requests, _ = services
    monkeypatch.setattr("tiptop.doosan.readiness.registered_object", lambda name: {})
    result = asyncio.run(check_readiness(options(url)))
    assert result["success"] and result["check_only"]
    assert result["motion_commands"] == result["model_inference_calls"] == result["gemini_api_calls"] == 0
    assert all(method == "GET" for method, path in requests)
    assert requests.count(("GET", "/camera/observe")) == 1
    assert any(path.startswith("/sam3/reference?") for method, path in requests)


def test_reports_all_unavailable_services_without_inference(services, monkeypatch):
    url, _, requests, failures = services
    failures.update(["/pose/health", "/depth/health", "/sam3/reference"])
    monkeypatch.setattr("tiptop.doosan.readiness.registered_object", lambda name: {})
    result = asyncio.run(check_readiness(options(url)))
    assert not result["success"]
    assert {row["name"] for row in result["checks"] if not row["success"]} == {
        "foundation_pose",
        "foundation_stereo",
        "sam3_reference",
    }
    assert all(method == "GET" for method, path in requests)


@pytest.mark.parametrize("endpoint", ["/pose/health", "/depth/health"])
def test_required_model_failure_prevents_paid_identification(services, monkeypatch, tmp_path, endpoint):
    from tiptop.perception import gemini, sam2
    from tiptop.scripts import run_known_pick_place as runner

    url, _, requests, failures = services
    failures.add(endpoint)
    args = options(url)
    args.perception_mode, args.output = "gemini_er", tmp_path
    monkeypatch.setattr(sam2, "sam2_client", lambda: None)
    monkeypatch.setattr(gemini, "detect_and_translate_async", lambda *a, **kw: pytest.fail("No paid call expected"))
    with pytest.raises(Exception, match="503"):
        asyncio.run(runner.run(args))
    assert all(method == "GET" for method, path in requests)
    assert not (tmp_path / "gemini-call-consumed.json").exists()


def test_sensor_depth_does_not_require_stereo_service(services, monkeypatch):
    url, packet, requests, failures = services
    packet.pop("stereo_right_rgb")
    packet["depth"] = np.ones((4, 6), np.float32)
    failures.add("/depth/health")
    monkeypatch.setattr("tiptop.doosan.readiness.registered_object", lambda name: {})
    assert asyncio.run(check_readiness(options(url)))["success"]
    assert ("GET", "/depth/health") not in requests


def test_real_camera_without_measured_profile_fails(services, monkeypatch):
    url, packet, _, _ = services
    packet["camera_model"] = "orbbec/serial"
    monkeypatch.setattr("tiptop.doosan.readiness.registered_object", lambda name: {})
    result = asyncio.run(check_readiness(options(url)))
    assert not result["success"]
    failure = next(row for row in result["checks"] if row["name"] == "camera_observation")
    assert "hardware-profile" in failure["error"]


def test_physical_check_validates_measured_camera_and_gateway_without_commands(services, monkeypatch, tmp_path):
    url, packet, requests, _ = services
    packet.update(camera_model="orbbec/serial", camera_serial="serial", world_frame="base_link", robot_system=0)
    monkeypatch.setattr("tiptop.doosan.readiness.registered_object", lambda name: {})
    monkeypatch.setattr(
        "tiptop.doosan.hardware.load_hardware_profile",
        lambda path: dict(_camera_calibration=dict(camera_serial="serial")),
    )
    args = options(url)
    args.hardware_profile, args.gateway_url = tmp_path / "measured.json", url + "/gateway"
    result = asyncio.run(check_readiness(args))
    assert result["success"]
    assert ("GET", "/gateway/state") in requests
    assert all(method == "GET" for method, path in requests)


def gateway_state():
    return dict(
        success=True,
        robot_system=0,
        motion_status=0,
        joint_positions=[0] * 6,
        read_only=False,
        gripper=dict(backend="uon_native", feedback_fresh=True),
        connections=dict(
            trajectory_action_ready=True,
            stop_service_ready=True,
            gripper_command_subscribers=1,
            gripper_command_publishers=1,
            gripper_state_publishers=1,
        ),
    )


@pytest.mark.parametrize(
    "update",
    [
        dict(robot_system=1),
        dict(motion_status=1),
        dict(success=False),
        dict(joint_positions=[0] * 5),
        dict(read_only=True),
        dict(gripper=dict(backend="model", feedback_fresh=True)),
        dict(gripper=dict(backend="uon_native", feedback_fresh=False)),
        dict(execution=dict(status="failed")),
        dict(execution=dict(status="cancelled")),
        dict(connections={}),
    ],
)
def test_execution_readiness_rejects_disconnected_or_wrong_backend(update):
    with pytest.raises(ValueError):
        check_gateway_state(gateway_state() | update, execute=True)


def test_read_only_gateway_is_valid_for_planning_but_not_execution():
    state = gateway_state() | dict(read_only=True, gripper=dict(backend="unavailable_in_state_only_mode"))
    assert check_gateway_state(state, execute=False)["idle"]
    with pytest.raises(ValueError, match="changed"):
        check_gateway_state(gateway_state(), execute=True, observed_q=np.ones(6))


@pytest.mark.parametrize(
    "key,value",
    [
        ("trajectory_action_ready", False),
        ("stop_service_ready", False),
        ("gripper_command_subscribers", 0),
        ("gripper_command_publishers", 2),
        ("gripper_state_publishers", 0),
    ],
)
def test_missing_or_duplicate_driver_interfaces_are_not_execution_ready(key, value):
    state = gateway_state()
    state["connections"][key] = value
    with pytest.raises(ValueError, match="driver"):
        check_gateway_state(state, execute=True)


def test_cli_check_with_execute_never_dispatches_task(monkeypatch, tmp_path, capsys):
    import sys

    from tiptop.scripts import run_known_pick_place as runner

    async def check(args):
        assert args.execute and args.check
        return dict(success=True, check_only=True, motion_commands=0)

    monkeypatch.setattr("tiptop.doosan.readiness.check_readiness", check)
    monkeypatch.setattr(runner, "run_task", lambda *a: pytest.fail("Check must not run a task"))
    monkeypatch.setattr(sys, "argv", ["check", "--perception-mode", "sam3_reference", "--check", "--execute"])
    monkeypatch.chdir(tmp_path)
    runner.main()
    assert json.loads(capsys.readouterr().out)["check_only"]
    assert not (tmp_path / "tiptop_outputs").exists()
