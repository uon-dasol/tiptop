"""Repeated tasks, result ownership and failure recovery without model/device calls."""

import asyncio
import json
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest

from simulation.isaac_sim.observation_server import ObservationServer
from simulation.isaac_sim.task_runtime import ExecutionStore, evaluate_pick_place
from tiptop.doosan.known_task import approach_tilt_mask
from tiptop.scripts import run_known_pick_place as runner


def test_approach_bound_defaults_to_three_degrees_and_never_expands():
    from scipy.spatial.transform import Rotation

    angles = [0, 3, 3.01, 10, 15, 16]
    poses = np.repeat(np.eye(4)[None], len(angles), axis=0)
    poses[:, :3, :3] = Rotation.from_euler("y", np.asarray(angles)[:, None], degrees=True).as_matrix() @ np.diag(
        [1, -1, -1]
    )
    assert approach_tilt_mask(poses).tolist() == [True, True, False, False, False, False]
    assert approach_tilt_mask(poses, 15).tolist() == [True, True, True, True, True, False]
    poses[0, :3, :3] = Rotation.from_euler("z", 127, degrees=True).as_matrix() @ np.diag([1, -1, -1])
    assert approach_tilt_mask(poses)[0]  # Yaw does not incline the approach axis.
    for bad in [-1, 91, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="degrees"):
            approach_tilt_mask(poses, bad)


def evaluation(**overrides):
    values = dict(
        baseline=[0, 0, 0.5],
        final=[0.2, 0, 0.42],
        trace=[dict(position=[0, 0, 0.56], orientation_wxyz=[0.7, 0.7, 0, 0])],
        forces=[0, 0, 0],
        support=0.2,
        region=dict(center_xy_m=[0.2, 0], size_xy_m=[0.1, 0.1]),
        max_other_displacement=0,
    )
    values.update(overrides)
    return evaluate_pick_place(**values)


def test_lying_placement_passes_despite_large_origin_height_change():
    result = evaluation()
    assert result["success"] and result["physical_placement_verified"]
    assert result["final_origin_height_change_m"] == pytest.approx(-0.08)
    assert result["orientation_required"] is False


@pytest.mark.parametrize(
    "change",
    [
        dict(unintended_drop=True),
        dict(forces=[0.2, 0.2, 0]),
        dict(support=0),
        dict(final=[0.4, 0, 0.42]),
        dict(max_other_displacement=0.02),
    ],
)
def test_actual_delivery_failures_are_not_hidden_by_orientation_policy(change):
    assert evaluation(**change)["success"] is False


def test_two_executions_keep_results_separate_and_reject_replay(tmp_path):
    store = ExecutionStore(tmp_path)
    store.accept(dict(observation_id="frame-1"))
    with pytest.raises(ValueError, match="active"):
        store.accept(dict(observation_id="frame-2"))
    store.finish(dict(observation_id="frame-1", finished=True, success=False))
    store.accept(dict(observation_id="frame-2"))
    assert store.result()["observation_id"] == "frame-2" and not store.result()["finished"]
    assert store.result("frame-1")["success"] is False
    store.finish(dict(observation_id="frame-2", finished=True, success=True))
    assert store.result("frame-2")["success"] is True
    assert json.loads((tmp_path / "executions/frame-1/execution.json").read_text())["success"] is False
    with pytest.raises(FileExistsError):
        store.accept(dict(observation_id="frame-1"))
    with pytest.raises(FileExistsError):
        ExecutionStore(tmp_path).accept(dict(observation_id="frame-2"))
    with pytest.raises(ValueError, match="Invalid"):
        store.result("../escape")


def test_result_polling_is_task_specific_without_waiting_for_kit(tmp_path):
    store = ExecutionStore(tmp_path)
    store.accept(dict(observation_id="old"))
    store.finish(dict(observation_id="old", finished=True, success=False))
    store.accept(dict(observation_id="current"))
    server = ObservationServer(port=0, executions=store)
    try:
        for identity, finished in [("current", False), ("old", True)]:
            with urlopen(
                f"http://127.0.0.1:{server.server.server_port}/result?observation_id={identity}", timeout=2
            ) as response:
                result = json.load(response)
            assert result["observation_id"] == identity and result["finished"] is finished
    finally:
        server.close()


def test_interactive_failure_then_next_instruction_uses_new_task_and_no_roi(tmp_path, monkeypatch):
    calls = []

    async def task(args):
        assert args.active_task is None
        args._observation_id = f"new-frame-{len(calls)}"
        calls.append((args.task, args.output))
        args._stage = "perception"
        if len(calls) == 1:
            raise ValueError("Target not recognized in current image")
        return dict(status="completed", success=True, executed=True)

    monkeypatch.setattr(runner, "run", task)
    instructions = iter(["A 집고 빈 곳에 놓아", "A 집고 빈 곳에 놓아", "/quit"])
    args = SimpleNamespace(output=tmp_path, active_task=None)
    asyncio.run(runner.run_interactive(args, read_instruction=lambda _: next(instructions)))
    assert len(calls) == 2 and calls[0][1] != calls[1][1]
    results = [json.loads((path / "task-result.json").read_text()) for _, path in calls]
    assert results[0]["status"] == "failed" and results[0]["failed_stage"] == "perception"
    assert results[1]["success"] is True
    assert results[0]["observation_id"] != results[1]["observation_id"]


def test_polling_rejects_old_task_result_even_if_finished():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self, **kwargs):
            return self.payload

    class Session:
        def post(self, *args, **kwargs):
            return Response(dict(accepted=True, observation_id="current"))

        def get(self, url):
            assert "observation_id=current" in url
            return Response(dict(observation_id="old", finished=True, success=True))

    with pytest.raises(ValueError, match="another task"):
        asyncio.run(
            runner.execute_live_plan(Session(), "http://local", dict(observation_id="current"), poll_interval=0)
        )


def test_real_http_frontend_acceptance_and_matching_result(tmp_path):
    import aiohttp

    store = ExecutionStore(tmp_path)
    server = ObservationServer(port=0, executions=store)

    def execute(path, body):
        assert path == "/execute"
        plan = json.loads(body)
        store.accept(plan)
        store.finish(dict(observation_id=plan["observation_id"], finished=True, success=True))
        return json.dumps(dict(accepted=True, observation_id=plan["observation_id"])).encode()

    async def session():
        async with aiohttp.ClientSession() as http:
            for identity in ["first", "second"]:
                task = asyncio.create_task(
                    runner.execute_live_plan(
                        http,
                        f"http://127.0.0.1:{server.server.server_port}",
                        dict(observation_id=identity),
                        poll_interval=0,
                    )
                )
                while not task.done():
                    server.poll(execute)
                    await asyncio.sleep(0.001)
                result = await task
                assert result["success"] and result["observation_id"] == identity

    try:
        asyncio.run(session())
    finally:
        server.close()


def test_cli_passes_requested_planning_limits_before_running_task(monkeypatch):
    import sys

    captured = []

    async def run_task(args):
        captured.append(args.planning_settings)
        return dict(success=True)

    monkeypatch.setattr(runner, "run_task", run_task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_known_pick_place",
            "--perception-mode",
            "sam3_reference",
            "--motion-timeout-s",
            "2",
            "--placement-support-tolerance-m",
            "0.002",
            "--placement-xy-tolerance-m",
            "0.007",
            "--motion-position-tolerance-m",
            "0.005",
            "--motion-rotation-tolerance",
            "0.05",
            "--release-clearance-m",
            "0.015",
        ],
    )
    runner.main()
    settings = captured[0]
    assert settings.motion_timeout_s == 2
    assert settings.placement_support_tolerance_m == 0.002
    assert settings.placement_xy_tolerance_m == 0.007
    assert settings.motion_position_tolerance_m == 0.005
    assert settings.motion_rotation_tolerance == 0.05
    assert settings.release_clearance_m == 0.015
    # Invalid height budgets fail before entering perception or planning.
    monkeypatch.setattr(sys, "argv", ["run_known_pick_place", "--release-clearance-m", "0.003"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2 and len(captured) == 1
