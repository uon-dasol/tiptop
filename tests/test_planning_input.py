"""Offline checks for snapshot ownership, overlapping IO and task cleanup."""

import asyncio
import json
import pickle
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from tiptop import planning_input
from tiptop.scripts import run_known_pick_place as runner


@dataclass
class Scene:
    object_meshes: dict
    object_pcds: dict
    perception_results: dict
    grasps: dict


class UncopyablePointCloud:
    def __deepcopy__(self, memo):
        raise AssertionError("Excluded point clouds must not be copied")


def inputs():
    mesh = np.arange(12, dtype=np.float64).reshape(4, 3)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    env = SimpleNamespace(mesh=mesh, goal={"target": [1, 2]})
    scene = Scene(
        {"target": mesh}, {"target": UncopyablePointCloud()},
        {"rgb": rgb, "placement_region": {"source": "before"}, "pcd_downsampled": UncopyablePointCloud()},
        {"target": {"scores": np.array([0.9, 0.8], dtype=np.float32)}},
    )
    observation = SimpleNamespace(frame=SimpleNamespace(rgb=rgb), q_init=np.arange(6, dtype=float))
    return env, scene, observation


def test_snapshot_preserves_values_and_internal_aliases_but_detaches_planner_data(tmp_path):
    import torch

    env, scene, observation = inputs()
    scene.grasps["target"]["tensor"] = torch.tensor([1.0, 2.0])
    snapshot = planning_input.snapshot_planning_input(env, scene, observation)
    env.mesh[:] = -1
    env.goal["target"].append(3)
    scene.perception_results["placement_region"]["source"] = "after"
    scene.grasps["target"]["tensor"].add_(10)
    observation.frame.rgb[:] = 255
    observation.q_init[:] = 99
    path = tmp_path / "planning-input.pkl"
    planning_input.write_planning_input(path, snapshot)
    with path.open("rb") as stream:
        saved_env, saved_scene, saved_observation = pickle.load(stream)
    np.testing.assert_array_equal(saved_env.mesh, np.arange(12).reshape(4, 3))
    assert saved_env.mesh.dtype == np.float64
    assert saved_env.mesh is saved_scene.object_meshes["target"]
    assert saved_scene.perception_results["rgb"] is saved_observation.frame.rgb
    assert saved_env.goal == {"target": [1, 2]}
    assert saved_scene.object_pcds == {} and "pcd_downsampled" not in saved_scene.perception_results
    assert saved_scene.perception_results["placement_region"]["source"] == "before"
    torch.testing.assert_close(saved_scene.grasps["target"]["tensor"], torch.tensor([1.0, 2.0]))
    np.testing.assert_array_equal(saved_observation.q_init, np.arange(6))
    assert not saved_observation.frame.rgb.any()


def test_failed_pickle_never_publishes_partial_file_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "planning-input.pkl"
    path.write_bytes(b"existing complete file")

    def fail(snapshot, stream):
        stream.write(b"partial pickle")
        raise OSError("test disk failure")

    monkeypatch.setattr(planning_input.pickle, "dump", fail)
    with pytest.raises(OSError, match="disk failure"):
        planning_input.write_planning_input(path, ())
    assert path.read_bytes() == b"existing complete file"
    assert not path.with_suffix(".pkl.tmp").exists()


@pytest.mark.parametrize("plan_fails", [False, True])
@pytest.mark.parametrize("save_fails", [False, True])
def test_planning_proceeds_while_save_is_blocked_and_result_waits_for_save(
    tmp_path, monkeypatch, plan_fails, save_fails,
):
    started, release = threading.Event(), threading.Event()
    writer = planning_input.write_planning_input

    def blocked_save(path, snapshot):
        started.set()
        assert release.wait(5)
        if save_fails:
            raise OSError("planning input disk failure")
        writer(path, snapshot)

    monkeypatch.setattr(planning_input, "write_planning_input", blocked_save)

    async def task(args):
        env, scene, observation = inputs()
        args._planning_input_save = planning_input.save_planning_input_in_background(
            env, scene, observation, args.output / "planning-input.pkl",
            background_tasks=args._background, timing=args._timing_trace,
        )
        args._stage = "planning"
        args._timing_trace.phase("planning")
        assert await asyncio.to_thread(started.wait, 5)
        env.mesh[:] = -1  # Planning may mutate inputs while saving its private snapshot.
        if plan_fails:
            raise ValueError("test planning failure")
        return dict(status="planned", success=None, executed=False)

    monkeypatch.setattr(runner, "run", task)

    async def exercise():
        args = SimpleNamespace(output=tmp_path, task="test", perception_mode="sam3_reference")
        pending = asyncio.create_task(runner.run_task(args))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            # Let planning finish and run_task reach its background drain.
            for _ in range(100):
                if args._timing_trace._name == "background_save_wait":
                    break
                await asyncio.sleep(.001)
            assert args._timing_trace._name == "background_save_wait"
            assert not pending.done()
            assert not (tmp_path / "task-result.json").exists()
        finally:
            release.set()
        return await pending

    result = asyncio.run(exercise())
    assert json.loads((tmp_path / "task-result.json").read_text()) == result
    assert result["status"] == ("failed" if plan_fails else "planned")
    assert result.get("planning_input_save_errors") == (["planning input disk failure"] if save_fails else None)
    assert "perception_save_errors" not in result
    if not save_fails:
        with (tmp_path / "planning-input.pkl").open("rb") as stream:
            env, _, _ = pickle.load(stream)
        np.testing.assert_array_equal(env.mesh, np.arange(12).reshape(4, 3))
    details = {entry["name"]: entry for entry in json.loads((tmp_path / "timings.json").read_text())["details"]}
    assert details["planning_input_write"]["status"] == ("failed" if save_fails else "completed")
    assert details["planning_input_snapshot"]["status"] == "completed"


@pytest.mark.parametrize("cancel_during_planning", [False, True])
def test_cancellation_drains_disk_thread_before_task_exits(tmp_path, monkeypatch, cancel_during_planning):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    writer = planning_input.write_planning_input

    def blocked_save(path, snapshot):
        started.set()
        assert release.wait(5)
        writer(path, snapshot)
        finished.set()

    monkeypatch.setattr(planning_input, "write_planning_input", blocked_save)

    async def task(args):
        args._planning_input_save = planning_input.save_planning_input_in_background(
            *inputs(), args.output / "planning-input.pkl", background_tasks=args._background,
            timing=args._timing_trace,
        )
        if cancel_during_planning:
            await asyncio.Event().wait()
        return dict(status="planned", success=None, executed=False)

    monkeypatch.setattr(runner, "run", task)

    async def exercise():
        args = SimpleNamespace(output=tmp_path, task="test", perception_mode="sam3_reference")
        pending = asyncio.create_task(runner.run_task(args))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            await asyncio.sleep(.01)
            assert not pending.done() and not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert finished.is_set()
        assert not args._planning_input_save.cancelled()

    asyncio.run(exercise())
    assert (tmp_path / "planning-input.pkl").exists()
    assert not (tmp_path / "task-result.json").exists()
