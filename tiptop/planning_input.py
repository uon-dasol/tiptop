"""Save a detached pre-planning snapshot without blocking the planner on disk IO."""

import asyncio
import logging
import pickle
from copy import deepcopy
from dataclasses import replace
from pathlib import Path


def snapshot_planning_input(env, scene, observation):
    # Open3D point clouds are deliberately omitted, as in the original export.
    # Detach the complete object graph before the planner can mutate meshes,
    # arrays, tensors or nested perception dictionaries. One deepcopy preserves
    # aliases within the snapshot without sharing mutable data with planning.
    cached_scene = replace(
        scene,
        object_pcds={},
        perception_results={
            key: value for key, value in scene.perception_results.items() if key != "pcd_downsampled"
        },
    )
    return deepcopy((env, cached_scene, observation))


def write_planning_input(path, snapshot):
    """Publish only a complete pickle, keeping the existing tuple/file format."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(snapshot, stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_planning_input_in_background(env, scene, observation, path, *, background_tasks, timing):
    """Copy synchronously, then queue only serialization and writing in a thread."""
    with timing.measure("planning_input_snapshot"):
        snapshot = snapshot_planning_input(env, scene, observation)

    def save():
        with timing.measure("planning_input_write"):
            write_planning_input(path, snapshot)
        logging.getLogger(__name__).info("Saved planning input to %s", path)

    task = asyncio.create_task(asyncio.to_thread(save))
    background_tasks.append(task)
    return task
