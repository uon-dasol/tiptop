"""Bounded, serialized planner reuse; no cached scenes, solutions or trajectories."""

import hashlib
import json
import threading
import time
from functools import wraps
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

_LOCK = threading.RLock()
_ENTRIES = {}


def serialized_planning(function):
    @wraps(function)
    def run(*args, **kwargs):
        with _LOCK:
            try:
                return function(*args, **kwargs)
            except BaseException:
                _ENTRIES.clear()
                raise
    return run


def configuration_key(*values):
    """Include generated model contents, so output path changes do not force a miss."""
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, np.ndarray):
            return dict(dtype=str(value.dtype), shape=value.shape, values=value.tolist())
        if isinstance(value, str) and value.endswith(('.urdf', '.yml', '.yaml', '.npy')):
            path = Path(value)
            if path.is_file():
                contents = path.read_bytes()
                resources = []
                if path.suffix == '.urdf':
                    robot = ET.fromstring(contents)
                    for mesh in robot.findall('.//mesh'):
                        resource = (path.parent / mesh.get('filename')).resolve()
                        mesh.set('filename', str(resource))
                        if resource.is_file():
                            stat = resource.stat()
                            resources.append((str(resource), stat.st_mtime_ns, stat.st_size))
                    contents = ET.tostring(robot)
                    parent = None
                else:
                    parent = str(path.parent.resolve())
                return dict(contents=hashlib.sha256(contents).hexdigest(), parent=parent, resources=resources)
        if isinstance(value, np.generic):
            return value.item()
        return value
    return json.dumps(normalize(values), sort_keys=True, allow_nan=False)


def cached_planner(kind, key, build, reset, *, enabled=True):
    """Call only under serialized_planning; retain at most one object of each kind."""
    started = time.perf_counter()
    entry = _ENTRIES.get(kind) if enabled else None
    reused = entry is not None and entry[0] == key
    reset_error = None
    if reused:
        value = entry[1]
        try:
            reset(value)
        except Exception as error:
            reset_error = str(error)
            _ENTRIES.pop(kind, None)
            reused = False
    if not reused:
        _ENTRIES.pop(kind, None)
        # Release the old GPU object before allocating its replacement.
        entry = value = None
        value = build()
        if enabled:
            _ENTRIES[kind] = (key, value)
    return value, dict(reused=reused, seconds=time.perf_counter()-started, reset_error=reset_error)


def reset_ik(solver, world):
    solver.world_coll_checker.clear_cache()
    solver.update_world(world)
    solver.reset_shape()
    solver.reset_seed()


def reset_motion_gen(solver, world):
    solver.detach_object_from_robot('attached_object')
    solver.clear_world_cache()
    solver.update_world(world)
    solver.reset(reset_seed=True)


def unique_path_goals_first(particles, grasp_keys):
    """Keep every ranked trial, trying each exact grasp/home pair before repeats."""
    import torch

    n = len(particles['q0'])
    if n < 2 or not grasp_keys:
        return particles, dict(total=n, unique_goals=n, deferred_duplicates=0)
    keys = ['q0', *grasp_keys]
    packed = torch.cat([particles[key].reshape(n, -1) for key in keys], dim=1).detach().cpu().numpy()
    seen, first, repeats = set(), [], []
    for i, row in enumerate(packed):
        signature = row.tobytes()
        if signature in seen:
            repeats.append(i)
        else:
            seen.add(signature)
            first.append(i)
    order = first + repeats
    output = {key: value[order] if value is not None else None for key, value in particles.items()}
    return output, dict(total=n, unique_goals=len(first), deferred_duplicates=len(repeats))
