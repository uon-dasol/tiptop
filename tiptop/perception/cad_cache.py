"""Per-process cache of registered CAD geometry, keyed on file identity."""

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

_load_lock = threading.Lock()


@lru_cache(maxsize=8)
def _scaled_cad(path, mtime_ns, size, scale_to_m):
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices) * scale_to_m, np.asarray(mesh.faces)


def ensure_cad_cache_capacity(objects):
    """Grow before sequence preparation; each caller still receives its own mesh."""
    global _scaled_cad
    if not isinstance(objects, int) or isinstance(objects, bool) or objects <= 0:
        raise ValueError("CAD cache capacity must be a positive integer")
    with _load_lock:
        capacity = _scaled_cad.cache_parameters()["maxsize"]
        if objects > capacity:
            _scaled_cad = lru_cache(maxsize=objects)(_scaled_cad.__wrapped__)
        return _scaled_cad.cache_parameters()["maxsize"]


def load_cad_mesh(path, scale_to_m):
    """Metre-scale CAD mesh; the file is read once per process and each caller gets its own copy."""
    stat = Path(path).stat()
    # The lock lets a background prefetch and the main task share one read instead of racing.
    with _load_lock:
        vertices, faces = _scaled_cad(str(path), stat.st_mtime_ns, stat.st_size, float(scale_to_m))
    return trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
