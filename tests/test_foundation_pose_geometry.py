"""Exercise the shared diameter function without loading CUDA or the pose server."""

import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def compute_mesh_diameter():
    source = Path(__file__).resolve().parents[1] / "FoundationPose/Utils.py"
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "compute_mesh_diameter")
    namespace = dict(np=np)
    # Use the actual function without importing Utils' unrelated GPU dependencies.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[fn.name]


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("count", [1, 2, 37, 300])
def test_pairwise_distance_preserves_original_arithmetic(compute_mesh_diameter, dtype, count):
    points = (np.random.default_rng(count).normal(size=(count, 3))*100).astype(dtype)
    expected = np.linalg.norm(points[None]-points[:, None], axis=-1).max()
    result = compute_mesh_diameter(model_pts=points, n_sample=None)
    assert result == expected
    assert result.dtype == expected.dtype


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.array([[np.inf, 0, 0]]),
                                  np.array([[np.nan, 0, 0], [1, 2, 3]])])
def test_empty_and_nonfinite_inputs_keep_original_behavior(compute_mesh_diameter, points):
    with np.errstate(invalid="ignore"):
        if not len(points):
            with pytest.raises(ValueError):
                compute_mesh_diameter(model_pts=points, n_sample=None)
        else:
            assert np.isnan(compute_mesh_diameter(model_pts=points, n_sample=None))


@pytest.mark.parametrize("seed", [0, 3, 41])
def test_actual_utils_function_keeps_sample_and_rng_state(compute_mesh_diameter, seed):
    points = np.random.default_rng(19).normal(size=(500, 3))
    np.random.seed(seed)
    sampled = points[np.random.choice(500, 300, replace=False)]
    expected = np.linalg.norm(sampled[None]-sampled[:, None], axis=-1).max()
    next_random = np.random.random()
    np.random.seed(seed)
    assert compute_mesh_diameter(model_pts=points, n_sample=300) == expected
    assert np.random.random() == next_random
