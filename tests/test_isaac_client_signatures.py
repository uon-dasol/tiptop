"""The Isaac gripper clients stand in for DoosanUONClient and must accept every keyword the executor sends."""

import ast
import inspect
from pathlib import Path

import pytest

from tiptop.doosan.client import DoosanUONClient

ISAAC = Path(__file__).resolve().parents[1] / "simulation/isaac_sim"


def method_parameters(path, name):
    tree = ast.parse(path.read_text())
    return [{arg.arg for arg in node.args.args} for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]


@pytest.mark.parametrize("script", ["plan_client.py", "live_perception.py", "pick_pepper.py"])
@pytest.mark.parametrize("method", ["close_gripper", "open_gripper"])
def test_isaac_gripper_methods_accept_the_real_clients_keywords(script, method):
    real = set(inspect.signature(getattr(DoosanUONClient, method)).parameters) - {"self"}
    found = method_parameters(ISAAC / script, method)
    assert all(real <= parameters - {"self"} for parameters in found), f"{script}: {method} lacks {real}"
