"""The minimal adapter must preserve validation and leave upstream globals intact."""
from dataclasses import replace

import numpy as np
import pytest

from tiptop.doosan.tamp import RecordingClient, m1013_entrypoint
from tiptop.execute_plan import execute_cutamp_plan


def test_entrypoint_keeps_robot_identity_and_restores_after_failure():
    import cutamp.algorithm as algorithm
    from cutamp.config import TAMPConfiguration

    validate, load = algorithm.validate_tamp_config, algorithm.load_robot_container
    config = TAMPConfiguration(robot="m1013_uon", enable_visualizer=False)
    container = object()
    with pytest.raises(ValueError, match="Invalid embodiment"):
        validate(config)
    with pytest.raises(RuntimeError, match="abort"):
        with m1013_entrypoint(container):
            algorithm.validate_tamp_config(config)
            assert config.robot == "m1013_uon"
            assert algorithm.load_robot_container(config.robot, None) is container
            with pytest.raises(ValueError, match="num_particles"):
                algorithm.validate_tamp_config(replace(config, num_particles=0))
            with pytest.raises(ValueError, match="Invalid embodiment"):
                algorithm.validate_tamp_config(replace(config, robot="unknown"))
            with pytest.raises(ValueError, match="enable_visualizer"):
                algorithm.validate_tamp_config(replace(config, enable_visualizer=True))
            raise RuntimeError("abort")
    assert algorithm.validate_tamp_config is validate
    assert algorithm.load_robot_container is load


def test_existing_executor_passes_six_axes_and_both_gripper_commands():
    q = np.zeros((3, 6))
    q[:, 0] = [0., .01, .02]
    client = RecordingClient(np.zeros(6))
    execute_cutamp_plan([
        {"type": "trajectory", "label": "pick", "positions": q,
         "velocities": np.zeros_like(q), "dt": .02},
        {"type": "gripper", "label": "pick", "action": "close"},
        {"type": "trajectory", "label": "place", "positions": q[::-1],
         "velocities": np.zeros_like(q), "dt": .02},
        {"type": "gripper", "label": "place", "action": "open"},
    ], client)
    assert client.calls == [{"type": "trajectory", "waypoints": 3, "dof": 6},
                            {"type": "gripper", "action": "close"},
                            {"type": "trajectory", "waypoints": 3, "dof": 6},
                            {"type": "gripper", "action": "open"}]
    np.testing.assert_array_equal(client.q, np.zeros(6))


@pytest.mark.parametrize("positions,velocities,durations", [
    ([[0.] * 7], [[0.] * 7], [.02]),
    ([[0.] * 6], [[0.] * 6], [0.]),
    ([[0.] * 6], [[float("nan")] * 6], [.02]),
    ([[.1] * 6], [[0.] * 6], [.02]),
])
def test_executor_probe_rejects_invalid_or_discontinuous_paths(positions, velocities, durations):
    client = RecordingClient(np.zeros(6))
    with pytest.raises(ValueError):
        client.execute_joint_impedance_path(positions, velocities, durations)
    assert client.calls == []


def test_actual_client_and_gateway_validator_accept_stationary_endpoint():
    from tiptop.doosan.planning import prepare_joint_path
    from tiptop.doosan.tamp import verify_gateway_contract
    from tiptop.execute_plan import ExecutionFailure
    step = {"type": "trajectory", "label": "cuTAMP path", "positions": [[0.] * 6, [.001] * 6],
            "velocities": [[0.] * 6, [.0001] * 6], "dt": .02}
    plan = {"q_init": [0.] * 6, "steps": [step]}
    with pytest.raises(ExecutionFailure, match="zero velocity"):
        verify_gateway_contract(plan)
    original = np.array(step["positions"])
    q, v, _ = prepare_joint_path(step["positions"], step["velocities"], step["dt"])
    step.update(positions=q, velocities=v)
    calls = verify_gateway_contract(plan)
    np.testing.assert_array_equal(q, original)
    assert calls == [{"type": "trajectory", "waypoints": len(q), "dof": 6}]


def test_motion_world_conversion_preserves_movable_and_static_scene_membership():
    from types import SimpleNamespace
    from curobo.geom.types import Cuboid
    from tiptop.doosan.tamp import motion_world_cfg

    pepper = Cuboid(name="pepper", dims=[.05] * 3, pose=[.55, 0, .5, 1, 0, 0, 0])
    table = Cuboid(name="table", dims=[.8, .8, .04], pose=[.65, 0, .38, 1, 0, 0, 0])
    env = SimpleNamespace(movables=[pepper], statics=[table])
    for _ in range(2):
        world = motion_world_cfg(env)
        assert [obj.name for obj in world.cuboid] == ["pepper", "table"]
        assert env.movables == [pepper]
        assert env.statics == [table]
