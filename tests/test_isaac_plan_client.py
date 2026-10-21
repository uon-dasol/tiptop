"""Isaac client timing/cancellation contracts without launching Isaac Sim."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tiptop.execute_plan import ExecutionFailure, execute_cutamp_plan

_path = Path(__file__).resolve().parents[1] / 'simulation/isaac_sim/plan_client.py'
_spec = importlib.util.spec_from_file_location('isaac_plan_client', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Array:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return np.asarray(self.data)[None]


class Drive:
    def __init__(self):
        self.q, self.v = np.zeros(7), np.zeros(7)
        self.target, self.velocity_target = self.q.copy(), self.v.copy()

    def set_dof_position_targets(self, values, *, dof_indices):
        self.target[dof_indices] = values

    def set_dof_velocity_targets(self, values, *, dof_indices):
        self.velocity_target[dof_indices] = values

    def get_dof_positions(self, dof_indices=None):
        return Array(self.q if dof_indices is None else self.q[dof_indices])

    def get_dof_velocities(self, dof_indices=None):
        return Array(self.v if dof_indices is None else self.v[dof_indices])


def client_fixture(stop_tick=None):
    client = _module.IsaacPlanClient.__new__(_module.IsaacPlanClient)
    client.robot, client.dt = Drive(), 0.01
    client.arm, client.spinner, client.controlled = list(range(6)), 6, list(range(7))
    client.cancelled, client.calls = False, []
    samples = []

    def tick():
        if client.cancelled or len(samples) == stop_tick:
            client.stop()
            raise RuntimeError('Execution stopped')
        client.robot.q = client.robot.target.copy()
        client.robot.v = client.robot.velocity_target.copy()
        samples.append((client.robot.q.copy(), client.robot.v.copy()))

    client._tick = tick
    return client, samples


def test_nonuniform_durations_preserve_cubic_positions_velocities_and_first_hold():
    client, samples = client_fixture()
    # A single analytical cubic sampled at nonuniform times, with zero endpoint velocities.
    times = np.array([0.02, 0.05, 0.10])
    u = (times - 0.02) / 0.08
    q = np.repeat((0.03 * (3 * u**2 - 2 * u**3))[:, None], 6, axis=1)
    v = np.repeat((0.03 / 0.08 * (6 * u - 6 * u**2))[:, None], 6, axis=1)
    result = client.execute_joint_impedance_path(q, v, [0.02, 0.03, 0.05])
    assert result['success'], result
    assert result['requested_duration_s'] == pytest.approx(0.1)
    for i, (observed_q, observed_v) in enumerate(samples[:10], 1):
        u = max(0, (i * 0.01 - 0.02) / 0.08)
        np.testing.assert_allclose(observed_q[:6], 0.03 * (3 * u**2 - 2 * u**3), atol=1e-12)
        np.testing.assert_allclose(observed_v[:6], 0.03 / 0.08 * (6 * u - 6 * u**2), atol=1e-12)


def test_cancel_holds_observed_state_and_prevents_following_gripper_action():
    client, samples = client_fixture(stop_tick=5)
    steps = [
        {'type': 'trajectory', 'label': 'move', 'positions': [[0] * 6, [0.02] * 6],
         'velocities': [[0] * 6] * 2, 'dt': 0.1},
        {'type': 'gripper', 'label': 'close', 'action': 'close'},
    ]
    with pytest.raises(ExecutionFailure, match='Execution stopped'):
        execute_cutamp_plan(steps, client)
    assert client.cancelled and len(client.calls) == 1
    np.testing.assert_array_equal(client.robot.target, samples[-1][0])
    np.testing.assert_array_equal(client.robot.velocity_target, np.zeros(7))


def test_discontinuous_start_is_rejected_before_simulation_advances():
    client, samples = client_fixture()
    result = client.execute_joint_impedance_path([[0.1] * 6], [[0] * 6], [0.1])
    assert not result['success'] and 'First waypoint' in result['error']
    assert not samples
