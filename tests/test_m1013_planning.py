import numpy as np
import pytest
from tiptop.doosan.planning import prepare_joint_path


def test_terminal_hold_preserves_original_samples():
    positions = np.array([[0] * 6, [.01] * 6])
    velocities = np.array([[0] * 6, [8.8e-5] * 6])
    q, v, durations = prepare_joint_path(positions, velocities, .02)
    np.testing.assert_array_equal(q[:2], positions)
    np.testing.assert_array_equal(v[:2], velocities)
    np.testing.assert_array_equal(q[2:], np.repeat(positions[-1:], 5, axis=0))
    assert np.count_nonzero(v[2:]) == 0
    assert durations == [.02] * 7


def test_moving_endpoint_cannot_be_relabelled_as_stopped():
    with pytest.raises(ValueError, match='too large'):
        prepare_joint_path([[0] * 6], [[.1] * 6], .02)


def test_showcase_reopens_before_each_arm_move_and_returns_to_start(monkeypatch):
    from tiptop.doosan import planning
    from tiptop.execute_plan import execute_cutamp_plan

    initial = np.array([.1, -.3, .5, .1, .2, -.1])
    expected = initial + np.radians([20, -20, 20, 15, -15, 25])

    def plan_moves(start, targets):
        np.testing.assert_array_equal(start, initial)
        np.testing.assert_allclose(targets, [expected, initial])
        return [{'type': 'trajectory', 'positions': [a.tolist(), b.tolist()],
                 'velocities': [[0.] * 6] * 2, 'dt': 10.}
                for a, b in [(start, targets[0]), (targets[0], targets[1])]]

    monkeypatch.setattr(planning, '_plan_joint_moves', plan_moves)
    plan = planning.plan_showcase(initial)
    calls = []

    class Client:
        opened = False
        joints = initial.copy()

        def open_gripper(self, **kwargs):
            self.opened = True
            calls.append('open')
            return {'success': True}

        def close_gripper(self, **kwargs):
            self.opened = False
            calls.append('close')
            return {'success': True}

        def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
            assert self.opened, 'Arm geometry is planned with the gripper open'
            np.testing.assert_allclose(joint_confs[0], self.joints)
            self.joints = joint_confs[-1]
            calls.append('arm')
            return {'success': True}

    client = Client()
    execute_cutamp_plan(plan['steps'], client)
    assert calls == ['close', 'open', 'arm', 'close', 'open', 'arm']
    np.testing.assert_allclose(client.joints, initial)
    np.testing.assert_array_equal(initial, plan['q_init'])


def test_plan_cli_defaults_to_showcase_without_sending_motion(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from tiptop.doosan import __main__ as cli, planning

    class ReadOnlyClient:
        def __init__(self, url):
            pass

        def get_joint_positions(self):
            return [0.] * 6

    def plan_moves(start, targets):
        return [{'type': 'trajectory', 'positions': [a.tolist(), b.tolist()],
                 'velocities': [[0.] * 6] * 2, 'dt': 10.}
                for a, b in [(start, targets[0]), (targets[0], targets[1])]]

    monkeypatch.setattr(cli, 'DoosanUONClient', ReadOnlyClient)
    monkeypatch.setattr(planning, '_plan_joint_moves', plan_moves)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['tiptop.doosan', 'plan'])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / 'tiptop_outputs/m1013/showcase.json').read_text())
    assert result['success'] and not result['sent_to_robot']
    assert result['steps'] == len(saved['steps']) == 6
    assert result['returns_to_start']
    assert saved['steps'][0]['action'] == 'close'
    np.testing.assert_array_equal(saved['steps'][-1]['positions'][-1], saved['q_init'])
