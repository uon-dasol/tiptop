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


def test_showcase_fixed_sequence_returns_home_with_closed_gripper(monkeypatch):
    from tiptop.doosan import planning
    from tiptop.execute_plan import execute_cutamp_plan

    initial = np.array([.1, -.3, .5, .1, .2, -.1])
    expected = initial + np.radians([60, -30, 40, 15, -15, 25])

    def plan_moves(start, targets, gripper_positions, world=None):
        if gripper_positions == [0.]:
            np.testing.assert_array_equal(start, initial)
            np.testing.assert_allclose(targets, [expected])
        else:
            np.testing.assert_allclose(start, expected)
            np.testing.assert_allclose(targets, [initial])
            assert gripper_positions == [1.5009832]
        return [{'type': 'trajectory', 'positions': [a.tolist(), b.tolist()],
                 'velocities': [[0.] * 6] * 2, 'dt': 10.}
                for a, b in [(start, targets[0])]]

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
            assert self.opened == (calls == ['open']), 'Outward is open; home return is closed'
            np.testing.assert_allclose(joint_confs[0], self.joints)
            self.joints = joint_confs[-1]
            calls.append('arm')
            return {'success': True}

    client = Client()
    execute_cutamp_plan(plan['steps'], client)
    assert calls == ['open', 'arm', 'close', 'arm', 'open']
    np.testing.assert_array_equal(plan['q_home'], initial)
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

    def plan_moves(start, targets, gripper_positions, world=None):
        return [{'type': 'trajectory', 'positions': [a.tolist(), b.tolist()],
                 'velocities': [[0.] * 6] * 2, 'dt': 10.}
                for a, b in [(start, targets[0])]]

    monkeypatch.setattr(cli, 'DoosanUONClient', ReadOnlyClient)
    monkeypatch.setattr(planning, '_plan_joint_moves', plan_moves)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['tiptop.doosan', 'plan'])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / 'tiptop_outputs/m1013/showcase.json').read_text())
    assert result['success'] and not result['sent_to_robot']
    assert result['steps'] == len(saved['steps']) == 5
    assert result['returns_to_start']
    assert saved['steps'][0]['action'] == saved['steps'][-1]['action'] == 'open'
    np.testing.assert_array_equal(saved['steps'][-2]['positions'][-1], saved['q_init'])


def test_box_clearance_uses_whole_robot_and_ignores_inactive_attachment_spheres():
    from scripts.compare_m1013_obstacle import sphere_box_clearance
    spheres = np.array([[0., 0., 0., -.1], [.2, 0., 0., .05], [0., .11, 0., .02]])
    # The off-axis sphere penetrates the box by 1 cm; checking only a TCP would miss it.
    assert sphere_box_clearance(spheres, np.zeros(3), [.2] * 3) == pytest.approx(-.01)


def test_comparison_failure_preserves_inputs_partial_paths_and_reason(monkeypatch, tmp_path):
    import json
    import sys
    from scripts import compare_m1013_obstacle as comparison

    def fail(inputs, output, report):
        report['stage'] = 'planning_with_box'
        comparison.save_json(output / 'comparison-paths.json', {'without_box': {'positions': [[0.] * 6]}})
        raise RuntimeError('MotionGenStatus.TRAJOPT_FAIL')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['compare', '--offline'])
    monkeypatch.setattr(comparison, 'compare', fail)
    assert comparison.main() == 1
    output, = (tmp_path / 'tiptop_outputs/m1013/comparisons').iterdir()
    report = json.loads((output / 'report.json').read_text())
    assert report['failure']['stage'] == 'planning_with_box'
    assert report['failure']['reason'] == 'MotionGenStatus.TRAJOPT_FAIL'
    assert not report['sent_to_robot'] and not report['execution_plan_created']
    assert json.loads((output / 'inputs.json').read_text())['q_start_rad'] == [0.] * 6
    assert (output / 'comparison-paths.json').exists()
    assert 'RuntimeError: MotionGenStatus.TRAJOPT_FAIL' in (output / 'run.log').read_text()


def test_comparison_replay_uses_recorded_inputs_without_gateway(monkeypatch, tmp_path):
    import json
    import sys
    from scripts import compare_m1013_obstacle as comparison

    inputs = {'q_start_rad': [.1] * 6, 'target_rad': [.2] * 6, 'seed': 0,
              'box': {'dims': [.08] * 3, 'pose': [0., 0., 1., 1., 0., 0., 0.]}}
    source = tmp_path / 'inputs.json'
    source.write_text(json.dumps(inputs))
    observed = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['compare', '--input', str(source)])
    monkeypatch.setattr(comparison, 'compare', lambda received, output, report: observed.append(received))
    assert comparison.main() == 0
    assert observed == [inputs]


@pytest.mark.parametrize('allow_partial', [False, True])
def test_showcase_return_failure_is_shared_and_prefix_is_explicit(monkeypatch, allow_partial):
    from tiptop.doosan import planning
    calls = []
    world = object()
    def plan_moves(start, targets, world=None, gripper_positions=None):
        calls.append((world, gripper_positions))
        if len(calls) == 2:
            raise RuntimeError('TRAJOPT_FAIL')
        return [{'type': 'trajectory', 'positions': [start.tolist(), targets[0].tolist()],
                 'velocities': [[0.] * 6] * 2, 'dt': .02}]
    monkeypatch.setattr(planning, '_plan_joint_moves', plan_moves)
    if allow_partial:
        partial = planning.plan_showcase(np.zeros(6), world=world, allow_partial=True)
        assert [s.get('action', s['type']) for s in partial['steps']] == ['open', 'trajectory', 'close']
        assert not partial['complete_sequence'] and not partial['returns_to_start']
        assert partial['expected_steps'] == 5
        assert partial['planning_failure']['reason'] == 'TRAJOPT_FAIL'
    else:
        with pytest.raises(RuntimeError, match='TRAJOPT_FAIL'):
            planning.plan_showcase(np.zeros(6), world=world)
    assert calls == [(world, [0.]), (world, [1.5009832])]


@pytest.mark.parametrize('block', [None, 'start_mismatch', 'rviz_missing', 'arm_controller_missing'])
def test_obstacle_run_reuses_executor_and_stops_before_unplanned_return(monkeypatch, tmp_path, block):
    import json
    from types import SimpleNamespace
    from scripts import compare_m1013_obstacle as comparison
    from tiptop.doosan import client as client_module

    calls = []
    class Client:
        def get_joint_positions(self):
            return [1. if block == 'start_mismatch' else 0.] * 6

        def open_gripper(self, **kwargs):
            calls.append('open')
            return {'success': True}

        def close_gripper(self, **kwargs):
            calls.append('close')
            return {'success': True}

        def execute_joint_impedance_path(self, **kwargs):
            calls.append('arm')
            return {'success': True}

    monkeypatch.setattr(client_module, 'DoosanUONClient', Client)
    published = []
    def publish(command, **kwargs):
        if 'ActionClient' in command[-1]:
            return SimpleNamespace(returncode=1 if block == 'arm_controller_missing' else 0,
                                   stdout='', stderr='')
        published.append(json.loads(command[-1]))
        return SimpleNamespace(returncode=1 if block == 'rviz_missing' else 0, stdout='', stderr='')
    monkeypatch.setattr(comparison.subprocess, 'run', publish)
    trajectory = {'type': 'trajectory', 'positions': [[0.] * 6], 'velocities': [[0.] * 6], 'dt': .02}
    plan = {'q_init': [0.] * 6, 'complete_sequence': False,
            'planning_failure': {'reason': 'TRAJOPT_FAIL'},
            'steps': [{'type': 'gripper', 'label': 'open', 'action': 'open'},
                      dict(trajectory, label='outward'),
                      {'type': 'gripper', 'label': 'close', 'action': 'close'}]}
    comparison.save_json(tmp_path / 'with-box-plan.json', plan)
    comparison.save_json(tmp_path / 'obstacle-markers.json', {'markers': [
        {'id': i, 'color': {'a': 1.}} for i in [0, 1, 3]]})
    assert comparison.run_record(tmp_path, 'with_box') == (1 if block else 0)
    output, = (tmp_path / 'executions').iterdir()
    result = json.loads((output / 'result.json').read_text())
    if block:
        assert calls == [] and not result['sent_to_robot']
        assert result['status'] == 'failed'
    else:
        assert calls == ['open', 'arm', 'close']
        assert result['completed_steps'] == 3 and not result['complete_sequence_planned']
        assert result['status'] == 'stopped_before_unplanned_return'
        assert published[0]['markers'][0]['color']['a'] == 1.
        assert published[0]['markers'][1]['color']['a'] == 0.
        assert published[0]['markers'][2]['color']['a'] == 1.
