"""Placement frame and release/failure contracts, without starting Isaac."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiptop.execute_plan import ExecutionFailure

path = Path(__file__).resolve().parents[1] / 'simulation/isaac_sim/place_pepper.py'
spec = importlib.util.spec_from_file_location('isaac_placement', path)
place = importlib.util.module_from_spec(spec)
spec.loader.exec_module(place)


def test_retreat_uses_tool_approach_axis_and_body_offset_is_pose_invariant():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.4, -.7, .8]).as_matrix()
    transform[:3, 3] = [.55, -.12, .44]
    _, _, release, retreat = place.placement_poses(transform, [0., .12])
    np.testing.assert_allclose(retreat[:3, 3] - release[:3, 3], -.12 * transform[:3, 2])
    local = np.array([.01, -.02, .03])
    for angles in ([0., 0., 0.], [.4, -.7, .8]):
        rotation = Rotation.from_euler('xyz', angles)
        xyzw = rotation.as_quat()
        body = np.array([.2, .3, .4])
        world = body + rotation.apply(local)
        np.testing.assert_allclose(place.body_relative_position(world, body, xyzw[[3, 0, 1, 2]]), local)


def run_place(*, open_ok=True, support=1.):
    motion = dict(type='trajectory', label='move', positions=[[0.] * 6], velocities=[[0.] * 6], dt=.2)
    record = dict(post_pick_steps=[motion, motion, dict(type='gripper', label='release', action='open'), motion],
                  placement=dict(offset_xy_m=[0., .12]))
    calls, report = [], dict(trace=[])

    class Client:
        dt = .01

        def execute_joint_impedance_path(self, **kwargs):
            calls.append(('motion', report['phase']))
            return dict(success=True)

        def open_gripper(self, **kwargs):
            calls.append(('open', report['phase']))
            return dict(success=open_ok, error='release rejected')

        def _tick(self): pass

        def get_state(self): return dict(success=True)

    array = SimpleNamespace(numpy=lambda: np.zeros((1, 3)))
    obj = SimpleNamespace(get_velocities=lambda: (array, array))

    def sample(label):
        report['trace'].append(dict(finger_forces_n=[1., 1., 1.] if label == 'before_release' else [0.] * 3,
                                    table_object_force_n=support))
        return np.array([.55, 0., .44])

    return (Client(), record, obj, [.55, -.12, .44], sample, lambda: None, report), calls, report


def test_failed_release_prevents_retreat():
    args, calls, report = run_place(open_ok=False)
    with pytest.raises(ExecutionFailure, match='release rejected'):
        place.execute_placement(*args)
    assert [c[0] for c in calls] == ['motion', 'motion', 'open']
    assert report['physical_placement_verified'] is False
    assert calls[-1][1] == 'release_and_retreat'


def test_placement_needs_table_support_after_release():
    args, _, report = run_place(support=0.)
    with pytest.raises(RuntimeError, match='supported by the table'):
        place.execute_placement(*args)
    assert report['physical_placement_verified'] is False
    args, calls, report = run_place(support=1.)
    place.execute_placement(*args)
    assert report['physical_placement_verified'] is True
    assert calls[-1] == ('motion', 'release_and_retreat')
