from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tiptop.doosan import planning_runtime as runtime


@pytest.fixture(autouse=True)
def isolated_cache():
    runtime._ENTRIES.clear()
    yield
    runtime._ENTRIES.clear()


def test_reuse_updates_world_and_resets_state_and_failed_reset_rebuilds():
    built, events = [], []
    world = dict(obstacle='new geometry with reused name')
    def build():
        solver = SimpleNamespace(world_coll_checker=SimpleNamespace(clear_cache=lambda: events.append('clear')),
            update_world=lambda current: events.append(deepcopy(current)),
            reset_shape=lambda: events.append('shape'), reset_seed=lambda: events.append('seed'))
        built.append(solver)
        return solver
    first, audit = runtime.cached_planner('ik', 'model-A', build, lambda x: runtime.reset_ik(x, world))
    assert not audit['reused']
    second, audit = runtime.cached_planner('ik', 'model-A', build, lambda x: runtime.reset_ik(x, world))
    assert first is second and audit['reused'] and events == ['clear', world, 'shape', 'seed']
    def broken(value):
        raise RuntimeError('buffer cannot reset')
    third, audit = runtime.cached_planner('ik', 'model-A', build, broken)
    assert third is not first and not audit['reused'] and audit['reset_error']
    fourth, audit = runtime.cached_planner('ik', 'model-B', build, broken)
    assert fourth is not third and len(runtime._ENTRIES) == 1


def test_motion_reset_drops_attachments_and_mesh_cache_before_loading_current_world():
    events = []
    solver = SimpleNamespace(detach_object_from_robot=lambda name: events.append(('detach', name)),
        clear_world_cache=lambda: events.append('clear'), update_world=lambda w: events.append(w),
        reset=lambda **kw: events.append(kw))
    runtime.reset_motion_gen(solver, 'fresh world')
    assert events == [('detach', 'attached_object'), 'clear', 'fresh world', dict(reset_seed=True)]


def test_model_key_uses_contents_and_resolved_relative_resources(tmp_path):
    first, second = tmp_path/'one', tmp_path/'two'
    first.mkdir(); second.mkdir()
    mesh = tmp_path/'mesh.stl'; mesh.write_text('old mesh')
    text = f'<robot><link><visual><geometry><mesh filename="{mesh}"/></geometry></visual></link></robot>'
    a, b = first/'model.urdf', second/'model.urdf'
    a.write_text(text); b.write_text(text)
    before = runtime.configuration_key(dict(urdf_path=str(a)))
    assert before == runtime.configuration_key(dict(urdf_path=str(b)))
    mesh.write_text('changed mesh surface')
    assert before != runtime.configuration_key(dict(urdf_path=str(a)))
    a.write_text(text.replace(str(mesh), 'relative.stl'))
    b.write_text(a.read_text())
    assert runtime.configuration_key(str(a)) != runtime.configuration_key(str(b))


def test_unique_first_retains_all_trials_with_their_joint_seeds_and_confidences():
    particles = dict(q0=torch.zeros((6, 6)), grasp=torch.tensor([[4.], [4.], [7.], [7.], [9.], [4.]]),
                     conf=torch.arange(6), other=None)
    particles['q0'][5, 0] = .1  # Same grasp with another home remains a distinct goal.
    original = {k:v.clone() if v is not None else None for k,v in particles.items()}
    result, audit = runtime.unique_path_goals_first(particles, ['grasp'])
    assert result['conf'].tolist() == [0, 2, 4, 5, 1, 3]
    assert audit == dict(total=6, unique_goals=4, deferred_duplicates=2)
    assert sorted(result['conf'].tolist()) == list(range(6))
    for k, value in original.items():
        if value is not None:
            torch.testing.assert_close(particles[k], value)


def test_failure_does_not_leave_half_updated_runtime_for_the_next_task():
    @runtime.serialized_planning
    def failure():
        runtime._ENTRIES['old'] = object()
        raise ValueError('failed task')
    with pytest.raises(ValueError):
        failure()
    assert not runtime._ENTRIES
