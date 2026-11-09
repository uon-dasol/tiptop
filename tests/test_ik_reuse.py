"""CPU recording backend for reuse guards; real GPU plans are checked offline."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from curobo.geom.types import Cuboid, Sphere, WorldConfig

from tiptop.doosan.ik_reuse import ScreenedIKReuse


class Goal:
    def __init__(self, matrices):
        self.matrices = matrices

    def get_matrix(self):
        return self.matrices

    def distance(self, other):
        return (
            torch.linalg.vector_norm(self.matrices[:, :3, 3] - other.matrices[:, :3, 3], dim=1),
            torch.linalg.matrix_norm(self.matrices[:, :3, :3] - other.matrices[:, :3, :3], dim=(1, 2)),
        )


def setup():
    matrices = torch.eye(4).repeat(3, 1, 1)
    matrices[:, 0, 3] = torch.tensor([.1, .2, .3])
    goals = Goal(matrices)
    result = SimpleNamespace(
        solution=torch.zeros(3, 2, 6), success=torch.tensor([[False, True], [True, False], [False, False]]),
    )
    result.solution[:, :, 0] = matrices[:, 0, 3, None]
    result.solution[0, 0, 0] = 99  # Never use a failed seed, even if it is first.
    world = WorldConfig(cuboid=[Cuboid(name="table", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, .1])])

    class Solver:
        joint_names = [f"j{i}" for i in range(6)]
        position_threshold, rotation_threshold = .001, .01

        def __init__(self):
            self.world_coll_checker = SimpleNamespace(world_model=world)
            self.kinematics = SimpleNamespace(get_state=self.fk)
            self.calls, self.checked, self.blocked, self.fk_offset = [], [], False, 0.

        def update_world(self, updated):
            self.world_coll_checker.world_model = updated

        def fk(self, q):
            poses = torch.eye(4).repeat(len(q), 1, 1)
            poses[:, :3, 3] = q[:, :3] + self.fk_offset
            return SimpleNamespace(ee_pose=Goal(poses))

        def check_constraints(self, state):
            self.checked.append(state.position.clone())
            return SimpleNamespace(feasible=torch.full((len(state.position), 1), not self.blocked))

        def solve_batch(self, goal, *args, **kwargs):
            self.calls.append((goal, args, kwargs))
            return "ordinary IK"

    solver = Solver()
    return solver, goals, result, ScreenedIKReuse(solver, goals, result)


def test_reordered_and_duplicate_goals_use_successful_seeds_and_independent_storage():
    solver, goals, original, cache = setup()
    requested = Goal(goals.matrices[[1, 0, 1]].clone())
    # Upstream results may be buffer-backed and change after construction.
    original.solution.fill_(123)
    original.success.fill_(False)
    cache.update_world(deepcopy(solver.world_coll_checker.world_model))
    result = cache.solve_batch(requested, seed_config=None)
    torch.testing.assert_close(result.solution[:, 0, 0], torch.tensor([.2, .1, .2]))
    assert result.success.all() and not solver.calls and len(solver.checked) == 1
    assert cache.stats["reused_goals"] == 3
    result.solution.fill_(456)
    second = cache.solve_batch(requested)
    torch.testing.assert_close(second.solution[:, 0, 0], torch.tensor([.2, .1, .2]))


@pytest.mark.parametrize("change", ["pose", "dims", "add", "remove", "mesh_type", "joints", "kinematics", "world_type"])
def test_changed_world_or_model_never_reuses(change):
    solver, goals, _, cache = setup()
    world = deepcopy(solver.world_coll_checker.world_model)
    if change == "pose":
        world.objects[0].pose[0] += .001
    elif change == "dims":
        world.objects[0].dims[0] += .001
    elif change == "add":
        world = WorldConfig(cuboid=[*world.objects, Cuboid(name="camera", pose=[1, 0, 0, 1, 0, 0, 0], dims=[1]*3)])
    elif change == "remove":
        world = WorldConfig()
    elif change == "mesh_type":
        world = WorldConfig(sphere=[Sphere(name="ball", pose=[0, 0, 0, 1, 0, 0, 0], radius=.1)])
    elif change == "joints":
        solver.joint_names = list(reversed(solver.joint_names))
    elif change == "kinematics":
        solver.kinematics = SimpleNamespace(get_state=solver.fk)
    else:
        world = [world]
    cache.update_world(world)
    requested = Goal(goals.matrices[:1])
    assert cache.solve_batch(requested) == "ordinary IK"
    assert len(solver.calls) == 1 and not solver.checked


@pytest.mark.parametrize("options", [dict(seed_config=torch.zeros(1, 1, 6)), dict(return_seeds=2),
                                    dict(retract_config=None), dict(newton_iters=1), dict(link_poses={})])
def test_nondefault_ik_requests_are_forwarded(options):
    solver, goals, _, cache = setup()
    assert cache.solve_batch(Goal(goals.matrices[:1]), **options) == "ordinary IK"
    assert solver.calls[0][2] is not options
    assert solver.calls[0][2].keys() == options.keys()
    assert cache.stats["fallback_reasons"] == {"solve_options": 1}


@pytest.mark.parametrize("change", ["goal", "nan_goal", "no_seed", "collision", "fk", "nan_q", "near_goal"])
def test_current_feasibility_and_fk_are_required(change):
    solver, goals, result, _ = setup()
    if change == "nan_q":
        result.solution[0, 1, 0] = float("nan")
    cache = ScreenedIKReuse(solver, goals, result)
    request = goals.matrices[:1].clone()
    if change == "goal":
        request[0, 0, 3] += .002
    elif change == "nan_goal":
        request[0, 0, 3] = float("nan")
    elif change == "no_seed":
        request = goals.matrices[2:].clone()
    elif change == "collision":
        solver.blocked = True
    elif change == "fk":
        solver.fk_offset = .01
    elif change == "near_goal":
        # A matching matrix alone does not guarantee the new IK error passes.
        request[0, 0, 3] += 5e-7
        solver.fk_offset = -.001 / (3**.5)
    assert cache.solve_batch(Goal(request)) == "ordinary IK"
    assert len(solver.calls) == 1


def test_float32_goal_roundoff_is_allowed_but_revalidated():
    solver, goals, _, cache = setup()
    requested = goals.matrices[:1].clone()
    requested[0, 0, 3] += 5e-7
    result = cache.solve_batch(Goal(requested))
    assert result.success.all() and 0 < result.position_error.item() < 1e-6
    assert len(solver.checked) == 1 and not solver.calls
