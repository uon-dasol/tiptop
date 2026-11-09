"""Reuse screened IK solutions within one known-object planning call.

This adapter belongs to TiPToP; the upstream cuTAMP/cuRobo sources stay intact.
It only handles a single returned seed and unchanged cuboid-only worlds. Every
hit checks FK error and joint/collision feasibility again with the owning solver.
Unknown requests fall back to that solver's ordinary solve_batch.
"""

import time

import numpy as np
import torch
from curobo.geom.types import Cuboid
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKResult


def _world_signature(world):
    objects = getattr(world, "objects", None)
    if objects is None or not all(type(obj) is Cuboid for obj in objects):
        return None
    boxes = []
    for obj in objects:
        if obj.scale is not None:
            return None
        pose, dims = np.asarray(obj.pose, float), np.asarray(obj.dims, float)
        if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(dims).all() or np.any(dims <= 0)):
            return None
        boxes.append((obj.name, tuple(pose), tuple(dims)))
    return tuple(sorted(boxes))


class ScreenedIKReuse:
    """Plan-local proxy for the exact solver that produced the screened results."""

    def __init__(self, solver, goal_pose, result):
        self._solver = solver
        self._world = _world_signature(solver.world_coll_checker.world_model)
        self._joint_names = tuple(solver.joint_names)
        self._kinematics = solver.kinematics
        self._goals = goal_pose.get_matrix().detach().clone()
        self._success = result.success.detach().clone()
        self._solutions = result.solution.detach().clone()
        self.stats = dict(reused_batches=0, reused_goals=0, fallback_batches=0,
                          lookup_validation_seconds=0.0, fallback_reasons={})

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def update_world(self, world):
        # cuTAMP calls this even when the static geometry is unchanged.
        return self._solver.update_world(world)

    def _reuse(self, goal_pose, args, kwargs):
        if args or any(key not in {"seed_config", "return_seeds"} for key in kwargs):
            return None, "solve_options"
        if kwargs.get("seed_config") is not None or kwargs.get("return_seeds", 1) != 1:
            return None, "solve_options"
        if (self._world is None or self._world != _world_signature(self._solver.world_coll_checker.world_model)
                or self._joint_names != tuple(self._solver.joint_names)
                or self._kinematics is not self._solver.kinematics):
            return None, "world_or_joints_changed"
        goals = goal_pose.get_matrix()
        if (goals.ndim != 3 or self._goals.ndim != 3 or not len(goals) or not len(self._goals)
                or goals.device != self._goals.device or goals.dtype != self._goals.dtype):
            return None, "goal_shape_or_device"
        # World/object/TCP float32 conversions introduce sub-micrometre roundoff.
        differences = (goals[:, None] - self._goals[None]).abs().amax(dim=(2, 3))
        errors, rows = differences.min(dim=1)
        if not bool((errors <= 1e-6).all()):
            return None, "goal_changed"
        valid_seeds = self._success[rows]
        if not bool(valid_seeds.any(dim=1).all()):
            return None, "no_successful_seed"
        seeds = valid_seeds.to(torch.int32).argmax(dim=1)
        q = self._solutions[rows, seeds].clone()
        if not bool(torch.isfinite(q).all()):
            return None, "nonfinite_solution"
        state = JointState.from_position(q, joint_names=list(self._joint_names))
        fk = self._solver.kinematics.get_state(q).ee_pose
        position_error, rotation_error = goal_pose.distance(fk)
        feasible = self._solver.check_constraints(state).feasible.reshape(len(q), -1).all(dim=1)
        success = (feasible & (position_error.reshape(-1) <= self._solver.position_threshold)
                   & (rotation_error.reshape(-1) <= self._solver.rotation_threshold))
        if not bool(success.all()):
            return None, "current_validation_failed"
        q = q[:, None]
        p, r = position_error.reshape(-1, 1), rotation_error.reshape(-1, 1)
        return IKResult(
            js_solution=JointState.from_position(q, joint_names=list(self._joint_names)),
            goal_pose=goal_pose, solution=q, seed=None, success=success[:, None],
            position_error=p, rotation_error=r, error=p+r, solve_time=0.0,
        ), None

    def solve_batch(self, goal_pose, *args, **kwargs):
        started = time.perf_counter()
        result, reason = self._reuse(goal_pose, args, kwargs)
        elapsed = time.perf_counter() - started
        self.stats["lookup_validation_seconds"] += elapsed
        if result is not None:
            result.solve_time = elapsed
            self.stats["reused_batches"] += 1
            self.stats["reused_goals"] += len(result.solution)
            return result
        self.stats["fallback_batches"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return self._solver.solve_batch(goal_pose, *args, **kwargs)
