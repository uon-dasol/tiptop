"""Prepare cuRobo samples for blocking M1013 execution."""
import math

import numpy as np


SHOWCASE_OFFSETS_DEG = (60, -30, 40, 15, -15, 25)


def showcase_target(q_start):
    q = np.asarray(q_start, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite start joint angles in radians')
    return q + np.radians(SHOWCASE_OFFSETS_DEG)


def plan_showcase(q_start, *, world=None, target=None, allow_partial=False):
    """Plan the same five-step task in any world; optionally preserve a failed return's prefix."""
    import logging

    q = np.asarray(q_start, dtype=float)
    default_target = showcase_target(q)
    goal = default_target if target is None else np.asarray(target, dtype=float)
    if goal.shape != (6,) or not np.isfinite(goal).all():
        raise ValueError('Expected six finite target joint angles in radians')
    outward, = _plan_joint_moves(q, [goal], world=world, gripper_positions=[0.0])
    outward = dict(outward, label='planned outward')
    steps = [{'type': 'gripper', 'label': 'open at home', 'action': 'open'}, outward,
             {'type': 'gripper', 'label': 'close at outward pose', 'action': 'close'}]
    failure = None
    try:
        returning, = _plan_joint_moves(np.asarray(outward['positions'][-1]), [q],
                                      world=world, gripper_positions=[1.5009832])
    except (RuntimeError, ValueError) as error:
        if not allow_partial:
            raise
        failure = {'status': 'failed', 'error_type': type(error).__name__, 'reason': str(error)}
        logging.getLogger(__name__).warning('Showcase return planning failed', exc_info=True)
    else:
        steps += [dict(returning, label='planned return home with closed gripper'),
                  {'type': 'gripper', 'label': 'open after returning home', 'action': 'open'}]
    return {'version': '1.0.0', 'robot': 'm1013_uon',
            'scene': 'empty_test_world' if world is None else 'box_test_world',
            'q_init': q.tolist(), 'q_home': q.tolist(),
            'joint_offsets_deg': list(SHOWCASE_OFFSETS_DEG) if target is None else np.degrees(goal - q).tolist(),
            'returns_to_start': failure is None, 'complete_sequence': failure is None,
            'expected_steps': 5, 'planning_failure': failure, 'steps': steps}


def _plan_joint_moves(q_start, targets, world=None, gripper_positions=None):
    """Reuse one planner; plan every leg before returning any executable steps."""

    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from tiptop.doosan.model import m1013_uon_curobo_cfg

    tensor_args = TensorDeviceType()
    # This placeholder is outside the workspace. No camera/table scene is loaded.
    if world is None:
        world = WorldConfig(cuboid=[Cuboid(name='placeholder', dims=[.01] * 3,
                                          pose=[5, 5, 5, 1, 0, 0, 0])])
    if gripper_positions is None:
        gripper_positions = [0.0] * len(targets)
    if len(gripper_positions) != len(targets):
        raise ValueError('Each motion leg needs a gripper angle')
    planners = {}
    q = q_start
    steps = []
    for leg, target in enumerate(targets, 1):
        angle = gripper_positions[leg - 1]
        if angle not in planners:
            config = MotionGenConfig.load_from_robot_config(
                robot_cfg=m1013_uon_curobo_cfg(angle), world_model=world,
                use_cuda_graph=False, trajopt_particle_opt=False, collision_activation_distance=0.02,
                num_trajopt_seeds=8, num_graph_seeds=8,
                graph_trajopt_iters=200,
                position_threshold=0.01, rotation_threshold=0.1)
            planners[angle] = MotionGen(config)
        planner = planners[angle]
        start_state = JointState.from_position(tensor_args.to_device(q[None]))
        target_state = JointState.from_position(tensor_args.to_device(target[None]))
        for label, state in [('start', start_state), ('target', target_state)]:
            valid, reason = planner.check_start_state(state)
            if not bool(valid):
                raise ValueError(f'leg {leg} {label} rejected by the planning model: {reason}')
        result = planner.plan_single_js(
            start_state, target_state,
            MotionGenPlanConfig(max_attempts=8, timeout=40., enable_graph=True, time_dilation_factor=0.2))
        if not bool(result.success.all()):
            raise RuntimeError(f'Joint planning leg {leg} failed: {result.status}')
        path = result.get_interpolated_plan()
        dt = float(result.interpolation_dt)
        positions, velocities, _ = prepare_joint_path(
            path.position.cpu().numpy(), path.velocity.cpu().numpy(), dt)
        steps.append({'type': 'trajectory', 'positions': positions.tolist(),
                      'velocities': velocities.tolist(), 'dt': dt,
                      'gripper_position_rad': float(angle)})
        # The next leg starts at the actual planned endpoint, including solver tolerance.
        q = positions[-1]
    return steps


def prepare_joint_path(positions, velocities, dt):
    """Preserve sample positions and timing; zero only tiny endpoint residues."""
    q, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) == 0 or v.shape != q.shape:
        raise ValueError('Expected matching nonempty [N, 6] position and velocity arrays')
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Expected finite samples and positive dt')
    if np.max(np.abs(v[-1])) > 5e-3:
        raise ValueError('Terminal velocity is too large for a stationary endpoint')
    v = v.copy()
    v[-1] = 0
    return q, v, [float(dt)] * len(q)
