"""Prepare cuRobo samples for blocking M1013 execution."""
import math

import numpy as np


def plan_showcase(q_start):
    """Plan a visible round trip; reopen the model gripper before each arm move."""
    q = np.asarray(q_start, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite start joint angles in radians')
    offsets_deg = [20, -20, 20, 15, -15, 25]
    target = q + np.radians(offsets_deg)
    outward, returning = _plan_joint_moves(q, [target, q])
    outward['label'] = 'planned outward: joints +20/-20/+20/+15/-15/+25 deg'
    returning['label'] = 'planned return to start'
    return {'version': '1.0.0', 'robot': 'm1013_uon', 'scene': 'empty_test_world',
            'q_init': q.tolist(), 'joint_offsets_deg': offsets_deg,
            'returns_to_start': True, 'steps': [
                {'type': 'gripper', 'label': 'close at start', 'action': 'close'},
                {'type': 'gripper', 'label': 'reopen before outward motion', 'action': 'open'},
                outward,
                {'type': 'gripper', 'label': 'close at outward pose', 'action': 'close'},
                {'type': 'gripper', 'label': 'reopen before return', 'action': 'open'},
                returning,
            ]}


def _plan_joint_moves(q_start, targets):
    """Reuse one planner; plan every leg before returning any executable steps."""

    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from tiptop.doosan.model import m1013_uon_curobo_cfg

    tensor_args = TensorDeviceType()
    # This placeholder is outside the workspace. No camera/table scene is loaded.
    world = WorldConfig(cuboid=[Cuboid(name='placeholder', dims=[.01] * 3,
                                      pose=[5, 5, 5, 1, 0, 0, 0])])
    config = MotionGenConfig.load_from_robot_config(
        robot_cfg=m1013_uon_curobo_cfg(), world_model=world,
        use_cuda_graph=False, trajopt_particle_opt=False, collision_activation_distance=0.0,
        position_threshold=0.01, rotation_threshold=0.1)
    planner = MotionGen(config)
    q = q_start
    steps = []
    for leg, target in enumerate(targets, 1):
        start_state = JointState.from_position(tensor_args.to_device(q[None]))
        target_state = JointState.from_position(tensor_args.to_device(target[None]))
        for label, state in [('start', start_state), ('target', target_state)]:
            valid, reason = planner.check_start_state(state)
            if not bool(valid):
                raise ValueError(f'leg {leg} {label} rejected by the planning model: {reason}')
        result = planner.plan_single_js(
            start_state, target_state,
            MotionGenPlanConfig(max_attempts=3, enable_graph=True, time_dilation_factor=0.2))
        if not bool(result.success.all()):
            raise RuntimeError(f'Joint planning leg {leg} failed: {result.status}')
        path = result.get_interpolated_plan()
        dt = float(result.interpolation_dt)
        positions, velocities, _ = prepare_joint_path(
            path.position.cpu().numpy(), path.velocity.cpu().numpy(), dt)
        steps.append({'type': 'trajectory', 'positions': positions.tolist(),
                      'velocities': velocities.tolist(), 'dt': dt})
        # The next leg starts at the actual planned endpoint, including solver tolerance.
        q = positions[-1]
    return steps


def prepare_joint_path(positions, velocities, dt, settle_time=0.1):
    """Preserve planned samples and append a stationary terminal hold.

    Only tiny terminal velocity residues are accepted. A path ending in motion
    must be planned to stop rather than being silently converted into a hold.
    """
    q, v = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) == 0 or v.shape != q.shape:
        raise ValueError('Expected matching nonempty [N, 6] position and velocity arrays')
    if not np.isfinite(q).all() or not np.isfinite(v).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Expected finite samples and positive dt')
    if not math.isfinite(settle_time) or settle_time <= 0:
        raise ValueError('settle_time must be finite and positive')
    if np.max(np.abs(v[-1])) > 1e-3:
        raise ValueError('Terminal velocity is too large for a stationary settling hold')
    n = max(1, math.ceil(settle_time / dt))
    return (np.vstack([q, np.repeat(q[-1:], n, axis=0)]),
            np.vstack([v, np.zeros((n, 6))]), [float(dt)] * (len(q) + n))
