"""Blocking Isaac client for TiPToP's existing execute_cutamp_plan function."""

import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ros2_ws/src/execution/tiptop_execution'))

from tiptop_execution.trajectory import JOINT_NAMES, validate_trajectory


def load_plan(path):
    """Check the entire input before starting the simulator or issuing commands."""
    return validate_plan(json.loads(Path(path).read_text()))


def validate_plan(plan):
    """Validate an in-memory request before reserving an execution slot."""
    if plan.get('version') != '1.0.0' or plan.get('robot', 'm1013_uon') != 'm1013_uon':
        raise ValueError('Expected a version 1.0.0 M1013 plan')
    q = np.asarray(plan['q_init'], dtype=float)
    validate_trajectory({'positions': [q], 'velocities': [[0] * 6], 'durations': [0.02]})
    if not plan['steps']:
        raise ValueError('Plan has no steps')
    for step in plan['steps']:
        if not isinstance(step.get('label'), str):
            raise ValueError('Every step needs a label')
        if step['type'] == 'trajectory':
            positions, _, _ = validate_trajectory({
                'positions': step['positions'], 'velocities': step['velocities'],
                'durations': [step['dt']] * len(step['positions'])})
            if np.max(np.abs(positions[0] - q)) > math.radians(1):
                raise ValueError('Discontinuous planned trajectory start')
            q = positions[-1]
        elif step['type'] != 'gripper' or step.get('action') not in ('open', 'close'):
            raise ValueError(f'Unsupported plan step: {step.get("type")}')
    return plan


class IsaacPlanClient:
    """Position/velocity drive targets in rad and seconds, with observed completion.

    Scene creation stays in m1013_uon.py; action ordering stays in TiPToP's executor.
    This client opens no network connection and never teleports joint positions.
    """

    def __init__(self, app, *, realtime, physics_dt=0.01,
                 articulation_path='/World/Arm/m1013/root_joint', arm_joint_names=JOINT_NAMES):
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.timeline

        self.app, self.realtime = app, realtime
        self.sim, self.render = SimulationManager, RenderingManager
        self.timeline = omni.timeline.get_timeline_interface()
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('Expected a positive finite physics timestep')
        self.robot = Articulation(articulation_path)
        self.sim.set_physics_dt(physics_dt)
        self.timeline.play()
        app.update()
        self.names = self.robot.dof_names
        self.arm = [self.names.index(name) for name in arm_joint_names]
        self.spinner = self.names.index('spinner_joint')
        self.controlled = self.arm + [self.spinner]
        limit = ET.parse(ROOT / 'tiptop/doosan/assets/m1013_uon/m1013_uon.urdf').getroot().find(
            "joint[@name='gripper_spinner_joint']/limit")
        self.open_angle, self.close_angle = float(limit.get('lower')), float(limit.get('upper'))
        self.dt = self.sim.get_physics_dt()
        self.cancelled, self.calls, self.ticks = False, [], 0
        self.wall_start, self.sim_start = time.perf_counter(), self.sim.get_simulation_time()

    def _tick(self):
        if self.cancelled or not self.app.is_running() or not self.timeline.is_playing():
            raise RuntimeError('Execution stopped; remaining actions were cancelled')
        self.sim.step(update_fabric=self.sim.is_fabric_enabled())
        self.ticks += 1
        if self.ticks % (3 if self.realtime else 100) == 0:
            self.render.render()
        if self.realtime:
            delay = self.sim.get_simulation_time() - self.sim_start - (time.perf_counter() - self.wall_start)
            if delay > 0:
                time.sleep(min(delay, self.dt))

    def get_joint_positions(self):
        return self.robot.get_dof_positions(dof_indices=self.arm).numpy()[0].tolist()

    def get_state(self):
        q = self.robot.get_dof_positions().numpy()[0]
        v = self.robot.get_dof_velocities().numpy()[0]
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError('Non-finite simulated joint state')
        return {'success': True, 'joint_positions': q[self.arm].tolist(),
                'joint_velocities': v[self.arm].tolist(),
                'motion_status': 0 if np.max(np.abs(v[self.controlled])) < 0.01 else 1,
                'gripper': {'position_rad': float(q[self.spinner]), 'velocity_rad_s': float(v[self.spinner])},
                'simulation_time_s': self.sim.get_simulation_time()}

    def stop(self):
        self.cancelled = True
        q = self.robot.get_dof_positions(dof_indices=self.controlled).numpy()[0]
        self.robot.set_dof_position_targets(q, dof_indices=self.controlled)
        self.robot.set_dof_velocity_targets(0.0, dof_indices=self.controlled)
        return {'success': True, 'stop_requested': True}

    def _run(self, kind, operation):
        try:
            if self.cancelled:
                raise RuntimeError('Execution was cancelled')
            result = dict(operation(), success=True, type=kind)
        except (Exception, KeyboardInterrupt) as exc:
            result = {'success': False, 'type': kind, 'error': str(exc) or 'Interrupted'}
            try:
                self.stop()
                result['state_at_failure'] = self.get_state()
            except Exception as stop_error:
                result['stop_error'] = str(stop_error)
        self.calls.append(result)
        print('ISAAC_ACTION', len(self.calls), json.dumps(result), flush=True)
        return result

    def _settle(self, targets, indices, tolerance):
        stable = 0
        for _ in range(math.ceil(5 / self.dt)):
            self._tick()
            q = self.robot.get_dof_positions(dof_indices=indices).numpy()[0]
            v = self.robot.get_dof_velocities(dof_indices=indices).numpy()[0]
            error = float(np.max(np.abs(q - targets)))
            stable = stable + 1 if error <= tolerance and np.max(np.abs(v)) < 0.01 else 0
            if stable * self.dt >= 0.15:
                return error
        raise RuntimeError(f'Goal did not settle: error={error:.6f} rad, tolerance={tolerance:.6f} rad')

    def execute_joint_impedance_path(self, joint_confs, joint_vels, durations):
        def execute():
            q, v, stamps = validate_trajectory({'positions': joint_confs, 'velocities': joint_vels,
                                               'durations': durations})
            if np.max(np.abs(np.asarray(self.get_joint_positions()) - q[0])) > math.radians(1):
                raise ValueError('First waypoint differs from current joints by more than 1 degree')
            # Same position+velocity cubic interpolation as the ROS2 JTC default.
            # dt[0] remains the initial hold; no stride/subsampling or time scaling.
            curve = CubicHermiteSpline(np.r_[0, stamps], np.vstack([q[:1], q]),
                                      np.vstack([np.zeros((1, 6)), v]))
            peak_error = 0.0
            for tick in range(1, math.ceil(stamps[-1] / self.dt) + 1):
                t = min(tick * self.dt, stamps[-1])
                target, velocity = curve(t), curve(t, 1)
                self.robot.set_dof_position_targets(target, dof_indices=self.arm)
                self.robot.set_dof_velocity_targets(velocity, dof_indices=self.arm)
                self._tick()
                peak_error = max(peak_error, float(np.max(np.abs(
                    np.asarray(self.get_joint_positions()) - target))))
            self.robot.set_dof_velocity_targets(0.0, dof_indices=self.arm)
            error = self._settle(q[-1], self.arm, math.radians(0.5))
            return {'waypoints': len(q), 'requested_duration_s': float(stamps[-1]),
                    'peak_tracking_error_rad': peak_error, 'final_error_rad': error}

        return self._run('trajectory', execute)

    def _gripper(self, action, speed, force):
        def execute():
            if not math.isfinite(speed) or not 0.01 <= speed <= 1 or not math.isfinite(force) or not 0 <= force <= 1:
                raise ValueError('Expected speed in [0.01, 1] rad/s and force in [0, 1]')
            start = self.get_state()['gripper']['position_rad']
            target = self.open_angle if action == 'open' else self.close_angle
            travel = abs(target - start) / speed
            direction = np.sign(target - start)
            for tick in range(1, math.ceil(travel / self.dt) + 1):
                t = min(tick * self.dt, travel)
                self.robot.set_dof_position_targets(start + direction * speed * t, dof_indices=[self.spinner])
                self.robot.set_dof_velocity_targets(direction * speed if t < travel else 0.0,
                                                    dof_indices=[self.spinner])
                self._tick()
            self.robot.set_dof_position_targets(target, dof_indices=[self.spinner])
            self.robot.set_dof_velocity_targets(0.0, dof_indices=[self.spinner])
            error = self._settle(target, [self.spinner], 1e-4)
            return {'action': action, 'target_rad': target, 'final_error_rad': error,
                    'force_parameter_applied': False}

        return self._run('gripper', execute)

    def open_gripper(self, speed=1.0, force=0.1):
        return self._gripper('open', speed, force)

    def close_gripper(self, speed=1.0, force=0.1):
        return self._gripper('close', speed, force)


def replay_plan(app, plan, *, realtime, report):
    from tiptop.execute_plan import execute_cutamp_plan

    client = IsaacPlanClient(app, realtime=realtime)
    report.update(backend='isaac_sim', physical_grasp_verified=False, completed_steps=0,
                  calls=client.calls, requested_trajectory_duration_s=sum(
                      len(s['positions']) * s['dt'] for s in plan['steps'] if s['type'] == 'trajectory'))
    try:
        if np.max(np.abs(np.asarray(client.get_joint_positions()) - plan['q_init'])) > math.radians(1):
            raise ValueError('Plan q_init differs from current joints by more than 1 degree')
        report['initial_state'] = client.get_state()
        print(f'Isaac replay: {len(plan["steps"])} steps, '
              f'{report["requested_trajectory_duration_s"]:.2f} s of arm trajectories.', flush=True)
        execute_cutamp_plan(plan['steps'], client)
        report['success'] = True
    finally:
        report['completed_steps'] = sum(c['success'] for c in client.calls)
        try:
            report['final_state'] = client.get_state()
        except Exception as exc:
            report['final_state_error'] = str(exc)
            report['success'] = False
            raise
