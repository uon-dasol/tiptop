"""Compare the same M1013 task with/without one box; plan first, then explicitly run a recorded case."""
import argparse
import copy
import logging
import math
import subprocess
import json
import contextlib
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from tiptop.doosan.planning import plan_showcase, showcase_target


def sphere_box_clearance(spheres, center, dims):
    """Signed distance from every active robot sphere to an axis-aligned box."""
    d = np.abs(spheres[..., :3] - center) - np.asarray(dims) / 2
    signed = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)
    return np.min(np.where(spheres[..., 3] > 0, signed - spheres[..., 3], np.inf), axis=-1)


def _dense(q):
    q = np.asarray(q)
    # Check the planned samples and four intermediate joint-space samples per interval.
    return np.vstack([(q[:-1, None] + (q[1:] - q[:-1])[:, None]
                       * np.arange(5)[None, :, None] / 5).reshape(-1, 6), q[-1:]])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare(inputs, output, report):
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from tiptop.doosan.model import CLOSED_POSITION, m1013_uon_curobo_cfg
    import torch

    torch.manual_seed(inputs['seed'])
    np.random.seed(inputs['seed'])
    q_start = np.asarray(inputs['q_start_rad'], dtype=float)
    target = np.asarray(inputs['target_rad'], dtype=float)
    tensor = TensorDeviceType()
    cfg = copy.deepcopy(m1013_uon_curobo_cfg()['robot_cfg'])
    cfg['kinematics']['lock_joints'] = {}
    model = CudaRobotModel(RobotConfig.from_dict(cfg, tensor).kinematics)
    self_cost = SelfCollisionCost(SelfCollisionCostConfig(
        weight=1., tensor_args=tensor, self_collision_kin_config=model.get_self_collision_config()))

    def state(q, angle=0.):
        q = np.asarray(q)
        values = {f'joint_{j + 1}': q[:, j] for j in range(6)}
        values['gripper_spinner_joint'] = np.broadcast_to(angle, (len(q),))
        ordered = np.stack([values[name] for name in model.joint_names], axis=-1)
        return model.get_state(tensor.to_device(ordered))

    def checkpoint(stage):
        report['stage'] = stage
        save_json(output / 'report.json', report)
        print(stage, flush=True)

    checkpoint('planning_without_box')
    baseline_plan = plan_showcase(q_start, target=target, allow_partial=True)
    baseline = baseline_plan['steps'][1]
    paths = {'without_box': baseline}
    save_json(output / 'comparison-paths.json', paths)
    outward = np.asarray(baseline['positions'])
    spheres = state(outward).link_spheres_tensor.cpu().numpy()
    if 'box' not in inputs:
        dims = np.full(3, .08)
        choices = []
        # Block the original path while leaving both endpoints valid.
        for fraction in [.4, .5, .6]:
            index = int(fraction * (len(outward) - 1))
            for center in spheres[index, spheres[index, :, 3] > 0, :3]:
                gap = min(float(sphere_box_clearance(spheres[0], center, dims)),
                          float(sphere_box_clearance(spheres[-1], center, dims)))
                if gap > .025:
                    choices.append((gap, center.copy()))
        if not choices:
            raise RuntimeError('No 8 cm box fits between these endpoints; use another start pose')
        _, center = max(choices, key=lambda item: item[0])
        inputs['box'] = {'name': 'comparison_box', 'dims': dims.tolist(),
                         'pose': [*center.tolist(), 1, 0, 0, 0]}
        save_json(output / 'inputs.json', inputs)
    box = inputs['box']
    center, dims = np.asarray(box['pose'][:3]), np.asarray(box['dims'])
    world = WorldConfig(cuboid=[Cuboid(**box)])
    report['obstacles'] = {'frame_id': 'base_link', 'cuboid': [box]}
    checkpoint('planning_with_box')
    detour_plan = plan_showcase(q_start, target=target, world=world, allow_partial=True)
    detour = detour_plan['steps'][1]
    paths['with_box'] = detour
    save_json(output / 'comparison-paths.json', paths)

    def verify(step, angle=0., include_box=True):
        arm = _dense(step['positions'])
        min_gap, max_cost = float('inf'), 0.
        for start in range(0, len(arm), 256):
            angles = angle if np.ndim(angle) == 0 else angle[start:start + 256]
            s = state(arm[start:start + 256], angles).link_spheres_tensor
            if include_box:
                min_gap = min(min_gap, float(sphere_box_clearance(s.cpu().numpy(), center, dims).min()))
            max_cost = max(max_cost, float(self_cost(s[:, None]).max().item()))
        return {'min_box_clearance_m': min_gap if include_box else None, 'max_self_collision_cost': max_cost,
                'checked_configurations': len(arm)}

    checkpoint('checking_outward_paths')
    before, after = verify(baseline), verify(detour)
    report['comparison'] = {'without_box_path_against_box': before, 'with_box_path': after}
    if before['min_box_clearance_m'] >= 0:
        raise RuntimeError('The box did not block the original path; comparison is inconclusive')
    if after['min_box_clearance_m'] <= 0 or after['max_self_collision_cost'] > 0:
        raise RuntimeError(f'Planned path failed the independent dense sphere checks: {after}')
    progress = np.linspace(0., 1., 201)
    resampled = [np.stack([np.interp(progress, np.linspace(0., 1., len(step['positions'])),
                                    np.asarray(step['positions'])[:, j]) for j in range(6)], -1)
                 for step in [baseline, detour]]
    report['comparison']['max_joint_path_difference_deg'] = float(
        np.degrees(np.abs(resampled[0] - resampled[1]).max()))
    traces = []
    for step in [baseline, detour]:
        q = np.asarray(step['positions'])
        indices = np.linspace(0, len(q) - 1, min(120, len(q)), dtype=int)
        traces.append([state(q[indices]).ee_position.cpu().numpy().tolist()])
    save_json(output / 'obstacle-markers.json', markers(report, traces))
    plot(report, traces, output / 'obstacle-comparison.png')
    report['outward_comparison_success'] = True
    checkpoint('outward_comparison_complete')

    def check_sequence(steps, include_box):
        current = q_start
        for step in steps:
            if step['type'] == 'gripper':
                # Check both directions of the complete gripper sweep at this arm pose.
                # _dense subdivides 13 repeated poses into 61 samples.
                checked = verify({'positions': np.repeat(current[None], 13, axis=0)},
                                 np.linspace(0., CLOSED_POSITION, 61), include_box)
            else:
                checked = verify(step, step['gripper_position_rad'], include_box)
                current = np.asarray(step['positions'][-1])
            if checked['max_self_collision_cost'] > 0 or (
                    include_box and checked['min_box_clearance_m'] <= 0):
                raise RuntimeError(f"Sequence collision check failed at {step['label']}: {checked}")

    report['plans'] = {}
    for case, plan in [('without_box', baseline_plan), ('with_box', detour_plan)]:
        failure = plan['planning_failure']
        if case == 'with_box':
            report['return'] = failure or {'status': 'planned'}
        if plan['complete_sequence']:
            paths[case + '_closed_gripper_return'] = plan['steps'][3]
            save_json(output / 'comparison-paths.json', paths)
        try:
            check_sequence(plan['steps'], include_box=case == 'with_box')
        except RuntimeError as error:
            report['plans'][case] = {'exported': False, 'reason': str(error)}
            traceback.print_exc()
            continue
        filename = case.replace('_', '-') + '-plan.json'
        save_json(output / filename, plan)
        report['plans'][case] = {'exported': True, 'file': filename,
                                 'complete_sequence': plan['complete_sequence'],
                                 'steps': len(plan['steps']), 'planning_failure': failure}
    report['execution_plan_created'] = any(p['exported'] for p in report['plans'].values())
    checkpoint('finished')


def markers(plan, traces):
    """ROS MarkerArray JSON; publish with the existing ros2 topic command."""
    box = plan['obstacles']['cuboid'][0]
    common = {'header': {'frame_id': 'base_link'}, 'ns': 'm1013_comparison', 'action': 0,
              'pose': {'orientation': {'w': 1.}}}
    items = [dict(common, id=0, type=1,
                  pose={'position': dict(zip('xyz', box['pose'][:3])), 'orientation': {'w': 1.}},
                  scale=dict(zip('xyz', box['dims'])), color={'r': 1., 'g': .2, 'b': .2, 'a': .55})]
    for case, paths in enumerate(traces):
        for leg, points in enumerate(paths):
            color = {'r': 1., 'g': .6, 'b': 0., 'a': 1.} if case == 0 else {'r': .1, 'g': 1., 'b': .4, 'a': 1.}
            items.append(dict(common, id=1 + case * 2 + leg, type=4, scale={'x': .006},
                              color=color, points=[dict(zip('xyz', p)) for p in points]))
    return {'markers': items}


def plot(plan, traces, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    box = plan['obstacles']['cuboid'][0]
    center, half = np.asarray(box['pose'][:3]), np.asarray(box['dims']) / 2
    vertices = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * half + center
    faces = [vertices[idx] for idx in [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                                      [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]]
    all_points = np.vstack([np.array(p) for case in traces for p in case] + [vertices])
    lo, hi = all_points.min(axis=0) - .08, all_points.max(axis=0) + .08
    fig = plt.figure(figsize=(12, 5))
    for case, title in enumerate(['Without box: original path', 'With box: cuRobo detour']):
        ax = fig.add_subplot(1, 2, case + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(faces, alpha=.1 if case == 0 else .45,
                                           facecolor='red', edgecolor='red'))
        for leg, points in enumerate(traces[case]):
            p = np.asarray(points)
            ax.plot(*p.T, color='orange' if case == 0 else 'seagreen',
                    linestyle='-' if leg == 0 else '--', label='Outward' if leg == 0 else 'Home')
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
        ax.set_box_aspect(hi - lo); ax.legend()
    fig.suptitle('Same start/goal; outward motion with open gripper\n'
                 'Lines show TCP only; collision checks include the whole arm and gripper')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.log.flush()
        return len(message)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_record(record, case):
    from tiptop.doosan.client import DoosanUONClient
    from tiptop.execute_plan import execute_cutamp_plan

    root = record / 'executions'
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=root))
    result = {'case': case, 'completed_steps': 0, 'sent_to_robot': False, 'status': 'preparing'}
    print(f'Execution records: {output}', flush=True)
    with (output / 'run.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            logging.basicConfig(level=logging.INFO, stream=sys.stderr)
            try:
                plan = json.loads((record / (case.replace('_', '-') + '-plan.json')).read_text())
                result['complete_sequence_planned'] = plan['complete_sequence']
                client = DoosanUONClient()
                actual = np.asarray(client.get_joint_positions())
                expected = np.asarray(plan['q_init'])
                if actual.shape != (6,) or not np.isfinite(actual).all() or np.max(np.abs(actual - expected)) > math.radians(1):
                    raise RuntimeError('Current joints differ from plan start by more than 1 degree; regenerate from current feedback')
                # An HTTP status response does not guarantee a trajectory action server exists.
                check_code = (
                    'import rclpy,sys\n'
                    'from rclpy.action import ActionClient\n'
                    'from control_msgs.action import FollowJointTrajectory\n'
                    'rclpy.init()\n'
                    'node=rclpy.create_node("tiptop_trajectory_readiness_check")\n'
                    'client=ActionClient(node,FollowJointTrajectory,"/dsr01/dsr_moveit_controller/follow_joint_trajectory")\n'
                    'ready=client.wait_for_server(timeout_sec=3.)\n'
                    'print("trajectory_server_ready:",ready)\n'
                    'client.destroy(); node.destroy_node(); rclpy.shutdown()\n'
                    'sys.exit(0 if ready else 1)\n')
                checked = subprocess.run(
                    ['docker', 'exec', '--user', 'ros', 'ros2-dev', 'bash', '-lc',
                     'source /opt/ros/jazzy/setup.bash\npython3 -c "$1"', '_', check_code],
                    capture_output=True, text=True, timeout=15)
                (output / 'controller-check.log').write_text(checked.stdout + checked.stderr)
                if checked.returncode:
                    raise RuntimeError('FollowJointTrajectory unavailable before execution; start the virtual driver and activate dsr_moveit_controller (see docs/m1013-planning.md, step 1)')
                # Publish only after RViz has subscribed: a missing display must not start motion.
                marker_data = json.loads((record / 'obstacle-markers.json').read_text())
                for marker in marker_data['markers']:
                    if marker['id'] == 0 and case == 'without_box':
                        marker['color']['a'] = 0.
                    elif marker['id'] in [1, 3]:
                        marker['color']['a'] = 1. if marker['id'] == (1 if case == 'without_box' else 3) else 0.
                print('Waiting for RViz MarkerArray /uon_model/planning_preview (up to 30 seconds)...', flush=True)
                command = ['docker', 'exec', '--user', 'ros', 'ros2-dev', 'bash', '-lc',
                           'source /opt/ros/jazzy/setup.bash\nros2 topic pub --once -w 1 --max-wait-time-secs 20 /uon_model/planning_preview visualization_msgs/msg/MarkerArray "$1"',
                           '_', json.dumps(marker_data, allow_nan=False)]
                try:
                    published = subprocess.run(command, capture_output=True, text=True, timeout=30)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError('RViz display not ready: Add → By display type → MarkerArray; set Topic to /uon_model/planning_preview, then retry') from error
                (output / 'markers-publish.log').write_text(published.stdout + published.stderr)
                if published.returncode:
                    raise RuntimeError('Box publication failed: check RViz MarkerArray topic /uon_model/planning_preview and markers-publish.log')

                class RecordingClient:
                    def __getattr__(self, name):
                        method = getattr(client, name)
                        def call(*args, **kwargs):
                            result['sent_to_robot'] = True
                            result['status'] = 'running'
                            save_json(output / 'result.json', result)
                            response = method(*args, **kwargs)
                            result['last_result'] = response
                            if response and response.get('success'):
                                result['completed_steps'] += 1
                            save_json(output / 'result.json', result)
                            return response
                        return call

                execute_cutamp_plan(plan['steps'], RecordingClient())
                result['status'] = 'completed' if plan['complete_sequence'] else 'stopped_before_unplanned_return'
                result['planning_failure'] = plan['planning_failure']
                print(json.dumps(result, indent=2))
                return 0
            except Exception as error:
                result.update(status='failed', reason=str(error))
                traceback.print_exc()
                return 1
            finally:
                save_json(output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--offline', action='store_true', help='use the zero model pose instead of live feedback')
    group.add_argument('--input', type=Path, help='retry the recorded start/goal/box from inputs.json')
    group.add_argument('--run', type=Path, metavar='RECORD_DIR', help='execute a recorded plan through the existing virtual gateway')
    parser.add_argument('--case', choices=['without_box', 'with_box'], default='with_box')
    args = parser.parse_args()
    if args.run:
        return run_record(args.run, args.case)
    root = Path('tiptop_outputs/m1013/comparisons')
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=root))
    report = {'outward_comparison_success': False, 'execution_plan_created': False,
              'sent_to_robot': False, 'stage': 'reading_inputs', 'output_dir': str(output)}
    save_json(output / 'report.json', report)
    print(f'Records: {output}', flush=True)
    exit_code = 0
    with (output / 'run.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                if args.input:
                    inputs = json.loads(args.input.read_text())
                else:
                    if args.offline:
                        q = np.zeros(6)
                    else:
                        from tiptop.doosan.client import DoosanUONClient
                        q = np.asarray(DoosanUONClient().get_joint_positions())
                    inputs = {'q_start_rad': q.tolist(),
                              'target_rad': showcase_target(q).tolist(),
                              'seed': 0, 'source': 'offline_zero' if args.offline else 'gateway_feedback',
                              'gripper_outward_rad': 0., 'frame_id': 'base_link'}
                save_json(output / 'inputs.json', inputs)
                compare(inputs, output, report)
            except Exception as error:
                report['failure'] = {'stage': report['stage'], 'error_type': type(error).__name__,
                                     'reason': str(error)}
                traceback.print_exc()
                exit_code = 1
            finally:
                save_json(output / 'report.json', report)
                print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
