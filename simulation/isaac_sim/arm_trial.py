"""Prepare one arm-only J1 +5 degree / return trial; no hardware connection."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trial(path):
    """Check model freshness before starting a simulator trial."""
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan.get("trial") != "lab_arm_only" or plan.get("collision_checked") is not True:
        raise ValueError("Expected a collision-checked lab arm trial")
    for relative, expected in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Trial source changed; prepare again: {relative}")
    from plan_client import validate_plan

    validate_plan(plan)
    if len(plan["steps"]) != 2 or any(s["type"] != "trajectory" for s in plan["steps"]):
        raise ValueError("This trial permits two arm trajectories and no gripper commands")
    return plan


def prepare(output, runtime):
    # Use the actual lab's saved collision geometry, not the empty showcase world.
    from copy import deepcopy
    import torch
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from tiptop.doosan.fixed_base_contact import allow_fixed_base_mounting_contact, check_fixed_base_clearance
    from tiptop.doosan.model import ASSETS, m1013_uon_curobo_cfg
    from tiptop.doosan.virtual_camera import camera_planning_options
    from simulation.isaac_sim.lab_scene import LAB_HOME, LAB_SCENE, load_lab_home

    output.mkdir(parents=True, exist_ok=True)
    # A failed re-check must not leave an older executable trial behind.
    (output / "plan.json").unlink(missing_ok=True)
    (output / "sim-result.json").unlink(missing_ok=True)
    preparation_path = runtime / "camera-preparation.json"
    prepared = json.loads(preparation_path.read_text())
    boxes = camera_planning_options(prepared)["fixed_obstacles"]
    table = Cuboid(name="trial_full_table", dims=[.78, 2.5, .03], pose=[0, 0, -.015, 1, 0, 0, 0])
    # For this free-space trial, keep the entire object area out of the arm path.
    area = np.asarray(prepared["pick_place_bounds_xy_m"])
    protected = Cuboid(name="trial_protected_object_area", dims=[*(area[1]-area[0]), .4],
                       pose=[*((area[0]+area[1])/2), .2, 1, 0, 0, 0])
    world = WorldConfig(cuboid=[table, *boxes, protected])
    home = load_lab_home()
    target = home.copy()
    target[0] += np.deg2rad(5.)
    dt, duration = .1, 5.
    t = np.linspace(0., duration, round(duration/dt)+1)
    u = t / duration
    blend = 10*u**3 - 15*u**4 + 6*u**5
    rate = (30*u**2 - 60*u**3 + 30*u**4) / duration
    steps, samples = [], []
    speed, acceleration, jerk = 0., 0., 0.
    for label, start, end in [("J1 +5 deg", home, target), ("return home", target, home)]:
        q = start + blend[:, None] * (end-start)
        v = rate[:, None] * (end-start)
        # Match the position/velocity cubic interpolation used by both executors.
        stamps = np.arange(len(q)+1)*dt
        curve = CubicHermiteSpline(stamps, np.vstack([q[:1], q]), np.vstack([np.zeros((1,6)), v]))
        dense_t = np.linspace(0., stamps[-1], int(stamps[-1]*240)+1)
        samples.append(curve(dense_t))
        speed = max(speed, float(abs(curve(dense_t, 1)).max()))
        acceleration = max(acceleration, float(abs(curve(dense_t, 2)).max()))
        jerk = max(jerk, float(abs(curve(dense_t, 3)).max()))
        steps.append(dict(type="trajectory", label=label, positions=q.tolist(), velocities=v.tolist(), dt=dt))
    assert speed <= np.deg2rad(2.0) and acceleration <= np.deg2rad(1.3)
    dense = np.concatenate(samples)
    started = time.perf_counter()
    tensor = TensorDeviceType()
    checks = []
    # Test both model endpoints; the physical motor-to-model mapping is still provisional.
    for angle in [0., 1.5009832]:
        solver = IKSolver(IKSolverConfig.load_from_robot_config(
            deepcopy(m1013_uon_curobo_cfg(angle)["robot_cfg"]), world,
            num_seeds=1, use_cuda_graph=False, self_collision_check=True, self_collision_opt=True,
            collision_activation_distance=.003))
        check_fixed_base_clearance(solver.kinematics, home, [*boxes, protected])
        allow_fixed_base_mounting_contact(solver)
        for begin in range(0, len(dense), 256):
            metrics = solver.check_constraints(JointState.from_position(tensor.to_device(dense[begin:begin+256])))
            if not bool(metrics.feasible.all()):
                raise ValueError(f"Collision/joint constraint failed: gripper model {angle}, sample batch {begin}")
        checks.append(angle)
        del solver
        torch.cuda.empty_cache()
    sources = [ASSETS/"m1013_uon.urdf", ASSETS/"m1013_uon.yml", LAB_HOME, LAB_SCENE,
               preparation_path, ROOT/"ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml"]
    plan = dict(version="1.0.0", robot="m1013_uon", trial="lab_arm_only", q_init=home.tolist(),
                steps=steps, collision_checked=True, checked_gripper_model_angles_rad=checks,
                source_sha256={str(p.resolve().relative_to(ROOT)): digest(p) for p in sources},
                collision_check_s=time.perf_counter()-started, samples_checked_per_gripper=len(dense),
                peak_speed_deg_s=float(np.rad2deg(speed)), peak_acceleration_deg_s2=float(np.rad2deg(acceleration)),
                peak_jerk_deg_s3=float(np.rad2deg(jerk)), physical_commands=0,
                note="Arm-only test against the approximate lab model; not a measured hardware picking profile.")
    (output/"plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k,v in plan.items() if k not in {"steps", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"ros2_ws/arm_trial")
    parser.add_argument("--runtime", type=Path, default=ROOT/"tiptop_outputs/m1013/isaac/lab-home-selected-01/runtime")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.runtime.resolve())
