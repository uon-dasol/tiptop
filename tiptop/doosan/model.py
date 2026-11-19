"""M1013/UON planning at the existing provisional model TCP.

The arm has six planning joints; spinner and mimicked fingers are locked at the requested angle.
Dataset grasp -> model TCP calibration is still required before object tasks.
"""

import copy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from yourdfpy import URDF

from cutamp.robots import RobotContainer
from cutamp.robots.utils import RerunRobot

from tiptop.robot_limits import load_robot_limits

ASSETS = Path(__file__).parent / 'assets/m1013_uon'
m1013_uon_home = (0.0,) * 6  # Model reference pose, not a calibrated physical home.
OPEN_POSITION = 0.0
CLOSED_POSITION = 1.5009832  # Same model endpoint as the ROS2 gateway.


@lru_cache(maxsize=2)
def _m1013_uon_base_cfg(gripper_position):
    if not OPEN_POSITION <= gripper_position <= CLOSED_POSITION:
        raise ValueError('Gripper angle is outside the model limits')
    cfg = load_yaml(str(ASSETS / 'm1013_uon.yml'))
    cfg['robot_cfg']['kinematics']['lock_joints']['gripper_spinner_joint'] = float(gripper_position)
    cfg['robot_cfg']['kinematics']['external_asset_path'] = str(ASSETS)
    cfg['robot_cfg']['kinematics']['external_robot_configs_path'] = str(ASSETS)
    return cfg


def apply_robot_limits(cfg):
    """Set the six arm joints' acceleration and jerk in `cfg` to the limits currently saved for the planner."""
    cspace = cfg['robot_cfg']['kinematics']['cspace']
    limits = load_robot_limits()
    for key in ('max_acceleration', 'max_jerk'):
        gripper = cspace[key][-1] if isinstance(cspace[key], list) else cspace[key]  # the locked gripper joint
        cspace[key] = [*limits[key], gripper]
    return cfg


def m1013_uon_curobo_cfg(gripper_position=OPEN_POSITION):
    return apply_robot_limits(copy.deepcopy(_m1013_uon_base_cfg(gripper_position)))


def get_m1013_uon_kinematics_model(tensor_args=TensorDeviceType()):
    cfg = RobotConfig.from_dict(m1013_uon_curobo_cfg()['robot_cfg'], tensor_args)
    return CudaRobotModel(cfg.kinematics)


def get_m1013_uon_ik_solver(world_cfg: WorldConfig, num_seeds=12):
    cfg = IKSolverConfig.load_from_robot_config(
        m1013_uon_curobo_cfg()['robot_cfg'], world_cfg, num_seeds=num_seeds,
        self_collision_check=True, self_collision_opt=True)
    return IKSolver(cfg)


def get_m1013_uon_gripper_spheres(tensor_args=TensorDeviceType()):
    return tensor_args.to_device(np.load(ASSETS / 'gripper_spheres_open.npy'))


def load_m1013_uon_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_m1013_uon_kinematics_model(tensor_args)
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"
    # ee_link is the provisional model TCP; no Franka/Robotiq offset is reused.
    tool_from_ee = torch.eye(4, device=tensor_args.device, dtype=tensor_args.dtype)
    return RobotContainer('m1013_uon', kin_model, joint_limits,
                          get_m1013_uon_gripper_spheres(tensor_args), tool_from_ee)


def load_m1013_uon_rerun(load_mesh=True):
    # yourdfpy derives mesh_dir automatically only for a string filename.
    urdf = URDF.load(str(ASSETS / 'm1013_uon.urdf'))
    return RerunRobot('m1013_uon', urdf, q_neutral=[*m1013_uon_home, 0.0], load_mesh=load_mesh)
