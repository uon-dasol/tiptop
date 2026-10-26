"""Measured hardware model preparation; no device connection or motion commands.

Original URDF/CAD/DB files stay read-only. A derived URDF holds measured mount,
TCP and camera-envelope transforms. Colour optical hand-eye calibration is a
separate transform and must never be substituted with the body-envelope pose.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tiptop.doosan.observation import validate_camera_calibration
from tiptop.perception.cameras.geometry import _rigid_transform


def hardware_profile_fingerprint(profile):
    """Bind a saved physical plan to the reviewed geometry and optical calibration."""
    values = {key: value for key, value in profile.items() if not key.startswith("_")}
    values["resolved_camera_calibration"] = profile["_camera_calibration"]
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_hardware_profile(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get("version") != 1 or document.get("calibrated") is not True:
        raise ValueError("Hardware profile needs version=1 and completed measured calibrations")
    if document.get("length_units") != "metres" or document.get("joint_units") != "radians":
        raise ValueError("Hardware geometry must be metres and robot joints radians")
    tool = document["tool"]
    if tool.get("calibrated") is not True or tool.get("gripper_geometry_matches_reference") is not True:
        raise ValueError("Mounted gripper geometry and DB/model TCP alignment must be measured")
    for key in ["link_6_from_gripper_body", "gripper_body_from_tcp", "db_grasp_from_tcp"]:
        tool[key] = _rigid_transform(tool[key], key).tolist()
    camera = document["camera_profile"]
    if camera.get("parent_link") != "link_6" or camera.get("mount_calibrated") is not True:
        raise ValueError("Measured link_6 camera-body mounting is required")
    camera["parent_from_rig"] = _rigid_transform(camera["parent_from_rig"], "link_6_from_camera_body").tolist()
    for key in ["body_dims_m", "bracket_dims_m", "rig_from_bracket_xyz_m"]:
        value = np.asarray(camera[key], float)
        if (
            value.shape != (3,)
            or not np.isfinite(value).all()
            or (key != "rig_from_bracket_xyz_m" and np.any(value <= 0))
        ):
            raise ValueError(f"Invalid measured camera {key}")
    margin = float(camera["collision_margin_m"])
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Camera collision margin must be finite non-negative metres")
    calibration_path = (path.parent / document["camera_calibration"]).resolve()
    calibration = json.loads(calibration_path.read_text())
    validate_camera_calibration(calibration)
    workspace = document["workspace"]
    if workspace.get("calibrated") is not True or not workspace.get("fixed_obstacles"):
        raise ValueError("Measured full table/workspace obstacles in base_link are required")
    names = set()
    for row in workspace["fixed_obstacles"]:
        name = row["name"]
        dims = np.asarray(row["dims_m"], float)
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or dims.shape != (3,)
            or not np.isfinite(dims).all()
            or np.any(dims <= 0)
        ):
            raise ValueError("Fixed obstacles need unique names and positive metre dimensions")
        _rigid_transform(row["base_from_obstacle"], "base_from_obstacle")
        names.add(name)
    document["_camera_calibration"] = calibration
    document["_profile_path"] = str(path)
    return document


def derive_mount_urdf(source, destination, *, link_6_from_gripper_body, gripper_body_from_tcp):
    """Change only named fixed joints in a new file; retain original collision geometry."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Measured model must be derived into a separate URDF")
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))
    for joint_name, parent, child, matrix in [
        ("gripper_mount_joint", "link_6", "gripper_body", link_6_from_gripper_body),
        ("gripper_tcp_joint", "gripper_body", "gripper_tcp", gripper_body_from_tcp),
    ]:
        transform = _rigid_transform(matrix, joint_name)
        joint = root.find(f"joint[@name='{joint_name}']")
        if (
            joint is None
            or joint.get("type") != "fixed"
            or joint.find("parent").get("link") != parent
            or joint.find("child").get("link") != child
        ):
            raise ValueError(f"Original model does not have the expected {joint_name} frame contract")
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", " ".join(map(str, transform[:3, 3])))
        origin.set("rpy", " ".join(map(str, Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def prepare_hardware_model(profile, output):
    """Build camera-aware FK/collision config and open tool spheres for measured TCP."""
    import yaml
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig

    from tiptop.doosan.camera import with_camera_envelope
    from tiptop.doosan.model import ASSETS, m1013_uon_curobo_cfg

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tool = profile["tool"]
    measured = derive_mount_urdf(
        ASSETS / "m1013_uon.urdf",
        output / "m1013_uon_measured.urdf",
        link_6_from_gripper_body=tool["link_6_from_gripper_body"],
        gripper_body_from_tcp=tool["gripper_body_from_tcp"],
    )
    cfg = deepcopy(m1013_uon_curobo_cfg())
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(measured)
    cfg = with_camera_envelope(cfg, profile["camera_profile"], urdf_path=measured, output_dir=output / "camera")
    tensor = TensorDeviceType()
    kin = CudaRobotModel(RobotConfig.from_dict(deepcopy(cfg["robot_cfg"]), tensor).kinematics)
    state = kin.get_state(tensor.to_device([[0.0] * 6]))
    tcp = state.ee_pose.get_matrix().cpu().numpy()[0]
    indices = []
    for link in ["gripper_body", "gripper_spinner", "gripper_finger1", "gripper_finger2", "gripper_finger3"]:
        indices.extend(kin.kinematics_config.get_sphere_index_from_link_name(link).cpu().tolist())
    spheres = state.link_spheres_tensor[0, indices].cpu().numpy()
    spheres = spheres[spheres[:, 3] > 0].copy()
    inverse = np.linalg.inv(tcp)
    spheres[:, :3] = spheres[:, :3] @ inverse[:3, :3].T + inverse[:3, 3]
    np.save(output / "gripper_spheres_open.npy", spheres)
    (output / "robot.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    fixed = []
    for row in profile["workspace"]["fixed_obstacles"]:
        matrix = np.asarray(row["base_from_obstacle"])
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        fixed.append(Cuboid(name="configured_" + row["name"], dims=row["dims_m"], pose=[*matrix[:3, 3], *quat]))
    return dict(
        robot_config=cfg,
        kinematics=kin,
        gripper_spheres=spheres,
        fixed_obstacles=fixed,
        grasp_from_tcp=np.asarray(tool["db_grasp_from_tcp"]),
    )
