"""Measured hardware model preparation; no device connection or motion commands.

Original URDF/CAD/DB files stay read-only. A derived URDF holds measured mount,
TCP and camera-envelope transforms. Colour optical hand-eye calibration is a
separate transform and must never be substituted with the body-envelope pose.
"""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tiptop.camera_geometry import fixed_camera_envelopes
from tiptop.doosan.observation import validate_camera_calibration
from tiptop.perception.cameras.geometry import _rigid_transform


def hardware_profile_fingerprint(profile):
    """Bind a saved physical plan to the reviewed geometry and optical calibration."""
    values = {key: value for key, value in profile.items() if not key.startswith("_")}
    values["resolved_camera_calibration"] = profile["_camera_calibration"]
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_hardware_profile(path, *, allow_assumed_camera=False, allow_assumed_model=False):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    review = allow_assumed_model and document.get("geometry_source") == "assumed_lab_model"
    if document.get("version") != 1 or (document.get("calibrated") is not True and not review):
        raise ValueError("Hardware profile needs version=1 and completed measured calibrations")
    if document.get("length_units") != "metres" or document.get("joint_units") != "radians":
        raise ValueError("Hardware geometry must be metres and robot joints radians")
    tool = document["tool"]
    if not review and (tool.get("calibrated") is not True or tool.get("gripper_geometry_matches_reference") is not True):
        raise ValueError("Mounted gripper geometry and DB/model TCP alignment must be measured")
    for key in ["link_6_from_gripper_body", "gripper_body_from_tcp", "db_grasp_from_tcp"]:
        tool[key] = _rigid_transform(tool[key], key).tolist()
    calibration_path = (path.parent / document["camera_calibration"]).resolve()
    calibration = json.loads(calibration_path.read_text())
    validate_camera_calibration(calibration, allow_assumed=allow_assumed_camera)
    assumed = calibration.get("pose_source") == "assumed"
    camera = document["camera_profile"]
    if camera.get("parent_link") not in {"link_6", "base_link"} or (
        camera.get("mount_calibrated") is not True and not assumed
    ):
        raise ValueError("Measured link_6 (wrist) or base_link (fixed) camera-body mounting is required")
    mount = "fixed" if camera["parent_link"] == "base_link" else "wrist"
    if camera.setdefault("mount", mount) != mount:
        raise ValueError("Camera mount differs from its parent link")
    camera["parent_from_rig"] = _rigid_transform(camera["parent_from_rig"], "parent_from_rig").tolist()
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
    if calibration["parent_link"] != camera["parent_link"]:
        raise ValueError("Camera optical calibration and body mounting require the same parent link")
    workspace = document["workspace"]
    if (workspace.get("calibrated") is not True and not review) or not workspace.get("fixed_obstacles"):
        raise ValueError("Measured full table/workspace obstacles in base_link are required")
    bounds = workspace.get("pick_place_bounds_xy_m")
    if mount == "fixed" or bounds is not None:
        bounds = np.asarray(bounds, float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Measured fixed-camera pick/place bounds require [[min_x, min_y], [max_x, max_y]]")
    if not isinstance(workspace.get("fixed_base_world_contact", False), bool):
        raise ValueError("fixed_base_world_contact must be an explicit boolean")
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
    if mount == "fixed" and names.intersection({"fixed_camera_body", "fixed_camera_bracket"}):
        raise ValueError("Camera body/bracket obstacles are added automatically; use different structure names")
    document["_camera_calibration"] = calibration
    document["_profile_path"] = str(path)
    return document


def validate_hardware_observation(packet, profile):
    """Keep measured geometry and the observation's optical frame consistent."""
    calibration = profile["_camera_calibration"]
    parent = calibration["parent_link"]
    rotation = calibration.get("image_rotation_deg", 0)
    if (
        str(packet.get("camera_serial", "")) != calibration["camera_serial"]
        or str(packet.get("world_frame", "")) != "base_link"
        or str(packet.get("camera_frame", "")) != calibration["camera_frame"]
        or str(packet.get("camera_parent_link", "link_6")) != parent
        or str(packet.get("camera_pose_source", "measured")) != calibration.get("pose_source", "measured")
        or float(packet.get("image_rotation_deg", 0)) != rotation
        or int(packet.get("robot_system", -1)) != 0
    ):
        raise ValueError("Physical observation camera/robot/base frame, pose source or image rotation differs from the profile")
    if parent == "base_link":
        expected = np.asarray(calibration["parent_from_camera"], float)
        if rotation == 180:
            expected = expected @ np.diag([-1.0, -1.0, 1.0, 1.0])
        actual = _rigid_transform(packet["world_from_cam"], "world_from_cam")
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError("Fixed camera pose differs from the measured optical calibration")


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


HARDWARE_CACHE = Path(__file__).resolve().parents[2] / "tiptop_outputs/m1013/hardware-model-cache"
_PREPARED = {}


def cached_hardware_model(profile, record_dir=None):
    """`prepare_hardware_model` once per profile and process, built in a content-keyed folder that outlives tasks.

    The key covers the reviewed profile and the original model files. `record_dir` receives a copy of the
    generated files so a task folder still shows the model it used.
    """
    from tiptop.doosan.model import ASSETS

    digest = hashlib.sha256(hardware_profile_fingerprint(profile).encode())
    for name in ("m1013_uon.urdf", "m1013_uon.yml"):
        digest.update((ASSETS / name).read_bytes())
    key = digest.hexdigest()[:16]
    folder = HARDWARE_CACHE / key
    if key not in _PREPARED:
        _PREPARED[key] = prepare_hardware_model(profile, folder)
    cached = _PREPARED[key]
    prepared = dict(cached, robot_config=deepcopy(cached["robot_config"]),
                    fixed_obstacles=deepcopy(cached["fixed_obstacles"]), gripper_spheres=cached["gripper_spheres"].copy())
    if record_dir is not None:
        shutil.copytree(folder, record_dir, dirs_exist_ok=True)
    return prepared


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
    if profile["camera_profile"]["mount"] == "wrist":
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
    rows = list(profile["workspace"]["fixed_obstacles"])
    if profile["camera_profile"]["mount"] == "fixed":
        rows.extend(fixed_camera_envelopes(profile["camera_profile"]))
    for row in rows:
        matrix = np.asarray(row["base_from_obstacle"])
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        fixed.append(Cuboid(name="configured_" + row["name"], dims=row["dims_m"], pose=[*matrix[:3, 3], *quat]))
    return dict(
        robot_config=cfg,
        kinematics=kin,
        gripper_spheres=spheres,
        fixed_obstacles=fixed,
        fixed_base_world_contact=profile["workspace"].get("fixed_base_world_contact", False),
        grasp_from_tcp=np.asarray(tool["db_grasp_from_tcp"]),
    )
