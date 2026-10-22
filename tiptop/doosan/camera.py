"""Add a declared camera envelope to a derived planning model, preserving assets.

The caller supplies a profile and output directory. There is no implicit camera
or calibrated mount in the original M1013/UON model. Camera and bracket have
separate collision links so forearm collisions are not lost by inheriting the
wrist's existing adjacency exclusions.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tiptop.perception.cameras.geometry import camera_world_pose


def _box_spheres(dims, centre, margin, cells=6):
    dims, centre = np.asarray(dims, float), np.asarray(centre, float)
    if (
        dims.shape != (3,)
        or centre.shape != (3,)
        or not np.isfinite([dims, centre]).all()
        or np.any(dims <= 0)
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("Camera envelope dimensions must be positive finite metres")
    # Enclose each full box slice, including its corners. These approximate
    # collision volumes, not estimated surface samples or a photoreal CAD mesh.
    step = dims[0] / cells
    radius = np.linalg.norm([step, dims[1], dims[2]]) / 2 + margin
    return [
        dict(center=(centre + [(-dims[0] + step) / 2 + i * step, 0, 0]).tolist(), radius=float(radius))
        for i in range(cells)
    ]


def with_camera_envelope(robot_config, profile, *, urdf_path, output_dir):
    """Return a camera-aware copy of an existing cuRobo robot config.

    Box coordinates use camera rig axes (+X right, +Y down, +Z forward). The
    body origin is the provisional stereo midpoint at its front face, not a
    manufacturer-specified RGB/IR optical offset or a measured hand-eye result.
    """
    cfg = deepcopy(robot_config)
    kin = cfg["robot_cfg"]["kinematics"]
    transform = camera_world_pose(profile["parent_from_rig"], mount="fixed")
    parent = profile["parent_link"]
    source = Path(urdf_path).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    if root.find(f"link[@name='{parent}']") is None:
        raise ValueError(f"Camera parent link not found: {parent}")
    # Derived URDF lives elsewhere: keep all existing mesh references valid.
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if name and "://" not in name and not Path(name).is_absolute():
            mesh.set("filename", str((source.parent / name).resolve()))
    margin = float(profile["collision_margin_m"])
    body_dims = np.asarray(profile["body_dims_m"], float)
    body_centre = np.array([0.0, 0.0, -body_dims[2] / 2])
    parts = [
        ("wrist_camera_body", body_dims, body_centre, parent, transform),
        ("wrist_camera_bracket", profile["bracket_dims_m"], [0.0, 0.0, 0.0], "wrist_camera_body", np.eye(4)),
    ]
    parts[1][4][:3, 3] = profile["rig_from_bracket_xyz_m"]
    for name, dims, centre, parent_name, local in parts:
        if root.find(f"link[@name='{name}']") is not None:
            raise ValueError(f"Camera link already exists: {name}")
        spheres = _box_spheres(dims, centre, margin)
        link = ET.SubElement(root, "link", name=name)
        for kind in ["visual", "collision"]:
            geom = ET.SubElement(link, kind)
            ET.SubElement(geom, "origin", xyz=" ".join(map(str, centre)), rpy="0 0 0")
            ET.SubElement(ET.SubElement(geom, "geometry"), "box", size=" ".join(map(str, dims)))
        joint = ET.SubElement(root, "joint", name=name + "_joint", type="fixed")
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(map(str, local[:3, 3])),
            rpy=" ".join(map(str, Rotation.from_matrix(local[:3, :3]).as_euler("xyz"))),
        )
        kin["collision_link_names"].append(name)
        kin["link_names"].append(name)
        kin["collision_spheres"][name] = spheres
    ignore = kin["self_collision_ignore"]
    ignore["wrist_camera_body"] = [parent, "wrist_camera_bracket"]
    ignore["wrist_camera_bracket"] = [parent]
    kin.get("self_collision_buffer", {}).update(wrist_camera_body=0.0, wrist_camera_bracket=0.0)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "m1013_uon_with_camera.urdf"
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    kin["urdf_path"] = str(destination.resolve())
    return cfg
