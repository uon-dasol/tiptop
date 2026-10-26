"""Read-only M1013/Orbbec observation acquisition for the common perception path."""

import time
from functools import lru_cache

import numpy as np

from tiptop.perception.cameras.geometry import _rigid_transform


@lru_cache(maxsize=1)
def _robot_urdf():
    from pathlib import Path

    from yourdfpy import URDF

    return URDF.load(str(Path(__file__).parent / "assets/m1013_uon/m1013_uon.urdf"), load_meshes=False)


def world_from_parent(q, parent_link="link_6"):
    """FK in the robot base_link frame, not an assumed external world frame."""
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected six finite robot joints in radians")
    robot = _robot_urdf()
    if parent_link not in robot.link_map:
        raise ValueError(f"Unknown calibration parent link: {parent_link}")
    robot.update_cfg(dict(zip([f"joint_{i}" for i in range(1, 7)], q)) | {"gripper_spinner_joint": 0.0})
    return robot.get_transform(parent_link, "base_link").copy()


def stationary_state(client, *, expected_system=None):
    state = client.get_state()
    if not state.get("success"):
        raise RuntimeError(f"Robot state unavailable: {state.get('error', 'unknown error')}")
    q = np.asarray(state.get("joint_positions"), float)
    if q.shape != (6,) or not np.isfinite(q).all() or state.get("motion_status") != 0:
        raise ValueError("Observation requires fresh finite six-joint feedback and an idle robot")
    if expected_system is not None and state.get("robot_system") != expected_system:
        raise ValueError("Robot system differs from the requested physical/virtual backend; check /state robot_system")
    return q, state


def capture_stationary(
    camera, client, *, parent_from_camera, parent_link="link_6", source="foundation_stereo", expected_system=0
):
    """Bracket a fresh camera frame with idle joint reads; never issue commands.

    A stationary read interval avoids pretending that unrelated camera and ROS
    device clocks are synchronized. Moving captures require a proper shared
    timestamp/interpolation implementation and are rejected here.
    """
    if source not in {"foundation_stereo", "sensor"}:
        raise ValueError("Depth source must be foundation_stereo or sensor")
    transform = _rigid_transform(parent_from_camera, "parent_from_camera")
    before, _ = stationary_state(client, expected_system=expected_system)
    started = time.monotonic()
    frame = camera.read_camera()
    after, state = stationary_state(client, expected_system=expected_system)
    delta = float(np.max(np.abs(after - before)))
    if delta > 0.001:
        raise ValueError("Robot joints changed during camera acquisition; stationary capture rejected")
    packet = dict(
        rgb=frame.rgb.copy(),
        intrinsics=frame.intrinsics.copy(),
        q_init=after,
        world_from_cam=world_from_parent(after, parent_link) @ transform,
        camera_model=f"orbbec/{frame.serial}",
        camera_serial=str(frame.serial),
        timestamp=frame.timestamp,
        world_frame="base_link",
        camera_frame="undistorted_color_optical",
        depth_source=source,
        stationary_joint_delta_rad=delta,
        acquisition_interval_s=time.monotonic() - started,
        robot_system=state.get("robot_system", -1),
    )
    if source == "sensor":
        if frame.depth is None:
            raise ValueError("Sensor depth was selected but the camera depth stream is disabled")
        packet["depth"] = frame.depth.copy()
    else:
        intrinsic = camera.get_intrinsics()
        packet.update(
            ir_left=frame.ir_left.copy(),
            ir_right=frame.ir_right.copy(),
            K_ir=intrinsic.K_ir.copy(),
            baseline_ir=intrinsic.baseline_ir,
            T_color_from_ir=intrinsic.T_color_from_ir.copy(),
        )
    return packet


def validate_camera_calibration(document):
    """A physical colour-camera observation requires an explicitly measured frame."""
    if document.get("calibrated") is not True or document.get("consistent") is not True:
        raise ValueError("A reviewed measured hand-eye calibration is required; candidates are not active calibrations")
    if document.get("world_frame") != "base_link" or document.get("parent_link") != "link_6":
        raise ValueError("Physical observation currently requires base_link / link_6 calibration frames")
    if document.get("camera_frame") != "undistorted_color_optical" or not document.get("camera_serial"):
        raise ValueError("Calibration must identify the undistorted colour optical frame and camera serial")
    return _rigid_transform(document["parent_from_camera"], "parent_from_camera")
