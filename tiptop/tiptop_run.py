import asyncio
import json
import logging
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import open3d as o3d
import rerun as rr
import trimesh
import tyro
from curobo.geom.types import Cuboid, Mesh
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.tamp_domain import HandEmpty, Holding, Near, On
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun
from jaxtyping import Bool, Float
from scipy.spatial import KDTree

from tiptop.config import load_calibration, tiptop_cfg
from tiptop.execute_plan import execute_cutamp_plan
from tiptop.motion_planning import build_curobo_solvers, go_to_capture
from tiptop.perception.cad_cache import load_cad_mesh
from tiptop.perception.cameras import (
    Camera,
    DepthEstimator,
    Frame,
    ZedCamera,
    get_depth_estimator,
    get_external_camera,
    get_hand_camera,
)
from tiptop.perception.cameras.geometry import camera_world_pose
from tiptop.perception.m2t2 import generate_grasps_async, m2t2_to_tiptop_transform
from tiptop.perception.placement import (
    TABLE_BOX_LOWERING_M,
    TABLE_SUPPORT_MODES,
    check_support_gap,
    table_support_z,
)
from tiptop.perception.sam2 import sam2_client
from tiptop.perception.segmentation import (
    masked_object_points,
    segment_pointcloud_by_masks,
    segment_table_with_ransac,
)
from tiptop.perception.shape_completion import reconstruct_objects_with_recgen
from tiptop.perception.utils import convert_trimesh_box_to_curobo_cuboid, convert_trimesh_to_curobo_mesh
from tiptop.perception_wrapper import run_visual_perception
from tiptop.planning import build_tamp_config, run_planning, save_tiptop_plan, serialize_plan
from tiptop.recording import (
    record_cameras,
    save_perception_outputs,
    save_run_metadata,
    save_run_outputs,
)
from tiptop.utils import (
    NumpyEncoder,
    RobotClient,
    add_file_handler,
    check_cutamp_version,
    get_robot_client,
    get_robot_rerun,
    load_gripper_mask,
    print_tiptop_banner,
    remove_file_handler,
    setup_logging,
)
from tiptop.viz_utils import get_gripper_mesh, get_heatmap
from tiptop.workspace import workspace_cuboids

_log = logging.getLogger(__name__)
tensor_args = TensorDeviceType()

_executor_pool = None


class UserExitException(Exception):
    """Raised when user explicitly requests to exit."""


@dataclass(frozen=True)
class Observation:
    """Snapshot of sensor data and robot state needed for one perception+planning run."""

    frame: Frame
    world_from_cam: Float[np.ndarray, "4 4"]
    q_init: Float[np.ndarray | list, "n"]


@dataclass(frozen=True)
class _DemoContainer:
    """Container for storing things needed for the live robot demo."""

    robot: RobotClient
    cam: Camera
    external_cam: Camera | None
    enable_recording: bool
    camera_parent_from_cam: Float[np.ndarray, "4 4"]
    depth_estimator: DepthEstimator | None

    gripper_mask: Bool[np.ndarray, "h w"] | None

    ik_solver: IKSolver
    motion_gen: MotionGen
    camera_mount: str = "wrist"


@dataclass
class ProcessedScene:
    """Processed 3D scene ready for TAMP."""

    table_cuboid: Cuboid
    object_meshes: dict[str, Mesh]
    object_pcds: dict[str, o3d.geometry.PointCloud]
    grasps: dict[str, dict]  # Label -> grasp data with tensor versions
    perception_results: dict | None = None
    table_support: dict | None = None  # Mode, support height and per-target CAD-bottom gap of known-DB scenes


def capture_live_observation(container: _DemoContainer) -> Observation:
    """Read current robot state and a wrist/fixed camera into one Observation."""
    q_curr = container.robot.get_joint_positions()
    world_from_ee = None
    if container.camera_mount == "wrist":
        q_curr_pt = tensor_args.to_device(q_curr)
        world_from_ee = container.motion_gen.kinematics.get_state(q_curr_pt).ee_pose.get_numpy_matrix()[0]
    world_from_cam = camera_world_pose(
        container.camera_parent_from_cam, mount=container.camera_mount, world_from_ee=world_from_ee
    )
    frame = container.cam.read_camera()
    return Observation(frame=frame, world_from_cam=world_from_cam, q_init=q_curr)


def get_demo_container(
    num_particles: int, num_spheres: int, collision_activation_distance: float, enable_recording: bool = False
) -> _DemoContainer:
    """Cache and warm-up everything needed for the live demo."""
    _log.info("Starting demo warmup...")
    cfg = tiptop_cfg()
    camera_mount = cfg.cameras.hand.get("mount", "wrist")
    depth_source = cfg.perception.get("depth_source", "foundation_stereo")
    if camera_mount not in {"wrist", "fixed"}:
        raise ValueError("cameras.hand.mount must be 'wrist' or 'fixed'")
    if depth_source not in {"sensor", "foundation_stereo"}:
        raise ValueError("perception.depth_source must be 'sensor' or 'foundation_stereo'")
    client = get_robot_client()

    # Setup cameras
    cam = get_hand_camera(depth=depth_source == "sensor")
    external_cam = get_external_camera() if enable_recording else None
    camera_parent_from_cam = load_calibration(cam.serial)

    # External camera for recording (if enabled)
    if enable_recording:
        if not isinstance(cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED hand camera, got {type(cam).__name__}")
        if not isinstance(external_cam, ZedCamera):
            raise NotImplementedError(f"Recording requires a ZED external camera, got {type(external_cam).__name__}")

    # Check perception servers are all up before slow cuRobo warmups
    async def _check() -> None:
        async with aiohttp.ClientSession() as session:
            await check_server_health(session)

    asyncio.run(_check())

    # Create depth estimator once — closed over camera intrinsics
    # Cache the SAM2 client
    if cfg.perception.get("mode", "gemini_er") == "gemini_er":
        sam2_client()

    # Warm-up IK solver and motion generator
    ik_solver, motion_gen, _ = build_curobo_solvers(num_particles, num_spheres, collision_activation_distance)
    return _DemoContainer(
        robot=client,
        cam=cam,
        external_cam=external_cam,
        enable_recording=enable_recording,
        camera_parent_from_cam=camera_parent_from_cam,
        camera_mount=camera_mount,
        depth_estimator=get_depth_estimator(cam, source=depth_source),
        gripper_mask=load_gripper_mask() if camera_mount == "wrist" else None,
        ik_solver=ik_solver,
        motion_gen=motion_gen,
    )


async def check_server_health(session: aiohttp.ClientSession):
    """Check the configured perception services; sensor depth needs no depth server."""
    if tiptop_cfg().perception.get("mode", "gemini_er") == "sam3_reference":
        from tiptop.perception.sam3_reference import check_reference_health

        await check_reference_health(session, tiptop_cfg().perception.sam3.url)
    from tiptop.perception.foundation_stereo import check_health_status as fs_check_health_status
    from tiptop.perception.m2t2 import check_health_status as m2t2_check_health_status
    from tiptop.perception.recgen import check_health_status as recgen_check_health_status

    cfg = tiptop_cfg()
    depth_source = cfg.perception.get("depth_source", "foundation_stereo")
    if depth_source not in {"sensor", "foundation_stereo"}:
        raise ValueError("perception.depth_source must be 'sensor' or 'foundation_stereo'")
    provider = cfg.perception.get("grasp_provider", "m2t2")
    if provider == "m2t2":
        health_checks = [m2t2_check_health_status(session, cfg.perception.m2t2.url)]
    elif provider == "known_db":

        async def check_pose():
            url = cfg.perception.get("foundation_pose_url", "http://127.0.0.1:13346")
            async with session.get(url.rstrip("/") + "/health") as response:
                response.raise_for_status()

        health_checks = [check_pose()]
    else:
        raise ValueError(f"Unknown grasp provider: {provider}")
    if depth_source == "foundation_stereo":
        health_checks.append(fs_check_health_status(session, cfg.perception.foundation_stereo.url))
    if cfg.perception.recgen.enabled:
        health_checks.append(recgen_check_health_status(session, cfg.perception.recgen.url))
    await asyncio.gather(*health_checks)
    _log.info("Server health checks successful!")


def _label_rollout(save_dir: Path, output_dir: str, date_str: str, timestamp: str) -> None:
    """Prompt user to label rollout as success/failure, moving it out of eval/. Loops on invalid input."""
    try:
        while True:
            user_input = (
                input(
                    "\nWas the execution successful? Enter 'y' for success, 'n' for failure, or leave empty to skip: "
                )
                .strip()
                .lower()
            )
            if user_input == "y":
                dest = Path(output_dir) / "success" / date_str / timestamp
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(save_dir, dest)
                _log.info(f"Moved rollout to success directory: {dest}")
                return
            elif user_input == "n":
                dest = Path(output_dir) / "failure" / date_str / timestamp
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(save_dir, dest)
                _log.info(f"Moved rollout to failure directory: {dest}")
                return
            elif user_input == "":
                _log.info(f"Keeping rollout in eval directory: {save_dir}")
                return
            else:
                print("Invalid input. Please enter 'y', 'n', or leave empty to skip.")
    except EOFError:
        _log.info("No input received, keeping rollout in eval directory")


def _get_task_instruction() -> str:
    task_instruction = ""
    while not task_instruction:
        try:
            task_instruction = input(
                "Enter task instruction (e.g., 'place the red cup on the table', or 'exit' to quit): "
            ).strip()
            if task_instruction.lower() == "exit":
                raise UserExitException("User requested exit")
        except KeyboardInterrupt:
            raise UserExitException("User interrupted with Ctrl+C")

    return task_instruction


def create_tamp_environment(
    object_meshes: dict[str, Mesh], table_cuboid: Cuboid, grounded_atoms: list[dict], include_workspace: bool
) -> tuple[TAMPEnvironment, list[Cuboid | Mesh]]:
    # Reject goals that reference objects not present in the perceived scene.
    # Without this, cuTAMP's BFS runs without stopping, expanding the move-chain on an unreachable goal.
    known_labels = set(object_meshes.keys()) | {table_cuboid.name}
    for atom in grounded_atoms:
        for arg in atom.get("args", []):
            if arg not in known_labels:
                raise ValueError(
                    f"Goal predicate {atom['predicate']}({', '.join(atom['args'])}) "
                    f"references unknown object '{arg}'. Known objects: {sorted(known_labels)}"
                )

    # Identify which objects are used as surfaces (second arg in on(x, y))
    surface_labels = set()
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            surface_labels.add(atom["args"][1])

    # Separate movables and surfaces
    movables = []
    surfaces = []
    for label, mesh in object_meshes.items():
        if label in surface_labels:
            surfaces.append(mesh)
        else:
            movables.append(mesh)
    _log.info(f"Movables: {[m.name for m in movables]}")
    _log.info(f"Surfaces: {[s.name for s in surfaces]}")

    # Create goal state from grounded atoms
    goal_state: set = set()
    has_holding = False
    for atom in grounded_atoms:
        if atom["predicate"] == "on" and len(atom["args"]) == 2:
            movable_label, surface_label = atom["args"]
            goal_state.add(On.ground(movable_label, surface_label))
            _log.info(f"Goal: {movable_label} on {surface_label}")
        elif atom["predicate"] == "holding" and len(atom["args"]) == 1:
            has_holding = True
            movable_label = atom["args"][0]
            goal_state.add(Holding.ground(movable_label))
            _log.info(f"Goal: holding {movable_label}")
        elif atom["predicate"] == "near" and len(atom["args"]) == 2:
            # KNOWN LIMITATION: if reference_label is a surface (e.g. "put X in the bowl and Y
            # next to the bowl"), cuTAMP rejects the goal because PlaceNear's reference must be a
            # movable. See https://github.com/tiptop-robot/cuTAMP/issues/26
            movable_label, reference_label = atom["args"]
            goal_state.add(Near.ground(movable_label, reference_label))
            _log.info(f"Goal: {movable_label} near {reference_label}")
        else:
            _log.warning(f"Ignoring unexpected grounded atom from Gemini: {atom}")
    if not has_holding:
        goal_state.add(HandEmpty.ground())

    # All surfaces include table and detected surface objects
    all_surfaces = [table_cuboid, *surfaces]
    statics = list(workspace_cuboids()) if include_workspace else []
    for surface in all_surfaces:
        statics.append(surface)

    # Create TAMP environment
    env = TAMPEnvironment(
        name="tiptop_cutamp",
        movables=movables,
        statics=statics,
        type_to_objects={"Movable": movables, "Surface": all_surfaces},
        goal_state=frozenset(goal_state),
    )
    _log.info(f"Created TAMP environment with {len(movables)} movables, {len(all_surfaces)} surfaces")
    return env, all_surfaces


def declared_table_cuboid(declared: Cuboid) -> Cuboid:
    """Table box from a configured table, lowered like the RANSAC box so both feed the same support-plane logic."""
    pose, dims = np.asarray(declared.pose, float), np.asarray(declared.dims, float)
    if abs(pose[3]) < 1 - 1e-9:
        raise ValueError(f"Configured table {declared.name} must be axis-aligned")
    return Cuboid(name="table", dims=dims.tolist(), pose=[pose[0], pose[1], pose[2] - TABLE_BOX_LOWERING_M, 1, 0, 0, 0])


def process_scene_geometry(
    xyz_map: np.ndarray,
    rgb_map: np.ndarray,
    masks: np.ndarray,
    bboxes: list,
    grasps: dict,
    recgen_meshes: dict[str, trimesh.Trimesh] | None = None,
    *,
    registered_grasps: dict | None = None,
    table_support: str = "ransac-cad",
    declared_table: Cuboid | None = None,
) -> ProcessedScene:
    """Process perception results into 3D scene geometry for TAMP.

    Args:
        xyz_map: World-space XYZ coordinates (H, W, 3)
        rgb_map: RGB image (H, W, 3) in 0-1 range
        masks: Segmentation masks from SAM2
        bboxes: Bounding boxes from Gemini
        grasps: Grasp predictions from M2T2
        recgen_meshes: Per-object meshes in world frame (e.g. from RecGen). When provided, replaces the convex-hull path.
        table_support: "ransac-cad" scores RANSAC planes with the posed-CAD bottoms (default), "ransac-mask" with the
            mask-bottom points, "declared" skips RANSAC and uses `declared_table`, which then also needs the CAD-bottom
            gap check that the 3 cm RANSAC contact threshold otherwise provides.

    Returns:
        ProcessedScene with table cuboid, object meshes, pcds, and filtered grasps
    """
    if table_support not in TABLE_SUPPORT_MODES:
        raise ValueError(f"Unknown table support {table_support!r}; choose one of {TABLE_SUPPORT_MODES}")
    if table_support == "declared":
        if declared_table is None:
            raise ValueError("Table support 'declared' needs a configured full table")
        table_cuboid = declared_table_cuboid(declared_table)
        table_box_top_z = float(table_cuboid.pose[2] + table_cuboid.dims[2] / 2)
    else:
        # Segment table with RANSAC (returns trimesh Box)
        table_options = {}
        if (
            table_support == "ransac-cad"
            and registered_grasps
            and all("support_point_world" in item for item in registered_grasps.values())
        ):
            table_options["support_points_world"] = np.stack(
                [item["support_point_world"] for item in registered_grasps.values()]
            )
        table_trimesh = segment_table_with_ransac(xyz_map, rgb_map, masks, **table_options)
        table_cuboid = convert_trimesh_box_to_curobo_cuboid(table_trimesh, name="table")
        table_box_top_z = float(table_trimesh.bounds[1, 2])
    log_curobo_mesh_to_rerun("world/table", table_cuboid.get_mesh(), static_transform=True)

    # For filtering to table plane height
    config = TAMPConfiguration()
    table_top_z = table_box_top_z + config.world_activation_distance + config.coll_sphere_radius * 2
    if recgen_meshes is not None:
        # Derive each object's point cloud from its mask, then apply the same table-relative z-filter + outlier removal
        # the convex-hull path uses. Drop objects with too few above-table points. Keep object_trimeshes in sync with
        # the surviving labels so downstream loops don't KeyError.
        erode_pixels = tiptop_cfg().perception.mask_erosion_pixels
        masks_2d = masks.squeeze(1).astype(bool)
        object_pcds = {}
        for mask_2d, bbox in zip(masks_2d, bboxes):
            label = bbox["label"]
            if label not in recgen_meshes:
                _log.warning(f"Skipping {label}: no RecGen mesh for this label")
                continue
            points = masked_object_points(mask_2d, xyz_map, rgb_map, erode_pixels, label)
            if points is None:
                continue
            _, xyz_obj, rgb_obj = points
            keep = xyz_obj[:, 2] > table_top_z
            if keep.sum() <= 10:
                _log.warning(f"Skipping {label}: {keep.sum()} points above table (need > 10)")
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_obj[keep])
            pcd.colors = o3d.utility.Vector3dVector(rgb_obj[keep])
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            object_pcds[label] = pcd
        object_trimeshes = {label: mesh for label, mesh in recgen_meshes.items() if label in object_pcds}
    else:
        object_trimeshes, object_pcds = segment_pointcloud_by_masks(
            xyz_map,
            rgb_map,
            masks,
            bboxes,
            table_top_z,
            return_pcd=True,
            erode_pixels=tiptop_cfg().perception.mask_erosion_pixels,
        )

    if not object_pcds:
        raise ValueError("No objects with sufficient point cloud data; cannot associate grasps.")

    if registered_grasps is not None:
        # DB output already contains model TCP poses in the world frame. Applying
        # the Panda/M2T2 correction below would corrupt those transforms.
        object_meshes = {label: convert_trimesh_to_curobo_mesh(mesh, label) for label, mesh in object_trimeshes.items()}
        support_z = table_support_z(table_box_top_z)
        if table_support == "ransac-cad":
            # The RANSAC contact threshold already rejects far-off poses; the gap is only recorded.
            support_gaps = {
                label: float(item["support_point_world"][2] - support_z)
                for label, item in registered_grasps.items()
                if "support_point_world" in item
            }
        else:
            support_gaps = {
                label: check_support_gap(label, item["support_point_world"][2], support_z)
                for label, item in registered_grasps.items()
            }
        table_info = dict(mode=table_support, support_z_m=support_z, cad_bottom_gap_m=support_gaps)
        supplied = {}
        for label, result in registered_grasps.items():
            if label not in object_meshes:
                raise ValueError(f"Registered target {label} has no observed collision geometry")
            world_from_obj = np.eye(4)
            world_from_obj[:3, 3] = object_meshes[label].pose[:3]
            data = result["grasps"]
            supplied[label] = dict(
                grasps_obj=tensor_args.to_device(np.linalg.inv(world_from_obj) @ data["world_from_tcp"]),
                confidences_pt=tensor_args.to_device(data["scores"]),
                registered=result,
            )
        return ProcessedScene(table_cuboid, object_meshes, object_pcds, supplied, table_support=table_info)

    # Associate grasps with objects by checking contact point proximity
    # Build a single KDTree from all object points with label tracking
    obj_labels = list(object_pcds.keys())
    all_points = []
    point_to_label = []  # Maps each point index to its object label
    for label, pcd in object_pcds.items():
        obj_points = np.asarray(pcd.points)
        all_points.append(obj_points)
        point_to_label.extend([label] * len(obj_points))

    all_points = np.vstack(all_points)
    point_to_label = np.array(point_to_label)
    combined_kdtree = KDTree(all_points)

    # Re-associate grasps to objects based on contact point proximity
    # Collect all valid grasps in flat arrays first
    all_poses, all_confs, all_contacts, all_labels = [], [], [], []
    for _, grasp_dict in grasps.items():
        poses, confs, contacts = grasp_dict["poses"], grasp_dict["confidences"], grasp_dict["contacts"]
        if len(contacts) == 0:
            continue

        dists, nearest_idxs = combined_kdtree.query(contacts)
        nearest_labels = point_to_label[nearest_idxs]
        within_thresh = dists < tiptop_cfg().perception.contact_threshold_m
        all_poses.append(poses[within_thresh])
        all_confs.append(confs[within_thresh])
        all_contacts.append(contacts[within_thresh])
        all_labels.append(nearest_labels[within_thresh])

    # Group by object label using boolean masks
    filtered_grasps = {}
    if all_poses:
        all_poses = np.concatenate(all_poses)
        all_confs = np.concatenate(all_confs)
        all_contacts = np.concatenate(all_contacts)
        all_labels = np.concatenate(all_labels)

        for label in obj_labels:
            mask = all_labels == label
            filtered_grasps[label] = {
                "poses": all_poses[mask],
                "confidences": all_confs[mask],
                "contacts": all_contacts[mask],
            }
            count = mask.sum()
            if count > 0:
                _log.info(
                    f"Object {label}: Associated {count} grasps (within {tiptop_cfg().perception.contact_threshold_m * 100:.1f}cm)"
                )
            else:
                _log.warning(f"Object {label}: No grasps within threshold")
    else:
        for label in obj_labels:
            filtered_grasps[label] = {
                "poses": np.array([]).reshape(0, 4, 4),
                "confidences": np.array([]),
                "contacts": np.array([]).reshape(0, 0, 3),
            }
            _log.warning(f"Object {label}: No grasps within threshold")

    gripper_mesh = get_gripper_mesh()
    vertices = np.asarray(gripper_mesh.vertices)
    vertices_hom = np.c_[vertices, np.ones(len(vertices))]  # Add homogeneous coordinate
    faces = np.asarray(gripper_mesh.triangles)
    viz_grasp_dur = 0.0

    # Convert trimesh objects to cuRobo meshes and log to Rerun
    object_meshes = {}
    for label, trimesh_obj in object_trimeshes.items():
        curobo_mesh = convert_trimesh_to_curobo_mesh(trimesh_obj, label)
        object_meshes[label] = curobo_mesh
        label_clean = label.replace(" ", "-")
        log_curobo_mesh_to_rerun(f"world/objects/{label_clean}", curobo_mesh.get_mesh(), static_transform=True)

        # Log the point cloud
        pcd = object_pcds[label]
        rr.log(f"obj_pcd/{label_clean}", rr.Points3D(positions=pcd.points, colors=pcd.colors))

        # Transform grasps to tcp frame
        grasp_dict = filtered_grasps[label]
        world_from_obj = np.eye(4)
        curobo_pose = np.array(curobo_mesh.pose)
        assert np.allclose(curobo_pose[3:], np.array([1.0, 0.0, 0.0, 0.0]))
        world_from_obj[:3, 3] = curobo_pose[:3]
        obj_from_world = np.linalg.inv(world_from_obj)

        world_from_grasp = grasp_dict["poses"] @ m2t2_to_tiptop_transform()
        obj_from_grasp = obj_from_world @ world_from_grasp
        filtered_grasps[label]["grasps_obj"] = tensor_args.to_device(obj_from_grasp)
        filtered_grasps[label]["confidences_pt"] = tensor_args.to_device(filtered_grasps[label]["confidences"])

        if len(world_from_grasp) == 0:
            continue

        # Visualize the resulting grasps
        viz_start = time.perf_counter()
        my_vertices_hom = vertices_hom.copy()

        # Convert to tiptop convention and select top grasps
        grasp_poses = world_from_grasp[:30]
        confidences = filtered_grasps[label]["confidences"][:30]
        transformed_verts = np.einsum("nij,mj->nmi", grasp_poses, my_vertices_hom)[..., :3]
        colors = get_heatmap(confidences)

        for grasp_idx, (verts, color) in enumerate(zip(transformed_verts, colors)):
            rr.log(
                f"grasps/{label}/{grasp_idx:04d}",
                rr.Mesh3D(
                    vertex_positions=verts, triangle_indices=faces, vertex_colors=np.tile(color, (len(verts), 1))
                ),
                static=True,
            )
        viz_grasp_dur += time.perf_counter() - viz_start

    _log.info(f"Visualizing grasps took: {viz_grasp_dur:.2f}s")
    return ProcessedScene(
        table_cuboid=table_cuboid,
        object_meshes=object_meshes,
        object_pcds=object_pcds,
        grasps=filtered_grasps,
    )


async def run_perception(
    session: aiohttp.ClientSession,
    observation: Observation,
    task_instruction: str,
    save_dir: Path,
    depth_estimator: DepthEstimator | None = None,
    gripper_mask: Bool[np.ndarray, "h w"] | None = None,
    include_workspace: bool = True,
    log_to_rerun: bool = True,
    *,
    grasp_provider: str | None = None,
    client=None,
    active_task=None,
    in_place_return=False,
    registered_grasp_from_tcp=None,
    perception_mode=None,
    sam3_url=None,
    reference_scene=None,
    pick_place_bounds_xy=None,
    cad_mask_min_iou=0.75,
    preview=None,
    background_tasks: list | None = None,
    table_support: str = "ransac-cad",
    declared_table: Cuboid | None = None,
) -> tuple[TAMPEnvironment, list, ProcessedScene, list[dict]]:
    """Run perception. With `background_tasks`, saving outputs is appended there instead of awaited."""
    if background_tasks is not None and log_to_rerun:
        raise ValueError("log_to_rerun needs the saved visualizations, so it cannot defer saving")
    start_time = time.perf_counter()

    cfg = tiptop_cfg()
    frame = observation.frame
    rgb = frame.rgb
    provider = grasp_provider or cfg.perception.get("grasp_provider", "m2t2")
    if provider not in {"m2t2", "known_db"}:
        raise ValueError(f"Unknown grasp provider: {provider}")
    if log_to_rerun:
        rr.log("rgb", rr.Image(rgb))

    object_context = None
    if provider == "known_db":
        from tiptop.perception.known_objects import known_object_detection_context

        object_context = known_object_detection_context()
    # Share the RGB/depth/mask front end with simulation and future CAD-pose grasps.
    results = await run_visual_perception(
        session,
        frame,
        observation.world_from_cam,
        task_instruction,
        depth_estimator=depth_estimator,
        gripper_mask=gripper_mask,
        client=client,
        active_task=active_task,
        object_context=object_context,
        perception_mode=perception_mode,
        sam3_url=sam3_url,
        reference_scene=reference_scene,
        on_update=None if preview is None else preview.update,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "semantic-task.json").write_text(
        json.dumps(
            dict(
                instruction=task_instruction,
                bboxes=results["bboxes"],
                grounded_atoms=results["grounded_atoms"],
                semantic_source=results["semantic_source"],
                reference_diagnostics=results.get("reference_diagnostics"),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    registered_grasps = None
    if provider == "m2t2":
        results["grasps"] = await generate_grasps_async(
            session,
            cfg.perception.m2t2.url,
            scene_xyz=results["xyz_downsampled"],
            scene_rgb=results["rgb_downsampled"],
            apply_bounds=cfg.perception.m2t2.apply_bounds,
        )
    elif provider == "known_db":
        from tiptop.perception.known_objects import estimate_registered_grasps, registered_target

        label, entry = registered_target(results["bboxes"], results["grounded_atoms"])
        registered_grasps = {
            label: await estimate_registered_grasps(
                session,
                frame,
                results,
                observation.world_from_cam,
                entry=entry,
                label=label,
                server_url=cfg.perception.get("foundation_pose_url", "http://127.0.0.1:13346"),
                world_from_planner_object=np.eye(4),
                grasp_from_tcp=np.eye(4) if registered_grasp_from_tcp is None else registered_grasp_from_tcp,
                cad_mask_min_iou=cad_mask_min_iou,
            )
        }
        results["grasps"] = {}
        (save_dir / "registered-pose.json").write_text(
            json.dumps(
                dict(
                    label=label,
                    class_id=entry["class_id"],
                    pose=registered_grasps[label]["pose"],
                    support_point_world=registered_grasps[label]["support_point_world"],
                ),
                cls=NumpyEncoder,
                indent=2,
            )
            + "\n"
        )
        if preview is not None:
            await asyncio.to_thread(
                preview.grasps, registered_grasps[label]["grasps"], observation.world_from_cam, frame.intrinsics,
                max_tilt_deg=preview.max_tilt_deg,
            )
    else:
        raise ValueError(f"Unknown grasp provider: {provider}")
    _log.info(f"Capturing observation and running perception APIs took {time.perf_counter() - start_time:.2f}s")

    # Save results (ProcessPoolExecutor for live mode, default thread pool for h5 mode)
    loop = asyncio.get_running_loop()
    save_future = loop.run_in_executor(
        _executor_pool,
        save_perception_outputs,
        rgb,
        frame.intrinsics,
        results["depth_map"],
        results["xyz_map"],
        results["rgb_map"],
        results["bboxes"],
        results["masks"],
        save_dir,
        gripper_mask,
    )

    if log_to_rerun:
        rr.log(
            "pcd",
            rr.Points3D(positions=results["xyz_map"].reshape(-1, 3), colors=results["rgb_map"].reshape(-1, 3)),
        )

    # Reconstruct objects with RecGen (when enabled).
    recgen_meshes: dict[str, trimesh.Trimesh] | None = None
    if cfg.perception.recgen.enabled:
        recgen_meshes = await reconstruct_objects_with_recgen(
            session,
            cfg.perception.recgen.url,
            rgb_cam=rgb,
            depth_cam=results["depth_map"],
            masks=results["masks"],
            bboxes=results["bboxes"],
            intrinsics=frame.intrinsics,
            world_from_cam=observation.world_from_cam,
            target_faces=cfg.perception.recgen.target_faces,
            concurrency=cfg.perception.recgen.concurrency,
        )

    # Run scene geometry processing while saving
    proc_st = time.perf_counter()
    process_coroutine = asyncio.to_thread(
        process_scene_geometry,
        results["xyz_map"],
        results["rgb_map"],
        results["masks"],
        results["bboxes"],
        results["grasps"],
        recgen_meshes=recgen_meshes,
        registered_grasps=registered_grasps,
        table_support=table_support,
        declared_table=declared_table,
    )
    if background_tasks is None:
        processed_scene, save_result = await asyncio.gather(process_coroutine, save_future)
    else:
        background_tasks.append(save_future)
        processed_scene = await process_coroutine
    processed_scene.perception_results = results

    if in_place_return:
        if provider != "known_db":
            raise ValueError("In-place return requires a known CAD target")
        from tiptop.doosan.in_place import source_region
        from tiptop.perception.known_objects import registered_target, require_return_in_place

        if results["semantic_source"] == "gemini":
            require_return_in_place(results["grounded_atoms"])
        label, entry = registered_target(results["bboxes"], results["grounded_atoms"])
        estimate = processed_scene.grasps[label]["registered"]["pose"]["world_from_cad"]
        vertices = load_cad_mesh(entry["cad"], entry["cad_scale_to_m"]).vertices
        vertices = vertices @ estimate[:3, :3].T + estimate[:3, 3]
        table_bounds = (np.asarray(processed_scene.table_cuboid.pose[:3])
                        + np.array([-.5, .5])[:, None]*np.asarray(processed_scene.table_cuboid.dims))
        results["placement_region"] = source_region(table_bounds, vertices, pick_place_bounds_xy)
        results["placement_region"]["table_support"] = processed_scene.table_support
        if preview is not None:
            await asyncio.to_thread(preview.side_scene, vertices, results["placement_region"]["support_z_m"], observation.world_from_cam)
        results["grounded_atoms"] = [{"predicate": "holding", "args": [label]}]
        (save_dir / "placement-region.json").write_text(json.dumps(results["placement_region"], indent=2)+"\n")

    if log_to_rerun:
        bbox_viz, masks_viz = save_result
        rr.log("bboxes", rr.Image(bbox_viz))
        rr.log("masks", rr.Image(masks_viz))

    env, all_surfaces = create_tamp_environment(
        processed_scene.object_meshes,
        processed_scene.table_cuboid,
        results["grounded_atoms"],
        include_workspace,
    )
    _log.info(f"Processing scene and perception results took {time.perf_counter() - proc_st:.2f}s")
    _log.info(f"Perception pipeline completed, took {time.perf_counter() - start_time:.2f}s")
    return env, all_surfaces, processed_scene, results["grounded_atoms"]


async def async_entrypoint(container: _DemoContainer, config: TAMPConfiguration, output_dir: str, execute_plan: bool):
    """Main async entrypoint for the live robot demo."""
    cfg = tiptop_cfg()

    # Force TCP handshake for every request
    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                _log.debug(f"Preparing TiPToP for next run...")
                await check_server_health(session)

                # Go to capture pose and ask user for instruction
                if container.camera_mount == "wrist":
                    _log.debug("Moving robot to capture joint positions")
                    go_to_capture(time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=container.motion_gen)
                container.robot.open_gripper()  # ensure gripper is open
                task_instruction = _get_task_instruction()  # Let UserExitException propagate
                _log.info(f"User entered instruction: {task_instruction}")

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                iso_timestamp = now.isoformat(timespec="seconds")
                date_str = now.strftime("%Y-%m-%d")
                rr.init("tiptop_run", recording_id=timestamp, spawn=True)
                # Log workspace for visualization purposes
                robot_rr = get_robot_rerun()
                for obj in workspace_cuboids():
                    log_curobo_mesh_to_rerun(f"world/workspace/{obj.name}", obj.get_mesh(), static_transform=True)

                save_dir = Path(output_dir) / "eval" / timestamp
                _log.info(f"Saving logs, results, and visualizations to {save_dir}")

                # Add log file handler for this run
                file_handler = add_file_handler(save_dir / "tiptop_run.log")
                try:
                    # Capture robot state and compute camera pose
                    observation = capture_live_observation(container)
                    robot_rr.set_joint_positions(observation.q_init)

                    # Now we're ready! Start timing
                    _log.info("Running Perception...")
                    perception_start = time.perf_counter()
                    env, all_surfaces, processed_scene, grounded_atoms = await run_perception(
                        session,
                        observation,
                        task_instruction,
                        save_dir,
                        depth_estimator=container.depth_estimator,
                        gripper_mask=container.gripper_mask,
                    )
                    perception_duration = time.perf_counter() - perception_start

                    cutamp_plan = None
                    planning_duration = None
                    failure_reason = None
                    try:
                        _log.info("Running Planning...")
                        cutamp_plan, planning_duration, failure_reason = run_planning(
                            env,
                            config,
                            q_init=observation.q_init,
                            ik_solver=container.ik_solver,
                            grasps=processed_scene.grasps,
                            motion_gen=container.motion_gen,
                            all_surfaces=all_surfaces,
                            experiment_dir=save_dir / "cutamp",
                        )
                        _log.info(
                            f"Perception and cuTAMP planning took: {perception_duration + planning_duration:.2f}s"
                        )
                        if cutamp_plan is not None:
                            plan_path = save_dir / "tiptop_plan.json"
                            save_tiptop_plan(serialize_plan(cutamp_plan, observation.q_init), plan_path)
                            _log.info(f"Saved TiPToP plan to {plan_path}")

                        if cutamp_plan is not None and execute_plan:
                            _log.info("Executing plan...")
                            # Execute with optional recording
                            if container.enable_recording:
                                cameras_to_record = [
                                    (
                                        container.external_cam,
                                        save_dir / "external_cam.svo",
                                        save_dir / "external_cam.mp4",
                                    ),
                                ]
                                if isinstance(container.cam, ZedCamera):
                                    cameras_to_record.append(
                                        (container.cam, save_dir / "hand_cam.svo", save_dir / "hand_cam.mp4"),
                                    )
                                with record_cameras(cameras_to_record):
                                    execute_cutamp_plan(cutamp_plan, client=container.robot)
                            else:
                                execute_cutamp_plan(cutamp_plan, client=container.robot)
                            _log.info("Finished executing plan!")
                        elif cutamp_plan is not None:
                            _log.info("Skipping cuTAMP plan execution on real robot")
                        else:
                            _log.warning(f"No plan found: {failure_reason}")

                        _log.debug(f"Finished run for instruction: {task_instruction}")
                    finally:
                        # Always save env, grasps, metadata, and artifacts regardless of success
                        save_run_outputs(save_dir, env, processed_scene.grasps)
                        save_run_metadata(
                            save_dir=save_dir,
                            timestamp=iso_timestamp,
                            task_instruction=task_instruction,
                            q_at_capture=observation.q_init,
                            world_from_cam=observation.world_from_cam,
                            perception_duration=perception_duration,
                            grounded_atoms=grounded_atoms,
                            planning_success=cutamp_plan is not None,
                            planning_failure_reason=failure_reason,
                            planning_duration=planning_duration,
                        )
                        _log.info(f"Logs, results, and visualizations saved to {save_dir}")

                    if execute_plan:
                        _label_rollout(save_dir, output_dir, date_str, timestamp)

                        # If the goal was just to pick an object (i.e., goal contains Holding), ask the user to confirm before
                        # opening the gripper so the object can drop safely
                        if cutamp_plan is not None and any(atom.fluent.name == Holding.name for atom in env.goal_state):
                            print("WARNING: object will drop when gripper opens, so be ready to catch it.")
                            while True:
                                try:
                                    response = input("Open gripper? [y]: ").strip().lower()
                                except KeyboardInterrupt:
                                    raise UserExitException("User interrupted with Ctrl+C")
                                if response == "y":
                                    break
                            container.robot.open_gripper()
                except Exception:
                    _log.exception("TiPToP run failed")
                    raise
                finally:
                    # Always remove the file handler after the run
                    remove_file_handler(file_handler)
            except (UserExitException, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    _log.info("Interrupted by user (Ctrl+C)")
                else:
                    _log.info("User requested exit")
                break


def _sync_entrypoint(
    output_dir: str = "tiptop_outputs",
    max_planning_time: float = 60.0,
    opt_steps_per_skeleton: int = 500,
    execute_plan: bool = True,
    cutamp_visualize: bool = False,
    num_particles: int = 256,
    enable_recording: bool = False,
):
    """
    TiPToP live robot runner. Runs continuously on the real robot.

    Args:
        output_dir: Top-level directory to save outputs to; a timestamped subdirectory is created per run.
        max_planning_time: Maximum time to spend planning with cuTAMP across all skeletons (approximate).
        opt_steps_per_skeleton: Number of optimization steps per skeleton in cuTAMP.
        execute_plan: Whether to execute the plan on the real robot.
        cutamp_visualize: Whether to visualize cuTAMP optimization.
        num_particles: Number of particles for cuTAMP; decrease if running out of GPU memory.
        enable_recording: Whether to record external camera video during execution.
    """
    assert max_planning_time > 0
    assert opt_steps_per_skeleton > 0
    assert num_particles > 0

    print_tiptop_banner()
    check_cutamp_version()

    cfg = tiptop_cfg()
    config = build_tamp_config(
        num_particles=num_particles,
        max_planning_time=max_planning_time,
        opt_steps=opt_steps_per_skeleton,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
        near_placement=cfg.experimental.pick_place_next_to,
        collision_activation_distance=0.0,
        enable_visualizer=cutamp_visualize,
    )

    global _executor_pool
    setup_logging(level=logging.DEBUG)

    container = get_demo_container(num_particles, config.coll_n_spheres, 0.0, enable_recording)
    # Workers ignore SIGINT so only the main process handles Ctrl+C for clean shutdown
    _executor_pool = ProcessPoolExecutor(
        max_workers=4, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)
    )

    exit_code = 1
    try:
        asyncio.run(async_entrypoint(container, config, output_dir, execute_plan))
        exit_code = 0
    except (UserExitException, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            _log.info("Interrupted during startup/shutdown (Ctrl+C)")
        else:
            _log.debug("Exit detected")
        exit_code = 0
    finally:
        if container is not None:
            _log.debug("Tearing down cameras and robot...")
            container.cam.close()
            if container.external_cam is not None:
                container.external_cam.close()
            container.robot.close()
        if _executor_pool is not None:
            _executor_pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(exit_code)


def entrypoint():
    tyro.cli(_sync_entrypoint)


if __name__ == "__main__":
    entrypoint()
