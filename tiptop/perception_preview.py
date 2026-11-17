"""Optional task snapshots for the RGB/depth viewer; no camera or robot access."""

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)
_COLORS = [(52, 211, 153), (96, 165, 250), (251, 191, 36), (244, 114, 182)]


def finger_tip(vertices, length_m=0.015):
    """Centre of the 15 mm finger end nearest the TCP, in TCP metres (+Z points toward the object)."""
    vertices = np.asarray(vertices)
    return vertices[vertices[:, 2] >= vertices[:, 2].max() - length_m].mean(axis=0)


@lru_cache(maxsize=8)
def gripper_mesh_parts(urdf_path):
    """Read the existing UON CAD at its open joint configuration, in TCP metres."""
    import trimesh
    from scipy.spatial.transform import Rotation

    urdf_path = Path(urdf_path).resolve()
    robot = ET.parse(urdf_path).getroot()

    def origin(element):
        matrix = np.eye(4)
        if element is not None:
            matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
            matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        return matrix

    body_from_link = {"gripper_body": np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        found = []
        for joint in pending:
            parent = joint.find("parent").get("link")
            if parent in body_from_link:
                body_from_link[joint.find("child").get("link")] = body_from_link[parent] @ origin(joint.find("origin"))
                found.append(joint)
        if not found:
            break
        pending = [joint for joint in pending if joint not in found]
    tcp_from_body = np.linalg.inv(body_from_link["gripper_tcp"])
    parts = []
    for link in robot.findall("link"):
        name = link.get("name")
        if name not in body_from_link or name == "gripper_tcp":
            continue
        for visual in link.findall("visual"):
            geometry = visual.find("geometry/mesh")
            if geometry is None:
                continue
            mesh = trimesh.load(urdf_path.parent / geometry.get("filename"), force="mesh", process=False)
            # The convex exterior of each CAD part gives an uncluttered 2D silhouette.
            vertices = np.asarray(mesh.convex_hull.vertices) * np.fromstring(geometry.get("scale", "1 1 1"), sep=" ")
            transform = tcp_from_body @ body_from_link[name] @ origin(visual.find("origin"))
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            parts.append((name, vertices))
    return parts


def project_grasps(world_from_tcp, world_from_cam, intrinsics, spheres, ids, scores, *, limit=32, tool_parts=()):
    """Project TCP axes, optional CAD outlines, and open-tool collision spheres."""
    poses = np.linalg.inv(np.asarray(world_from_cam)) @ np.asarray(world_from_tcp)
    k, spheres = np.asarray(intrinsics), np.asarray(spheres)
    # +Z is the DB approach axis; the gripper body lies on its negative side.
    axes = np.array([[0, 0, 0], [.035, 0, 0], [0, .035, 0], [0, 0, .07]])
    output = []
    for i, pose in enumerate(poses[:limit]):
        camera = axes @ pose[:3, :3].T + pose[:3, 3]
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .001):
            continue
        pixels = camera @ k.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        centers = spheres[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        circles = []
        for center, radius in zip(centers, spheres[:, 3]):
            if center[2] <= radius or not np.isfinite(center).all():
                continue
            pixel = k @ center
            circles.append([float(pixel[0]/pixel[2]), float(pixel[1]/pixel[2]),
                            float(k[0, 0]*radius/center[2]), float(k[1, 1]*radius/center[2])])
        outlines = []
        for name, vertices in tool_parts:
            points = vertices @ pose[:3, :3].T + pose[:3, 3]
            if np.any(points[:, 2] <= .001):
                continue
            projected = points @ k.T
            projected = projected[:, :2]/projected[:, 2:]
            polygon = cv2.convexHull(projected.astype(np.float32)).reshape(-1, 2)
            outline = dict(part=name, points=polygon.tolist())
            if "finger" in name:
                tip = finger_tip(vertices) @ pose[:3, :3].T + pose[:3, 3]
                if tip[2] > .001:
                    pixel = k @ tip
                    outline["tip"] = (pixel[:2] / pixel[2]).tolist()
            outlines.append(outline)
        output.append(dict(id=int(ids[i]), score=float(scores[i]), axes=pixels.tolist(), spheres=circles,
                           outlines=outlines, world_from_tcp=np.asarray(world_from_tcp)[i].tolist()))
    return output


def project_grasp_boxes(boxes, world_from_db, world_from_cam, intrinsics):
    """Project the three recorded sweep rectangles, already in object-zero metres."""
    boxes = np.asarray(boxes, float)
    if boxes.shape != (3, 4, 3) or not np.isfinite(boxes).all():
        raise ValueError("Expected three grasp_boxes, each with four 3D corners")
    world_from_db = np.asarray(world_from_db)
    camera_from_world = np.linalg.inv(np.asarray(world_from_cam))
    output = []
    for finger, corners in enumerate(boxes, 1):
        world = corners @ world_from_db[:3, :3].T + world_from_db[:3, 3]
        camera = world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        if np.any(camera[:, 2] <= .001):
            continue
        pixels = camera @ np.asarray(intrinsics).T
        output.append(dict(finger=finger, points=(pixels[:, :2]/pixels[:, 2:]).tolist(),
                           world_points=world.tolist()))
    return output


def rgb_camera_view(world_from_cam, intrinsics, width, height):
    """Use the RGB optical calibration for perspective rendering of world geometry."""
    return dict(camera_from_world=np.linalg.inv(np.asarray(world_from_cam, float)).tolist(),
                intrinsics=np.asarray(intrinsics, float).tolist(), width=int(width), height=int(height))


def build_obstacle_scene(object_meshes, objects, bboxes, support_z, target_label, *, point_limit=1200,
                         collision_labels=None):
    """Display masked observations and the actual planning meshes in world metres."""
    from scipy.spatial.transform import Rotation

    instances = []
    for index, box in enumerate(bboxes):
        label = box["label"]
        if collision_labels is not None and label not in collision_labels:
            continue
        points = np.asarray(objects.get(label, {}).get("world_xyz", []), float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        count = len(points)
        if count > point_limit:
            points = points[np.linspace(0, count-1, point_limit, dtype=int)]
        vertices, faces = np.empty((0, 3)), np.empty((0, 3), dtype=int)
        mesh = object_meshes.get(label)
        if mesh is not None:
            geometry = mesh.get_trimesh_mesh(process=False)
            pose = np.asarray(mesh.pose, float)
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            vertices = np.asarray(geometry.vertices) @ rotation.T + pose[:3]
            faces = np.asarray(geometry.faces, int)
        instances.append(dict(label=label, class_id=box.get("class_id", label),
                              points=np.round(points, 5).tolist(), observed_points=count,
                              vertices=np.round(vertices, 5).tolist(), faces=faces.tolist(),
                              color=list(_COLORS[index % len(_COLORS)]), collision_mesh=mesh is not None))
    return dict(objects=instances, support_z_m=float(support_z), target_label=target_label,
                frame="world", units="m", source="planning_meshes")


class TaskPreview:
    """Publish one observation's immutable image layers and atomic metadata."""

    def __init__(self, output, *, observation_id, task, depth_source, recorded=False, max_tilt_deg=3):
        self.directory = Path(output) / "viewer"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.revision = 0
        self.max_tilt_deg = max_tilt_deg
        self.spheres = None
        self.side_frame = None
        self._database_context = None
        self._candidate_tcp = {}
        self.urdf_path = Path(__file__).parent / "doosan/assets/m1013_uon/m1013_uon.urdf"
        self.state = dict(observation_id=observation_id, task=task, depth_source=depth_source,
                          recorded=recorded, images={}, boxes=[], grasps=[], final_grasp=None, ready=[], status="관측 준비")

    def _image(self, name, pixels):
        filename = f"{name}-{self.revision}.png"
        Image.fromarray(pixels).save(self.directory / filename, compress_level=1)
        self.state["images"][name] = filename

    def update(self, event, data):
        """Visualization failures must not change perception or execution behavior."""
        try:
            with self.lock:
                self.revision += 1
                if event == "observation":
                    self.rgb = np.asarray(data["rgb"])
                    self.state.update(width=self.rgb.shape[1], height=self.rgb.shape[0])
                    self._image("rgb", self.rgb)
                    if data.get("depth") is not None:
                        self._depth(data["depth"])
                    self.state["status"] = "RGB 관측 완료 · 인식 대기"
                elif event == "depth":
                    self._depth(data["depth_map"])
                elif event == "detection":
                    rgba = np.zeros((*self.rgb.shape[:2], 4), np.uint8)
                    for i, mask in enumerate(np.asarray(data["masks"])[:, 0]):
                        rgba[mask.astype(bool)] = [*_COLORS[i % len(_COLORS)], 110]
                    self._image("mask", rgba)
                    self.state["boxes"] = data["bboxes"]
                    self.state["detector"] = data.get("semantic_source", "gemini_sam2")
                    self.state["target_label"] = next(
                        (atom["args"][0] for atom in data.get("grounded_atoms", []) if atom.get("args")), None
                    )
                    self.state["status"] = "물체 인식 완료 · 최종 경로 계획 대기"
                elif event == "target":
                    self.state["target_label"] = data["label"]
                elif event == "obstacles":
                    self.state["obstacle_scene"] = data
                elif event == "final_grasp":
                    self.state["final_grasp"] = data
                    self.state["status"] = "전체 경로 계획 완료 · 실행 계획의 최종 파지 표시"
                elif event == "side_scene":
                    self.state["side_scene"] = data
                elif event == "grasps":
                    self.state.update(data)
                    if self.state["final_grasp"] is None:
                        self.state["status"] = "인식 완료 · 최종 경로 계획 대기"
                elif event == "planning":
                    self.state["feasible_ids"] = data["supplied_candidate_ids"]
                elif event == "placement":
                    self.state["placement"] = data
                elif event == "error":
                    self.state["status"] = str(data["message"])
                if event not in self.state["ready"]:
                    self.state["ready"].append(event)
                self.state["revision"] = self.revision
                temporary = self.directory / "state.tmp"
                temporary.write_text(json.dumps(self.state, ensure_ascii=False, allow_nan=False))
                temporary.replace(self.directory / "state.json")
        except Exception as error:
            _log.warning("Perception viewer update skipped: %s", error)

    def _depth(self, depth):
        depth = np.asarray(depth, float)
        valid = np.isfinite(depth) & (depth > 0)
        # Fix the range for this observation so later overlays cannot recolor it.
        low, high = np.percentile(depth[valid], [2, 98]) if valid.any() else (0., 1.)
        high = max(float(high), float(low) + .001)
        scaled = np.clip((np.where(valid, depth, low)-low)/(high-low), 0, 1)
        colors = cv2.applyColorMap((scaled*255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
        colors[~valid] = 0
        self._image("depth", colors)
        self.state["depth_range_m"] = [float(low), high]
        self.state["depth_valid_percent"] = round(float(valid.mean()*100), 1)
        if "depth" not in self.state["ready"]:
            self.state["ready"].append("depth")

    def obstacles(self, scene, visual, *, world_from_cam=None, intrinsics=None, collision_labels=None):
        """Export display samples without rebuilding or modifying collision geometry."""
        try:
            from tiptop.perception.placement import table_support_z

            table = scene.table_cuboid
            support_z = table_support_z(table.pose[2] + table.dims[2] / 2)
            target = next(iter(scene.grasps), self.state.get("target_label"))
            data = build_obstacle_scene(scene.object_meshes, visual["objects"], visual["bboxes"], support_z, target,
                                        collision_labels=collision_labels)
            data["surrounding_obstacles"] = visual.get("surrounding_obstacles", True)
            if world_from_cam is not None and intrinsics is not None:
                data["camera"] = rgb_camera_view(world_from_cam, intrinsics, self.state["width"], self.state["height"])
            path = self.directory / "obstacles.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
            self.update("obstacles", data)
        except Exception as error:
            _log.warning("Perception viewer obstacle scene skipped: %s", error)

    def placement(self, region, world_from_cam, intrinsics, *, is_destination=True):
        """Show the configured physical box and release plane in this observation."""
        try:
            center, half = np.asarray(region["center_xy_m"]), np.asarray(region["size_xy_m"]) / 2
            xy = center + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
            top, release = region["support_z_m"], region["release_bottom_z_m"]
            bottom = region.get("box", {}).get("box_bottom_z_m", top)
            layers = {key: np.column_stack((xy, np.full(4, z)))
                      for key, z in (("bottom", bottom), ("top", top), ("release", release))}
            camera = np.linalg.inv(np.asarray(world_from_cam))

            def project(points):
                points = points @ camera[:3, :3].T + camera[:3, 3]
                if np.any(points[:, 2] <= .001):
                    raise ValueError("Placement is behind the RGB camera")
                pixels = points @ np.asarray(intrinsics).T
                return (pixels[:, :2] / pixels[:, 2:]).tolist()

            self.update("placement", dict(region=region, is_destination=is_destination,
                                           **{key: project(points) for key, points in layers.items()}))
            if region["source"] == "fixed_box":
                vertices = np.concatenate([layers["bottom"], layers["top"]])
                faces = [[0,1,2], [0,2,3], [4,6,5], [4,7,6], [0,4,5], [0,5,1],
                         [1,5,6], [1,6,2], [2,6,7], [2,7,3], [3,7,4], [3,4,0]]
                obstacle = dict(label=region["destination"], class_id="고정 박스", points=[],
                                observed_points=0, vertices=vertices.tolist(), faces=faces,
                                color=[120,212,224], collision_mesh=True, source="configured")
                scene = dict(self.state.get("obstacle_scene", {}))
                scene["objects"] = [obj for obj in scene.get("objects", []) if obj["label"] != obstacle["label"]] + [obstacle]
                scene.setdefault("support_z_m", bottom)
                scene["camera"] = rgb_camera_view(world_from_cam, intrinsics, self.state["width"], self.state["height"])
                (self.directory / "obstacles.json").write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False))
                self.update("obstacles", scene)
        except Exception as error:
            _log.warning("Placement preview skipped: %s", error)

    def grasps(self, data, world_from_cam, intrinsics, *, max_tilt_deg=10, spheres=None):
        """Retain DB candidates for optional inspection; they are never the final grasp."""
        try:
            tcp = np.asarray(data["world_from_tcp"])
            keep = np.flatnonzero(-tcp[:, 2, 2] >= np.cos(np.deg2rad(max_tilt_deg))-1e-12)
            if spheres is None:
                spheres = self.spheres
            if spheres is None:
                spheres = np.load(Path(__file__).parent / "doosan/assets/m1013_uon/gripper_spheres_open.npy")
            projected = project_grasps(tcp[keep], world_from_cam, intrinsics, spheres,
                                       np.asarray(data["candidate_indices"])[keep], np.asarray(data["scores"])[keep],
                                       tool_parts=gripper_mesh_parts(str(self.urdf_path)))
            self._candidate_tcp = dict(zip(map(int, data["candidate_indices"]), tcp))
            for grasp in projected:
                self._attach_db_boxes(grasp, grasp["id"], world_from_cam, intrinsics)
            if self.side_frame is not None:
                for grasp in projected:
                    grasp["side"] = self._side_grasp(np.asarray(grasp["world_from_tcp"]))
            self.update("grasps", dict(grasps=projected, db_total=len(tcp), tilt_candidates=len(keep),
                                       max_tilt_deg=float(max_tilt_deg)))
        except Exception as error:
            _log.warning("Perception viewer candidate projection skipped: %s", error)

    def _load_database(self):
        if self._database_context is None:
            path = self.directory.parent / "registered-pose.json"
            if not path.is_file():
                return None
            from tiptop.doosan.grasp_database import database_grasp_matrices
            from tiptop.object_catalog import registered_object

            registered = json.loads(path.read_text())
            entry = registered_object(registered["class_id"])
            document = json.loads(entry["db"].read_text())
            if (document["metadata"]["coordinate_frame"] != "object_zero_pose"
                    or document["metadata"]["object"] != registered["class_id"]):
                raise ValueError("Viewer DB object/frame does not match the registered pose")
            self.state["db_source"] = dict(path=str(entry["db"]), class_id=entry["class_id"],
                                           coordinate_frame="object_zero_pose", units="m",
                                           count=len(document["data"]))
            matrices = database_grasp_matrices(document)
            rows = [dict(row, grasp_mat=matrix.tolist()) for row, matrix in zip(document["data"], matrices)]
            self._database_context = (rows, np.asarray(registered["pose"]["world_from_cad"]))
        return self._database_context

    def _attach_db_boxes(self, grasp, index, world_from_cam, intrinsics):
        try:
            context = self._load_database()
            if context is None:
                return
            rows, world_from_db = context
            row = rows[index]
            grasp["db_reference"] = dict(index=int(index), scene_id=row["scene_id"])
            for key in ("source_pregrasp_index", "source_output_index"):
                if key in row:
                    grasp["db_reference"][key] = row[key]
            if row.get("grasp_boxes") is not None:
                # Keep the recorded object-fixed geometry for reference. The visible sweep template is
                # the DB grasp frame placed at the displayed TCP (DB-to-TCP correction undone), so it
                # follows the final FK endpoint residual and matches the recorded boxes at the DB target.
                grasp["db_grasp_boxes_original"] = project_grasp_boxes(row["grasp_boxes"], world_from_db,
                                                                      world_from_cam, intrinsics)
                grasp_mat = np.asarray(row["grasp_mat"])
                candidate_tcp = self._candidate_tcp.get(int(index))
                grasp_from_tcp = (np.linalg.inv(world_from_db @ grasp_mat) @ candidate_tcp
                                  if candidate_tcp is not None else np.eye(4))
                world_from_template = np.asarray(grasp["world_from_tcp"]) @ np.linalg.inv(grasp_mat @ grasp_from_tcp)
                grasp["grasp_boxes"] = project_grasp_boxes(row["grasp_boxes"], world_from_template,
                                                           world_from_cam, intrinsics)
                grasp["grasp_boxes_frame"] = "displayed_tcp"
                self._side_boxes(grasp)
        except Exception as error:
            _log.warning("Viewer DB sweep boxes unavailable: %s", error)

    def _side_boxes(self, grasp):
        if self.side_frame is not None:
            for box in grasp.get("grasp_boxes", []):
                box["side_points"] = self._side_points(box["world_points"]).tolist()

    def _attach_final_db_boxes(self, final, plan, world_from_cam, intrinsics):
        """Use the successful plan's goal; reconcile old FK-only plans conservatively."""
        allowed = self.state.get("feasible_ids", plan.get("supplied_candidate_ids"))
        candidates = [(i, pose) for i, pose in self._candidate_tcp.items() if allowed is None or i in allowed]
        if not candidates:
            return
        tcp = np.asarray(final["world_from_tcp"])
        settings = plan.get("planning_settings", {})
        position_tolerance = settings.get("motion_position_tolerance_m", .01)
        rotation_tolerance = settings.get("motion_rotation_tolerance", .1)
        selected_target = plan.get("selected_grasp", {}).get("world_from_tcp")
        if selected_target is not None:
            selected_target = np.asarray(selected_target, float)
            if selected_target.shape != (4, 4) or not np.isfinite(selected_target).all():
                final["db_match_note"] = "계획의 선택 파지 정보가 유효하지 않아 범위 박스를 표시하지 않습니다."
                return
        matches = []
        for index, pose in candidates:
            distance = float(np.linalg.norm(pose[:3, 3]-tcp[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(pose[:3, :3].T @ tcp[:3, :3])-1)/2, -1, 1)))
            goal_matches = selected_target is None or np.allclose(pose, selected_target, atol=1e-6, rtol=0)
            if goal_matches and distance <= position_tolerance and angle <= rotation_tolerance:
                matches.append((index, distance, angle))
        if len(matches) != 1:
            final["db_match_note"] = "최종 TCP와 DB 행의 대응이 불명확하여 범위 박스를 표시하지 않습니다."
            return
        index, distance, angle = matches[0]
        self._attach_db_boxes(final, index, world_from_cam, intrinsics)
        if "db_reference" in final:
            method = "planned_target" if selected_target is not None else "final_tcp_pose"
            final["db_reference"].update(match_method=method, position_error_mm=distance*1000,
                                         rotation_error_deg=float(np.degrees(angle)))

    def side_scene(self, vertices_world, support_z, world_from_cam):
        """Look from the Right-table end toward the robot, with the table at zero."""
        try:
            vertices = np.asarray(vertices_world, float)
            # The operator sits at +Y looking toward -Y. Screen right is -X,
            # and screen up is +Z, independently of the RGB camera orientation.
            horizontal = np.array([-1., 0., 0.])
            center = vertices.mean(axis=0)
            center[2] = support_z
            self.side_frame = (center, horizontal)
            points = self._side_points(vertices)
            # Display sampling only; all height limits use the complete CAD.
            sampled = points[np.linspace(0, len(points)-1, min(6000, len(points)), dtype=int)]
            with self.lock:
                for grasp in self.state["grasps"]:
                    if "world_from_tcp" in grasp:
                        grasp["side"] = self._side_grasp(np.asarray(grasp["world_from_tcp"]))
                    self._side_boxes(grasp)
                final = self.state.get("final_grasp")
                if final is not None:
                    final["side"] = self._side_grasp(np.asarray(final["world_from_tcp"]))
                    self._side_boxes(final)
                self.update("side_scene", dict(points=sampled.tolist(), object_top_m=float(points[:, 1].max()),
                                               viewpoint="right_table_end", view_label="Right table 끝 → 로봇",
                                               horizontal_world=horizontal.tolist()))
        except Exception as error:
            _log.warning("Perception viewer side scene skipped: %s", error)

    def _side_points(self, world):
        center, horizontal = self.side_frame
        relative = np.asarray(world)-center
        return np.column_stack((relative @ horizontal, relative[:, 2]))

    def _side_grasp(self, tcp):
        parts, finger_heights = [], []
        for name, vertices in gripper_mesh_parts(str(self.urdf_path)):
            points = self._side_points(vertices @ tcp[:3, :3].T + tcp[:3, 3])
            part = dict(part=name, points=cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).tolist())
            if "finger" in name:
                finger_heights.append(float(points[:, 1].min()))
                tip = finger_tip(vertices) @ tcp[:3, :3].T + tcp[:3, 3]
                part["tip"] = self._side_points([tip])[0].tolist()
            parts.append(part)
        axes = np.array([[0, 0, 0], [0, 0, .07]]) @ tcp[:3, :3].T + tcp[:3, 3]
        return dict(outlines=parts, axes=self._side_points(axes).tolist(),
                    finger_lowest_m=min(finger_heights) if finger_heights else None)

    def publish_plan(self, plan, world_from_cam, intrinsics):
        """Show the exact commanded arm pose immediately before Pick.close."""
        try:
            from yourdfpy import URDF

            q = final_grasp_joints(plan)
            robot = URDF.load(str(self.urdf_path.resolve()), load_meshes=False)
            robot.update_cfg(dict(zip((f"joint_{i}" for i in range(1, 7)), q)) | {"gripper_spinner_joint": 0.0})
            tcp = robot.get_transform("gripper_tcp", "base_link")
            spheres = self.spheres
            if spheres is None:
                spheres = np.load(Path(__file__).parent / "doosan/assets/m1013_uon/gripper_spheres_open.npy")
            projected = project_grasps([tcp], world_from_cam, intrinsics, spheres, [0], [0.],
                                       tool_parts=gripper_mesh_parts(str(self.urdf_path)))
            if not projected:
                raise ValueError("Final planned TCP is behind the RGB camera")
            final = projected[0]
            final.pop("id")
            final.pop("score")
            final.update(joint_positions_rad=q.tolist(), world_from_tcp=tcp.tolist(),
                         source_step="Pick.descend", before_action="Pick.close")
            if self.side_frame is not None:
                final["side"] = self._side_grasp(tcp)
            self._attach_final_db_boxes(final, plan, world_from_cam, intrinsics)
            self.update("final_grasp", final)
        except Exception as error:
            _log.warning("Perception viewer final plan projection skipped: %s", error)


def final_grasp_joints(plan):
    """The execution endpoint before close is authoritative, not DB scores or IK seeds."""
    for i, step in enumerate(plan["steps"]):
        if step.get("label") == "Pick.close" and step.get("action") == "close":
            previous = plan["steps"][i-1] if i else {}
            if previous.get("type") != "trajectory" or previous.get("label") != "Pick.descend":
                raise ValueError("Final grasp must come from the Pick.descend trajectory before Pick.close")
            return np.asarray(previous["positions"][-1], float)
    raise ValueError("No final Pick.close in the completed plan")


def preview_saved_task(output):
    """Rebuild a saved-task preview from RGB-D/JSON only; never deserialize pickle."""
    output = Path(output)
    with np.load(output / "observation.npz", allow_pickle=False) as packet:
        observation = {key: packet[key] for key in packet.files}
    result = json.loads((output / "task-result.json").read_text())
    preview = TaskPreview(output, observation_id=str(observation["observation_id"]),
                          task=result["task"], depth_source=str(observation.get("depth_source", "unknown")),
                          recorded=True)
    spheres_path = output / "hardware-model/gripper_spheres_open.npy"
    if spheres_path.is_file():
        preview.spheres = np.load(spheres_path, allow_pickle=False)
    urdf_path = output / "hardware-model/m1013_uon_measured.urdf"
    if urdf_path.is_file():
        preview.urdf_path = urdf_path
    depth_path = output / "perception/depth.png"
    depth = np.asarray(Image.open(depth_path)).astype(float)/1000 if depth_path.is_file() else observation.get("depth")
    preview.update("observation", dict(rgb=observation["rgb"], depth=depth))
    obstacle_path = preview.directory / "obstacles.json"
    if obstacle_path.is_file():
        try:
            obstacle_scene = json.loads(obstacle_path.read_text())
            obstacle_scene["camera"] = rgb_camera_view(observation["world_from_cam"], observation["intrinsics"],
                                                        preview.state["width"], preview.state["height"])
            preview.update("obstacles", obstacle_scene)
        except (OSError, ValueError) as error:
            _log.warning("Saved obstacle scene unavailable: %s", error)
    semantic_path = output / "semantic-task.json"
    if semantic_path.is_file() and (output / "perception/masks.npz").is_file():
        semantic = json.loads(semantic_path.read_text())
        with np.load(output / "perception/masks.npz", allow_pickle=False) as masks:
            preview.update("detection", dict(masks=masks["arr_0"], bboxes=semantic["bboxes"],
                                              semantic_source=semantic["semantic_source"],
                                              grounded_atoms=semantic.get("grounded_atoms", [])))
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
    region_file = output / "placement-region.json"
    if region_file.is_file():
        region = json.loads(region_file.read_text())
        if region.get("source") in {"fixed_box", "perceived_surface"}:
            preview.placement(region, observation["world_from_cam"], observation["intrinsics"])
        elif plan and plan.get("task_settings", {}).get("fixed_box_present"):
            from tiptop.task_presets import TaskSettings

            settings = TaskSettings(**plan["task_settings"])
            preview.placement(settings.box_region(), observation["world_from_cam"], observation["intrinsics"],
                              is_destination=False)
    pose_path = output / "registered-pose.json"
    region_path = output / "placement-region.json"
    if pose_path.is_file() and region_path.is_file():
        try:
            import trimesh
            from tiptop.object_catalog import registered_object

            registered = json.loads(pose_path.read_text())
            entry = registered_object(registered["class_id"])
            pose = np.asarray(registered["pose"]["world_from_cad"])
            vertices = np.asarray(trimesh.load(entry["cad"], force="mesh", process=False).vertices) * entry["cad_scale_to_m"]
            region = json.loads(region_path.read_text())
            preview.side_scene(vertices @ pose[:3, :3].T + pose[:3, 3],
                               region.get("source_support_z_m", region["support_z_m"]), observation["world_from_cam"])
        except Exception as error:
            _log.warning("Saved side scene unavailable: %s", error)
    if pose_path.is_file():
        try:
            from tiptop.doosan.grasp_database import database_grasp_matrices
            from tiptop.object_catalog import registered_object

            pose = json.loads(pose_path.read_text())
            entry = registered_object(pose["class_id"])
            document = json.loads(entry["db"].read_text())
            rows = document["data"]
            order = np.argsort(-np.asarray([row["score"] for row in rows]), kind="stable")
            grasp_from_tcp = (plan or {}).get("db_grasp_from_tcp")
            profile = (plan or {}).get("hardware_profile")
            if grasp_from_tcp is None and profile and Path(profile).is_file():
                grasp_from_tcp = np.asarray(json.loads(Path(profile).read_text())["tool"]["db_grasp_from_tcp"])
            if grasp_from_tcp is None:
                preparation = json.loads(str(observation.get("camera_preparation", "{}")))
                grasp_from_tcp = preparation.get("db_grasp_from_tcp", np.eye(4))
            grasp_from_tcp = np.asarray(grasp_from_tcp, float)
            world_tcp = np.asarray(pose["pose"]["world_from_cad"]) @ database_grasp_matrices(document)[order] @ grasp_from_tcp
            preview.grasps(dict(world_from_tcp=world_tcp, candidate_indices=order, scores=[rows[i]["score"] for i in order]),
                           observation["world_from_cam"], observation["intrinsics"],
                           max_tilt_deg=(plan or {}).get("max_approach_tilt_deg", 10))
        except Exception as error:
            _log.warning("Saved DB candidates unavailable: %s", error)
    report_path = output / "planning-result.json"
    if report_path.is_file():
        preview.update("planning", json.loads(report_path.read_text()))
    if plan_path.is_file():
        preview.publish_plan(plan, observation["world_from_cam"], observation["intrinsics"])
    return preview
