"""DB 파지 → TCP 정렬 가정을 계획 모델의 손가락 표면과 대조한다 (CPU, 물리 시뮬레이션 아님).

저장된 접촉점(object 좌표)을 가정한 DB→TCP 변환으로 gripper_body 좌표에 옮기고, 기록된 닫힘 각도의
손가락 표면까지 거리를 잰다. 접촉점은 손가락별 평균이며 표면의 대응점이 아니므로 6D 보정을 확정하지 않는다.
raw 기록(output_grasp)이 있는 single_object_grasp_compact DB만 지원한다.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from experiments.grasping.paths import PROJECT_ROOT, WORK_ROOT
from tiptop.object_catalog import DEFAULT_CATALOG, asset_path, load_catalog

PLANNING_URDF = PROJECT_ROOT / "tiptop/doosan/assets/m1013_uon/m1013_uon.urdf"
HARDWARE_PROFILE = PROJECT_ROOT / "tiptop/config/m1013_fixed_hardware.declared.json"
SPINNER_JOINT = "gripper_spinner_joint"


def rigid(rotation=None, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def _origin(element):
    """URDF origin: xyz in metres, rpy as fixed-axis roll-pitch-yaw."""
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    return rigid(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


@lru_cache(maxsize=None)
def _mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def gripper_fingers(spinner_rad, urdf=PLANNING_URDF):
    """Finger surfaces in the gripper_body frame at a spinner angle (fingers mimic the spinner)."""
    root = ET.parse(urdf).getroot()
    joints = [j for j in root.findall("joint") if j.get("name").startswith("gripper_")
              and j.get("name") not in ("gripper_mount_joint", "gripper_tcp_joint")]
    poses = {"gripper_body": np.eye(4)}
    while joints:
        pending = [j for j in joints if j.find("parent").get("link") in poses]
        if not pending:
            raise ValueError("Disconnected gripper joint graph in the planning URDF")
        for joint in pending:
            mimic = joint.find("mimic")
            if joint.get("name") == SPINNER_JOINT:
                angle = spinner_rad
            elif mimic is not None and mimic.get("joint") == SPINNER_JOINT:
                angle = float(mimic.get("multiplier")) * spinner_rad
            else:
                raise ValueError(f"Unexpected movable gripper joint {joint.get('name')}")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            motion = rigid(Rotation.from_rotvec(axis * angle).as_matrix())
            parent_pose = poses[joint.find("parent").get("link")]
            poses[joint.find("child").get("link")] = parent_pose @ _origin(joint.find("origin")) @ motion
            joints.remove(joint)
    fingers = {}
    for index in (1, 2, 3):
        link = root.find(f"link[@name='gripper_finger{index}']")
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh")
        vertices, faces = _mesh(str((Path(urdf).parent / mesh.get("filename")).resolve()))
        vertices = vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        body_from_mesh = poses[f"gripper_finger{index}"] @ _origin(visual.find("origin"))
        fingers[f"finger{index}"] = (transform(vertices, body_from_mesh), faces)
    return fingers


def finger_scenes(fingers):
    scenes = {}
    for name, (vertices, faces) in fingers.items():
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.int32))))
        scenes[name] = scene
    return scenes


def hypotheses(declared_grasp_from_tcp):
    """DB grasp → TCP 후보: 선언값, 위치 0의 Z 회전 60° 간격, 예전 가정 Rx(π)."""
    result = {"declared": np.asarray(declared_grasp_from_tcp, dtype=float)}
    for degrees in range(0, 360, 60):
        result[f"Rz({degrees}°)"] = rigid(Rotation.from_euler("z", degrees, degrees=True).as_matrix())
    result["Rx(180°)"] = rigid(np.diag([1.0, -1.0, -1.0]))
    return result


def contact_distances_mm(scenes, contacts, object_from_grasp, grasp_from_tcp, body_from_tcp):
    """접촉점별 (같은 번호 손가락까지, 가장 가까운 손가락까지) 거리 [mm]."""
    object_from_body = object_from_grasp @ grasp_from_tcp @ np.linalg.inv(body_from_tcp)
    result = {}
    for name, point in contacts.items():
        point_body = transform(point, np.linalg.inv(object_from_body)).astype(np.float32)[None]
        distances = {finger: float(scene.compute_distance(o3d.core.Tensor(point_body), nthreads=1).numpy()[0] * 1000)
                     for finger, scene in scenes.items()}
        result[name] = (distances[name], min(distances.values()))
    return result


def load_candidates(db_path, indices):
    """merged DB 행과 대응하는 raw 기록의 (접촉점, 닫힘 각도, object→grasp)."""
    merged = json.loads(Path(db_path).read_text())
    if merged["metadata"].get("schema") != "single_object_grasp_compact":
        raise ValueError("raw 기록과 직접 대응하는 single_object_grasp_compact DB만 지원한다")
    raw_dir, raw_files, candidates = Path(db_path).parent / "output_grasp", {}, []
    for index in indices:
        row = merged["data"][index]
        if row["scene_id"] not in raw_files:
            raw_files[row["scene_id"]] = json.loads((raw_dir / f"{row['scene_id']}.json").read_text())
        matches = [r for r in raw_files[row["scene_id"]] if r["source_pregrasp_index"] == row["source_pregrasp_index"]]
        if len(matches) != 1:
            raise ValueError(f"행 {index}에 대응하는 raw 기록이 하나가 아니다")
        raw, grasp = matches[0], np.asarray(row["grasp_mat"], dtype=float)
        if np.abs(np.linalg.inv(raw["quality"]["object_initial_mat"]) @ raw["grasp_mat"] - grasp).max() > 1e-5:
            raise ValueError(f"행 {index}: merged와 raw의 파지 행렬이 다르다")
        candidates.append(dict(
            row_index=index, object_from_grasp=grasp,
            spinner_rad=float(raw["quality"]["grasp_complete_joint_pos_rad"]["spinner_joint"]),
            contacts={c["sensor"].split("/")[-1]: np.asarray(c["position"], dtype=float)
                      for c in raw["contact_points_object_local"]}))
    return candidates


def summarize(values):
    values = np.asarray(values, dtype=float)
    return dict(median_mm=float(np.median(values)), p95_mm=float(np.percentile(values, 95)),
                rms_mm=float(np.sqrt(np.mean(values ** 2))), count=int(values.size))


def evaluate(candidates, body_from_tcp, grasp_from_tcp_options, urdf=PLANNING_URDF):
    matched = {name: [] for name in grasp_from_tcp_options}
    nearest = {name: [] for name in grasp_from_tcp_options}
    for candidate in candidates:
        scenes = finger_scenes(gripper_fingers(candidate["spinner_rad"], urdf))
        for name, grasp_from_tcp in grasp_from_tcp_options.items():
            distances = contact_distances_mm(
                scenes, candidate["contacts"], candidate["object_from_grasp"], grasp_from_tcp, body_from_tcp)
            for same, any_finger in distances.values():
                matched[name].append(same)
                nearest[name].append(any_finger)
    return {name: dict(same_finger=summarize(matched[name]), nearest_finger=summarize(nearest[name]))
            for name in grasp_from_tcp_options}


def axial_sweep(candidates, body_from_tcp, grasp_from_tcp, urdf=PLANNING_URDF, span_mm=80):
    """grasp Z 방향 이동만 바꿔 가장 가까운 손가락 RMS를 본다 (위치 0 가정의 점검용)."""
    offsets = np.arange(-span_mm, span_mm + 1) / 1000
    squares = [[] for _ in offsets]
    for candidate in candidates:
        scenes = finger_scenes(gripper_fingers(candidate["spinner_rad"], urdf))
        for k, offset in enumerate(offsets):
            shifted = grasp_from_tcp.copy()
            shifted[:3, 3] += [0.0, 0.0, offset]
            squares[k] += [d ** 2 for _, d in contact_distances_mm(
                scenes, candidate["contacts"], candidate["object_from_grasp"], shifted, body_from_tcp).values()]
    rms = np.sqrt([np.mean(s) for s in squares])
    best = int(np.argmin(rms))
    return dict(best_offset_mm=float(offsets[best] * 1000), rms_at_best_mm=float(rms[best]),
                rms_at_zero_mm=float(rms[len(offsets) // 2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="black_pepper_shaker", help="등록 물체 class_id")
    parser.add_argument("--num-candidates", type=int, default=30, help="DB 행 전체에서 같은 간격으로 고를 후보 수")
    parser.add_argument("--hardware-profile", type=Path, default=HARDWARE_PROFILE)
    parser.add_argument("--urdf", type=Path, default=PLANNING_URDF)
    parser.add_argument("--output", type=Path, default=WORK_ROOT / "audit/db-tcp-alignment")
    args = parser.parse_args(argv)
    catalog = load_catalog(DEFAULT_CATALOG)
    entry = next((e for e in catalog["objects"] if e["class_id"] == args.object), None)
    if entry is None:
        parser.error(f"등록되지 않은 물체: {args.object}")
    db_path = asset_path(catalog, entry, "db")
    rows = len(json.loads(db_path.read_text())["data"])
    indices = sorted(set(np.linspace(0, rows - 1, args.num_candidates).astype(int).tolist()))
    tool = json.loads(args.hardware_profile.read_text())["tool"]
    body_from_tcp = np.asarray(tool["gripper_body_from_tcp"], dtype=float)
    options = hypotheses(tool["db_grasp_from_tcp"])
    candidates = load_candidates(db_path, indices)
    results = evaluate(candidates, body_from_tcp, options, args.urdf)
    sweep = axial_sweep(candidates, body_from_tcp, options["declared"], args.urdf)
    print(f"{args.object}: DB 행 {len(candidates)}개, 접촉점 {results['declared']['nearest_finger']['count']}개")
    print(f"{'가정':<10} {'같은 번호 손가락 median/p95 [mm]':>36} {'가장 가까운 손가락 median/p95 [mm]':>38}")
    for name, value in sorted(results.items(), key=lambda item: item[1]["nearest_finger"]["rms_mm"]):
        a, b = value["same_finger"], value["nearest_finger"]
        same, nearest = f"{a['median_mm']:.2f} / {a['p95_mm']:.2f}", f"{b['median_mm']:.2f} / {b['p95_mm']:.2f}"
        print(f"{name:<10} {same:>24} {nearest:>30}")
    print(f"선언값에서 grasp Z 이동 탐색: 최소 RMS {sweep['rms_at_best_mm']:.2f} mm @ {sweep['best_offset_mm']:+.0f} mm "
          f"(이동 0에서는 {sweep['rms_at_zero_mm']:.2f} mm)")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(object=args.object, db=str(db_path), rows=indices, hypotheses=results, declared_axial_sweep=sweep,
                  hardware_profile=str(args.hardware_profile), physics_run=False, transform_adopted=False,
                  limitations=["Unsigned distances to visual triangles, not a PhysX collision test.",
                               "Contacts are per-finger averages, not corresponding material points.",
                               "Does not determine a unique 6D calibration."])
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"저장: {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
