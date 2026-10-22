"""Read local CAD+UON DB, screen all candidates with two other classes present.

World collision uses conservative full-object AABBs for non-target objects.
No source assets or DBs are copied or modified.
"""

import copy
import hashlib
import json
import time
import traceback

import numpy as np
import torch
import trimesh
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from cutamp.utils.shapes import sample_greedy_surface_spheres
from scipy.spatial.transform import Rotation

from experiments.grasping import pick_multi
from experiments.grasping.paths import PROJECT_ROOT as ROOT
from experiments.grasping.paths import WORK_ROOT as W
from experiments.grasping.paths import ensure_workdirs, read_json
from tiptop.doosan.model import m1013_uon_curobo_cfg

ensure_workdirs()
names = ["black_pepper_shaker", "car_leather_conditioner_bottle", "disney_mickey_handled_dipper"]
assets = {}
rows = {}
vertices = {}
proxies = {}
source_hashes = {}
for name in names:
    base = ROOT / "data" / name
    cad = base / "3d_model/edited" / f"{name}.obj"
    usd = base / "3d_model/edited" / f"{name}_rigid.usd"
    db = base / "UON_3finger_gripper"
    merged = db / f"{name}_merged_grasp_zero_pose.json"
    data = read_json(merged)
    assert data["metadata"]["coordinate_frame"] == "object_zero_pose"
    assert data["metadata"]["gripper_model"] == "UON_3finger_gripper"
    scale = data["metadata"]["object_config_zero_pose"]["scale"]
    assert scale == [0.1, 0.1, 0.1]
    mesh = trimesh.load(str(cad), force="mesh", process=False)
    mesh.apply_scale(0.001)
    v = np.asarray(mesh.vertices)
    vertices[name] = v
    rows[name] = data["data"]
    if name == "black_pepper_shaker":
        proxies[name] = np.load(ROOT / "tiptop_outputs/m1013/tamp/diagnosis-wide/initial.npz")["local_object_spheres"]
    else:
        torch.manual_seed(0)
        np.random.seed(0)
        shape = Mesh(name=name, pose=[0, 0, 0, 1, 0, 0, 0], vertices=v.tolist(), faces=mesh.faces.tolist())
        proxies[name] = sample_greedy_surface_spheres(shape, n_spheres=32, sphere_radius=0.003).cpu().numpy()
    assets[name] = dict(
        class_name=name,
        asset=str(usd),
        cad=str(cad),
        db_directory=str(db),
        asset_scale=scale,
        mass_source="existing rigid USD",
        db_count=len(rows[name]),
    )
    for p in [usd, cad, merged]:
        source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
(W / "audit/multiscene-source-sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
tool_spheres = np.load(ROOT / "tiptop/doosan/assets/m1013_uon/gripper_spheres_open.npy")
old = read_json(ROOT / "tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json")
summary = []
layouts = {
    "wide": [(0.50, -0.20, 0), (0.50, 0.12, 0), (0.82, -0.06, 0)],
    "near": [(0.55, -0.17, 0), (0.57, 0.13, 15), (0.76, -0.10, -15)],
}
for layout, coords in layouts.items():
    entries = []
    for name, (x, y, yaw) in zip(names, coords):
        t = np.eye(4)
        t[:3, :3] = (
            Rotation.from_euler("z", yaw, degrees=True).as_matrix()
            @ Rotation.from_euler("x", 90, degrees=True).as_matrix()
        )
        t[:2, 3] = [x, y]
        t[2, 3] = 0.4 - float((vertices[name] @ t[:3, :3].T)[:, 2].min()) + 0.002
        v = vertices[name] @ t[:3, :3].T + t[:3, 3]
        entry = {
            "class": name,
            **{k: v for k, v in assets[name].items() if k != "class_name"},
            "instance_id": name + "-01",
            "world_from_object": t.tolist(),
            "world_bounds": [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
        }
        entries.append(entry)
    (W / "plans" / f"scene-{layout}.json").write_text(json.dumps(entries, indent=2) + "\n")
    for target_index, target0 in enumerate(entries):
        torch.manual_seed(0)
        np.random.seed(0)
        started = time.monotonic()
        name = target0["class"]
        label = f"multi-{layout}-{name}"
        out = W / "plans" / label
        out.mkdir(exist_ok=True)
        target = copy.deepcopy(target0)
        target.update(root_path="/World/Pepper", prim_path="/World/Pepper/Asset/" + name)
        distractors = []
        boxes = []
        for i, other0 in enumerate(x for j, x in enumerate(entries) if j != target_index):
            other = copy.deepcopy(other0)
            other.update(root_path=f"/World/Distractor_{i}", prim_path=f"/World/Distractor_{i}/Asset/{other['class']}")
            distractors.append(other)
            low, high = np.array(other["world_bounds"])
            center = (low + high) / 2
            boxes.append(
                dict(
                    name="obstacle_" + other["class"],
                    dims=(high - low + 0.006).tolist(),
                    pose=[*center.tolist(), 1, 0, 0, 0],
                )
            )
        try:
            t = np.array(target["world_from_object"])
            g = np.array([r["grasp_mat"] for r in rows[name]])
            scores = np.array([r["score"] for r in rows[name]])
            tcp = t @ g
            tool = np.einsum("nij,kj->nki", tcp[:, :3, :3], tool_spheres[:, :3]) + tcp[:, None, :3, 3]
            local = proxies[name]
            objworld = local[:, :3] @ t[:3, :3].T + t[:3, 3]
            delta = tool[:, :, None, :] - objworld[None, None, :, :]
            overlap = np.maximum(
                tool_spheres[None, :, 3, None]
                + local[None, None, :, 3]
                - np.sqrt(np.sum(delta * delta, axis=-1) + 1e-8),
                0,
            ).sum(axis=(1, 2))
            clearance = (tool[:, :, 2] - tool_spheres[None, :, 3] - 0.4).min(axis=1)
            downward = -tcp[:, 2, 2] >= np.cos(np.deg2rad(15))
            valid = np.flatnonzero(downward & (overlap <= 0.001) & (clearance >= 0.002))
            valid = valid[np.argsort(-scores[valid], kind="stable")]
            world = WorldConfig(
                cuboid=[Cuboid(name="target", dims=[0.8, 0.8, 0.04], pose=[0.65, 0, 0.38, 1, 0, 0, 0])]
                + [Cuboid(name=b["name"], dims=b["dims"], pose=b["pose"]) for b in boxes]
            )
            solver = IKSolver(
                IKSolverConfig.load_from_robot_config(
                    copy.deepcopy(m1013_uon_curobo_cfg(0.0)["robot_cfg"]),
                    world,
                    num_seeds=64,
                    self_collision_check=True,
                    self_collision_opt=True,
                    position_threshold=0.0005,
                    rotation_threshold=0.003,
                    high_precision=True,
                    use_cuda_graph=False,
                )
            )
            successes = []
            if len(valid):
                result = solver.solve_batch(Pose.from_matrix(solver.tensor_args.to_device(tcp[valid])), return_seeds=8)
                limits = solver.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
                ref = np.array(old["successes"][0]["q"])
                for i, index in enumerate(valid):
                    choices = result.solution[i][result.success[i]].cpu().numpy()
                    if not len(choices):
                        continue
                    wrapped = choices + 2 * np.pi * np.round((ref - choices) / (2 * np.pi))
                    choices = np.where((wrapped >= limits[0]) & (wrapped <= limits[1]), wrapped, choices)
                    q = choices[np.linalg.norm(choices - ref, axis=1).argmin()]
                    successes.append(
                        dict(
                            candidate_index=int(index),
                            score=float(scores[index]),
                            q=q.tolist(),
                            world_from_tcp=tcp[index].tolist(),
                        )
                    )
            source = dict(
                screened=len(valid),
                world_from_object=t.tolist(),
                successes=successes,
                target_object=target,
                distractors=distractors,
                distractor_collision_cuboids=boxes,
                scene_layout=layout,
                screening=dict(
                    total=len(g),
                    topdown=int(downward.sum()),
                    sphere_clear=int((downward & (overlap <= 0.001)).sum()),
                    table_clear_candidates=len(valid),
                    ik_success=len(successes),
                    obstacles="two full CAD world AABBs + 3 mm margin each side",
                ),
                object_spheres_proxy_only=True,
            )
            (out / "candidate-ik.json").write_text(json.dumps(source, indent=2) + "\n")
            np.savez(out / "initial.npz", local_object_spheres=local)
            if not successes:
                raise RuntimeError("No candidate after target sphere/table screen and all-object IK")
            # Screen all DB candidates, then prepare only the first feasible-IK candidate.
            # The retired score-ordered retry policy is not duplicated here.
            record = pick_multi.prepare(out, 0, candidates_path=out / "candidate-ik.json", save=True)
            item = dict(
                name=label,
                status="prepared",
                pick_path=str(out / "pick.json"),
                target_class=name,
                candidate_index=record["candidate_index"],
                rank=0,
                screening=source["screening"],
                repetitions=1,
            )
        except Exception as e:
            item = dict(name=label, status="planning_rejected", target_class=name, error=str(e))
            (out / "error.txt").write_text(traceback.format_exc())
        item["seconds"] = time.monotonic() - started
        summary.append(item)
        print(json.dumps(item), flush=True)
        (W / "plans/multi-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
