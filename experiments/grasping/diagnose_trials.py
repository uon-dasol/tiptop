import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import PROJECT_ROOT as ROOT
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json

config = yaml.safe_load((ROOT / "ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml").read_text())[
    "tcp"
]
body_tcp = np.eye(4)
body_tcp[:3, :3] = Rotation.from_euler("xyz", config["rpy"]).as_matrix()
body_tcp[:3, 3] = config["xyz"]


def matrix(row, prefix):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.array(row[prefix + "_orientation"])[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = row[prefix + "_position"]
    return t


def rel(row):
    return np.linalg.inv(matrix(row, "body") @ body_tcp) @ matrix(row, "object")


ensure_workdirs()
summary = []
for check in sorted((W / "runs").glob("*/check.json")):
    d = read_json(check)
    r = read_json(check.parent / "pick.json")
    labels = {x["label"]: x for x in d.get("trace", [])}
    if not {"closed", "awake_held", "settled"}.issubset(labels):
        continue
    closed = rel(labels["closed"])
    held = rel(labels["awake_held"])
    world_rotation = np.degrees(
        Rotation.from_matrix(
            matrix(labels["settled"], "object")[:3, :3].T @ matrix(labels["awake_held"], "object")[:3, :3]
        ).magnitude()
    )
    mismatch = None
    candidate_path = r.get("candidate_source_path")
    if candidate_path:
        source = read_json(Path(candidate_path))
    else:
        source = read_json(ROOT / "tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json")
    if any(x["candidate_index"] == r["candidate_index"] for x in source["successes"]):
        selected = next(x for x in source["successes"] if x["candidate_index"] == r["candidate_index"])
        nominal = np.linalg.inv(np.array(selected["world_from_tcp"])) @ np.array(r["world_from_object"])
        mismatch = dict(
            translation_mm=float(np.linalg.norm(held[:3, 3] - nominal[:3, 3]) * 1000),
            rotation_deg=float(np.degrees(Rotation.from_matrix(nominal[:3, :3].T @ held[:3, :3]).magnitude())),
        )
    actual_xy = actual_z = None
    if "placed" in labels:
        goal = np.asarray(labels["settled"]["object_position"]).copy()
        goal[:2] += r["placement"]["offset_xy_m"]
        delta = np.asarray(labels["placed"]["object_position"]) - goal
        actual_xy = float(np.linalg.norm(delta[:2]) * 1000)
        actual_z = float(abs(delta[2]) * 1000)
    item = dict(
        name=check.parent.name,
        target_class=r.get("target_object", {}).get("class", "black_pepper_shaker"),
        candidate=r["candidate_index"],
        success=d.get("success"),
        error=d.get("error"),
        phase=d.get("phase"),
        closed_to_held_relative_translation_mm=float(np.linalg.norm(held[:3, 3] - closed[:3, 3]) * 1000),
        closed_to_held_relative_rotation_deg=float(
            np.degrees(Rotation.from_matrix(closed[:3, :3].T @ held[:3, :3]).magnitude())
        ),
        settled_to_held_world_rotation_deg=float(world_rotation),
        nominal_to_observed_held=mismatch,
        actual_final_xy_error_mm=actual_xy,
        actual_final_z_error_mm=actual_z,
        held_pose_replay_error=d.get("held_pose_replay_error"),
        max_carry_slip_mm=d.get("max_carry_slip_m", 0) * 1000,
        required_actions=len(r["steps"]) + len(r.get("post_pick_steps", [])),
        completed_calls=len(d.get("calls", [])),
        action_success_flags=[c.get("success") for c in d.get("calls", [])],
        check=str(check),
    )
    summary.append(item)
(WORK_ROOT / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
print("diagnosed", len(summary))
for x in summary:
    if not x["success"] or x["closed_to_held_relative_translation_mm"] > 10:
        print(
            json.dumps(
                {k: v for k, v in x.items() if k not in ["check", "action_success_flags", "nominal_to_observed_held"]},
                ensure_ascii=False,
            )
        )
