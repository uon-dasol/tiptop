"""Placement from observed held pose; retain the existing arm/executor and no force controller."""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from experiments.grasping import place_multi
from experiments.grasping.audit_multiscene_geometry import audit
from experiments.grasping.paths import DATA_ROOT, ensure_workdirs, read_json
from experiments.grasping.paths import PROJECT_ROOT as ROOT
from experiments.grasping.paths import WORK_ROOT as W


def pose(position, orientation):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(orientation)[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


mode = sys.argv[1] if len(sys.argv) > 1 else "initial"
ensure_workdirs()
summary = []
for check in sorted((DATA_ROOT / "runs").glob("multi-*/check.json")):
    if mode == "extensions" and not check.parent.name.startswith(("multi-rotated-", "multi-shifted-", "multi-dense-")):
        continue
    if mode == "initial" and check.parent.name.startswith("multi-alt-"):
        continue
    if mode == "alternatives" and not check.parent.name.startswith("multi-alt-"):
        continue
    d = read_json(check)
    r = read_json(check.parent / "pick.json")
    if not d.get("success") or not d.get("multi_object_verified") or r.get("post_pick_steps"):
        continue
    label = "measured-place-" + check.parent.name.removesuffix("-r01")
    out = W / "plans" / label
    out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    started = time.monotonic()
    try:
        held = next(x for x in d["trace"] if x["label"] == "awake_held")
        settled = next(x for x in d["trace"] if x["label"] == "settled")
        config = yaml.safe_load(
            (ROOT / "ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml").read_text()
        )["tcp"]
        body_tcp = np.eye(4)
        body_tcp[:3, :3] = Rotation.from_euler("xyz", config["rpy"]).as_matrix()
        body_tcp[:3, 3] = config["xyz"]
        world_tcp = pose(held["body_position"], held["body_orientation"]) @ body_tcp
        measured = np.linalg.inv(world_tcp) @ pose(held["object_position"], held["object_orientation"])
        r["body_from_tcp"] = body_tcp.tolist()
        r["observed_tcp_from_object"] = measured.tolist()
        r["observed_held_object_pose"] = pose(held["object_position"], held["object_orientation"]).tolist()
        r["observed_settled_object_pose"] = pose(settled["object_position"], settled["object_orientation"]).tolist()
        close = next(x for x in d["calls"] if x.get("action") == "close")
        r["observed_gripper_angle_rad"] = close["position_rad"]
        r["placement_source_sha256"] = d["source_sha256"]
        r["placement_spinner_torque_limit_nm"] = 5.0
        r["placement_observation_source"] = str(check)
        r = place_multi.prepare_placement(r, [0, 0.12], Path(r["candidate_source_path"]))
        geometry = audit(r, Path(r["candidate_source_path"]))
        (out / "full-geometry-audit.json").write_text(json.dumps(geometry, indent=2) + "\n")
        if not geometry["full_object_box_audit_passed"]:
            raise RuntimeError("Full target object box carry audit rejected")
        (out / "pick.json").write_text(json.dumps(r, indent=2) + "\n")
        item = dict(
            name=label,
            status="prepared",
            target_class=r["target_object"]["class"],
            candidate_index=r["candidate_index"],
            pick_path=str(out / "pick.json"),
            repetitions=1,
        )
    except Exception as e:
        item = dict(name=label, status="planning_rejected", error=str(e))
        (out / "error.txt").write_text(traceback.format_exc())
    item["seconds"] = time.monotonic() - started
    summary.append(item)
    print(json.dumps(item), flush=True)
    (W / "plans" / f"measured-{mode}-placement-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
