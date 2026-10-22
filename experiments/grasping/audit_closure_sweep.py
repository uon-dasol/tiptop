"""Nominal closing sweep against table and both non-target conservative boxes."""

import json

import numpy as np
import yaml
from yourdfpy import URDF

from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import PROJECT_ROOT as ROOT
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json

cfg = yaml.safe_load((ROOT / "tiptop/doosan/assets/m1013_uon/m1013_uon.yml").read_text())["robot_cfg"]["kinematics"]
urdf = URDF.load(str(ROOT / "tiptop/doosan/assets/m1013_uon/m1013_uon.urdf"), load_meshes=False)
ensure_workdirs()
summary = []
for path in sorted((W / "plans").glob("multi-*/pick.json")):
    r = read_json(path)
    q = np.array(r["steps"][0]["positions"][-1])
    angle = r["grasp_joint_rad"]
    scope = "DB recorded closure estimate"
    result = W / "runs" / (path.parent.name + "-r01") / "check.json"
    if result.is_file():
        d = read_json(result)
        close = next((x for x in d.get("calls", []) if x.get("action") == "close" and x.get("success")), None)
        if close:
            angle = close["position_rad"]
            scope = "previous observed actual closure"
    boxes = [dict(name="table", dims=r["target_dims"], pose=[*r["target_xyz"], 1, 0, 0, 0])] + r[
        "distractor_collision_cuboids"
    ]
    best = float("inf")
    worst = None
    for a in np.linspace(0, angle, 61):
        urdf.update_cfg({**dict(zip([f"joint_{i}" for i in range(1, 7)], q)), "gripper_spinner_joint": float(a)})
        for link, spheres in cfg["collision_spheres"].items():
            if not link.startswith("gripper_"):
                continue
            transform = urdf.get_transform(link, "base_link")
            centers = np.array([s["center"] for s in spheres]) @ transform[:3, :3].T + transform[:3, 3]
            radii = np.array([s["radius"] for s in spheres])
            for box in boxes:
                distance = (
                    np.linalg.norm(
                        np.maximum(np.abs(centers - np.array(box["pose"][:3])) - np.array(box["dims"]) / 2, 0), axis=1
                    )
                    - radii
                )
                if float(distance.min()) < best:
                    best = float(distance.min())
                    worst = dict(gripper_angle_rad=float(a), link=link, obstacle=box["name"])
    item = dict(
        name=path.parent.name,
        angles=61,
        angle_scope=scope,
        min_nominal_sphere_box_clearance_m=best,
        passed=best >= 0,
        worst=worst,
        limit="Fixed nominal arm pose; checks gripper versus table/non-targets, not intended target contact or self-collision during the sweep.",
    )
    summary.append(item)
(WORK_ROOT / "audit/closure-sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
print("nominal closing sweeps", len(summary), "passed", sum(x["passed"] for x in summary))
for x in summary:
    if not x["passed"]:
        print(json.dumps(x))
