"""Conservative full-CAD-box carry checks supplement sparse object sphere proxies."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from scipy.interpolate import CubicHermiteSpline

from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json
from tiptop.doosan.model import m1013_uon_curobo_cfg


def audit(record, candidates_path):
    source = read_json(candidates_path)
    selected = next(x for x in source["successes"] if x["candidate_index"] == record["candidate_index"])
    tcp_obj = (
        np.asarray(record["observed_tcp_from_object"])
        if record.get("observed_tcp_from_object")
        else np.linalg.inv(np.asarray(selected["world_from_tcp"])) @ np.asarray(record["world_from_object"])
    )
    cad = Path(record["target_object"]["cad"])
    vertices = (
        np.asarray(
            [[float(x) for x in line.split()[1:4]] for line in cad.read_text().splitlines() if line.startswith("v ")]
        )
        * 0.001
    )
    model = CudaRobotModel(
        RobotConfig.from_dict(
            copy.deepcopy(
                m1013_uon_curobo_cfg(record.get("observed_gripper_angle_rad", record["grasp_joint_rad"]))["robot_cfg"]
            ),
            TensorDeviceType(),
        ).kinematics
    )
    results = []
    for step in record["steps"] + record.get("post_pick_steps", []):
        if step["type"] != "trajectory" or step["label"] not in ("lift", "transfer", "lower to place"):
            continue
        q = np.array(step["positions"])
        v = np.array(step["velocities"])
        stamps = np.arange(len(q) + 1) * step["dt"]
        curve = CubicHermiteSpline(stamps, np.vstack([q[:1], q]), np.vstack([np.zeros((1, 6)), v]))
        time = np.linspace(0, stamps[-1], int(round(stamps[-1] * 240)) + 1)
        tcp = model.get_state(model.tensor_args.to_device(curve(time))).ee_pose.get_matrix().cpu().numpy()
        attach = (
            np.linalg.inv(np.asarray(selected["world_from_tcp"])) @ np.asarray(record["world_from_object"])
            if step["label"] == "lift"
            else tcp_obj
        )
        object_rotation = tcp[:, :3, :3] @ attach[:3, :3]
        object_position = np.einsum("nij,j->ni", tcp[:, :3, :3], attach[:3, 3]) + tcp[:, :3, 3]
        lows = []
        highs = []
        for begin in range(0, len(tcp), 16):
            transformed = (
                np.einsum("nij,kj->nki", object_rotation[begin : begin + 16], vertices)
                + object_position[begin : begin + 16, None, :]
            )
            lows.append(transformed.min(axis=1))
            highs.append(transformed.max(axis=1))
        low, high = np.concatenate(lows), np.concatenate(highs)
        min_gap = float("inf")
        collision_samples = 0
        for box in record["distractor_collision_cuboids"]:
            center = np.array(box["pose"][:3])
            half = np.array(box["dims"]) / 2
            # Positive gap in at least one axis proves these conservative AABBs disjoint.
            gap = np.maximum(center - half - high, low - center - half).max(axis=1)
            min_gap = min(min_gap, float(gap.min()))
            collision_samples += int(np.count_nonzero(gap < 0))
        table_xy = np.array(record["target_xyz"])[:2]
        half = np.array(record["target_dims"])[:2] / 2
        tabletop = float(record["target_xyz"][2] + record["target_dims"][2] / 2)
        edge = np.minimum(low[:, :2] - (table_xy - half), (table_xy + half) - high[:, :2]).min()
        result = dict(
            label=step["label"],
            attachment_model="nominal_at_lift_start"
            if step["label"] == "lift"
            else "observed_held"
            if record.get("observed_tcp_from_object")
            else "nominal",
            samples=len(time),
            min_full_object_box_table_clearance_m=float(low[:, 2].min() - tabletop),
            min_full_object_box_distractor_separating_gap_m=min_gap,
            full_object_box_collision_samples=collision_samples,
            min_full_object_box_table_edge_margin_m=float(edge),
        )
        results.append(result)
    success = all(
        x["min_full_object_box_table_clearance_m"] >= -0.001
        and x["full_object_box_collision_samples"] == 0
        and x["min_full_object_box_table_edge_margin_m"] >= 0.005
        for x in results
    )
    return dict(
        full_object_box_audit_passed=success,
        bounds_source="all CAD vertices transformed at each sample",
        checks=results,
        note="Conservative object AABBs sampled at 240 Hz; does not prove continuous collision freedom or the gripper open/close sweep.",
    )


if __name__ == "__main__":
    ensure_workdirs()
    summary = []
    patterns = (
        ["multi-rotated-*/pick.json", "multi-shifted-*/pick.json", "multi-dense-*/pick.json"]
        if "--extensions" in sys.argv
        else ["multi-alt-*/pick.json"]
        if "--alternatives" in sys.argv
        else ["multi-*/pick.json", "place-multi-*/pick.json"]
    )
    for path in [x for pattern in patterns for x in sorted((W / "plans").glob(pattern))]:
        r = read_json(path)
        out = audit(r, Path(r["candidate_source_path"]))
        out["name"] = path.parent.name
        destination = WORK_ROOT / "plans" / path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "full-geometry-audit.json").write_text(json.dumps(out, indent=2) + "\n")
        summary.append(out)
        print(json.dumps(out), flush=True)
    (
        WORK_ROOT
        / "audit"
        / (
            "multiscene-extension-geometry.json"
            if "--extensions" in sys.argv
            else "multiscene-alternative-geometry.json"
            if "--alternatives" in sys.argv
            else "multiscene-full-geometry.json"
        )
    ).write_text(json.dumps(summary, indent=2) + "\n")
