import csv
import json

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json


def rotation(row):
    def quat(k):
        return Rotation.from_quat(np.asarray(row[k])[[1, 2, 3, 0]])

    return quat("body_orientation").inv() * quat("object_orientation")


def percentiles(a):
    return np.percentile(a, [0, 5, 50, 95, 100]).tolist() if len(a) else None


ensure_workdirs()
rows = []
for path in sorted((W / "runs").glob("*/check.json")):
    d = read_json(path)
    r = read_json(path.parent / "pick.json")
    name = path.parent.name
    close = next((c for c in d.get("calls", []) if c.get("action") == "close"), {})
    labels = {x["label"]: x for x in d.get("trace", [])}
    expected_place = bool(r.get("post_pick_steps"))
    params = d.get("robot_parameters", {})
    armatures = dict(zip(params.get("dof_names", []), params.get("armatures", [])))
    validated = (
        d.get("success") is True
        and d.get("physical_grasp_verified") is True
        and not d.get("object_attached", True)
        and d.get("spinner_max_effort_effective_nm") == 5
        and armatures.get("joint_6") == 1
        and abs(d.get("physics_dt_s", 0) - 1 / 240) < 1e-12
        and all(x.get("success") is True for x in d.get("calls", []))
        and (not r.get("target_object") or d.get("multi_object_verified") is True)
        and (not expected_place or (d.get("physical_placement_verified") is True and d.get("phase") == "placed"))
    )
    passive = d.get("passive_trace", [])
    carry = [x for x in passive if x["phase"] == "transfer_and_place"]
    awake = [x for x in carry if not x["object_sleeping"]]
    hold = [x for x in passive if x["phase"] == "hold"]
    ah = [x for x in hold if not x["object_sleeping"]]
    ref = labels.get("awake_held")
    rel_rot_max = None
    if ref and carry:
        rr = rotation(ref)
        rel_rot_max = float(np.degrees(max((rr.inv() * rotation(x)).magnitude() for x in carry)))
    wrist = max(
        (
            float(abs(np.degrees(np.asarray(x["joint_positions"])[-1] - np.asarray(x["joint_targets"])[-1])))
            for x in passive
        ),
        default=None,
    )
    offset = r.get("placement", {}).get("offset_xy_m")
    hold_slip = hold_rotation = None
    if ref and hold:
        qb = Rotation.from_quat(np.asarray(ref["body_orientation"])[[1, 2, 3, 0]])
        pb = qb.inv().apply(np.asarray(ref["object_position"]) - np.asarray(ref["body_position"]))
        rr = rotation(ref)
        hold_slip = float(
            max(
                np.linalg.norm(
                    Rotation.from_quat(np.asarray(x["body_orientation"])[[1, 2, 3, 0]])
                    .inv()
                    .apply(np.asarray(x["object_position"]) - np.asarray(x["body_position"]))
                    - pb
                )
                for x in hold
            )
            * 1000
        )
        hold_rotation = float(np.degrees(max((rr.inv() * rotation(x)).magnitude() for x in hold)))
    row = dict(
        name=name,
        scene_layout=r.get("scene_layout"),
        target_class=r.get("target_object", {}).get("class", "black_pepper_shaker"),
        multi_object=bool(r.get("target_object")),
        multi_object_verified=d.get("multi_object_verified"),
        candidate=r["candidate_index"],
        offset_xy_m=offset,
        object_xyz=np.array(r["world_from_object"])[:3, 3].tolist(),
        expected_place=expected_place,
        success=d.get("success", False),
        validated=bool(validated),
        phase=d.get("phase"),
        error=d.get("error"),
        lift_mm=None if d.get("min_hold_lift_m") is None else d["min_hold_lift_m"] * 1000,
        slip_mm=None if d.get("max_carry_slip_m") is None else d["max_carry_slip_m"] * 1000,
        placement_xy_mm=None if d.get("placement_xy_error_m") is None else d["placement_xy_error_m"] * 1000,
        placement_z_mm=None if d.get("placement_height_error_m") is None else d["placement_height_error_m"] * 1000,
        closed_force_n=sum(close.get("contact_force_n", [])) if close else None,
        closed_fingers=close.get("contact_fingers"),
        close_angle_rad=close.get("position_rad"),
        close_target_rad=close.get("target_rad"),
        threshold_reached=close.get("contact_stop_threshold_reached"),
        awake_hold_force_n=sum(labels["awake_held"]["finger_forces_n"]) if ref else None,
        before_release_force_n=sum(labels["before_release"]["finger_forces_n"]) if "before_release" in labels else None,
        carry_force_min_p05_median_p95_max_n=percentiles([sum(x["finger_forces_n"]) for x in awake]),
        hold_force_min_p05_median_p95_max_n=percentiles([sum(x["finger_forces_n"]) for x in ah]),
        carry_samples=len(carry),
        carry_awake_samples=len(awake),
        hold_samples=len(hold),
        hold_awake_samples=len(ah),
        max_hold_relative_displacement_mm=hold_slip,
        max_hold_relative_rotation_deg=hold_rotation,
        max_distractor_displacement_mm=max(
            (x["displacement_m"] * 1000 for x in d.get("distractor_trace", [])), default=None
        ),
        max_robot_distractor_force_n=d.get("max_robot_distractor_force_n"),
        distractor_final=d.get("distractor_final"),
        max_relative_rotation_deg=rel_rot_max,
        max_observed_wrist_tracking_deg=wrist,
        final_table_support_n=labels.get("placed", {}).get("table_object_force_n"),
        final_finger_force_n=sum(labels["placed"]["finger_forces_n"]) if "placed" in labels else None,
        input_path=str(path.parent / "pick.json"),
        result_path=str(path),
    )
    rows.append(row)
(WORK_ROOT / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
if rows:
    with (WORK_ROOT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "completed",
        len(rows),
        "validated_success",
        sum(x["validated"] for x in rows),
        "place",
        sum(x["validated"] and x["expected_place"] for x in rows),
        "pick_only",
        sum(x["validated"] and not x["expected_place"] for x in rows),
    )
    for key in [
        "closed_force_n",
        "awake_hold_force_n",
        "slip_mm",
        "placement_xy_mm",
        "max_relative_rotation_deg",
        "max_observed_wrist_tracking_deg",
    ]:
        vals = [x[key] for x in rows if x["validated"] and x[key] is not None]
        if vals:
            print(key, "min/median/max", np.round(np.percentile(vals, [0, 50, 100]), 5).tolist())
    for x in rows:
        if not x["validated"]:
            print("FAILED", x["name"], x["phase"], x["error"])
