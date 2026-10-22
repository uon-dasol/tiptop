import json
from pathlib import Path

import numpy as np

from experiments.grasping.known_scene_adapter import Instance, db_to_planner
from experiments.grasping.paths import DATA_ROOT as W
from experiments.grasping.paths import WORK_ROOT, ensure_workdirs, read_json

ensure_workdirs()
summary = []
for cls in ["black_pepper_shaker", "car_leather_conditioner_bottle", "disney_mickey_handled_dipper"]:
    record = read_json(W / "plans" / f"multi-wide-{cls}/pick.json")
    source = read_json(Path(record["candidate_source_path"]))
    merged = read_json(Path(record["target_object"]["db_directory"]) / f"{cls}_merged_grasp_zero_pose.json")
    t = np.asarray(record["world_from_object"])
    cad = Path(record["target_object"]["cad"])
    v = (
        np.array(
            [[float(x) for x in line.split()[1:4]] for line in cad.read_text().splitlines() if line.startswith("v ")]
        )
        * 0.001
    )
    center = (v.min(axis=0) + v.max(axis=0)) / 2
    cad_from_center = np.eye(4)
    cad_from_center[:3, 3] = center
    world_planner = t @ cad_from_center
    instance = Instance(cls + "-01", cls, t, "Isaac_scene_initialization_ground_truth")
    ids = [x["candidate_index"] for x in source["successes"][:16]]
    rows = [dict(candidate_index=i, **merged["data"][i]) for i in ids]
    result = db_to_planner(rows, instance, world_planner)
    errors = []
    for pose, meta in zip(result["grasps_obj"], result["metadata"]):
        expected = t @ np.asarray(merged["data"][meta["candidate_index"]]["grasp_mat"])
        errors.append(float(np.abs(world_planner @ pose - expected).max()))
    assert max(errors) < 1e-10
    summary.append(
        dict(class_name=cls, candidates=len(ids), max_frame_roundtrip_error=max(errors), planner_object_recentered=True)
    )
(WORK_ROOT / "audit/local-db-bridge.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
