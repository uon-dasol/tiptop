"""Per-observation execution records and orientation-independent evaluation.

Physics measurements belong to evaluation only, never to perception/planning.
"""

import json
from pathlib import Path
import re
from threading import Lock

import numpy as np


class ExecutionStore:
    """One active motion, immutable task identities, and queryable old results."""

    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active_id = self.latest_id = None
        self.lock = Lock()

    def directory(self, observation_id):
        if not isinstance(observation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", observation_id):
            raise ValueError("Invalid observation ID")
        return self.output / "executions" / observation_id

    @staticmethod
    def write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def accept(self, plan):
        with self.lock:
            if self.active_id is not None:
                raise ValueError("An execution is already active")
            observation_id = plan["observation_id"]
            directory = self.directory(observation_id)
            # Also reject replay after a process restart; do not overwrite evidence.
            directory.mkdir(parents=True, exist_ok=False)
            self.write(directory / "accepted-plan.json", plan)
            pending = dict(observation_id=observation_id, finished=False, success=False)
            self.write(directory / "execution.json", pending)
            self.write(self.output / "accepted-plan.json", plan)
            self.write(self.output / "execution.json", pending)
            self.active_id = self.latest_id = observation_id
            return directory

    def finish(self, report):
        with self.lock:
            if report["observation_id"] != self.active_id:
                raise ValueError("Result does not belong to the active execution")
            if report.get("finished") is not True:
                raise ValueError("Execution result is not terminal")
            self.write(self.directory(self.active_id) / "execution.json", report)
            self.write(self.output / "execution.json", report)
            self.active_id = None

    def result(self, observation_id=None):
        with self.lock:
            observation_id = observation_id or self.latest_id
            if observation_id is None:
                return dict(finished=False)
            path = self.directory(observation_id) / "execution.json"
            if not path.is_file():
                return dict(observation_id=observation_id, finished=False)
            return json.loads(path.read_text())


def evaluate_pick_place(
    *, baseline, final, trace, forces, support, region, max_other_displacement, unintended_drop=False
):
    """Accept lying placements. Neither orientation nor final origin Z is a gate."""
    baseline, final = np.asarray(baseline, float), np.asarray(final, float)
    forces = np.asarray(forces, float)
    positions = np.asarray([row["position"] for row in trace], float)
    if (
        baseline.shape != (3,)
        or final.shape != (3,)
        or positions.ndim != 2
        or positions.shape[1] != 3
        or not len(positions)
        or not forces.size
        or not all(np.isfinite(value).all() for value in (baseline, final, positions, forces))
        or not np.isfinite([support, max_other_displacement]).all()
    ):
        raise ValueError("Evaluation requires finite measured positions, contacts and support")
    size = np.asarray(region.get("size_xy_m", region.get("size_m")), float)
    center = np.asarray(region["center_xy_m"], float)
    if size.shape not in ((), (2,)) or center.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Invalid observed placement region")
    in_region = bool(np.all(np.abs(final[:2] - center) <= size / 2 + 0.01))
    lift = float(np.max(positions[:, 2] - baseline[2]))
    moved = float(np.linalg.norm(final[:2] - baseline[:2]))
    checks = dict(
        lifted=lift > 0.03,
        transported=moved > 0.1,
        inside_selected_region=in_region,
        table_supported=support > 0.01,
        gripper_released=float(forces.max()) < 0.1,
        other_objects_undisturbed=max_other_displacement <= 0.005,
        no_unintended_drop=not unintended_drop,
    )
    success = all(checks.values())
    return dict(
        success=success,
        checks=checks,
        maximum_lift_m=lift,
        horizontal_displacement_m=moved,
        final_origin_height_change_m=float(final[2] - baseline[2]),
        final_forces_n=forces.tolist(),
        table_support_force_n=float(support),
        inside_selected_region=in_region,
        empty_region=region,
        unintended_drop_detected=bool(unintended_drop),
        physical_grasp_verified=checks["lifted"],
        physical_placement_verified=success,
        orientation_required=False,
    )
