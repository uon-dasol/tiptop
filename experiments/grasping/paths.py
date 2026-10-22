"""Paths shared by experiment commands; importing this module does no work."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("TIPTOP_EXPERIMENT_DATA", PROJECT_ROOT / "tiptop_outputs/experiments/grasping/20261020")
).resolve()
WORK_ROOT = Path(os.environ.get("TIPTOP_EXPERIMENT_WORK", PROJECT_ROOT / "tiptop_outputs/experiments/work")).resolve()


def relocated_path(value):
    """Resolve former absolute experiment paths without rewriting saved evidence."""
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    markers = (
        "/simulation/isaac_sim/experiments/nextstage-20261020",
        "/tmp/tiptop-nextstage-20261020",
        "/tiptop_outputs/experiments/grasping/20261020",
    )
    for marker in markers:
        prefix, found, suffix = value.partition(marker)
        if found and (not suffix or suffix.startswith("/")):
            return str(DATA_ROOT) + suffix
    return value


def read_json(path):
    """Load relocated inputs and remap path-valued keys and values recursively."""

    def visit(value):
        if isinstance(value, dict):
            return {relocated_path(key): visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return relocated_path(value)

    return visit(json.loads(Path(relocated_path(str(path))).read_text()))


def ensure_workdirs():
    """Create only explicitly requested scratch output directories."""
    if WORK_ROOT == DATA_ROOT or WORK_ROOT.is_relative_to(DATA_ROOT):
        raise ValueError("Experiment work directory must be outside the input archive")
    for name in ("plans", "runs", "audit", "logs"):
        (WORK_ROOT / name).mkdir(parents=True, exist_ok=True)
