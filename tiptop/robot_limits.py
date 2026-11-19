"""Planner limits editable from the launcher; defaults live here, saved edits in config/robot_limits.json."""

import json
import math
from pathlib import Path

PATH = Path(__file__).parent / "config/robot_limits.json"
NUM_JOINTS = 6
DEFAULTS = dict(
    max_acceleration=[5.0, 5.0, 5.0, 5.0, 5.0, 7.0],  # rad/s^2, J1..J6
    max_jerk=[160.0, 160.0, 160.0, 160.0, 160.0, 224.0],  # rad/s^3, J1..J6
    terminal_velocity_tolerance_rad_s=0.015,
)
# Typo guards only; they are not limits of the arm.
BOUNDS = dict(max_acceleration=(0.1, 20.0), max_jerk=(1.0, 1000.0), terminal_velocity_tolerance_rad_s=(1e-4, 0.1))


def _number(name, value):
    low, high = BOUNDS[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")
    return float(value)


def validate_limits(values: dict) -> dict:
    """Return a clean copy with every field present and in range."""
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown robot limit fields: {sorted(unknown)}")
    result = {}
    for name in ("max_acceleration", "max_jerk"):
        entries = values.get(name, DEFAULTS[name])
        if not isinstance(entries, (list, tuple)) or len(entries) != NUM_JOINTS:
            raise ValueError(f"{name} needs {NUM_JOINTS} values, one per joint")
        result[name] = [_number(name, entry) for entry in entries]
    name = "terminal_velocity_tolerance_rad_s"
    result[name] = _number(name, values.get(name, DEFAULTS[name]))
    return result


def load_robot_limits(path: Path | None = None) -> dict:
    """The saved limits, or the defaults when nothing was saved."""
    path = Path(PATH if path is None else path)
    return validate_limits(json.loads(path.read_text())) if path.is_file() else validate_limits(DEFAULTS)


def save_robot_limits(values: dict, path: Path | None = None) -> dict:
    path = Path(PATH if path is None else path)
    clean = validate_limits(values)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean, indent=2) + "\n")
    temporary.replace(path)
    return clean
