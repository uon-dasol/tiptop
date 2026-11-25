"""Command construction for the run_known_pick_place launcher; no process or device access."""

import json
import math
import os
import shlex
import sys
from pathlib import Path

from tiptop.object_catalog import load_catalog
from tiptop.task_presets import TASK_PRESETS, TaskSettings, task_defaults

PRESETS_PATH = Path(__file__).parent / "config/launcher_presets.json"
RUNNER_MODULE = "tiptop.scripts.run_known_pick_place"
MODES = ("sam3_reference", "gemini_er")
KINDS = ("plan", "execute", "check")
TABLE_SUPPORT = ("ransac-cad", "ransac-mask", "declared")

_SAM3_ONLY = ["sam3_reference"]
# hidden fields have no input box; their preset value still goes into the command.
FIELDS = (
    dict(key="task_preset", label="작업 프리셋", type="select", options=list(TASK_PRESETS),
         option_labels={key: value["label"] for key, value in TASK_PRESETS.items()}, modes=None, hidden=True),
    dict(key="task", label="지시문", type="text", modes=None),
    dict(key="release_height_m", label="목적지 상단에서 놓는 높이 (m)", type="number", modes=None,
         tasks=["fixed_box", "perceived_surface"], hidden=True),
    *[dict(key=key, label=label, type="number", modes=None, tasks=["fixed_box"], hidden=True) for key, label in (
        ("box_center_x_m", "박스 중심 X (m)"), ("box_center_y_m", "박스 중심 Y (m)"),
        ("box_bottom_z_m", "박스 바닥 Z (m)"), ("box_size_x_m", "박스 X 길이 (m)"),
        ("box_size_y_m", "박스 Y 길이 (m)"), ("box_size_z_m", "박스 높이 (m)"),
    )],
    dict(key="worker_socket", label="작업 워커 소켓 (비우면 워커 없이 실행)", type="text", modes=_SAM3_ONLY, hidden=True),
    dict(key="reference_scene", label="SAM3 참조 장면 (비우면 설정 기본값)", type="text", modes=_SAM3_ONLY),
    dict(key="camera_url", label="관측 서버 URL", type="text", modes=None, hidden=True),
    dict(key="hardware_profile", label="하드웨어 프로필", type="text", modes=None, hidden=True),
    dict(key="gateway_url", label="Gateway URL", type="text", modes=None, hidden=True),
    dict(key="max_approach_tilt_deg", label="최대 접근 기울기 (deg)", type="number", modes=None),
    dict(key="cad_mask_min_iou", label="CAD-마스크 최소 IoU", type="number", modes=None),
    dict(key="min_table_clearance_m", label="최소 테이블 여유 (m)", type="number", modes=None),
    dict(key="max_wrist_travel_deg", label="최대 손목 이동 (deg)", type="number", modes=None),
    dict(key="footprint_overhang_tolerance_mm", label="박스 윗면 이탈 허용 (mm)", type="number", modes=None),
    dict(key="min_engagement_depth_m", label="최소 파지 깊이 (m)", type="number", modes=None),
    dict(key="num_particles", label="후보 파지 수 (= 입자 수 상한)", type="number", modes=None),
    dict(key="sam3_confidence_threshold", label="SAM3 신뢰도 임계값 (초과한 후보만)", type="number", modes=_SAM3_ONLY),
    dict(key="appearance_ranking", label="외형 유사도로 후보 순위 보정", type="checkbox", modes=_SAM3_ONLY),
    dict(key="appearance_weight", label="외형 유사도 가중치 (SAM3 점수 + 가중치 × 유사도)", type="number", modes=_SAM3_ONLY),
    dict(key="table_support", label="테이블 높이 방식", type="select", options=list(TABLE_SUPPORT), modes=None),
    dict(key="viewer", label="뷰어 표시", type="checkbox", modes=None, hidden=True),
    dict(key="surrounding_obstacles", label="주변 물체 장애물 반영", type="checkbox", modes=None),
    dict(key="minimal_records", label="데모 모드 (기록 최소화)", type="checkbox", modes=None),
)
_NUMBERS = tuple((field["key"], field["modes"], field.get("tasks")) for field in FIELDS if field["type"] == "number")
_INTEGERS = ("num_particles",)
_MODE_KEYS = tuple(field["key"] for field in FIELDS if field["modes"])


def expand(value):
    """Fill the {uid} placeholder used by per-user socket paths."""
    return value.replace("{uid}", str(os.getuid())) if isinstance(value, str) else value


def load_presets(path: Path = PRESETS_PATH) -> dict:
    document = json.loads(Path(path).read_text())
    presets = {mode: {"surrounding_obstacles": True, "minimal_records": False, **task_defaults(), **document["common"], **document[mode]} for mode in MODES}
    presets = {mode: {key: expand(value) for key, value in values.items()} for mode, values in presets.items()}
    for values in presets.values():
        registered = TaskSettings(preset=values["task_preset"])
        values.update({key: getattr(registered, key) for key in TaskSettings.__dataclass_fields__
                       if key.startswith("box_") or key in {"fixed_box_present", "release_height_m"}})
    services = [{key: expand(value) for key, value in service.items()} for service in document["services"]]
    return dict(presets=presets, services=services, task_presets=TASK_PRESETS)


def save_preset(mode: str, values: dict, path: Path = PRESETS_PATH) -> None:
    """Store edited values as the new defaults; mode-only keys go to that mode's block."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    known = {field["key"] for field in FIELDS} | {"fixed_box_present"}
    if set(values) - known:
        raise ValueError(f"Unknown launcher fields: {sorted(set(values) - known)}")
    document = json.loads(Path(path).read_text())
    for key, value in values.items():
        if key.startswith("box_") or key in {"fixed_box_present", "release_height_m"}:
            continue  # Registered task/workspace geometry has one source: task_presets.json.
        block = document[mode] if key in _MODE_KEYS else document["common"]
        block[key] = value.replace(str(os.getuid()), "{uid}") if key == "worker_socket" and isinstance(value, str) else value
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _text(values: dict, key: str) -> str:
    value = values.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _number(values: dict, key: str) -> str:
    try:
        number = float(values[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    if key in _INTEGERS and number != int(number):
        raise ValueError(f"{key} must be an integer")
    return format(number, "g")


def class_id_task(task: str) -> str:
    """Replace a leading Korean display name with the catalog class_id the runner expects."""
    text = task.strip()
    names = sorted(
        ((entry["name_ko"], entry["class_id"]) for entry in load_catalog()["objects"] if entry.get("name_ko")),
        key=lambda pair: -len(pair[0]),
    )
    for name, class_id in names:
        if text.startswith(name) and (len(text) == len(name) or text[len(name)].isspace()):
            return class_id + text[len(name):]
    return task


def build_command(mode: str, values: dict, kind: str) -> list[str]:
    """Arguments for run_known_pick_place; options only valid for another mode or run kind are left out."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if kind not in KINDS:
        raise ValueError(f"Unknown run kind: {kind}")
    robot = kind in ("execute", "check")
    for key in ("task", "camera_url") + (("hardware_profile", "gateway_url") if robot else ()):
        if not _text(values, key):
            raise ValueError(f"{key} is required" + (" for robot runs and connection checks" if key in ("hardware_profile", "gateway_url") else ""))
    if values.get("table_support") == "declared" and not _text(values, "hardware_profile"):
        raise ValueError("table_support 'declared' needs a hardware_profile")
    if values.get("table_support") not in TABLE_SUPPORT:
        raise ValueError(f"table_support must be one of {TABLE_SUPPORT}")
    preset = values.get("task_preset", "fixed_box")
    task = TaskSettings(preset=preset)
    task.validate_perception_mode(mode)
    task.perception_instruction(_text(values, "task"))

    command = [sys.executable, "-m", RUNNER_MODULE]
    if mode == "sam3_reference" and _text(values, "worker_socket") and kind != "check":
        command += ["--worker-socket", _text(values, "worker_socket")]
    command += ["--camera-url", _text(values, "camera_url")]
    if _text(values, "hardware_profile"):
        command += ["--hardware-profile", _text(values, "hardware_profile")]
    if robot:
        command += ["--gateway-url", _text(values, "gateway_url")]
    command += ["--perception-mode", mode]
    if mode == "sam3_reference" and _text(values, "reference_scene"):
        command += ["--reference-scene", _text(values, "reference_scene")]
    if mode == "sam3_reference" and values.get("appearance_ranking"):
        command += ["--appearance-ranking"]
    if mode == "gemini_er" and kind != "check":
        command += ["--allow-gemini-once"]
    command += ["--task", class_id_task(_text(values, "task"))]
    command += ["--task-preset", preset]
    for key, modes, tasks in _NUMBERS:
        if key.startswith("box_") or key == "release_height_m":
            continue  # Runner reads registered geometry; stale UI values cannot override it.
        if (modes and mode not in modes) or (tasks and preset not in tasks):
            continue
        command += [f"--{key.replace('_', '-')}", _number(values, key)]
    command += ["--table-support", values["table_support"]]
    if not values.get("surrounding_obstacles", True) and kind != "check":
        command += ["--no-surrounding-obstacles"]
    if kind == "execute":
        command += ["--execute"]
    if kind == "check":
        command += ["--check"]
    if values.get("viewer") and kind != "check":
        command += ["--viewer"]
    if values.get("minimal_records") and kind != "check":
        command += ["--minimal-records"]
    return command


def display_command(command: list[str]) -> str:
    """Shell-style text in the layout of the hand-written commands."""
    if command[2:3] != [RUNNER_MODULE]:
        return shlex.join(command)
    parts, line = [], []
    for token in command[3:]:
        if token.startswith("--") and line:
            parts.append(" ".join(line))
            line = []
        line.append(shlex.quote(token))
    parts.append(" ".join(line))
    return " \\\n  ".join(["pixi run python -m " + RUNNER_MODULE, *parts])
