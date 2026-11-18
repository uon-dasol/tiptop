"""One line per finished task, appended to a single file and shown in the viewer."""

import json
import os
import re
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from tiptop.object_catalog import grasp_effort_counts, load_catalog

ROOT = Path(__file__).resolve().parents[1]
TASK_LOG_PATH = ROOT / "tiptop_outputs/m1013/task-log.jsonl"
TASK_MARKS_PATH = ROOT / "tiptop_outputs/m1013/task-marks.json"
SHOWN_ENTRIES = 100
SUCCESS = "성공"
FAILURE_REASONS = ("파지 실패", "이송 중 놓침", "충돌")
MARK_CHOICES = dict(perception=("정상 인식", "오인식"), result=(SUCCESS, *FAILURE_REASONS))
_MARKS_LOCK = threading.Lock()
EMPTY_CLOSE_MAX_COUNTS = 315  # fingers closed without an object end at or below this, whatever the closing current


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def grasp_outcome(close_position_counts):
    """`성공` when the fingers stopped on an object when closing, `실패` when they closed fully; None if not closed."""
    if close_position_counts is None:
        return None
    return "실패" if close_position_counts <= EMPTY_CLOSE_MAX_COUNTS else "성공"


def task_log_entry(output, result, wall_seconds, demo=False):
    """What the log keeps of one task."""
    output = Path(output)
    number = re.fullmatch(r"task-(\d+)", output.name)
    plan = _read_json(output / "plan.json") or {}
    execution = _read_json(output / "physical-execution.json") or {}
    close = next(
        (event.get("result") or {} for event in execution.get("events", []) if event.get("method") == "close_gripper"),
        {},
    )
    class_id = plan.get("class_id") or (_read_json(output / "registered-pose.json") or {}).get("class_id")
    names = {item["class_id"]: item["name_ko"] for item in load_catalog()["objects"]}
    effort = next((step.get("effort_counts") for step in plan.get("steps", []) if step.get("label") == "Pick.close"), None)
    if effort is None and class_id:  # stopped before a plan existed: the value that would have been used
        effort = grasp_effort_counts(class_id)
    return dict(
        task=int(number[1]) if number else None,
        time=datetime.now().isoformat(timespec="seconds"),
        output=str(output),
        class_id=class_id,
        name_ko=names.get(class_id, class_id),
        effort_counts=close.get("current_limit_counts", effort),
        close_position_counts=close.get("position_counts"),
        close_current_counts=close.get("current_counts"),
        grasp=grasp_outcome(close.get("position_counts")),
        duration_s=round(wall_seconds, 1),
        status=result.get("status"),
        executed=result.get("executed"),
        demo=bool(demo),
        failed_stage=result.get("failed_stage"),
    )


def append_task_log(output, result, wall_seconds, path=None, demo=False):
    """Add one line for a finished task; the log must never make a task fail."""
    try:
        path = Path(TASK_LOG_PATH if path is None else path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(task_log_entry(output, result, wall_seconds, demo), ensure_ascii=False) + "\n")
    except Exception as error:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("Could not write the task log: %s", error)


def log_cells(entry):
    """The log line's columns; one that does not apply is empty so the page can keep every row's columns aligned."""
    effort, position, current = (
        entry.get("effort_counts"),
        entry.get("close_position_counts"),
        entry.get("close_current_counts"),
    )
    if entry.get("status") == "failed":
        outcome = f"작업 중단({entry.get('failed_stage') or '?'})"
    else:
        outcome = "작업 완료" if entry.get("executed") else ""
    return [
        f"TASK {entry['task']}" if entry.get("task") is not None else "TASK",
        entry.get("name_ko") or "-",
        f"effort {effort}" if effort is not None else "effort -",
        f"전류 {current:g}" if current is not None else "전류 -",
        f"위치 {position:g}" if position is not None else "위치 -",
        f"{entry['duration_s']}초" if entry.get("duration_s") is not None else "-",
        f"파지 {entry['grasp']}" if entry.get("grasp") else "",
        outcome,
        "데모 모드" if entry.get("demo") else "",
    ]


def log_line(entry):
    """`TASK 877 · 바가지 · effort 50 · 전류 -49 · 위치 1272 · 18.4초 · 파지 성공 · 작업 완료 · 데모 모드`"""
    return " · ".join(cell for cell in log_cells(entry) if cell)


def read_task_log(path=None, limit=SHOWN_ENTRIES):
    """The last `limit` entries, newest first; unreadable lines are skipped."""
    path = Path(TASK_LOG_PATH if path is None else path)
    if not path.is_file():
        return []
    latest = deque(maxlen=limit)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                latest.append(json.loads(line))
            except ValueError:
                continue
    return list(reversed(latest))


def read_task_marks(path=None):
    """What a person entered per task: `{task number: {"perception": ..., "result": ...}}`; unreadable means none."""
    marks = _read_json(TASK_MARKS_PATH if path is None else path)
    return marks if isinstance(marks, dict) else {}


def set_task_mark(task, field, value, path=None):
    """Record what a person saw for a logged task; an empty value removes that entry."""
    if field not in MARK_CHOICES:
        raise ValueError(f"Unknown field {field!r}; use one of {', '.join(MARK_CHOICES)}")
    if value not in ("", *MARK_CHOICES[field]):
        raise ValueError(f"Unknown {field} {value!r}; use one of {', '.join(MARK_CHOICES[field])}")
    if task not in {entry["task"] for entry in read_task_log()}:
        raise ValueError(f"Task {task} is not in the task log")
    path = Path(TASK_MARKS_PATH if path is None else path)
    with _MARKS_LOCK:
        marks = read_task_marks(path)
        entered = dict(marks.get(str(task), {}))
        if value:
            entered[field] = value
        else:
            entered.pop(field, None)
        if entered:
            marks[str(task)] = entered
        else:
            marks.pop(str(task), None)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(marks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def task_log_rows():
    """What the viewer shows: the log line and the person's entries for each of the newest tasks."""
    marks = read_task_marks()
    blank = dict(perception="", result="")
    return [dict(task=entry["task"], line=log_line(entry), cells=log_cells(entry), **{**blank, **marks.get(str(entry["task"]), {})})
            for entry in read_task_log()]
