"""Task terminal output is saved to terminal.log next to the task results."""
import asyncio
import logging
import sys
import time
from types import SimpleNamespace

from tiptop.run_manager import RunManager
from tiptop.scripts import run_known_pick_place as runner
from tiptop.task_logging import LOG_OWNER_ENV, capture_terminal_log


def test_capture_saves_stdout_stderr_and_console_logging_without_ansi(tmp_path, capsys):
    path = tmp_path / "terminal.log"
    logger = logging.getLogger("task-logging-test")
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        with capture_terminal_log(path):
            print("\x1b[32mgreen\x1b[0m line")
            print("to stderr", file=sys.stderr)
            logger.warning("logged warning")
    finally:
        logger.removeHandler(handler)
    text = path.read_text()
    assert "green line" in text and "\x1b" not in text
    assert "to stderr" in text and "logged warning" in text
    captured = capsys.readouterr()
    assert "green" in captured.out and "logged warning" in captured.err


def test_capture_survives_default_stderr_handler_and_unwritable_path(tmp_path, capsys):
    logging.getLogger("task-logging-last-resort").handlers.append(logging.lastResort)
    try:
        with capture_terminal_log(tmp_path / "missing" / "terminal.log"):
            print("still runs")
        with capture_terminal_log(tmp_path / "terminal.log"):
            print("saved")
    finally:
        logging.getLogger("task-logging-last-resort").handlers.clear()
    assert "Terminal log unavailable" in capsys.readouterr().err
    assert "saved" in (tmp_path / "terminal.log").read_text()


def test_run_task_writes_terminal_log_in_reserved_task_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(LOG_OWNER_ENV, raising=False)

    async def fake(args):
        print(f"TASK_OUTPUT {args.output}")
        return dict(status="planned", success=None)

    monkeypatch.setattr(runner, "_run_task", fake)
    args = SimpleNamespace(output=tmp_path / "task-001")
    asyncio.run(runner.run_task(args))
    assert f"TASK_OUTPUT {tmp_path / 'task-001'}" in (tmp_path / "task-001" / "terminal.log").read_text()


def test_run_task_leaves_logging_to_the_launcher(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG_OWNER_ENV, "launcher")

    async def fake(args):
        print("launcher owns the log")

    monkeypatch.setattr(runner, "_run_task", fake)
    args = SimpleNamespace(output=tmp_path / "task-001")
    asyncio.run(runner.run_task(args))
    assert not (tmp_path / "task-001" / "terminal.log").exists()


def test_launcher_run_manager_saves_output_from_before_and_after_task_output(tmp_path):
    (tmp_path / "out" / "task-001").mkdir(parents=True)
    script = "print('before'); print('TASK_OUTPUT out/task-001', flush=True); print('after')"
    manager = RunManager(cwd=tmp_path, build=lambda mode, values, kind: [sys.executable, "-c", script])
    manager.start("sam3_reference", {}, "plan")
    deadline = time.time() + 10
    while manager.running() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.2)
    lines = (tmp_path / "out" / "task-001" / "terminal.log").read_text().splitlines()
    assert lines[:3] == ["before", "TASK_OUTPUT out/task-001", "after"]
    assert lines[-1].startswith("[종료 코드")


def test_launcher_run_manager_refuses_log_paths_outside_the_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    script = f"print('TASK_OUTPUT {tmp_path / 'elsewhere'}', flush=True)"
    manager = RunManager(cwd=project, build=lambda mode, values, kind: [sys.executable, "-c", script])
    manager.start("sam3_reference", {}, "plan")
    deadline = time.time() + 10
    while manager.running() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.2)
    assert not (tmp_path / "elsewhere").exists()
