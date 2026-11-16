"""One run_known_pick_place process at a time for the launcher, with its console output kept for polling."""

import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

from tiptop.launcher import build_command, display_command
from tiptop.task_logging import LOG_OWNER_ENV, TerminalLog


class RunManager:
    def __init__(self, cwd: Path, *, build=build_command, max_lines: int = 4000):
        self.cwd = Path(cwd)
        self.build = build
        self._lock = threading.Lock()
        self._lines = deque(maxlen=max_lines)
        self._count = 0
        self._process = None
        self._kind = None
        self._command = ""

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, mode: str, values: dict, kind: str) -> str:
        command = self.build(mode, values, kind)
        with self._lock:
            if self.running():
                raise RuntimeError("A run is already in progress")
            self._lines.clear()
            self._count = 0
            self._kind = kind
            self._command = display_command(command)
            process = subprocess.Popen(
                command, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1", LOG_OWNER_ENV: "launcher"},
            )
            self._process = process
        threading.Thread(target=self._read, args=(process,), daemon=True).start()
        return self._command

    def _append(self, text: str) -> None:
        with self._lock:
            self._lines.append(text)
            self._count += 1

    def _read(self, process) -> None:
        log, pending = None, []

        def record(text):
            nonlocal log
            if log is not None:
                log.write(text + "\n")
            else:
                pending.append(text)
                if text.startswith("TASK_OUTPUT "):
                    try:
                        output = (self.cwd / text.removeprefix("TASK_OUTPUT ")).resolve()
                        if not output.is_relative_to(self.cwd.resolve()):
                            raise ValueError("Task log output is outside the project directory")
                        log = TerminalLog(output / "terminal.log")
                        for line in pending:
                            log.write(line + "\n")
                        pending.clear()
                    except (OSError, ValueError) as error:
                        self._append(f"[로그 저장 실패] {error}")
            self._append(text)

        try:
            for line in process.stdout:
                record(line.rstrip("\n"))
            record(f"[종료 코드 {process.wait()}]")
        finally:
            if log is not None:
                log.close()

    def stop(self) -> bool:
        """Send Ctrl+C to the script. This is not a robot emergency stop."""
        if not self.running():
            return False
        self._process.send_signal(signal.SIGINT)
        return True

    def log(self, since: int) -> dict:
        with self._lock:
            first = self._count - len(self._lines)
            lines = list(self._lines)[max(since - first, 0):]
            return dict(next=self._count, lines=lines)

    def state(self) -> dict:
        returncode = None if self._process is None else self._process.poll()
        return dict(running=self.running(), kind=self._kind, command=self._command, returncode=returncode)
