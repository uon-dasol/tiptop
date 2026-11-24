"""Start and stop the viewer's local services. Services outlive the viewer; their state is kept in per-service files."""

import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

KEEP_LOGS = 10
TERM_AFTER_S = 10.0
KILL_AFTER_S = 20.0
_PASSTHROUGH_ENV = (
    "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)


def terminal_argv(command: str) -> list[str]:
    """Run the command as in a freshly opened terminal: only the login-session variables plus what ~/.bashrc sets."""
    session = [f"{key}={os.environ[key]}" for key in _PASSTHROUGH_ENV if key in os.environ]
    return ["env", "-i", *session, "TERM=xterm", "PYTHONUNBUFFERED=1", "bash", "-ic", command]


def process_start_ticks(pid: int):
    """Start time of a live process in clock ticks since boot; None if it is gone or a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return None if fields[0] == "Z" else int(fields[19])


class ServiceManager:
    def __init__(self, cwd: Path, log_root: Path, *, wrap: Callable[[str], list[str]] = terminal_argv,
                 term_after: float = TERM_AFTER_S, kill_after: float = KILL_AFTER_S, keep_logs: int = KEEP_LOGS):
        self.cwd, self.log_root, self.wrap = Path(cwd), Path(log_root), wrap
        self.term_after, self.kill_after, self.keep_logs = term_after, kill_after, keep_logs
        self._lock = threading.RLock()
        self._children = {}

    def _record_path(self, service_id: str) -> Path:
        return self.log_root / service_id / "current.json"

    def _load(self, service_id: str):
        try:
            return json.loads(self._record_path(service_id).read_text())
        except (OSError, ValueError):
            return None

    def _save(self, service_id: str, record: dict) -> None:
        path = self._record_path(service_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)

    def _alive(self, service_id: str, record: dict) -> bool:
        child = self._children.get(service_id)
        if child is not None:
            child.poll()
        ticks = process_start_ticks(record["pid"])
        return ticks is not None and ticks == record["ticks"]

    def status(self, service_id: str, up: bool) -> dict:
        """State is one of off, starting, up, stopping, failed; `managed` is true for a process started here."""
        with self._lock:
            record = self._load(service_id)
            if record is not None and record["ended"] is None:
                if self._alive(service_id, record):
                    state = "stopping" if record["stop_requested"] else ("up" if up else "starting")
                    return dict(state=state, managed=True)
                child = self._children.get(service_id)
                finished_cleanly = record.get("oneshot") and (child is None or child.returncode == 0)
                record["ended"] = "stopped" if record["stop_requested"] or finished_cleanly else "failed"
                self._save(service_id, record)
            if up:
                return dict(state="up", managed=False)
            return dict(state="failed" if record is not None and record["ended"] == "failed" else "off", managed=False)

    def start(self, service_id: str, command: str, *, up: bool, oneshot: bool = False) -> None:
        with self._lock:
            state = self.status(service_id, up)["state"]
            if state not in ("off", "failed"):
                raise RuntimeError(f"{service_id} is already running ({state})")
            directory = self.log_root / service_id
            directory.mkdir(parents=True, exist_ok=True)
            logs = sorted(directory.glob("*.log"))
            for old in logs[: max(len(logs) - self.keep_logs + 1, 0)]:
                old.unlink()
            log_path = directory / f"{datetime.now():%Y%m%d-%H%M%S}.log"
            with log_path.open("ab") as log:
                log.write(f"$ {command}\n".encode())
                log.flush()
                process = subprocess.Popen(
                    self.wrap(command), cwd=self.cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self._children[service_id] = process
            self._save(service_id, dict(pid=process.pid, ticks=process_start_ticks(process.pid), log=str(log_path),
                                        stop_requested=False, ended=None, oneshot=oneshot))

    def stop(self, service_id: str, signal_command: str | None = None, *, leader_only: bool = False) -> bool:
        """Send SIGINT to the service's process group, as Ctrl+C in a terminal does, then escalate if it lingers.

        `signal_command` replaces the process-group signal for services whose real process runs elsewhere (e.g. in a
        container); its `{signal}` placeholder is INT, TERM or KILL. `leader_only` sends the first SIGINT to the
        launcher process alone, for `pixi run`, which forwards it to its child when there is no terminal.
        """
        with self._lock:
            record = self._load(service_id)
            if record is None or record["ended"] is not None or record["stop_requested"]:
                return False
            if not self._alive(service_id, record):
                return False
            record["stop_requested"] = True
            self._save(service_id, record)
        self._send(record, signal.SIGINT, signal_command, group=not leader_only)
        threading.Thread(target=self._escalate, args=(service_id, record, signal_command), daemon=True).start()
        return True

    @staticmethod
    def _send(record: dict, sig: signal.Signals, signal_command: str | None, group: bool = True) -> None:
        if signal_command is not None:
            try:
                subprocess.run(["bash", "-c", signal_command.replace("{signal}", sig.name.removeprefix("SIG"))],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
            return
        try:
            (os.killpg if group else os.kill)(record["pid"], sig)
        except ProcessLookupError:
            pass

    def _escalate(self, service_id: str, record: dict, signal_command: str | None) -> None:
        start = time.monotonic()
        for after, sig in ((self.term_after, signal.SIGTERM), (self.kill_after, signal.SIGKILL)):
            while time.monotonic() - start < after:
                if not self._alive(service_id, record):
                    return
                time.sleep(0.1)
            self._send(record, sig, signal_command)


class CommandProbes:
    """Readiness checks that run a command; later calls return the latest result while a refresh runs in the background.

    Probes with the same command share one run, so each may match a different part of its output.
    """

    def __init__(self, timeout: float = 8.0, first_wait: float = 1.0):
        self.timeout, self.first_wait = timeout, first_wait
        self._lock = threading.Lock()
        self._results = {}
        self._running = set()

    def _run(self, command: list[str]) -> tuple[bool, str]:
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                                    timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired):
            return False, ""
        return result.returncode == 0, result.stdout

    def _refresh(self, key: tuple, command: list[str]) -> None:
        succeeded, output = self._run(command)
        with self._lock:
            self._results[key] = (succeeded, output, time.monotonic())
            self._running.discard(key)

    def up(self, probe: dict) -> bool:
        command = probe["command"]
        key = tuple(command)
        with self._lock:
            cached = self._results.get(key)
            refresh = key not in self._running and (
                cached is None or time.monotonic() - cached[2] > probe.get("interval", 3.0))
            if refresh:
                self._running.add(key)
        if refresh:
            thread = threading.Thread(target=self._refresh, args=(key, command), daemon=True)
            thread.start()
            if cached is None:
                thread.join(self.first_wait)
                with self._lock:
                    cached = self._results.get(key)
        if cached is None:
            return False
        succeeded, output, _ = cached
        return succeeded and (probe.get("match") is None or re.search(probe["match"], output) is not None)
