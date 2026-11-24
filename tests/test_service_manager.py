import json
import shlex
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from tiptop.service_manager import ServiceManager, process_start_ticks, terminal_argv


def plain_shell(command):
    return ["bash", "-c", command]


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def python_command(code):
    return f"exec {sys.executable} -c {shlex.quote(code)}"


def listen_after(port, delay):
    return python_command(
        f"import socket,time\ntime.sleep({delay})\ns=socket.socket()\ns.bind(('127.0.0.1',{port}))\ns.listen()\ntime.sleep(60)"
    )


def wait_for(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@pytest.fixture
def manager(tmp_path):
    created = []

    def make(**options):
        created.append(ServiceManager(tmp_path, tmp_path / "services", wrap=plain_shell, **options))
        return created[-1]

    yield make
    for item in created:
        for record in (item._load(path.parent.name) for path in item.log_root.glob("*/current.json")):
            if record and record["ended"] is None and process_start_ticks(record["pid"]) == record["ticks"]:
                os.killpg(record["pid"], 9)


def state(manager, service_id, port):
    return manager.status(service_id, port_open(port))["state"]


def test_service_goes_from_off_to_starting_to_up_and_back_to_off_when_stopped(manager):
    service, port = manager(), free_port()
    assert state(service, "demo", port) == "off"
    service.start("demo", listen_after(port, 0.8), up=False)
    assert state(service, "demo", port) == "starting"
    wait_for(lambda: state(service, "demo", port) == "up")
    assert service.status("demo", True)["managed"] is True
    assert service.stop("demo") is True
    assert service.stop("demo") is False
    wait_for(lambda: state(service, "demo", port) == "off")


def test_process_that_exits_on_its_own_is_failed_until_started_again(manager):
    service = manager()
    service.start("demo", "exit 3", up=False)
    wait_for(lambda: service.status("demo", False)["state"] == "failed")
    service.start("demo", python_command("import time; time.sleep(60)"), up=False)
    assert service.status("demo", False)["state"] == "starting"
    service.stop("demo")


def test_start_is_rejected_while_running_or_when_the_port_is_already_served(manager):
    service = manager()
    service.start("demo", python_command("import time; time.sleep(60)"), up=False)
    with pytest.raises(RuntimeError, match="already running"):
        service.start("demo", "true", up=False)
    service.stop("demo")
    wait_for(lambda: service.status("demo", False)["state"] == "off")
    with pytest.raises(RuntimeError, match="already running"):
        service.start("demo", "true", up=True)
    assert service.status("demo", True) == dict(state="up", managed=False)


def test_stop_sends_sigint_like_ctrl_c(manager, tmp_path):
    service = manager()
    marker = tmp_path / "interrupted"
    code = (
        "import time,pathlib\ntry:\n  time.sleep(60)\nexcept KeyboardInterrupt:\n"
        f"  pathlib.Path({str(marker)!r}).write_text('x')"
    )
    service.start("demo", python_command(code), up=False)
    time.sleep(0.5)
    service.stop("demo")
    wait_for(marker.exists)
    wait_for(lambda: service.status("demo", False)["state"] == "off")


def test_stop_escalates_to_terminate_and_kill_when_interrupt_is_ignored(manager):
    service = manager(term_after=0.3, kill_after=0.8)
    stubborn = "import signal,time\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)"
    service.start("demo", python_command(stubborn), up=False)
    time.sleep(0.5)
    service.stop("demo")
    assert service.status("demo", False)["state"] == "stopping"
    wait_for(lambda: service.status("demo", False)["state"] == "off", timeout=5)


def test_new_manager_adopts_a_running_service_and_can_stop_it(manager):
    first, port = manager(), free_port()
    first.start("demo", listen_after(port, 0.3), up=False)
    second = manager()
    wait_for(lambda: state(second, "demo", port) == "up")
    assert second.status("demo", True)["managed"] is True
    assert second.stop("demo") is True
    wait_for(lambda: state(second, "demo", port) == "off")


def test_only_the_latest_logs_are_kept_and_the_log_records_the_command_and_output(manager):
    service = manager(keep_logs=2)
    directory = service.log_root / "demo"
    directory.mkdir(parents=True)
    for name in ("20200101-000000.log", "20200102-000000.log", "20200103-000000.log"):
        (directory / name).write_text("old")
    service.start("demo", "echo hello; exit 1", up=False)
    wait_for(lambda: service.status("demo", False)["state"] == "failed")
    logs = sorted(path.name for path in directory.glob("*.log"))
    assert len(logs) == 2 and "20200103-000000.log" in logs
    latest = Path(json.loads((directory / "current.json").read_text())["log"])
    assert latest.read_text() == "$ echo hello; exit 1\nhello\n"


def test_terminal_command_runs_in_a_fresh_interactive_shell():
    argv = terminal_argv("fp && exec bash scripts/run_foundation_pose_server.sh")
    assert argv[:2] == ["env", "-i"]
    assert argv[-3:] == ["bash", "-ic", "fp && exec bash scripts/run_foundation_pose_server.sh"]
    assert "PYTHONUNBUFFERED=1" in argv
    assert not any(item.startswith(("PIXI_", "CONDA_")) for item in argv)


def test_one_shot_command_returns_to_off_when_it_succeeds_and_fails_when_it_does_not(manager):
    service = manager()
    service.start("task", "true", up=False, oneshot=True)
    wait_for(lambda: service.status("task", False)["state"] == "off")
    service.start("task", "exit 2", up=False, oneshot=True)
    wait_for(lambda: service.status("task", False)["state"] == "failed")


def test_stop_uses_the_signal_command_and_escalates_through_it(manager, tmp_path):
    service = manager(term_after=0.4, kill_after=0.9)
    sent = tmp_path / "signals"
    stubborn = "import signal,time\nsignal.signal(signal.SIGINT,signal.SIG_IGN)\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)"
    pidfile = tmp_path / "inner.pid"
    service.start("demo", f"echo $$ > {pidfile}; {python_command(stubborn)}", up=False)
    wait_for(pidfile.exists)
    time.sleep(0.3)
    signal_command = f"echo {{signal}} >> {sent}; kill -{{signal}} $(cat {pidfile})"
    assert service.stop("demo", signal_command) is True
    wait_for(lambda: service.status("demo", False)["state"] == "off", timeout=5)
    assert sent.read_text().split() == ["INT", "TERM", "KILL"]


def test_command_probes_match_the_output_share_one_run_and_never_block_on_a_slow_command(tmp_path):
    from tiptop.service_manager import CommandProbes

    counter = tmp_path / "runs"
    command = ["bash", "-c", f"echo run >> {counter}; echo '/dsr01/node_a'; echo '/gripper_node'"]
    probes = CommandProbes()
    assert probes.up(dict(command=command, match="/gripper_node")) is True
    assert probes.up(dict(command=command, match="/dsr01/node_a")) is True
    assert probes.up(dict(command=command, match="/missing")) is False
    assert counter.read_text().count("run") == 1
    slow = ["bash", "-c", "sleep 3; echo ok"]
    started = time.monotonic()
    assert CommandProbes(first_wait=0.2).up(dict(command=slow)) is False
    assert time.monotonic() - started < 1.5
    assert CommandProbes().up(dict(command=["bash", "-c", "echo ok; exit 1"])) is False


def test_leader_only_interrupt_reaches_a_forwarding_launcher_once_instead_of_twice(manager, tmp_path):
    child = (
        "import signal,time,pathlib\n"
        f"signal.signal(signal.SIGINT,lambda *a: pathlib.Path({str(tmp_path / 'count')!r}).open('a').write('x'))\n"
        f"pathlib.Path({str(tmp_path / 'child-ready')!r}).write_text('x')\n"
        "time.sleep(60)"
    )
    launcher = (
        "import signal,subprocess,sys,time\n"
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        "signal.signal(signal.SIGINT,lambda *a: child.send_signal(signal.SIGINT))\n"
        f"import pathlib; pathlib.Path({str(tmp_path / 'launcher-ready')!r}).write_text('x')\n"
        "child.wait()"
    )

    def interrupts(leader_only):
        count, ready = tmp_path / "count", [tmp_path / "child-ready", tmp_path / "launcher-ready"]
        count.unlink(missing_ok=True)
        for marker in ready:
            marker.unlink(missing_ok=True)
        service = manager(term_after=4.0, kill_after=8.0)
        service.start("demo", python_command(launcher), up=False)
        wait_for(lambda: all(marker.exists() for marker in ready), timeout=20)
        service.stop("demo", leader_only=leader_only)
        time.sleep(2.0)  # Count the interrupts before the escalation to SIGTERM at 4 s.
        interrupts_seen = len(count.read_text()) if count.exists() else 0
        wait_for(lambda: service.status("demo", False)["state"] == "off", timeout=15)
        return interrupts_seen

    assert interrupts(leader_only=False) == 2
    assert interrupts(leader_only=True) == 1
