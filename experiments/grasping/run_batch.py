"""Sequential bounded Isaac subprocesses; no source edits, outputs only this experiment directory."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from experiments.grasping.paths import PROJECT_ROOT as ROOT
from experiments.grasping.paths import TOOLS_ROOT, ensure_workdirs, read_json
from experiments.grasping.paths import WORK_ROOT as W

ensure_workdirs()
ISAAC = os.environ.get("ISAAC_PYTHON", str(Path.home() / "isaac-sim/6.2.0/python.sh"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path, help="Prepared experiment manifest; existing results are skipped")
MODE = str(parser.parse_args().manifest)
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT), CUDA_CACHE_PATH=str(W / "cache/cuda"))
queue = []
for item in read_json(Path(MODE)):
    if item.get("status") == "prepared":
        for rep in range(1, item.get("repetitions", 1) + 1):
            queue.append((f"{item['name']}-r{rep:02d}", Path(item["pick_path"])))
(W / "logs").mkdir(exist_ok=True)
state = []
for name, source in queue:
    out = W / "runs" / name
    if (out / "check.json").exists():
        state.append({"name": name, "status": "existing_result"})
        continue
    out.mkdir(exist_ok=True)
    record = read_json(source)
    if record.get("post_pick_steps"):
        record["placement_spinner_torque_limit_nm"] = 5.0
    (out / "pick.json").write_text(json.dumps(record, indent=2) + "\n")
    runner = (
        TOOLS_ROOT / "pick_multi.py" if record.get("target_object") else ROOT / "simulation/isaac_sim/pick_pepper.py"
    )
    (out / "runner-source-sha256.json").write_text(
        json.dumps({x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in TOOLS_ROOT.glob("*.py")}, indent=2)
        + "\n"
    )
    argv = [
        ISAAC,
        "--no-ros-env",
        str(runner),
        "--headless",
        "--check",
        "--physics-hz",
        "240",
        "--spinner-max-torque",
        "5",
        "--output",
        str(out),
    ]
    if name.endswith("r01"):
        argv.append("--snapshot")
    started = time.monotonic()
    print("START", name, flush=True)
    with (out / "isaac.log").open("w") as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (W / "logs/active-process.json").write_text(
            json.dumps({"batch_pid": os.getpid(), "child_pid": proc.pid, "name": name, "output": str(out)}) + "\n"
        )
        try:
            rc = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
    item = {"name": name, "exit_code": rc, "wall_seconds": time.monotonic() - started}
    if (out / "check.json").is_file():
        report = read_json(out / "check.json")
        item.update(
            success=report.get("success"),
            phase=report.get("phase"),
            error=report.get("error"),
            lift_m=report.get("min_hold_lift_m"),
            slip_m=report.get("max_carry_slip_m"),
            placement_error_m=report.get("placement_xy_error_m"),
        )
    else:
        item.update(success=False, error="No check.json; inspect process log")
    state.append(item)
    (W / "logs" / f"batch-{Path(MODE).stem}.json").write_text(json.dumps(state, indent=2) + "\n")
    print("DONE", json.dumps(item), flush=True)
    # Avoid blindly repeating a common launch/configuration failure.
    if len(state) >= 2 and all(x.get("error") == "No check.json; inspect process log" for x in state[-2:]):
        print("STOP: repeated process startup failure", flush=True)
        break
(W / "logs/active-process.json").unlink(missing_ok=True)
