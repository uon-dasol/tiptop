"""Run from the top workspace: python -m tiptop.doosan status|open|close|stop|run PLAN.json."""

import argparse
import json
import math
import logging
import sys
import time
from pathlib import Path

from tiptop.doosan.client import DoosanUONClient, validate_plan_backend
from tiptop.execute_plan import execute_cutamp_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("command", choices=["status", "open", "close", "stop", "run"])
    parser.add_argument("plan", nargs="?", type=Path)
    args = parser.parse_args()
    client = DoosanUONClient(args.url)
    if args.command == "status":
        result = client.get_state()
    elif args.command == "stop":
        result = client.stop()
    elif args.command in {"open", "close"}:
        result = getattr(client, f"{args.command}_gripper")()
    else:
        if args.plan is None:
            parser.error("run requires PLAN.json")
        plan = json.loads(args.plan.read_text())
        if plan.get("version") != "1.0.0":
            parser.error("Expected TiPToP plan version 1.0.0")
        if plan.get("robot_system") == 0:
            from tiptop.doosan.execution import execute_measured_plan

            logging.basicConfig(level=logging.INFO)
            report_path = args.plan.with_name(args.plan.stem + f".execution-{time.time_ns()}.json")
            print(f"PHYSICAL_EXECUTION_REPORT {report_path}", file=sys.stderr, flush=True)
            result = execute_measured_plan(plan, client, report_path=report_path)
            print(json.dumps(result, indent=2))
            if result.get("execution_success") is not True:
                raise SystemExit(1)
            return
        q_init = plan["q_init"]
        state = client.get_state()
        try:
            validate_plan_backend(plan, state)
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        actual = state["joint_positions"]
        if len(q_init) != 6 or any(not math.isfinite(v) for v in q_init):
            parser.error("q_init must contain six finite joint angles (rad)")
        if max(abs(a - b) for a, b in zip(actual, q_init)) > math.radians(1):
            parser.error("Plan q_init differs from current joints by more than 1 degree")
        log_format = logging.BASIC_FORMAT
        if sys.stderr.isatty():
            log_format = f"\033[93m{log_format}\033[0m"
        logging.basicConfig(level=logging.INFO, format=log_format)
        execute_cutamp_plan(plan["steps"], client)
        result = client.get_state()
        result["completed_steps"] = len(plan["steps"])
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
