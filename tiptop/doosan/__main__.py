"""Run from the top workspace: python -m tiptop.doosan status|open|close|stop|demo|plan|run PLAN.json."""

import argparse
import json
import math
import sys
from pathlib import Path

from tiptop.doosan.client import DoosanUONClient
from tiptop.execute_plan import execute_cutamp_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("command", choices=["status", "open", "close", "stop", "demo", "plan", "run"])
    parser.add_argument("plan", nargs="?", type=Path)
    args = parser.parse_args()
    client = DoosanUONClient(args.url)
    if args.command == "status":
        result = client.get_state()
    elif args.command == "stop":
        result = client.stop()
    elif args.command in {"open", "close"}:
        result = getattr(client, f"{args.command}_gripper")()
    elif args.command == "plan":
        if args.plan is None:
            args.plan = Path("tiptop_outputs/m1013/showcase.json")
        from tiptop.doosan.planning import plan_showcase

        try:
            q = client.get_joint_positions()
            plan = plan_showcase(q)
            args.plan.parent.mkdir(parents=True, exist_ok=True)
            args.plan.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
            trajectories = [step for step in plan["steps"] if step["type"] == "trajectory"]
            result = {"success": True, "plan_file": str(args.plan.resolve()),
                      "scene": plan["scene"], "steps": len(plan["steps"]),
                      "sequence": [step["label"] for step in plan["steps"]],
                      "waypoints": sum(len(step["positions"]) for step in trajectories),
                      "requested_duration_s": sum(len(step["positions"]) * step["dt"] for step in trajectories),
                      "joint_offsets_deg": plan["joint_offsets_deg"], "returns_to_start": True,
                      "sent_to_robot": False}
        except Exception as exc:
            result = {"success": False, "error": str(exc), "sent_to_robot": False}
    else:
        if args.command == "demo":
            q = client.get_joint_positions()
            offset = [math.radians(v) for v in [5, -5, 5, 2.5, 2.5, -2.5]]
            middle = [a + b / 2 for a, b in zip(q, offset)]
            target = [a + b for a, b in zip(q, offset)]
            plan = {"version": "1.0.0", "q_init": q, "steps": [
                {"type": "gripper", "label": "open", "action": "open"},
                {"type": "trajectory", "label": "outward", "positions": [q, middle, target],
                 "velocities": [[0.0] * 6] * 3, "dt": 2.0},
                {"type": "gripper", "label": "close", "action": "close"},
                {"type": "trajectory", "label": "return", "positions": [target, middle, q],
                 "velocities": [[0.0] * 6] * 3, "dt": 2.0},
                {"type": "gripper", "label": "open", "action": "open"},
            ]}
        else:
            if args.plan is None:
                parser.error("run requires PLAN.json")
            plan = json.loads(args.plan.read_text())
        if plan.get("version") != "1.0.0":
            parser.error("Expected TiPToP plan version 1.0.0")
        q_init = plan["q_init"]
        actual = client.get_joint_positions()
        if len(q_init) != 6 or any(not math.isfinite(v) for v in q_init):
            parser.error("q_init must contain six finite joint angles (rad)")
        if max(abs(a - b) for a, b in zip(actual, q_init)) > math.radians(1):
            parser.error("Plan q_init differs from current joints by more than 1 degree")
        import logging

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
