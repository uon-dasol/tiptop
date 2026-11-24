"""Play a saved plan in Rerun: the M1013 moving through its legs, the TCP trail, the box, and joint plots over time."""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from tiptop.doosan.model import load_m1013_uon_rerun
from tiptop.robot_limits import load_robot_limits
from tiptop.scripts.viz_plan import JOINT_COLORS, LEG_COLORS, gripper_durations, load_legs

TIMELINE = "time"
JOINTS = [f"J{i}" for i in range(1, 7)]


def hex_to_rgb(color):
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


def log_box(region):
    (cx, cy), (sx, sy), top = region["center_xy_m"], region["size_xy_m"], region["support_z_m"]
    rr.log("world/box", rr.Boxes3D(centers=[[cx, cy, top / 2]], half_sizes=[[sx / 2, sy / 2, top / 2]],
                                   colors=[[120, 212, 224]], fill_mode="majorwireframe"), static=True)


def log_series_styles():
    """Each joint keeps one colour and name in every plot."""
    for name in ("position_deg", "velocity_deg_s", "acceleration_rad_s2"):
        for joint, color in zip(JOINTS, JOINT_COLORS):
            rr.log(f"plots/{name}/{joint}", rr.SeriesLines(colors=[hex_to_rgb(color)], names=[joint], widths=[1.5]), static=True)


def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Motion", origin="/", contents=["/world/**", "/m1013_uon/**"],
                              eye_controls=rrb.EyeControls3D(position=[1.6, -0.9, 1.0], look_target=[0.0, 0.35, 0.2]),
                              line_grid=rrb.LineGrid3D(visible=True)),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Joint position (deg)", origin="/plots/position_deg"),
                rrb.TimeSeriesView(name="Joint velocity (deg/s)", origin="/plots/velocity_deg_s"),
                rrb.TimeSeriesView(name="Joint acceleration (rad/s²) and its limits", origin="/plots",
                                   contents=["/plots/acceleration_rad_s2/**", "/plots/acceleration_limit/**"]),
                rrb.TextLogView(name="Legs and gripper steps (start - end, length)", origin="/legs"),
                row_shares=[2, 2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline=TIMELINE),
        rrb.SelectionPanel(state="collapsed"),
        collapse_panels=False,
    )


def log_leg(robot, leg):
    """Robot pose, TCP marker and joint signals of one leg, sampled at the plan's own timestamps."""
    columns = robot.get_rr_columns(torch.as_tensor(leg["q"], dtype=torch.float32))
    index = [rr.TimeColumn(TIMELINE, duration=leg["t"])]
    for path, component in columns.items():
        rr.send_columns(path, indexes=index, columns=component)
    rr.send_columns("world/tcp", indexes=index, columns=rr.Points3D.columns(positions=leg["tcp"]).partition([1] * len(leg["tcp"])))
    color = hex_to_rgb(LEG_COLORS.get(leg["label"], "#999999"))
    rr.log(f"world/trail/{leg['label']}", rr.LineStrips3D([leg["tcp"]], colors=[color], radii=0.003), static=True)
    for name, key, scale in (("position_deg", "q", 180 / np.pi), ("velocity_deg_s", "v", 180 / np.pi), ("acceleration_rad_s2", "a", 1.0)):
        for j, joint in enumerate(JOINTS):
            rr.send_columns(f"plots/{name}/{joint}", indexes=index, columns=rr.Scalars.columns(scalars=leg[key][:, j] * scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path, help="Task folder containing plan.json")
    parser.add_argument("--output", type=Path, help="Recording path (default: motion.rrd in the task folder)")
    parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer on the recording")
    args = parser.parse_args()
    plan = json.loads((args.task_dir / "plan.json").read_text())
    legs, gaps = load_legs(plan, gripper_durations(args.task_dir))
    output = args.output or args.task_dir / "motion.rrd"

    rr.init(f"plan_{args.task_dir.name}", spawn=False)
    rr.save(str(output))
    robot = load_m1013_uon_rerun()
    if plan.get("placement_region"):
        log_box(plan["placement_region"])
    for leg in legs:
        log_leg(robot, leg)
        rr.set_time(TIMELINE, duration=float(leg["t"][0]))
        rr.log("legs", rr.TextLog(f"{leg['label']}  {leg['t'][0]:.2f} - {leg['t'][-1]:.2f} s ({leg['t'][-1] - leg['t'][0]:.2f} s)"))
    for start, end, name, measured in gaps:
        rr.set_time(TIMELINE, duration=float(start))
        rr.log("legs", rr.TextLog(f"gripper {name}  {start:.2f} - {end:.2f} s ({end - start:.2f} s){'' if measured else ' (nominal)'}"))
    log_series_styles()
    limits = load_robot_limits()["max_acceleration"]
    end = max(legs[-1]["t"][-1], gaps[-1][1] if gaps else 0.0)
    index = [rr.TimeColumn(TIMELINE, duration=np.array([0.0, end]))]
    for name, level in (("J1-5 limit", max(limits[:5])), ("J6 limit", limits[5])):
        for sign, suffix in ((1, "+"), (-1, "-")):
            path = f"plots/acceleration_limit/{name}{suffix}"
            rr.log(path, rr.SeriesLines(colors=[[204, 68, 68] if name.startswith("J1") else [160, 40, 120]], names=[name], widths=[1.0]),
                   static=True)
            rr.send_columns(path, indexes=index, columns=rr.Scalars.columns(scalars=np.array([sign * level] * 2)))
    rr.send_blueprint(blueprint())
    print(f"Saved {output}")
    if args.spawn:
        import subprocess

        subprocess.Popen(["rerun", str(output)])


if __name__ == "__main__":
    main()
