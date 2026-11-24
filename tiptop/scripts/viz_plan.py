"""Draw the motion of a saved plan: TCP path over the box, and joint position, velocity and acceleration over time."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from yourdfpy import URDF

from tiptop.doosan.model import ASSETS
from tiptop.robot_limits import load_robot_limits

LEG_COLORS = {"Pick.approach": "#2475b8", "Pick.descend": "#0f9bab", "Place.lift": "#8d6bd1",
              "Place.transport": "#e08a1e", "Home": "#3aa655"}
JOINT_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
GRIPPER_GAP_S = 1.0  # width of a gripper step when no execution record gives its measured duration


def tcp_positions(robot, positions):
    points = []
    for q in positions:
        robot.update_cfg(dict(zip((f"joint_{i}" for i in range(1, 7)), q)) | {"gripper_spinner_joint": 0.0})
        points.append(robot.get_transform("gripper_tcp", "base_link")[:3, 3])
    return np.array(points)


def gripper_durations(task_dir):
    """Measured seconds of each executed gripper step by label; empty when the task was not executed."""
    path = Path(task_dir) / "physical-execution.json"
    if not path.is_file():
        return {}
    events = json.loads(path.read_text()).get("events", [])
    return {event["label"]: float(event["duration_s"]) for event in events
            if event.get("type") == "gripper" and event.get("status") == "completed"}


def load_legs(plan, durations):
    """Trajectory legs with their time axis; gripper steps become gaps of their measured duration."""
    robot = URDF.load(str(ASSETS / "m1013_uon.urdf"), load_meshes=False)
    legs, gaps, clock = [], [], 0.0
    for step in plan["steps"]:
        if step["type"] == "gripper":
            measured = step["label"] in durations
            width = durations.get(step["label"], GRIPPER_GAP_S)
            gaps.append((clock, clock + width, step["label"].split(".")[-1], measured))
            clock += width
            continue
        q, v, dt = np.asarray(step["positions"], float), np.asarray(step["velocities"], float), step["dt"]
        legs.append(dict(label=step["label"], t=clock + dt * np.arange(len(q)), q=q, v=v,
                         a=np.gradient(v, dt, axis=0), tcp=tcp_positions(robot, q)))
        clock += dt * (len(q) - 1)
    return legs, gaps


def gaps_after(gaps, leg):
    return [gap for gap in gaps if gap[0] >= leg["t"][-1]]


def draw_leg_paths(top, side, leg, region, limits):
    """One leg's TCP path seen from above (x-y) and from the side (y-z), with its start and end marked."""
    color = LEG_COLORS.get(leg["label"], "#999")
    tcp = leg["tcp"]
    for axis, columns, name in ((top, (0, 1), "x-y"), (side, (1, 2), "y-z")):
        points = tcp[:, columns]
        axis.plot(points[:, 0], points[:, 1], color=color, lw=2.4)
        middle = len(points) // 2
        axis.annotate("", xy=points[middle + 1], xytext=points[middle - 1],
                      arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16))
        axis.plot(*points[0], "o", mfc="white", mec=color, mew=2, ms=9, label="start")
        axis.plot(*points[-1], "o", color=color, ms=9, label="end")
        axis.set(title=f"{leg['label']}: {'top view' if name == 'x-y' else 'side view'} ({name})", xlim=limits[name][0],
                 ylim=limits[name][1], xlabel=name[0] + " (m)", ylabel=name[-1] + " (m)")
        axis.grid(alpha=0.3)
    top.set_aspect("equal", adjustable="box")
    if region:
        (cx, cy), (sx, sy) = region["center_xy_m"], region["size_xy_m"]
        top.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, edgecolor="#888", lw=2, label="box top"))
        side.axhline(region["support_z_m"], color="#888", lw=2, label="box top")


def draw_all_paths(top, side, legs, region, limits):
    """Every leg in one top and one side view, with home, grasp and release marked."""
    for leg in legs:
        color = LEG_COLORS.get(leg["label"], "#999")
        top.plot(leg["tcp"][:, 0], leg["tcp"][:, 1], color=color, lw=2.2, label=leg["label"])
        side.plot(leg["tcp"][:, 1], leg["tcp"][:, 2], color=color, lw=2.2)
    marks = [("home", legs[0]["tcp"][0], "ko", 9), ("grasp", legs[0]["tcp"][-1], "k^", 10)]
    release = next((leg for leg in legs if leg["label"] == "Place.transport"), None)
    if release is not None:
        marks.append(("release", release["tcp"][-1], "ks", 9))
    for name, point, style, size in marks:
        top.plot(point[0], point[1], style, ms=size, label=name)
        side.plot(point[1], point[2], style, ms=size)
    for axis, name, title in ((top, "x-y", "All legs: top view (x-y)"), (side, "y-z", "All legs: side view (y-z)")):
        axis.set(title=title, xlim=limits[name][0], ylim=limits[name][1], xlabel=name[0] + " (m)", ylabel=name[-1] + " (m)")
        axis.grid(alpha=0.3)
    top.set_aspect("equal", adjustable="box")
    if region:
        (cx, cy), (sx, sy) = region["center_xy_m"], region["size_xy_m"]
        top.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, edgecolor="#888", lw=2, label="box top"))
        side.axhline(region["support_z_m"], color="#888", lw=2)
    top.legend(fontsize=8, loc="best")


def path_limits(legs, region):
    """Axis limits shared by every leg's path plots so the legs can be compared."""
    points = np.concatenate([leg["tcp"] for leg in legs])
    low, high = points.min(0), points.max(0)
    if region:
        (cx, cy), (sx, sy) = region["center_xy_m"], region["size_xy_m"]
        low[:2], high[:2] = np.minimum(low[:2], [cx - sx / 2, cy - sy / 2]), np.maximum(high[:2], [cx + sx / 2, cy + sy / 2])
        high[2] = max(high[2], region["support_z_m"])
    margin = 0.03
    return {"x-y": ((low[0] - margin, high[0] + margin), (low[1] - margin, high[1] + margin)),
            "y-z": ((low[1] - margin, high[1] + margin), (min(low[2], 0.0) - margin / 2, high[2] + margin))}


def draw(plan, legs, gaps, title):
    figure = plt.figure(figsize=(18, max(10.0, 3.4 * (len(legs) + 1) + 0.8)), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.1, 1.5])
    left = grid[0, 0].subgridspec(len(legs) + 1, 2, height_ratios=[1.25] + [1] * len(legs))
    right = grid[0, 1].subgridspec(3, 1)
    positions, velocities, accelerations = (figure.add_subplot(right[i, 0]) for i in range(3))
    region = plan.get("placement_region") or {}
    limits_xy = path_limits(legs, region)
    draw_all_paths(figure.add_subplot(left[0, 0]), figure.add_subplot(left[0, 1]), legs, region, limits_xy)
    for row, leg in enumerate(legs, 1):
        top, side = figure.add_subplot(left[row, 0]), figure.add_subplot(left[row, 1])
        draw_leg_paths(top, side, leg, region, limits_xy)
        if row == 1:
            top.legend(fontsize=8, loc="best")
    for leg in legs:
        color = LEG_COLORS.get(leg["label"], "#999")
        for axis in (positions, velocities, accelerations):
            axis.axvspan(leg["t"][0], leg["t"][-1], color=color, alpha=0.07)
        for j in range(6):
            kwargs = dict(color=JOINT_COLORS[j], lw=1.4, label=f"J{j + 1}" if leg is legs[0] else None)
            positions.plot(leg["t"], np.degrees(leg["q"][:, j]), **kwargs)
            velocities.plot(leg["t"], np.degrees(leg["v"][:, j]), **kwargs)
            accelerations.plot(leg["t"], leg["a"][:, j], **kwargs)
    limits = load_robot_limits()["max_acceleration"]
    for level, text in ((max(limits[:5]), "J1-5 limit"), (limits[5], "J6 limit")):
        for sign in (1, -1):
            accelerations.axhline(sign * level, color="#c44", ls="--", lw=1)
        accelerations.text(0.01, level, text, transform=accelerations.get_yaxis_transform(), color="#c44", va="bottom")
    for axis in (positions, velocities, accelerations):
        for start, end, name, measured in gaps:
            axis.axvspan(start, end, color="#999", alpha=0.2)
            for edge in (start, end):
                axis.axvline(edge, color="#555", lw=0.8)
            if axis is positions:
                note = "" if measured else " (nominal)"
                axis.text((start + end) / 2, 0.02, f"{name}\n{start:.2f} - {end:.2f} s ({end - start:.2f} s){note}",
                          transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    for leg in legs:
        last = leg is legs[-1] and not gaps_after(gaps, leg)
        positions.text(leg["t"][-1] if last else (leg["t"][0] + leg["t"][-1]) / 2, 0.98, f"{leg['label']}\n{leg['t'][0]:.2f} - {leg['t'][-1]:.2f} s ({leg['t'][-1] - leg['t'][0]:.2f} s)",
                       transform=positions.get_xaxis_transform(), ha="right" if last else "center", va="top", fontsize=8)
    positions.set(ylabel="deg", title="Joint position")
    velocities.set(ylabel="deg/s", title="Joint velocity")
    accelerations.set(ylabel="rad/s²", xlabel="time (s): planned leg time + measured gripper time", title="Joint acceleration")
    velocities.legend(ncol=6, loc="upper right", fontsize=8)
    end = max(legs[-1]["t"][-1], gaps[-1][1] if gaps else 0.0)
    for axis in (positions, velocities, accelerations):
        axis.set_xlim(0, end)
    for axis in (positions, velocities, accelerations):
        axis.grid(alpha=0.3)
    totals = ", ".join(f"{leg['label']} {leg['t'][-1] - leg['t'][0]:.2f}s" for leg in legs)
    figure.suptitle(f"{title}\n{totals}", fontsize=13)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path, help="Task folder containing plan.json")
    parser.add_argument("--output", type=Path, help="PNG path (default: motion.png in the task folder)")
    args = parser.parse_args()
    plan = json.loads((args.task_dir / "plan.json").read_text())
    legs, gaps = load_legs(plan, gripper_durations(args.task_dir))
    output = args.output or args.task_dir / "motion.png"
    draw(plan, legs, gaps, f"{args.task_dir.name} · {plan.get('class_id', '')}").savefig(output, dpi=110)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
