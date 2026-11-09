"""Compare table-support modes on saved task outputs. CPU only: no servers, camera or robot."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from experiments.grasping.paths import PROJECT_ROOT, WORK_ROOT
from tiptop.perception.placement import MAX_SUPPORT_GAP_M, check_support_gap, table_support_z
from tiptop.perception.segmentation import segment_table_with_ransac
from tiptop.perception.utils import depth_to_xyz

DEFAULT_TASKS = PROJECT_ROOT / "tiptop_outputs/m1013/tasks"
DEFAULT_PROFILE = PROJECT_ROOT / "tiptop/config/m1013_fixed_hardware.declared.json"
GAP_SWEEP_MM = (10, 15, 20, 30, 40)
REQUIRED = ("observation.npz", "registered-pose.json", "perception/depth.png", "perception/intrinsics.json",
            "perception/masks.npz")


def profile_table_top_z(path):
    """Top of the profile's axis-aligned full_table in the base frame."""
    rows = json.loads(Path(path).read_text())["workspace"]["fixed_obstacles"]
    table = next(row for row in rows if row["name"] == "full_table")
    pose = np.asarray(table["base_from_obstacle"], float)
    return float(pose[2, 3] + table["dims_m"][2] / 2)


def load_task(task_dir):
    """Rebuild the world-frame point map and target data the pipeline had before choosing the table."""
    depth = cv2.imread(str(task_dir / "perception/depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    intrinsics = np.asarray(json.loads((task_dir / "perception/intrinsics.json").read_text())["intrinsics"])
    world_from_cam = np.load(task_dir / "observation.npz", allow_pickle=False)["world_from_cam"]
    xyz = depth_to_xyz(depth, intrinsics) @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    xyz[~(np.isfinite(depth) & (depth > 0))] = np.nan
    pose = json.loads((task_dir / "registered-pose.json").read_text())
    return dict(
        xyz=xyz,
        rgb=np.zeros_like(xyz),
        masks=np.load(task_dir / "perception/masks.npz")["arr_0"],
        class_id=pose["class_id"],
        cad_bottom=np.asarray(pose["support_point_world"], float),
    )


def evaluate(column, task, declared_tops, max_gap_m):
    """One table-support mode on one task: support height, CAD-bottom gap and pass/fail."""
    if column.startswith("declared@"):
        support_z = declared_tops[column]
    else:
        options = dict(support_points_world=task["cad_bottom"][None]) if column == "ransac-cad" else {}
        try:
            table = segment_table_with_ransac(task["xyz"], task["rgb"], task["masks"], **options)
        except RuntimeError as error:
            return dict(ok=False, support_z_m=None, gap_mm=None, reason=str(error).split(" (tried")[0])
        support_z = table_support_z(table.bounds[1, 2])
    gap = float(task["cad_bottom"][2] - support_z)
    if column == "ransac-cad":
        # The pipeline relies on RANSAC's own 3 cm contact threshold here; the gap is informational.
        return dict(ok=True, support_z_m=support_z, gap_mm=gap * 1000, reason=None)
    try:
        check_support_gap(task["class_id"], task["cad_bottom"][2], support_z, max_gap_m)
    except ValueError as error:
        return dict(ok=False, support_z_m=support_z, gap_mm=gap * 1000, reason=str(error).split(": ", 1)[1])
    return dict(ok=True, support_z_m=support_z, gap_mm=gap * 1000, reason=None)


def cell(result):
    if result["ok"]:
        return f"ok {result['gap_mm']:+6.1f}"
    return "FAIL" if result["gap_mm"] is None else f"FAIL {result['gap_mm']:+6.1f}"


def task_number(path):
    return int(path.name.split("-")[1])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="저장된 작업 출력에서 책상 지지면 방식(ransac-cad / ransac-mask / declared)을 오프라인 비교한다."
    )
    p.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS, help="task-NNN 폴더가 있는 위치")
    p.add_argument("--task-range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="비교할 작업 번호 범위(포함)")
    p.add_argument("--hardware-profile", type=Path, default=DEFAULT_PROFILE, help="declared 방식의 full_table 출처")
    p.add_argument("--declared-support-z-m", type=float, nargs="+",
                   help="declared 지지 높이 후보(m); 지정하지 않으면 프로필 full_table 윗면")
    p.add_argument("--max-gap-m", type=float, default=MAX_SUPPORT_GAP_M,
                   help="ransac-mask/declared의 CAD 바닥-지지면 허용 차이(m), 기본 0.03")
    p.add_argument("--output", type=Path, help="새 결과 폴더 (기본: 작업 출력 위치의 table-support/<시각>)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    declared = args.declared_support_z_m or [profile_table_top_z(args.hardware_profile)]
    declared_tops = {f"declared@{z * 1000:.1f}mm": z for z in declared}
    columns = ["ransac-cad", "ransac-mask", *declared_tops]
    output = (args.output or WORK_ROOT / "table-support" / f"{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if output.exists():
        raise ValueError(f"출력 폴더가 이미 있습니다. 새 경로를 지정하세요: {output}")
    dirs = sorted((d for d in args.tasks_root.glob("task-*") if (d / "registered-pose.json").is_file()), key=task_number)
    if args.task_range:
        dirs = [d for d in dirs if args.task_range[0] <= task_number(d) <= args.task_range[1]]
    o3d.utility.random.seed(0)
    rows, skipped = [], []
    print(f"비교 대상 {len(dirs)}개 작업, 방식: {', '.join(columns)}", flush=True)
    for task_dir in dirs:
        if not all((task_dir / name).is_file() for name in REQUIRED):
            skipped.append(task_dir.name)
            continue
        task = load_task(task_dir)
        results = {column: evaluate(column, task, declared_tops, args.max_gap_m) for column in columns}
        rows.append(dict(task=task_dir.name, class_id=task["class_id"], results=results))
        print(f"{task_dir.name:9s} {task['class_id'][:28]:28s} "
              + "  ".join(f"{column}: {cell(results[column]):12s}" for column in columns), flush=True)
    print(f"\n작업 {len(rows)}개 비교 (필요 파일 누락으로 제외 {len(skipped)}개)")
    for column in columns:
        failed = [row for row in rows if not row["results"][column]["ok"]]
        print(f"  {column:18s} 통과 {len(rows) - len(failed):3d} / 실패 {len(failed):3d}")
    differing = [row for row in rows if len({row["results"][c]["ok"] for c in columns}) > 1]
    print(f"방식에 따라 결과가 갈린 작업 {len(differing)}개: {', '.join(row['task'] for row in differing[:30])}")
    for column in columns:
        reasons = Counter(
            "평면 선택 실패" if row["results"][column]["gap_mm"] is None else
            ("지지면보다 위" if row["results"][column]["gap_mm"] > 0 else "지지면보다 아래")
            for row in rows if not row["results"][column]["ok"]
        )
        if reasons:
            print(f"  {column} 실패 사유: " + ", ".join(f"{name} {count}" for name, count in reasons.items()))
    print("CAD 바닥-지지면 차이가 한계 이내인 작업 수 (평면 선택이 실패한 작업은 제외)")
    print("  " + "방식".ljust(18) + "".join(f"{limit:>8d}mm" for limit in GAP_SWEEP_MM))
    for column in columns:
        gaps = [abs(row["results"][column]["gap_mm"]) for row in rows if row["results"][column]["gap_mm"] is not None]
        print("  " + column.ljust(18) + "".join(f"{sum(g <= limit for g in gaps):>10d}" for limit in GAP_SWEEP_MM))
    output.mkdir(parents=True)
    note = ("저장된 depth.png(mm, 로봇 점 제거 후)·intrinsics·world_from_cam으로 점군을 다시 만들었다. RANSAC은 "
            "Open3D 난수를 0으로 고정해 실제 실행과 평면 선택이 다를 수 있다. ransac-cad의 gap은 기록용이다.")
    (output / "results.json").write_text(json.dumps(
        dict(note=note, columns=columns, max_gap_m=args.max_gap_m, declared_support_z_m=declared_tops,
             tasks_root=str(args.tasks_root), skipped=skipped, rows=rows), indent=2, ensure_ascii=False) + "\n")
    print(f"결과 저장: {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
