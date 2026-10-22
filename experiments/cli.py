"""Experiment entry point. Help, stored results and command previews use only the standard library."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "pepper": ("후추통", "black_pepper_shaker", "baseline-r01"),
    "lotion": (
        "컨디셔너 병",
        "car_leather_conditioner_bottle",
        "measured-place-multi-alt-car_leather_conditioner_bottle-c2463-r01",
    ),
    "dipper": (
        "미키 바가지",
        "disney_mickey_handled_dipper",
        "measured-place-multi-alt-disney_mickey_handled_dipper-c2435-r01",
    ),
}
TOOLS = {
    "check-frames": ("check_adapter", "CPU: CAD/DB/계획 좌표 변환·물체 식별 검사"),
    "check-db": ("verify_local_db_bridge", "CPU: 세 종류 로컬 DB → 계획 좌표 변환 대조"),
    "audit-close": ("audit_closure_sweep", "CPU: 닫힘 중 책상·다른 물체와의 형상 겹침 검사"),
    "audit-carry": ("audit_multiscene_geometry", "GPU: 운반 물체 전체의 충돌 형상 검사"),
    "prepare-scenes": ("prepare_multiscene", "GPU: 세 물체 배치의 후보·IK 및 접근/들기 경로 준비"),
    "prepare-measured": ("prepare_measured_placements", "GPU: 이전 시뮬 측정 자세를 쓴 놓기 연구"),
    "plan-check": ("probe_cutamp_scene_safe", "GPU: DB 후보의 cuTAMP 연결 확인, 물리 실행 없음"),
    "summarize": ("summarize", "CPU: check.json에서 힘·미끄러짐·놓기 오차 표 재생성"),
    "diagnose": ("diagnose_trials", "CPU: 닫기/들기 중 상대 자세와 DB 자세의 차이 계산"),
    "batch": ("run_batch", "Isaac/GPU: 지정 manifest의 실험 실행, 기존 결과는 건너뜀"),
}


def parser():
    p = argparse.ArgumentParser(
        description="실험·진단 도구. 본체의 새 관측 작업은 docs/known-object-pick-place.md 참고."
    )
    p.add_argument("--data-root", type=Path, help="읽을 결과/계획 폴더 (기본: 보존된 20261020 결과)")
    p.add_argument("--work-dir", type=Path, help="새 분석·계획 출력 폴더 (기본: tiptop_outputs/experiments/work)")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="사용 가능한 실험과 자원 요구 사항")
    results = commands.add_parser("results", help="저장된 결과 조회; 추론·파일 쓰기 없음")
    results.add_argument("--object", choices=list(CASES))
    results.add_argument("--failures-only", action="store_true")
    run = commands.add_parser("run", help="기존 물리 실험 한 건 재현; 기본은 명령 미리보기")
    run.add_argument("--case", choices=list(CASES), required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--isaac-python", type=Path, default=Path("/home/uon/isaac-sim/6.2.0/python.sh"))
    run.add_argument("--headless", action="store_true")
    run.add_argument("--execute", action="store_true", help="실제로 Isaac 실행; 미지정이면 미리보기만")
    perception = commands.add_parser("perception", help="파지 DB 영상 뷰어 또는 새 카메라 인식 진단")
    perception.add_argument("mode", choices=["view", "observe"])
    perception.add_argument("args", nargs=argparse.REMAINDER)
    tool = commands.add_parser("tool", help="고급 진단 도구; --execute 전에는 설명만 출력")
    tool.add_argument("name", choices=list(TOOLS))
    tool.add_argument("--execute", action="store_true")
    tool.add_argument("args", nargs="*", help="하위 도구 인자 앞에는 --를 붙인다")
    return p


def module_command(module, args, env):
    return subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env).returncode


def show_results(args, paths):
    records = paths.read_json(paths.DATA_ROOT / "results.json")
    if args.object:
        records = [row for row in records if row["target_class"] == CASES[args.object][1]]
    if not records:
        raise ValueError("결과가 없습니다. --data-root 또는 results.json을 확인하세요.")
    placed = [row for row in records if row["expected_place"]]
    print(f"물리 실험 {len(records)}건: 기준 통과 {sum(bool(r['validated']) for r in records)}건")
    print(f"전체 집기·놓기 {len(placed)}건: 기준 통과 {sum(bool(r['validated']) for r in placed)}건")
    print("저장된 시뮬 물리 시험이며, 카메라·자연어 전체 연결의 반복 성능이나 실물 성공률은 아닙니다.")
    for row in records:
        if args.failures_only and row["validated"]:
            continue
        print(f"{'PASS' if row['validated'] else 'FAIL'}  {row['name']}  phase={row['phase']}")
        if row.get("error"):
            print(f"  {row['error']}")
    print(f"원본 결과: {paths.DATA_ROOT}")


def run_case(args, paths, env):
    title, _, case_name = CASES[args.case]
    source = paths.DATA_ROOT / "runs" / case_name / "pick.json"
    record = paths.read_json(source)
    output = (args.output or paths.WORK_ROOT / "runs" / f"{args.case}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if not output.is_relative_to(ROOT / "tiptop_outputs") or output == ROOT / "tiptop_outputs":
        raise ValueError("출력은 프로젝트 tiptop_outputs 내부의 새 하위 폴더를 지정하세요.")
    if output == paths.DATA_ROOT or output.is_relative_to(paths.DATA_ROOT):
        raise ValueError("기존 입력 결과를 보존하기 위해 --data-root 외부에 출력하세요.")
    if output.exists():
        raise ValueError(f"출력 폴더가 이미 있습니다. 새 경로를 지정하세요: {output}")
    runner = (
        paths.TOOLS_ROOT / "pick_multi.py"
        if record.get("target_object")
        else ROOT / "simulation/isaac_sim/pick_pepper.py"
    )
    command = [
        str(args.isaac_python),
        "--no-ros-env",
        str(runner),
        "--check",
        "--physics-hz",
        "240",
        "--spinner-max-torque",
        "5",
        "--snapshot",
        "--output",
        str(output),
    ]
    if args.headless:
        command.append("--headless")
    print(f"{title}: 저장된 실험 입력의 물리 재현 (새 영상 인식·새 cuTAMP 계획 아님)", flush=True)
    print(f"입력: {source}\n출력: {output}", flush=True)
    print(shlex.join(command), flush=True)
    if not args.execute:
        print("미리보기만 했습니다. 실행하려면 같은 명령에 --execute를 추가하세요.")
        return 0
    if not args.isaac_python.is_file():
        raise FileNotFoundError(args.isaac_python)
    output.mkdir(parents=True)
    (output / "pick.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "experiment-source.json").write_text(
        json.dumps({"source": str(source), "case": args.case}, indent=2) + "\n"
    )
    print(f"실행 후 평가: {output / 'check.json'}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main(argv=None):
    p = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    # Advanced scripts use their own flags; keep them out of the common parser.
    extra = []
    if "--" in raw:
        index = raw.index("--")
        raw, extra = raw[:index], raw[index + 1 :]
    args = p.parse_args(raw)
    if args.command not in ("tool", "perception") and extra:
        p.error("하위 인자는 tool/perception에서만 사용할 수 있습니다.")
    if args.data_root:
        os.environ["TIPTOP_EXPERIMENT_DATA"] = str(args.data_root.resolve())
    if args.work_dir:
        os.environ["TIPTOP_EXPERIMENT_WORK"] = str(args.work_dir.resolve())
    from experiments.grasping import paths

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    try:
        if args.command == "list":
            for name, (title, _, _) in CASES.items():
                print(f"run --case {name:6}  {title}: 집기·놓기 물리 실험")
            print("perception view     저장된 실제 영상에 DB 원점·축·접촉점 표시 (CPU)")
            print("perception observe  현재 가상 카메라 → SAM/깊이/CAD 자세/DB 확인 (GPU, Gemini 호출 없음)")
            for name, (_, description) in TOOLS.items():
                print(f"tool {name:18} {description}")
            return 0
        if args.command == "results":
            show_results(args, paths)
            return 0
        if args.command == "run":
            return run_case(args, paths, env)
        if args.command == "perception":
            module = "view_grasps" if args.mode == "view" else "observe"
            return module_command(f"experiments.perception.{module}", [*args.args, *extra], env)
        module, description = TOOLS[args.name]
        print(description, flush=True)
        print(f"입력: {paths.DATA_ROOT}\n새 출력: {paths.WORK_ROOT}", flush=True)
        if not args.execute:
            print("설명만 표시했습니다. 실행하려면 --execute를 추가하세요. 하위 인자는 -- 다음에 지정합니다.")
            return 0
        paths.ensure_workdirs()
        return module_command(f"experiments.grasping.{module}", [*args.args, *extra], env)
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"오류: {error}\n")
