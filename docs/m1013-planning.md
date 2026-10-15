# M1013 경로 계획·RViz 동작 확인

같은 출발·목표에서 **박스 없는 경로와 박스를 우회하는 경로**를 계산하고,
기존 게이트웨이 → ROS2 가상 팔 → RViz 연결로 실행한다. 실물·물리 접촉 검증은 포함하지 않는다.

기존 `tiptop.doosan plan`은 박스 없는 showcase를 생성한다.
비교 스크립트도 같은 공통 계획 함수를 박스 없음/있음으로 호출하며, 비교·표시·로그를 추가한다.
목표 자세와 `열림 → 이동 → 닫힘 → 홈 → 열림` 순서는 두 방식이 공유한다.

## 1. 박스 유무 경로 계산

```bash
pixi run python scripts/compare_m1013_obstacle.py
```

현재 관절값을 출발 자세(이번 시험의 홈)로 사용한다. 목표는 관절 1~6에 각각
`+60°, -30°, +40°, +15°, -15°, +25°`를 더한 자세다.
중간을 막는 8cm 박스를 배치하며 팔·그리퍼 전체의 충돌 구를 검사한다.

출력 JSON의 `output_dir` 경로를 복사해 같은 터미널에서 아래 변수에 넣는다.

```bash
output_dir=tiptop_outputs/m1013/comparisons/실제_기록_폴더명
```

각 경우의 검사에 통과하면 실행할 계획이 아래 파일로 저장된다.

- `$output_dir/without-box-plan.json`: 박스 없는 동작
- `$output_dir/with-box-plan.json`: 박스를 고려한 동작

`outward_comparison_success: true`와 실행하려는 `plans` 항목의 `exported: true`를 확인한다.
박스 없는 경로의 `min_box_clearance_m`은 음수, 박스 우회 경로는 양수여야 한다.
`max_self_collision_cost`는 0이어야 한다. 구간 사이 추가 보간점도 검사하지만 연속 시간 전체의 무충돌 증명은 아니다.

## 2. RViz에서 동작 비교

계획 생성 후 다른 명령으로 팔을 움직이지 않는다. 먼저 박스 없는 동작을 실행한다.

```bash
pixi run python scripts/compare_m1013_obstacle.py --run "$output_dir" --case without_box
```

**주황색 경로**, 박스 없는 팔 움직임과 `열림 → 이동 → 닫힘 → 홈 → 열림`을 확인한다.
`status: completed`, `completed_steps: 5`로 끝나면 같은 홈에서 박스 우회 동작을 실행한다.

```bash
pixi run python scripts/compare_m1013_obstacle.py --run "$output_dir" --case with_box
```

**빨간 박스·초록색 경로**와 팔이 우회하는 모습을 확인한다. 선은 모델 말단 궤적이고,
충돌 검사는 팔·그리퍼 전체에 적용되므로 팔 몸체·손가락도 함께 본다.
박스는 계획·표시용이며 두산 에뮬레이터에 물리 충돌을 추가하지 않는다.

- 정상 순서: **열림 → 이동 → 닫힘 → 홈 → 열림**.
- 복귀 계획 실패 시: 검사된 **열림 → 이동 → 닫힘**까지만 실행하고 멈춘다.
  `status: stopped_before_unplanned_return`, `completed_steps: 3`이며 5단계 완료가 아니다.
  실패한 복귀를 역순 경로로 대체하지 않는다. `planning_failure`에 사유를 남긴다.
- 첫 동작이 홈에 돌아오지 못했다면 두 번째 명령을 이어서 실행하지 않는다.
  현재 자세에서 1번 계획을 새로 만든다. 시작 자세가 1° 넘게 다르면 실행이 차단된다.

## 3. 결과와 실패 기록 확인

```bash
cat "$output_dir/report.json"
xdg-open "$output_dir/obstacle-comparison.png"
```

계획 입력·성공 경로·실패 사유는 해당 폴더의 `inputs.json`, `comparison-paths.json`,
`report.json`, `run.log`에 남는다. 실행별 결과와 로그는 그 아래 `executions/`에 저장한다.
팔 완료 로그의 `motion_status: 0`, `max_joint_error_rad`도 확인한다.
`TRAJOPT_FAIL`만으로 복귀 불가능이나 특정 원인을 단정할 수는 없다.

같은 시작·목표·박스로 계산을 다시 시험하려면:

```bash
pixi run python scripts/compare_m1013_obstacle.py --input "$output_dir/inputs.json"
```

게이트웨이 없이 계산만 시험하려면 1번 명령에 `--offline`을 붙인다.
여섯 관절 모두 0rad인 자세를 가정하므로, 현재 팔이 다른 자세라면 그 결과를 바로 실행할 수 없다.

## 개발 검사·박스 없는 단독 실행

로봇 명령을 보내지 않는 자동 검사:

```bash
pixi run python -m pytest tests/test_m1013_planning.py tests/test_doosan_execution.py tests/test_config.py -q
```

충돌 모델·GPU 검사는 [충돌 검사 안내](m1013-collision.md)를 따른다.
박스 비교 없이 단독으로 계획·실행할 때 사용하는 별도 명령이다.
`pixi run python -m tiptop.doosan plan`이 성공하면 기본 경로
`tiptop_outputs/m1013/showcase.json`에 저장하고 출력의 `plan_file`로 경로를 알려준다.
`success: true`를 확인한 뒤 `pixi run python -m tiptop.doosan run tiptop_outputs/m1013/showcase.json`으로 실행한다.
