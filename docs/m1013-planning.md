# M1013 경로 계획·RViz 동작 확인

같은 출발·목표에서 **박스 없는 경로와 박스를 우회하는 경로**를 계산하고, gateway → ROS2 가상 팔 → RViz로 실행해 비교한다.
실물·물리 접촉 검증은 포함하지 않는다. 가상 ROS 환경 준비는 [가상 ROS 실행 확인](m1013-execution.md)을 따른다.

두 경로는 같은 계획 함수를 박스 없음/있음으로 호출하며, 순서는 `열림 → 이동 → 닫힘 → 홈 → 열림`으로 같다.

## 1. 경로 계산

```bash
pixi run python scripts/compare_m1013_obstacle.py
```

- 현재 관절값을 출발 자세(홈)로 쓰고, 목표는 관절 1~6에 각각 `+60°, -30°, +40°, +15°, -15°, +25°`를 더한 자세다.
- 중간을 막는 8 cm 박스를 놓고 팔·그리퍼 전체의 충돌 구를 검사한다.
- 출력 JSON의 `output_dir`을 변수로 둔다.

```bash
output_dir=tiptop_outputs/m1013/comparisons/<출력된 폴더명>
```

검사를 통과하면 `$output_dir/without-box-plan.json`, `$output_dir/with-box-plan.json`이 저장된다. 확인할 값:

- `outward_comparison_success: true`, 실행할 `plans` 항목의 `exported: true`
- 박스 없는 경로의 `min_box_clearance_m`은 음수, 우회 경로는 양수
- `max_self_collision_cost`는 0 (구간 사이 보간점까지 검사하지만 연속 시간 전체의 무충돌 증명은 아님)

## 2. RViz에서 비교

계획 생성 후 다른 명령으로 팔을 움직이지 않는다. 박스 없는 동작을 먼저 실행하고, 같은 홈에서 우회 동작을 실행한다.

```bash
pixi run python scripts/compare_m1013_obstacle.py --run "$output_dir" --case without_box
pixi run python scripts/compare_m1013_obstacle.py --run "$output_dir" --case with_box
```

- 박스 없는 동작은 **주황색 경로**, 우회 동작은 **빨간 박스·초록색 경로**로 표시된다. 선은 말단 궤적이고 충돌 검사는 팔·그리퍼 전체에 적용된다.
- 박스는 계획·표시용이며 두산 에뮬레이터에 물리 충돌을 추가하지 않는다.
- 정상 종료는 `status: completed`, `completed_steps: 5`다.
- 복귀 계획이 실패하면 `열림 → 이동 → 닫힘`까지만 실행하고 멈춘다(`status: stopped_before_unplanned_return`, `completed_steps: 3`). 역순 경로로 대체하지 않으며 사유는 `planning_failure`에 남는다.
- 첫 동작이 홈에 돌아오지 못했다면 두 번째 명령을 실행하지 않고 현재 자세에서 1번을 다시 한다. 시작 자세가 1° 넘게 다르면 실행이 차단된다.

## 3. 결과 확인

```bash
cat "$output_dir/report.json"
xdg-open "$output_dir/obstacle-comparison.png"
```

계획 입력·성공 경로·실패 사유는 `inputs.json`, `comparison-paths.json`, `report.json`, `run.log`에, 실행별 결과는 `executions/`에 남는다.
같은 시작·목표·박스로 다시 계산하려면 `--input "$output_dir/inputs.json"`, gateway 없이 계산만 하려면 `--offline`을 쓴다(여섯 관절 0 rad 자세를 가정하므로 다른 자세의 팔에는 바로 실행할 수 없다).

## 박스 없는 단독 실행

```bash
pixi run python -m tiptop.doosan plan    # 성공하면 tiptop_outputs/m1013/showcase.json 저장
pixi run python -m tiptop.doosan run tiptop_outputs/m1013/showcase.json
```

## 자동 검사

로봇에 명령을 보내지 않는다. 충돌 모델은 [충돌 설정](m1013-collision.md)을 따른다.

```bash
pixi run python -m pytest tests/test_m1013_planning.py tests/test_doosan_execution.py tests/test_config.py -q
```
