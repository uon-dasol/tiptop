# M1013+UON 최소 cuTAMP 연결 확인

후추통 하나·고정 자세·DB 파지 후보 하나로 cuTAMP 계획을 만들고,
기존 실행기를 통해 가상팔에서 실행한다. RViz 물체 부착은 시각화이며 물리 파지 검증은 Isaac 단계다.

프로젝트 루트에서 `pixi run python`으로 실행한다.
ROS2 준비는 [기존 실행 안내](m1013-execution.md)를 따른다.

```bash
cd /home/uon/workspace/tiptop
```

## 1. 계획 생성·실행기 입력 검증

```bash
OMP_NUM_THREADS=2 pixi run python -m tiptop.doosan.tamp \
  --data /media/uon/data1/Dataset_2026/isaacsim_grasp_data_gen/black_pepper_shaker/UON_3finger_gripper \
  --cad /media/uon/data1/3d_model/2026_real_world_test_objects/black_pepper_shaker/edited/black_pepper_shaker.obj \
  --output tiptop_outputs/m1013/tamp/verified
```

cuTAMP가 `MoveFree → Pick → MoveHolding → Place` 작업 순서와 6축 경로·개폐 명령을 만든다.
96개 충돌 구 모델로 실험 커널을 사용한다.
저장할 계획을 기존 실행기→Doosan 클라이언트→게이트웨이 입력 검사에 모의전달한다.
경로 끝의 미세한 속도는 기존 `prepare_joint_path`로 정지 유지 구간을 추가해 처리한다.
이 단계는 게이트웨이나 로봇에 명령을 보내지 않는다.

출력 폴더의 `result.json`에서 `success: true`, `stage: executor_verified`,
`gateway_contract_verified: true`를 확인한다. 실행할 계획은 `plan.json`이다.
실패 시 `stage`, `error`, `planning_failure`를 확인한다.
같은 출력 폴더로 재실행하면 계획과 결과가 갱신된다.

## 2. RViz에서 후추통과 함께 가상 실행

드라이버·컨트롤러·게이트웨이·RViz 모델 표시를 켠다.
RViz의 **Add → By display type → MarkerArray**를 추가하고,
Topic을 `/uon_model/planning_preview`로 설정한다.

```bash
pixi run python -m tiptop.doosan status
pixi run python -m tiptop.doosan open
OMP_NUM_THREADS=2 pixi run python scripts/rviz_m1013_tamp.py \
  tiptop_outputs/m1013/tamp/verified/plan.json --execute
```

기본 시작 관절은 모두 0rad다. 현재 자세가 다르면 1번 명령에
`--q-init q1 q2 q3 q4 q5 q6`을 추가해 현재 rad 값에서 계획을 다시 만든다.
시작 자세가 1° 넘게 다르면 실행기가 거부한다.

저장된 계획을 실행기→클라이언트→ROS2 게이트웨이→컨트롤러→드라이버→가상제어기로 전달한다.
후추통 CAD와 초록 목표 영역을 표시하며 계획을 다시 계산하지 않는다.
닫힘 완료 후 후추통이 모델 TCP를 따라가고, 열림 완료 후 그 위치에 남는다.
접근→닫힘→운반·배치→열기→팔 복귀를 관찰한다.

성공 시 `rviz-execution.json`의 `success: true`, `completed_steps: 9`와
`final_state.motion_status: 0`을 확인하고, 시작 자세 복귀와 그리퍼 열림을 확인한다.

## 선택 확인

- **초기화만 확인:** 1번 명령에 `--init-only`를 추가한다.
  `stage: initialized`, `joint_limits_shape: [2, 6]`, `motion_gen_dof: 6`을 확인한다.
  모델·작업 순서·계획기 구성을 확인하며 구체적인 이동 경로는 생성하지 않는다.
- **물체만 표시:** 2번의 RViz 실행 명령에서 `--execute`를 뺀다. 종료는 Ctrl+C다.
- **물체 표시 없이 실행:** 2번의 RViz 실행 명령 대신 아래 명령을 사용한다.
  두 실행 명령을 연달아 실행할 필요는 없다.

```bash
pixi run python -m tiptop.doosan run tiptop_outputs/m1013/tamp/verified/plan.json
```

## 입력과 한계

- `--data`, `--cad` 생략 시 `tiptop/config/known_objects.json`에 등록된 후추통의
  외부 DB 디렉터리와 CAD를 사용한다. 데이터 위치는 [실행 가이드](known-object-pick-place.md#물체-데이터)를 참고한다.
- CAD는 0.001m 축척으로 읽고 convex hull을 메모리에서 만든다.
  물체 위치 기본값은 `[0.55, -0.12, 0.5]`m, 회전 기본값은 단위 회전이다.
  `--object-rpy rx ry rz`로 월드 기준 XYZ Euler 회전(rad)을 지정할 수 있다.
- 목표는 중심 `[0.65, 0.0, 0.38]`m, 크기 `[0.8, 0.8, 0.04]`m인 책상 위 배치
  `On(black_pepper_shaker, target)`이다. 상판 높이는 0.4m이며 `--target-xyz`, `--target-dims`로 바꿀 수 있다.
  정확한 최종 6D 자세 지정은 아니다.
- 파지 후보는 DB index 0(`scene_id=0072`, `source_pregrasp_index=443`) 하나다.
  DB grasp→모델 TCP의 기본값은 단위 변환(추가 회전 없음·이동 0)이다.
  기록된 접촉점과 손가락 형상을 비교해 임시 `Rx(pi)`를 제거했다. 이동 0과 물리 파지는 미검증이며,
  `--grasp-from-tcp MATRIX.json`으로 변환을 지정할 수 있다. 기존 저장 계획은 재생성해야 반영된다.
- 계획 중 그리퍼 형상은 열린 상태로 고정한다. 실행 개폐는 기존 모델 endpoint를 사용한다.
  기존 `verified/plan.json`에는 이전 물체 배치·임시 회전이 남아 있으므로 현재 파지 검증 결과로 취급하지 않는다.
- 외부 cuTAMP는 수정하지 않는다. 전용 프로세스 안에서 로봇 이름 검증·container 로딩을 임시 연결하고 복구한다.
- 실환경 파지 전 DB grasp→body/TCP·flange 장착 변환을 확인·보정해야 한다.
  접촉·파지력·미끄러짐·물리적 들림은 이번 검증에 포함하지 않는다.
