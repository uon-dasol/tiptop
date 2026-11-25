# M1013+UON cuTAMP 연결 검사

고정 예제로 계획기·실행기 입력 연결을 검사한다. 현재 영상으로 작업하려면 [작업 실행](known-object-pick-place.md)을 쓴다.

## 계획 생성

```bash
OMP_NUM_THREADS=2 pixi run python -m tiptop.doosan.tamp \
  --data /nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/black_pepper_shaker/UON_3finger_gripper \
  --cad /nas/ochansol/3d_model/2026_real_world_test_objects/black_pepper_shaker/edited/black_pepper_shaker.obj \
  --output tiptop_outputs/m1013/tamp/verified
```

로봇에 연결하지 않는다. `result.json`의 `stage: executor_verified`와 `gateway_contract_verified: true`를 확인하고, 실행 계획은 `plan.json`이다.
`--init-only`는 초기화만 검사하고, `--q-init`에 6축 rad 값을 주면 다른 시작 자세로 계획한다. 종점의 미세한 속도만 0으로 맞추며 위치·샘플 수·시간은 유지한다.

## 가상 RViz 재생

[가상 ROS 환경](m1013-execution.md)을 준비한 뒤 실행한다.

```bash
OMP_NUM_THREADS=2 pixi run python scripts/rviz_m1013_tamp.py \
  tiptop_outputs/m1013/tamp/verified/plan.json --execute
```

`--execute`를 빼면 표시만 한다. MarkerArray 토픽은 `/uon_model/planning_preview`다.
RViz의 물체 부착은 시각화이며 실제 접촉 파지를 검증하지 않는다. 현재 자세와 계획 시작값이 다르면 다시 계획한다.
