# 가상 팔 왕복 시험

실험실 홈에서 J1을 +5° 이동하고 복귀한다. 편도 5초이며 그리퍼 명령은 없다. 모델 충돌 검사이며 실물 동작 검증이 아니다.

먼저 [가상 runtime](m1013-fixed-camera.md#실행)을 실행하고, 그 runtime의 카메라·구조물 설정으로 계획을 준비한다.

```bash
pixi run python simulation/isaac_sim/arm_trial.py \
  --runtime tiptop_outputs/m1013/isaac/lab-fixed-01/runtime \
  --output tiptop_outputs/m1013/arm-trial
```

기존 runtime을 종료한 뒤 새 출력 폴더로 시험한다.

```bash
/home/uon/isaac-sim/6.2.0/python.sh --no-ros-env \
  simulation/isaac_sim/live_perception.py --lab \
  --joint-test tiptop_outputs/m1013/arm-trial/plan.json \
  --output tiptop_outputs/m1013/isaac/arm-trial-01/runtime
```

`ARM TRIAL PASSED`와 계획 폴더의 `sim-result.json`을 확인한다. 모델·설정을 바꾸면 계획을 다시 만든다.
