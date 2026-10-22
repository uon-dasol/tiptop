# 후추통 인식부터 빈자리 놓기까지

저장 결과 조회·물리 실험 재현·영상/DB 진단은 [실험 도구 가이드](../experiments/README.md)를 참고한다.
아래 명령은 현재 카메라 관측에서 새로 인식하고 경로를 계획하는 본체 실행이다.

2026-10-22: 한국어 지시 `후추통 집고 빈 곳에 놓아`로 Gemini 1회 호출,
SAM2 → FoundationStereo → FoundationPose → UON DB → 기존 cuTAMP → Isaac 접촉 실행을 확인했다.
물리 실행은 1회이며, 약 86.6 mm 들기 → 307.1 mm 이동 → 해제 → 후퇴가 성공했다.
다른 두 물체의 측정 변위는 0, 해제 후 손가락 접촉 0, 책상 지지 약 0.86 N이었다.

## 지금 결과만 확인

- `tiptop_outputs/m1013/isaac/prompt-pick-place-20261022/summary.json`: 전체 요약·제한 사항.
- 같은 폴더 `final-window.png`: 완료된 Isaac 창.
- `runtime/final-camera.png`: 완료 후 실제 가상 카메라의 새 영상.
- `runtime/execution.json`: 물리 평가. `success`, `physical_placement_verified`가 true.
- `task/planning-result.json`, `task/plan.json`: 이번 관측에서 새로 계산한 계획.
- `task/gemini-call-consumed.json`: 유료 호출을 1회로 제한한 기록.

첫 시도에서 인식은 완료했으나 Open3D 점군의 중간 저장이 실패했다. 저장 코드를 고친 뒤
같은 지시의 라벨/ROI만 이어받아 새 영상으로 재관측했다. Gemini 추가 호출은 없었고,
저장된 영상·관절 경로를 재생하지 않았다. 실행 클라이언트는 결과 저장 후 종료할 때
native 충돌(exit 139)이 있었다. 이후 원인을 결과 이미지 저장 시 worker thread에서 생성된
Matplotlib Qt GUI로 분리 재현했다. 저장 전용 Agg canvas로 수정한 뒤 같은 최소재현과
저장된 영상의 SAM·점군·결과 저장 경로 모두 exit 0으로 종료했다. 전체 물리 작업은 반복하지 않았다.
이전 실행의 exit 139는 역사값으로 보존하고, 수정 후 증거는 `client-shutdown/result.json`에 별도로 남겼다.

종료 오류의 최소 회귀 검사는 API·로봇·GPU 없이 실행할 수 있다:

```bash
pixi run python -m pytest tests/test_perception_visualization.py -q
```

기대 결과는 `1 passed`와 정상 종료다.

## 다시 실행

프로젝트 루트에서 각 블록을 별도 터미널에서 실행한다. 현재 열린 Isaac은 결과 확인용이다.
새 작업은 그 창을 닫고, 이전 결과를 덮지 않는 **새 출력 폴더**로 시작한다.
실물 팔·카메라를 열지 않는다. Gemini를 쓸 때는 사용자에게 별도 허락을 받는다.

```bash
bash scripts/run_foundation_stereo_server.sh
```

```bash
bash scripts/run_foundation_pose_server.sh
```

```bash
# 가상 장면과 현재 관측/실행 backend. 실행할 때마다 새 폴더명을 사용한다.
/home/uon/isaac-sim/6.2.0/python.sh --no-ros-env \
  simulation/isaac_sim/live_perception.py \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/runtime
```

`LIVE_CAMERA_READY`와 두 서버의 준비 완료를 확인한 뒤:

```bash
# GEMINI_API_KEY 또는 GOOGLE_API_KEY가 등록된 터미널에서 실행.
# --allow-gemini-once는 실제 API를 사용한다. 자동 재시도는 하지 않는다.
pixi run python -m tiptop.scripts.run_known_pick_place \
  --task '후추통 집고 빈 곳에 놓아' \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/task \
  --allow-gemini-once --execute
```

`--execute`를 생략하면 계획 저장까지만 한다. 한 Isaac runtime은 한 번만 실행을 받는다.
실행 후 `runtime/execution.json`의 `finished=true`, `success=true`,
`inside_selected_region=true`, `physical_placement_verified=true`를 확인한다.
실패 시 자동 재실행하지 말고 해당 폴더의 로그를 확인한다.

## 코드가 연결된 위치

- `tiptop/tiptop_run.py`: 기존 `run_perception`에 `known_db` 선택과 빈자리 목표 연결.
  기존 M2T2 경로가 기본값이다. DB TCP에는 Panda/M2T2 보정을 적용하지 않는다.
- `tiptop/perception/known_objects.py`: 명시된 카탈로그 별칭으로 대상 하나만 연결.
- `tiptop/perception/placement.py`: 새 깊이에서 관측된 책상 면과 장애물로 빈 영역 선정.
  충분히 관측되지 않은 곳은 빈 곳으로 취급하지 않는다.
- `tiptop/doosan/known_task.py`: 기존 M1013 모델·카메라 충돌 외형·cuTAMP·계획 직렬화 사용.
- `tiptop/scripts/run_known_pick_place.py`: 현재 backend 관측 → 본체 인식 → 계획 → 실행 요청.
- `simulation/isaac_sim/live_perception.py`: 가상 장치와 물리 평가. 계획기는 포함하지 않는다.
- `simulation/isaac_sim/config/`: 가상 카메라와 세 물체의 초기 장면 설정.

현재 fixture의 실행 평가는 후추통에 한정된다. 나머지 두 종류의 파지나 실물 전환은 미검증이다.
카메라 외형·장착·TCP는 임시 가정이며, 5 N·m는 시뮬 장면의 상한이다.
시뮬 정답 물체 자세는 물리 평가에만 읽고 인식/운반 제어 입력으로 사용하지 않는다.
메인 `tiptop_run`의 기존 대화형 실행이나 websocket을 이 명령과 동일하게 모두 검증한 것은 아니다.
실시간 의미는 새 관측에서 인식·경로를 계산한다는 뜻이며 동작 중 영상 추적/재계획은 없다.

## 카메라 입력과 통신 경로

현재 가상 카메라는 336L 외형·카메라 간격을 참고한 **이상적인 좌우 RGB 카메라**다.
실물 336L의 IR 영상, 별도 컬러 렌즈, RGB/IR 정렬이나 노이즈를 재현한 모델은 아니다.
가상 연결 성공을 실물 336L의 깊이 정확도 검증으로 해석하지 않는다.

이 가이드의 실행 명령은 HTTP로 현재 좌우 RGB를 받아 실행 프로그램에서
FoundationStereo를 호출한 뒤 공통 TiPToP 인식·계획으로 넘긴다. TiPToP websocket은 거치지 않는다.
실물 336L 어댑터는 IR 양안으로 깊이를 계산하고 별도 컬러 영상에 정렬한다.

TiPToP websocket은 기존처럼 RGB＋이미 계산된 깊이를 입력받는다.
`droid-sim-evals-6.2`는 이 규약으로 Isaac 제공 깊이를 보낸다.
FoundationStereo를 쓰는 클라이언트도 먼저 깊이를 계산해서 같은 규약으로 보낼 수 있다.
websocket 서버 내부에서 양안 영상을 받아 FoundationStereo를 호출하는 추가 확장은 제거했다.
현재 가상 실행의 FoundationStereo 경로는 유지하며, 별도 Isaac 정답 깊이 선택 모드는 추가하지 않았다.
