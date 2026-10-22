# M1013 Auto Picking 가상 실행 가이드

작업 지시와 현재 카메라 영상을 바탕으로 물체를 인식하고, UON 파지 DB와 계획기를 이용해
Isaac Sim에서 집기·빈자리 배치를 수행한다.

저장 결과 조회·물리 실험 재현·영상/DB 진단은 [실험 도구 가이드](../experiments/README.md)를 참고한다.

## 실행 절차

프로젝트 루트에서 각 단계를 별도 터미널로 실행한다.
출력 경로의 `manual-pick-place-01`은 매 실행마다 새 이름으로 바꾸고, Isaac과 작업 실행 명령에 같은 이름을 사용한다.

### 1. FoundationStereo 서버 실행

```bash
bash scripts/run_foundation_stereo_server.sh
```

### 2. FoundationPose 서버 실행

```bash
bash scripts/run_foundation_pose_server.sh
```

### 3. Isaac Sim 실행

```bash
# 가상 장면과 현재 관측/실행 backend. 실행할 때마다 새 폴더명을 사용한다.
/home/uon/isaac-sim/6.2.0/python.sh --no-ros-env \
  simulation/isaac_sim/live_perception.py \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/runtime
```

`LIVE_CAMERA_READY`와 두 추론 서버의 준비 완료를 확인한다.

### 4. 작업 지시 입력

```bash
# GEMINI_API_KEY 또는 GOOGLE_API_KEY가 등록된 터미널에서 실행.
# --allow-gemini-once는 실제 API를 사용한다. 자동 재시도는 하지 않는다.
pixi run python -m tiptop.scripts.run_known_pick_place \
  --task '후추통 집고 빈 곳에 놓아' \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/task \
  --allow-gemini-once --execute
```

`--allow-gemini-once`는 Gemini API를 1회 호출한다.
`--execute`를 생략하면 계획 저장까지만 수행한다. 한 Isaac runtime은 한 번만 실행을 받는다.
다시 실행하려면 Isaac을 재시작하고 새 출력 경로를 지정한다.

## 결과 확인

실행 결과는 지정한 출력 경로 아래에 저장된다.

| 파일 | 내용 |
|---|---|
| `task/plan.json` | 현재 관측에서 계산한 실행 계획 |
| `task/planning-result.json` | 계획 결과 |
| `task/empty-region.json` | 선정된 배치 영역 |
| `runtime/execution.json` | 집기·배치 동작 평가 |

`runtime/execution.json`의 `finished`, `success`, `inside_selected_region`,
`physical_placement_verified`가 모두 `true`인지 확인한다.

## 구현 구성

- `tiptop/tiptop_run.py`: 기존 `run_perception`에 `known_db` 선택과 빈자리 목표 연결.
  기존 M2T2 경로가 기본값이다. DB TCP에는 Panda/M2T2 보정을 적용하지 않는다.
- `tiptop/perception/known_objects.py`: 명시된 카탈로그 별칭으로 대상 하나만 연결.
- `tiptop/perception/placement.py`: 새 깊이에서 관측된 책상 면과 장애물로 빈 영역 선정.
  충분히 관측되지 않은 곳은 빈 곳으로 취급하지 않는다.
- `tiptop/doosan/known_task.py`: 기존 M1013 모델·카메라 충돌 외형·cuTAMP·계획 직렬화 사용.
- `tiptop/scripts/run_known_pick_place.py`: 현재 backend 관측 → 본체 인식 → 계획 → 실행 요청.
- `simulation/isaac_sim/live_perception.py`: 가상 장치와 물리 평가. 계획기는 포함하지 않는다.
- `simulation/isaac_sim/config/`: 가상 카메라와 세 물체의 초기 장면 설정.

## 카메라 및 통신 구성

가상 카메라는 336L 외형·카메라 간격을 참고한 **이상적인 좌우 RGB 카메라**다.
실물 336L의 IR 영상, 별도 컬러 렌즈, RGB/IR 정렬이나 노이즈를 재현한 모델은 아니다.

가상 실행 프로그램은 HTTP로 현재 좌우 RGB를 받아 FoundationStereo로 깊이를 계산한 뒤,
공통 TiPToP 인식·계획으로 넘긴다. TiPToP websocket은 거치지 않는다.
실물 336L 어댑터는 IR 양안으로 깊이를 계산하고 별도 컬러 영상에 정렬한다.

TiPToP websocket은 RGB와 이미 계산된 깊이를 입력받는 별도 진입점이다.
FoundationStereo를 사용하는 클라이언트는 깊이를 계산한 뒤 이 규약으로 전달할 수 있다.

## 지원 범위 및 제한 사항

현재 가상 실행의 동작 평가는 후추통에 한정된다. 나머지 두 종류의 파지나 실물 전환은 미검증이다.
카메라 외형·장착·TCP는 임시 가정이며, 5 N·m는 시뮬 장면의 상한이다.
시뮬 정답 물체 자세는 물리 평가에만 읽고 인식/운반 제어 입력으로 사용하지 않는다.
메인 `tiptop_run`의 기존 대화형 실행이나 websocket을 이 명령과 동일하게 모두 검증한 것은 아니다.
실시간 의미는 새 관측에서 인식·경로를 계산한다는 뜻이며 동작 중 영상 추적/재계획은 없다.
