# M1013 실험실 가상 고정 카메라

`live_perception.py --lab`은 실물 운영 설정을 공유하는 Isaac 가상 환경이다. 실물 장치에는 연결하지 않는다.

## 설정

| 항목 | 기준 파일·값 |
|---|---|
| 실물·가상 공통 하드웨어 | `tiptop/config/m1013_fixed_hardware.declared.json` |
| 카메라 설치·영상 회전 | `tiptop/config/m1013_fixed_camera.declared.json` |
| SDK 내부 보정 | `simulation/isaac_sim/config/gemini_336l_factory.json` |
| 가상 시작 관절값 | `simulation/isaac_sim/config/lab_home.json` |
| 가상 물체 배치 | `simulation/isaac_sim/config/lab_objects.json` (물체 3개) |
| 카메라 본체 전면 중앙 | 베이스 기준 `(-0.01, 0.76, 1.02)` m, 수직 하향, RGB·Depth 180° 회전 |
| 책상 | 78×250 cm, 두께 3 cm, 상판 Z=0 |
| 작업 영역 | X `−0.365~0.365` m, Y `0.470~1.225` m |

- 가상 장면의 카메라 광학 자세·책상·구조물 15개·작업 영역이 공통 설정과 다르면 시작을 중단한다.
- 계획용 카메라 본체·브래킷은 상자 2개이며 여유는 하드웨어 설정의 `collision_margin_m`이다.
- 설정을 바꾸면 runtime을 재시작하고 새 관측으로 다시 계획한다.

그리퍼 장착·TCP·DB 보정값은 [그리퍼 설정](m1013-gripper-check.md#좌표와-모델)을 따른다.

## 실행

프로젝트 루트에서 실행한다. SAM3·FoundationPose 서버는 [작업 실행](known-object-pick-place.md#실행)을 따르며, 이 환경은 기하 깊이를 쓰므로 FoundationStereo는 필요 없다.
Isaac은 ROS/Pixi 환경이 섞이지 않은 별도 터미널에서 실행하고 출력 폴더는 매번 새 이름을 쓴다.

```bash
export ISAAC_PYTHON="$HOME/isaac-sim/6.2.0/python.sh"   # Isaac Sim 설치 위치에 맞게 지정
"$ISAAC_PYTHON" --no-ros-env simulation/isaac_sim/live_perception.py --lab \
  --output tiptop_outputs/m1013/isaac/lab-fixed-01/runtime
```

호스트 Pixi 터미널에서 관측·인식·계획을 확인한다.

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --camera-url http://127.0.0.1:13345 \
  --perception-mode sam3_reference --reference-scene Lab/Uon_1F_Lab/table \
  --task 'black_pepper_shaker 집고 박스에 놓아' \
  --table-support declared --viewer
```

- `--check`는 연결만 검사하고 `--execute`는 가상 동작까지 실행한다. 가상 실행에는 `--hardware-profile`·`--gateway-url`을 지정하지 않는다.
- 결과는 작업 출력 폴더와 runtime의 `executions/<관측 ID>/execution.json`에서 본다.

저장된 실물 계획을 가상 접촉으로 재생하려면 기존 runtime을 종료하고 실행한다. `plan.json`과 같은 폴더의 `registered-pose.json`이 필요하다.

```bash
"$ISAAC_PYTHON" --no-ros-env simulation/isaac_sim/live_perception.py --lab \
  --replay-plan tiptop_outputs/m1013/tasks/task-NNN/plan.json \
  --port 13349 --output tiptop_outputs/m1013/isaac/task-NNN-contact-01
```

실물 장치나 추론 서버에 연결하지 않으며, CAD와 USD의 경계가 3 mm 이상 다르면 중단한다.

## USD와 시점 저장

기본 장면은 `simulation/isaac_sim/scenes/lab_table_gripper_view.usda`이며 아래 4개 레이어의 참조 구조를 유지한다.

| 레이어 | 역할 |
|---|---|
| `lab_table_gripper_view.usda` | 책상·표시용 로봇·저장 시점 |
| `lab_table_keepout.usda` | 주변 구조물·Left 진입 제외 영역 |
| `lab_table_camera.usda` | 카메라 본체·브래킷·설치 위치 |
| `lab_table_camera_optics.usda` | SDK 값으로 생성한 RGB·Depth·IR 광학 |

- 실행기는 환경만 참조하고 별도의 제어 가능한 로봇을 쓴다. 손가락 180° 보정은 `live_perception.py --lab`이 적용하므로 USD를 직접 연 화면과 다르다.
- 시점은 **Save current view**로 저장하고 `--view simulation/isaac_sim/scenes/lab_pick_view.usda`로 복원한다. 일반 File → Save As로 물리 핸들이 무효화되면 runtime을 재시작한다.

## 실물과 다른 점

- 시작 자세와 물체 배치는 저장된 예제이며 실물 상태와 동기화되지 않는다.
- 깊이는 이상적인 기하 렌더다. 센서 잡음·반사·측정 실패와 실제 SDK 깊이 정렬은 재현하지 않는다.
- 가상 그리퍼의 관절각·5 N·m 토크 제한은 실물 모터 count·전류와 대응이 보정되지 않았다.
- 가상 팔은 링크 중력을 끈 PhysX 위치·속도 제어라 실제 제어기의 추종 오차·마찰·접촉과 다르다.
- 계획기의 보수적 충돌 형상과 PhysX 접촉 형상은 다르다.
- 운영 설정의 `calibrated:true`는 사용자 채택 상태이며 정밀 실측 보정이나 실물 파지 성공을 뜻하지 않는다.

실물 관측·보정은 [실물 준비](m1013-hardware-preparation.md)를 따른다.
