# M1013 실험실 가상 고정 카메라

`live_perception.py --lab`은 현재 실물 운영 설정을 공유하는 Isaac 가상 환경이다.
실물 장치에 연결하지 않으며, 손목 카메라(`wrist`) 예제는 별도로 유지한다.

## 설정

| 항목 | 기준 파일 또는 값 |
|---|---|
| 실물·가상 공통 하드웨어 | `tiptop/config/m1013_fixed_hardware.declared.json` |
| 카메라 설치·영상 회전 | `tiptop/config/m1013_fixed_camera.declared.json` |
| SDK 내부 보정 | `simulation/isaac_sim/config/gemini_336l_factory.json` |
| 가상 시작 관절값 | `simulation/isaac_sim/config/lab_home.json` |
| 가상 물체 배치 | `simulation/isaac_sim/config/lab_objects.json`의 물체 3개 |
| 카메라 본체 전면 중앙 | 베이스 기준 `(-0.01, 0.76, 1.02)m`, 수직 하향, RGB·Depth 180° 회전 |
| 책상·작업 영역 | 상판 78×250cm, Z=0. 작업 영역 X `−0.365~0.365m`, Y `0.470~1.225m` |

가상 장면의 카메라 광학 자세·책상·구조물 15개·작업 영역이 공통 설정과 다르면 시작을 중단한다.
계획용 카메라 본체·브래킷은 실물과 같은 상자 2개와 여유 2mm를 사용한다.
설정을 바꾸면 runtime을 재시작하고 새 관측으로 다시 계획한다.

그리퍼는 몸체 장착 −90°, 손가락 배치 180°, TCP 높이 236.8508mm, DB→TCP 회전 −60°다.
DB 원본은 유지하고 불러올 때 보정한다. 손가락이 120° 간격이므로 `180°−60°=120°`로 DB 배치와 맞는다.
−60°(=300°)는 11월 3일 세 물체의 오프라인 계획 비교 후 사용자가 채택한 임시 고정값이다.
60°·180°도 같은 손가락 배치를 만들지만 몸체·케이블 방향이 다르며, 최소 회전 자동 선택은 구현되지 않았다.

## 실행

프로젝트 루트에서 실행한다. SAM3·FoundationPose 서버는 [기존 실행 절차](known-object-pick-place.md)를 따른다.
이 가상 환경은 기하 깊이를 사용하므로 FoundationStereo 서버가 필요하지 않다.
Isaac은 ROS/Pixi 환경이 섞이지 않은 별도 터미널에서 실행하고, 출력 폴더는 매번 새 이름을 사용한다.

```bash
export ISAAC_PYTHON=/home/uon/isaac-sim/6.2.0/python.sh
"$ISAAC_PYTHON" --no-ros-env simulation/isaac_sim/live_perception.py --lab \
  --output tiptop_outputs/m1013/isaac/lab-fixed-01/runtime
```

호스트 Pixi 터미널에서 관측·인식·계획을 확인한다.

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --camera-url http://127.0.0.1:13345 \
  --perception-mode sam3_reference \
  --reference-scene Lab/Uon_1F_Lab/table \
  --task 'black_pepper_shaker를 집고 제자리에 놓아' \
  --max-approach-tilt-deg 20 --cad-mask-min-iou 0.6 \
  --min-table-clearance-m 0.01 --max-wrist-travel-deg 90 \
  --min-engagement-depth-m 0.02 --table-support declared \
  --viewer
```

연결만 검사하려면 `--check`, 가상 동작까지 실행하려면 `--execute`를 추가한다.
가상 실행에는 `--hardware-profile`·`--gateway-url`을 지정하지 않는다.
동작은 `접근 → 닫기 → 5cm 상승 → 1초 유지 → 그 위치에서 열기 → 시작 관절 자세 복귀`다.
결과는 작업 출력 폴더와 runtime의 `executions/<관측 ID>/execution.json`에서 확인한다.

저장된 실물 계획을 가상 접촉으로 재생하려면 기존 runtime을 종료하고 실행한다.
`plan.json`과 같은 폴더의 `registered-pose.json`이 필요하다.

```bash
"$ISAAC_PYTHON" --no-ros-env simulation/isaac_sim/live_perception.py --lab \
  --replay-plan tiptop_outputs/m1013/tasks/task-025/plan.json \
  --port 13349 --output tiptop_outputs/m1013/isaac/task-025-contact-01
```

이 재생은 실물 장치나 추론 서버에 연결하지 않는다. CAD와 USD의 경계가 3mm 이상 다르면 중단한다.
손목 카메라 예제는 `--lab` 없이 실행하며 `config/wrist_camera.json`을 사용한다.

## USD와 시점 저장

기본 장면은 `simulation/isaac_sim/scenes/lab_table_gripper_view.usda`다.
아래 4개는 역할이 달라 참조 구조를 유지한다.

| 레이어 | 역할 |
|---|---|
| `lab_table_gripper_view.usda` | 책상·표시용 로봇·저장 시점 |
| `lab_table_keepout.usda` | 주변 구조물·Left 진입 제외 영역 |
| `lab_table_camera.usda` | 카메라 본체·브래킷·설치 위치 |
| `lab_table_camera_optics.usda` | SDK 값으로 생성한 RGB·Depth·IR 광학 |

실행기는 환경만 참조하고 별도의 제어 가능한 로봇을 사용한다.
손가락 180° 보정은 `live_perception.py --lab`에서 적용하므로 USD를 직접 연 화면과 구분한다.
시점은 **Save current view**로 저장하고, 다음 실행에 `--view simulation/isaac_sim/scenes/lab_pick_view.usda`로 복원한다.
일반 File → Save As로 물리 핸들이 무효화되면 runtime을 재시작해야 한다.

## 실물과 남아 있는 차이

- 시작 자세는 10월 29일 저장한 값이고 물체는 예제 배치다. 현재 실물 자세·배치와 자동 동기화되지 않는다.
- 깊이는 이상적인 기하 렌더다. 센서 잡음·반사·측정 실패와 실제 SDK 깊이 정렬을 그대로 재현하지 않는다.
- 가상 그리퍼의 관절각·5Nm 토크 제한과 실물 모터 count·전류의 대응은 미보정이다.
- 가상 팔은 링크 중력을 끈 PhysX 위치·속도 제어를 사용한다. 실제 제어기의 추종 오차·마찰·접촉과 다르다.
- 계획기의 보수적 충돌 상자·구와 PhysX 접촉 형상은 다르며, PhysX 자기충돌도 계획기 검사와 구분한다.
- 운영 설정의 `calibrated:true`는 사용자 채택 상태다. 정밀 실측 보정 완료나 실물 파지 성공을 뜻하지 않는다.

실물 관측·보정은 [하드웨어 준비](m1013-hardware-preparation.md)를 따른다.
후속 코드 정리에서는 Doosan 모듈의 카메라 영상 회전·공통 검증을 카메라 공통 모듈로 옮기는 작업이 남아 있다.
